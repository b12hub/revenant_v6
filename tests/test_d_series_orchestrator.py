import pytest
import asyncio
from unittest.mock import patch
import logging

# -------------------------------------------------------
#  GLOBAL LOGGING FOR FULL TRACE VISIBILITY
# -------------------------------------------------------
logging.basicConfig(level=logging.DEBUG)

@pytest.fixture(autouse=True)
def verbose_logs():
    """Force DEBUG logs + print everything pytest normally hides."""
    logging.getLogger().setLevel(logging.DEBUG)
    print("\n[TEST-TRACE] verbose logging enabled\n")
    yield


from agents.d_series.orchestrator import (
    DSeriesOrchestrator,
    NodeExecutionError,
    CircuitOpenError
)


# -------------------------------------------------------
# MOCK AGENT + REGISTRY
# -------------------------------------------------------

class MockAgent:
    def __init__(self, response_data=None, should_fail=False):
        self.response_data = response_data or {}
        self.should_fail = should_fail
        self.run_call_count = 0

    async def run(self, input_data):
        self.run_call_count += 1
        print(f"[AGENT] run() called → count={self.run_call_count}, should_fail={self.should_fail}")

        await asyncio.sleep(0.01)

        if self.should_fail:
            print("[AGENT] raising Exception('Mock agent failure')")
            raise Exception("Mock agent failure")

        return {
            **self.response_data,
            "trace_id": input_data.get("trace_id"),
            "agent_called": True
        }


class MockRegistry:
    def __init__(self):
        self.agents = {}

    def add_agent(self, agent_id, agent_func):
        print(f"[REGISTRY] add_agent({agent_id})")
        self.agents[agent_id] = agent_func

    def get_agent(self, agent_id):
        print(f"[REGISTRY] get_agent({agent_id})")
        return self.agents.get(agent_id)


# -------------------------------------------------------
# ORCHESTRATOR FIXTURE
# -------------------------------------------------------

@pytest.fixture
def mock_workflow_spec():
    return {
        "name": "Test Workflow",
        "version": "1.0.0",
        "nodes": [
            {
                "id": "node1",
                "type": "call_agent",
                "config": {
                    "agent_id": "TestAgent1",
                    "timeout": 10,
                    "max_retries": 2
                }
            },
            {
                "id": "node2",
                "type": "call_agent",
                "config": {
                    "agent_id": "TestAgent2",
                    "timeout": 10
                }
            },
            {
                "id": "node3",
                "type": "evaluate",
                "config": {
                    "condition": "test_condition"
                }
            }
        ],
        "edges": [
            {"source": "node1", "target": "node2"},
            {"source": "node2", "target": "node3"}
        ]
    }


@pytest.fixture
def orchestrator(mock_workflow_spec):
    with patch.object(DSeriesOrchestrator, "_load_workflow_spec"):
        orch = DSeriesOrchestrator()
        orch.workflow_spec = mock_workflow_spec
        print("\n[ORCH] New orchestrator instance created\n")
        return orch


# -------------------------------------------------------
# BASIC WORKFLOW TESTS
# -------------------------------------------------------

@pytest.mark.asyncio
async def test_orchestrator_initialization():
    print("\n[TEST] test_orchestrator_initialization START\n")

    with patch("pathlib.Path.exists", return_value=True), \
         patch("pathlib.Path.open"), \
         patch("json.load", return_value={"name": "X", "version": "1", "nodes": [], "edges": []}):

        orch = DSeriesOrchestrator()
        assert orch.workflow_spec is not None
        assert orch.circuit_breaker is not None

    print("[TEST] test_orchestrator_initialization END")


@pytest.mark.asyncio
async def test_node_sequencing(orchestrator):
    print("\n[TEST] test_node_sequencing START\n")

    registry = MockRegistry()
    registry.add_agent("TestAgent1", lambda: MockAgent({"result": "a1"}))
    registry.add_agent("TestAgent2", lambda: MockAgent({"result": "a2"}))
    registry.add_agent("EvaluatorAgent", lambda: MockAgent({"eval": True}))
    orchestrator.registry = registry

    result = await orchestrator.run({
        "trace_id": "T1",
        "workflow_data": {"k": "v"}
    })

    assert list(result["node_results"].keys()) == ["node1", "node2", "node3"]
    assert result["metrics"]["node1"]["status"] == "completed"
    assert result["metrics"]["node2"]["status"] == "completed"
    assert result["metrics"]["node3"]["status"] == "completed"

    print("[TEST] test_node_sequencing END")


@pytest.mark.asyncio
async def test_trace_id_propagation(orchestrator):
    print("\n[TEST] test_trace_id_propagation START\n")

    a1 = MockAgent()
    a2 = MockAgent()

    reg = MockRegistry()
    reg.add_agent("TestAgent1", lambda: a1)
    reg.add_agent("TestAgent2", lambda: a2)
    reg.add_agent("EvaluatorAgent", lambda: MockAgent())
    orchestrator.registry = reg

    await orchestrator.run({"trace_id": "XYZ"})

    assert a1.run_call_count == 1
    assert a2.run_call_count == 1

    print("[TEST] test_trace_id_propagation END")


# -------------------------------------------------------
# RETRY BEHAVIOR
# -------------------------------------------------------

@pytest.mark.asyncio
async def test_retry_behavior(orchestrator):
    print("\n[TEST] test_retry_behavior START\n")

    call_count = 0

    class FlakyAgent:
        async def run(self, input_data):
            nonlocal call_count
            call_count += 1
            print(f"[FLAKY] call_count={call_count}")
            await asyncio.sleep(0.01)
            if call_count <= 2:
                raise Exception("Temporary fail")
            return {"ok": True}

    reg = MockRegistry()
    reg.add_agent("TestAgent1", FlakyAgent)
    orchestrator.registry = reg

    orchestrator.workflow_spec["nodes"] = [orchestrator.workflow_spec["nodes"][0]]
    orchestrator.workflow_spec["edges"] = []

    result = await orchestrator.run({"trace_id": "R"})

    assert call_count == 3
    assert result["metrics"]["node1"]["status"] == "completed"

    print("[TEST] test_retry_behavior END")


# -------------------------------------------------------
# CIRCUIT BREAKER — FULL TRACE
# -------------------------------------------------------

@pytest.mark.asyncio
async def test_circuit_breaker(orchestrator):
    print("\n\n===================== CIRCUIT BREAKER TEST TRACE =====================\n")

    failing = MockAgent(should_fail=True)

    reg = MockRegistry()
    reg.add_agent("TestAgent1", lambda: failing)
    orchestrator.registry = reg

    orchestrator.workflow_spec["nodes"] = [orchestrator.workflow_spec["nodes"][0]]
    orchestrator.workflow_spec["edges"] = []

    print("\n--- FIRST RUN (must raise NodeExecutionError) ---\n")

    with pytest.raises(NodeExecutionError):
        await orchestrator.run({"trace_id": "CB1"})

    print("\n--- SECOND RUN (must raise CircuitOpenError) ---\n")

    with pytest.raises(CircuitOpenError):
        await orchestrator.run({"trace_id": "CB2"})

    print("\n===================== END CIRCUIT BREAKER TEST =====================\n")


# -------------------------------------------------------
# CONCURRENCY
# -------------------------------------------------------

@pytest.mark.asyncio
async def test_concurrent_execution(orchestrator):
    print("\n[TEST] test_concurrent_execution START\n")

    orchestrator.workflow_spec["nodes"] = [
        {"id": "A", "type": "call_agent", "config": {"agent_id": "X"}},
        {"id": "B", "type": "call_agent", "config": {"agent_id": "Y"}},
    ]
    orchestrator.workflow_spec["edges"] = []

    order = []

    class A:
        async def run(self, _):
            print("[AGENT A] running")
            order.append("A")
            await asyncio.sleep(0.05)

    class B:
        async def run(self, _):
            print("[AGENT B] running")
            order.append("B")
            await asyncio.sleep(0.05)

    reg = MockRegistry()
    reg.add_agent("X", lambda: A())
    reg.add_agent("Y", lambda: B())
    orchestrator.registry = reg

    await orchestrator.run({"trace_id": "CC"})

    assert len(order) == 2
    assert "A" in order and "B" in order

    print("[TEST] test_concurrent_execution END")


# -------------------------------------------------------
# MEMORY / FUSION / TIMEOUT
# -------------------------------------------------------

@pytest.mark.asyncio
async def test_memory_operations(orchestrator):
    print("\n[TEST] test_memory_operations START\n")

    orchestrator.workflow_spec["nodes"] = [
        {"id": "mem", "type": "store_memory", "config": {"key": "t"}}
    ]
    orchestrator.workflow_spec["edges"] = []

    class Mem:
        async def run(self, data):
            print("[MEM] memory operation")
            return {"ok": True, "operation": data["operation"]}

    reg = MockRegistry()
    reg.add_agent("MemoryAgent", Mem)
    orchestrator.registry = reg

    result = await orchestrator.run({"trace_id": "M", "workflow_data": {"x": "y"}})

    assert result["node_results"]["mem"]["ok"] is True

    print("[TEST] test_memory_operations END")


@pytest.mark.asyncio
async def test_fusion_operations(orchestrator):
    print("\n[TEST] test_fusion_operations START\n")

    orchestrator.workflow_spec["nodes"] = [
        {"id": "f", "type": "fuse", "config": {"sources": ["a"], "strategy": "merge"}}
    ]
    orchestrator.workflow_spec["edges"] = []

    class Fusion:
        async def run(self, data):
            print("[FUSION] merging sources")
            return {"fusion": "done", "sources": data["sources"]}

    reg = MockRegistry()
    reg.add_agent("FusionAgent", Fusion)
    orchestrator.registry = reg

    res = await orchestrator.run({"trace_id": "F"})
    assert res["node_results"]["f"]["fusion"] == "done"

    print("[TEST] test_fusion_operations END")


@pytest.mark.asyncio
async def test_global_timeout(orchestrator):
    print("\n[TEST] test_global_timeout START\n")

    orchestrator.global_timeout = 0.1

    class Slow:
        async def run(self, data):
            print("[SLOW] sleeping…")
            await asyncio.sleep(1)

    reg = MockRegistry()
    reg.add_agent("TestAgent1", Slow)
    orchestrator.registry = reg

    orchestrator.workflow_spec["nodes"] = [orchestrator.workflow_spec["nodes"][0]]
    orchestrator.workflow_spec["edges"] = []

    with pytest.raises(NodeExecutionError) as ex:
        await orchestrator.run({"trace_id": "T"})
    assert "timeout" in str(ex.value).lower()

    print("[TEST] test_global_timeout END")


def test_dependency_graph_building(orchestrator):
    print("\n[TEST] test_dependency_graph_building START\n")

    edges = [
        {"source": "A", "target": "B"},
        {"source": "B", "target": "C"},
        {"source": "A", "target": "C"}
    ]

    deps = orchestrator._build_dependency_graph(edges)

    assert deps["B"] == ["A"]
    assert set(deps["C"]) == {"B", "A"}

    print("[TEST] test_dependency_graph_building END")
