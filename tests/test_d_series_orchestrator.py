import pytest
import asyncio
from unittest.mock import patch

from agents.d_series.orchestrator import (
    DSeriesOrchestrator,
    NodeExecutionError,
    CircuitOpenError
)

# pytestmark = pytest.mark.skip(reason="Legacy circuit breaker behavior not used in MetaCore v6+")


class MockAgent:
    def __init__(self, response_data=None, should_fail=False):
        self.response_data = response_data or {}
        self.should_fail = should_fail
        self.run_call_count = 0

    async def run(self, input_data):
        self.run_call_count += 1
        await asyncio.sleep(0.01)

        if self.should_fail:
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
        self.agents[agent_id] = agent_func

    def get_agent(self, agent_id):
        return self.agents.get(agent_id)


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
        return orch


# --------------------------
# BASIC WORKFLOW TESTS
# --------------------------

@pytest.mark.asyncio
async def test_orchestrator_initialization():
    with patch("pathlib.Path.exists", return_value=True), \
         patch("pathlib.Path.open"), \
         patch("json.load", return_value={"name": "X", "version": "1", "nodes": [], "edges": []}):

        orch = DSeriesOrchestrator()
        assert orch.workflow_spec is not None
        assert orch.circuit_breaker is not None


@pytest.mark.asyncio
async def test_node_sequencing(orchestrator):
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


@pytest.mark.asyncio
async def test_trace_id_propagation(orchestrator):
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


# --------------------------
# RETRY BEHAVIOR
# --------------------------

@pytest.mark.asyncio
async def test_retry_behavior(orchestrator):
    call_count = 0

    class FlakyAgent:
        async def run(self, input_data):
            nonlocal call_count
            call_count += 1
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


# --------------------------
# CIRCUIT BREAKER TEST (FIXED)
# --------------------------

@pytest.mark.asyncio
# @pytest.mark.skip(reason="Legacy behavior no longer supported")
async def test_circuit_breaker(orchestrator):
    failing = MockAgent(should_fail=True)

    reg = MockRegistry()
    reg.add_agent("TestAgent1", lambda: failing)
    orchestrator.registry = reg

    orchestrator.workflow_spec["nodes"] = [orchestrator.workflow_spec["nodes"][0]]
    orchestrator.workflow_spec["edges"] = []

    # 1️⃣ FIRST RUN — must fail as NodeExecutionError
    with pytest.raises(NodeExecutionError):
        await orchestrator.run({"trace_id": "CB1"})

    # 2️⃣ SECOND RUN — circuit MUST be open → CircuitOpenError
    with pytest.raises(CircuitOpenError):
        await orchestrator.run({"trace_id": "CB2"})


# --------------------------
# CONCURRENCY
# --------------------------

@pytest.mark.asyncio
async def test_concurrent_execution(orchestrator):
    orchestrator.workflow_spec["nodes"] = [
        {"id": "A", "type": "call_agent", "config": {"agent_id": "X"}},
        {"id": "B", "type": "call_agent", "config": {"agent_id": "Y"}},
    ]
    orchestrator.workflow_spec["edges"] = []

    order = []

    class A:
        async def run(self, _):
            order.append("A")
            await asyncio.sleep(0.05)

    class B:
        async def run(self, _):
            order.append("B")
            await asyncio.sleep(0.05)

    reg = MockRegistry()
    reg.add_agent("X", lambda: A())
    reg.add_agent("Y", lambda: B())
    orchestrator.registry = reg

    await orchestrator.run({"trace_id": "CC"})

    assert len(order) == 2
    assert "A" in order and "B" in order


# --------------------------
# MEMORY, FUSION, TIMEOUT
# --------------------------

@pytest.mark.asyncio
async def test_memory_operations(orchestrator):
    orchestrator.workflow_spec["nodes"] = [
        {"id": "mem", "type": "store_memory", "config": {"key": "t"}}
    ]
    orchestrator.workflow_spec["edges"] = []

    class Mem:
        async def run(self, data):
            return {"ok": True, "operation": data["operation"]}

    reg = MockRegistry()
    reg.add_agent("MemoryAgent", Mem)
    orchestrator.registry = reg

    result = await orchestrator.run({"trace_id": "M", "workflow_data": {"x": "y"}})

    assert result["node_results"]["mem"]["ok"] is True


@pytest.mark.asyncio
async def test_fusion_operations(orchestrator):
    orchestrator.workflow_spec["nodes"] = [
        {"id": "f", "type": "fuse", "config": {"sources": ["a"], "strategy": "merge"}}
    ]
    orchestrator.workflow_spec["edges"] = []

    class Fusion:
        async def run(self, data):
            return {"fusion": "done", "sources": data["sources"]}

    reg = MockRegistry()
    reg.add_agent("FusionAgent", Fusion)
    orchestrator.registry = reg

    res = await orchestrator.run({"trace_id": "F"})
    assert res["node_results"]["f"]["fusion"] == "done"


@pytest.mark.asyncio
async def test_global_timeout(orchestrator):
    orchestrator.global_timeout = 0.1

    class Slow:
        async def run(self, data):
            await asyncio.sleep(1)

    reg = MockRegistry()
    reg.add_agent("TestAgent1", Slow)
    orchestrator.registry = reg

    orchestrator.workflow_spec["nodes"] = [orchestrator.workflow_spec["nodes"][0]]
    orchestrator.workflow_spec["edges"] = []

    with pytest.raises(NodeExecutionError) as ex:
        await orchestrator.run({"trace_id": "T"})
    assert "timeout" in str(ex.value).lower()


def test_dependency_graph_building(orchestrator):
    edges = [
        {"source": "A", "target": "B"},
        {"source": "B", "target": "C"},
        {"source": "A", "target": "C"}
    ]

    deps = orchestrator._build_dependency_graph(edges)

    assert deps["B"] == ["A"]
    assert set(deps["C"]) == {"B", "A"}