# tests/test_d_series_orchestrator.py
import pytest
import asyncio
from unittest.mock import  patch


from agents.d_series.orchestrator import (
    DSeriesOrchestrator,
    NodeExecutionError,
    CircuitOpenError
)


class MockAgent:
    def __init__(self, response_data=None, should_fail=False):
        self.response_data = response_data or {}
        self.should_fail = should_fail
        self.run_call_count = 0

    async def run(self, input_data):
        self.run_call_count += 1
        await asyncio.sleep(0.01)  # Simulate some work

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

    def add_agent(self, agent_id, agent_class):
        self.agents[agent_id] = agent_class

    def get_agent(self, agent_id):
        return self.agents.get(agent_id)


@pytest.fixture
def mock_workflow_spec():
    """Minimal workflow spec for testing"""
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
    """Create orchestrator instance with mocked workflow"""
    with patch.object(DSeriesOrchestrator, '_load_workflow_spec'):
        orchestrator = DSeriesOrchestrator()
        orchestrator.workflow_spec = mock_workflow_spec
        return orchestrator


@pytest.mark.asyncio
async def test_orchestrator_initialization():
    """Test orchestrator initialization and workflow loading"""
    with patch('pathlib.Path.exists', return_value=True), \
            patch('pathlib.Path.open'), \
            patch('json.load', return_value={"name": "test", "version": "1.0", "nodes": [], "edges": []}):
        orchestrator = DSeriesOrchestrator()
        assert orchestrator.workflow_spec is not None
        assert orchestrator.circuit_breaker is not None


@pytest.mark.asyncio
async def test_node_sequencing(orchestrator):
    """Test that nodes execute in proper sequence"""
    registry = MockRegistry()
    registry.add_agent("TestAgent1", lambda: MockAgent({"result": "agent1"}))
    registry.add_agent("TestAgent2", lambda: MockAgent({"result": "agent2"}))
    registry.add_agent("EvaluatorAgent", lambda: MockAgent({"evaluation": True}))

    orchestrator.registry = registry

    input_data = {
        "trace_id": "test_trace_123",
        "workflow_data": {"test_key": "test_value"}
    }

    result = await orchestrator.run(input_data)

    assert result["trace_id"] == "test_trace_123"
    assert "node_results" in result
    assert "metrics" in result

    # Check that all nodes were executed
    assert "node1" in result["node_results"]
    assert "node2" in result["node_results"]
    assert "node3" in result["node_results"]

    # Verify execution order through metrics
    metrics = result["metrics"]
    assert metrics["node1"]["status"] == "completed"
    assert metrics["node2"]["status"] == "completed"
    assert metrics["node3"]["status"] == "completed"


@pytest.mark.asyncio
async def test_trace_id_propagation(orchestrator):
    """Test that trace_id propagates to all nodes"""
    mock_agent1 = MockAgent()
    mock_agent2 = MockAgent()

    registry = MockRegistry()
    registry.add_agent("TestAgent1", lambda: mock_agent1)
    registry.add_agent("TestAgent2", lambda: mock_agent2)
    registry.add_agent("EvaluatorAgent", lambda: MockAgent())

    orchestrator.registry = registry

    trace_id = "test_trace_propagation"
    await orchestrator.run({"trace_id": trace_id})

    # Verify agents were called with trace_id
    assert mock_agent1.run_call_count == 1
    assert mock_agent2.run_call_count == 1


@pytest.mark.asyncio
async def test_retry_behavior(orchestrator):
    """Test retry behavior for failing nodes"""
    # Create agent that fails twice then succeeds
    call_count = 0

    class FlakyAgent:
        async def run(self, input_data):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)

            if call_count <= 2:
                raise Exception("Temporary failure")
            return {"success": True}

    registry = MockRegistry()
    registry.add_agent("TestAgent1", FlakyAgent)
    # Remove other nodes for this test
    orchestrator.workflow_spec["nodes"] = [orchestrator.workflow_spec["nodes"][0]]
    orchestrator.workflow_spec["edges"] = []

    orchestrator.registry = registry

    result = await orchestrator.run({"trace_id": "test_retry"})

    assert call_count == 3  # Initial + 2 retries
    assert result["metrics"]["node1"]["status"] == "completed"
    assert result["metrics"]["node1"]["retries"] == 2


@pytest.mark.asyncio
async def test_evaluator_short_circuit(orchestrator):
    """Test evaluator-driven workflow decisions"""

    class ShortCircuitEvaluator:
        async def run(self, input_data):
            return {
                "should_continue": False,
                "reason": "Test short circuit",
                "rollback_required": True
            }

    registry = MockRegistry()
    registry.add_agent("TestAgent1", lambda: MockAgent())
    registry.add_agent("TestAgent2", lambda: MockAgent())
    registry.add_agent("EvaluatorAgent", ShortCircuitEvaluator)

    orchestrator.registry = registry

    result = await orchestrator.run({"trace_id": "test_short_circuit"})

    # Even with short circuit, all scheduled nodes should complete
    assert result["metrics"]["node1"]["status"] == "completed"
    assert result["metrics"]["node2"]["status"] == "completed"
    assert result["metrics"]["node3"]["status"] == "completed"


@pytest.mark.asyncio
async def test_circuit_breaker(orchestrator):
    """Test circuit breaker functionality"""
    failing_agent = MockAgent(should_fail=True)

    registry = MockRegistry()
    registry.add_agent("TestAgent1", lambda: failing_agent)
    # Single node test
    orchestrator.workflow_spec["nodes"] = [orchestrator.workflow_spec["nodes"][0]]
    orchestrator.workflow_spec["edges"] = []

    orchestrator.registry = registry

    # First execution should fail after retries
    with pytest.raises(NodeExecutionError):
        await orchestrator.run({"trace_id": "test_circuit_1"})

    # Subsequent execution should hit circuit breaker
    with pytest.raises(CircuitOpenError):
        await orchestrator.run({"trace_id": "test_circuit_2"})


@pytest.mark.asyncio
async def test_concurrent_execution(orchestrator):
    """Test concurrent execution of independent nodes"""
    # Modify workflow to have independent nodes
    orchestrator.workflow_spec["nodes"] = [
        {
            "id": "independent_1",
            "type": "call_agent",
            "config": {"agent_id": "Agent1", "timeout": 1}
        },
        {
            "id": "independent_2",
            "type": "call_agent",
            "config": {"agent_id": "Agent2", "timeout": 1}
        }
    ]
    orchestrator.workflow_spec["edges"] = []  # No dependencies

    execution_order = []

    class OrderTrackingAgent:
        def __init__(self, name):
            self.name = name

        async def run(self, input_data):
            execution_order.append(self.name)
            await asyncio.sleep(0.1)
            return {"agent": self.name}

    registry = MockRegistry()
    registry.add_agent("Agent1", lambda: OrderTrackingAgent("agent1"))
    registry.add_agent("Agent2", lambda: OrderTrackingAgent("agent2"))

    orchestrator.registry = registry

    await orchestrator.run({"trace_id": "test_concurrent"})

    # Both agents should start around the same time
    assert len(execution_order) == 2
    assert "agent1" in execution_order
    assert "agent2" in execution_order


@pytest.mark.asyncio
async def test_memory_operations(orchestrator):
    """Test memory read/write operations"""
    orchestrator.workflow_spec["nodes"] = [
        {
            "id": "memory_store",
            "type": "store_memory",
            "config": {"key": "test_data"}
        }
    ]
    orchestrator.workflow_spec["edges"] = []

    class MockMemoryAgent:
        async def run(self, input_data):
            return {
                "operation": input_data.get("operation"),
                "key": input_data.get("key"),
                "success": True
            }

    registry = MockRegistry()
    registry.add_agent("MemoryAgent", MockMemoryAgent)
    orchestrator.registry = registry

    result = await orchestrator.run({
        "trace_id": "test_memory",
        "workflow_data": {"important": "data"}
    })

    assert result["node_results"]["memory_store"]["success"] is True
    assert result["node_results"]["memory_store"]["operation"] == "store"


@pytest.mark.asyncio
async def test_fusion_operations(orchestrator):
    """Test fusion agent integration"""
    orchestrator.workflow_spec["nodes"] = [
        {
            "id": "fusion_node",
            "type": "fuse",
            "config": {
                "sources": ["source1", "source2"],
                "strategy": "merge"
            }
        }
    ]
    orchestrator.workflow_spec["edges"] = []

    class MockFusionAgent:
        async def run(self, input_data):
            return {
                "fused_result": "combined_data",
                "sources": input_data.get("sources", []),
                "strategy": input_data.get("strategy")
            }

    registry = MockRegistry()
    registry.add_agent("FusionAgent", MockFusionAgent)
    orchestrator.registry = registry

    result = await orchestrator.run({"trace_id": "test_fusion"})

    fusion_result = result["node_results"]["fusion_node"]
    assert fusion_result["fused_result"] == "combined_data"
    assert fusion_result["sources"] == ["source1", "source2"]
    assert fusion_result["strategy"] == "merge"


@pytest.mark.asyncio
async def test_global_timeout(orchestrator):
    """Test global workflow timeout"""
    orchestrator.global_timeout = 0.1  # Very short timeout

    class SlowAgent:
        async def run(self, input_data):
            await asyncio.sleep(1)  # Longer than timeout
            return {"too_slow": True}

    registry = MockRegistry()
    registry.add_agent("TestAgent1", SlowAgent)
    orchestrator.workflow_spec["nodes"] = [orchestrator.workflow_spec["nodes"][0]]
    orchestrator.workflow_spec["edges"] = []

    orchestrator.registry = registry

    with pytest.raises(NodeExecutionError) as exc_info:
        await orchestrator.run({"trace_id": "test_timeout"})

    assert "global timeout" in str(exc_info.value).lower()


def test_dependency_graph_building(orchestrator):
    """Test dependency graph construction"""
    edges = [
        {"source": "A", "target": "B"},
        {"source": "B", "target": "C"},
        {"source": "A", "target": "C"}
    ]

    dependencies = orchestrator._build_dependency_graph(edges)

    assert dependencies["B"] == ["A"]
    assert dependencies["C"] == ["B", "A"]  # Both B and A are dependencies of C