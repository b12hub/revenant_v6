from core.agent_base import RevenantAgentBase

class TestDSeriesImports:

    def test_import_memory_agent(self):
        from agents.d_series.memory_agent import MemoryAgent
        assert issubclass(MemoryAgent, RevenantAgentBase)
        assert MemoryAgent.metadata["series"] == "d_series"

    def test_import_planning_agent(self):
        from agents.d_series.coordinator_agent import CoordinatorAgent
        assert issubclass(CoordinatorAgent, RevenantAgentBase)
        assert CoordinatorAgent.metadata["series"] == "d_series"

    def test_import_security_agent(self):
        from agents.d_series.evaluator_agent import EvaluatorAgent
        assert issubclass(EvaluatorAgent, RevenantAgentBase)
        assert EvaluatorAgent.metadata["series"] == "d_series"

    def test_import_audit_agent(self):
        from agents.d_series.fusion_agent import FusionAgent
        assert issubclass(FusionAgent, RevenantAgentBase)
        assert FusionAgent.metadata["series"] == "d_series"
