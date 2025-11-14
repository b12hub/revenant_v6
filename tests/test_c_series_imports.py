import pytest
from core.agent_base import RevenantAgentBase

class TestCSeriesImports:

    def test_import_pipeline_agent(self):
        from agents.c_series.simulation_agent import SimulationAgent
        assert issubclass(SimulationAgent, RevenantAgentBase)
        assert SimulationAgent.metadata["series"] == "c_series"

    def test_import_orchestrator_agent(self):
        from agents.c_series.decision_making_agent import DecisionMakingAgent
        assert issubclass(DecisionMakingAgent, RevenantAgentBase)
        assert DecisionMakingAgent.metadata["series"] == "c_series"

    def test_import_registry_sync_agent(self):
        from agents.c_series.knowledge_graph_agent import KnowledgeGraphAgent
        assert issubclass(KnowledgeGraphAgent, RevenantAgentBase)
        assert KnowledgeGraphAgent.metadata["series"] == "c_series"

    def test_import_infra_monitor_agent(self):
        from agents.c_series.reasoning_agent import ReasoningAgent
        assert issubclass(ReasoningAgent, RevenantAgentBase)
        assert ReasoningAgent.metadata["series"] == "c_series"

    def test_import_scheduler_agent(self):
        from agents.c_series.research_agent import ResearchAgent
        assert issubclass(ResearchAgent, RevenantAgentBase)
        assert ResearchAgent.metadata["series"] == "c_series"
