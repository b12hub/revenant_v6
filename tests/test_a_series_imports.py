"""
Unit tests for A-Series agent imports and registry discovery.
Ensures all agents are importable and properly inherit from RevenantAgentBase.
"""

import pytest
import sys
from pathlib import Path

# Ensure project_root is in-path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.agent_base import RevenantAgentBase

pytest_plugins = ("pytest_asyncio",)

class TestASeriesImports:
    """Test suite for A-Series agent imports and basic structure."""

    def test_import_security_agent(self):
        """Test SecurityAgent can be imported and is properly structured."""
        from agents.a_series.security_agent import SecurityAgent

        assert issubclass(SecurityAgent, RevenantAgentBase)
        assert hasattr(SecurityAgent, 'metadata')
        assert SecurityAgent.metadata['series'] == 'a_series'
        assert SecurityAgent.metadata['name'] == 'SecurityAgent'

        # Test instantiation
        agent = SecurityAgent()
        assert agent.name == "SecurityAgent"
        assert hasattr(agent, 'run')
        assert hasattr(agent, 'setup')

    def test_import_analytics_agent(self):
        """Test AnalyticsAgent can be imported and is properly structured."""
        from agents.a_series.analytics_agent import AnalyticsAgent

        assert issubclass(AnalyticsAgent, RevenantAgentBase)
        assert hasattr(AnalyticsAgent, 'metadata')
        assert AnalyticsAgent.metadata['series'] == 'a_series'

        agent = AnalyticsAgent()
        assert agent.name == "AnalyticsAgent"

    def test_import_context_agent(self):
        """Test ContextAgent can be imported and is properly structured."""
        from agents.a_series.context_agent import ContextAgent

        assert issubclass(ContextAgent, RevenantAgentBase)
        assert hasattr(ContextAgent, 'metadata')
        assert ContextAgent.metadata['series'] == 'a_series'

        agent = ContextAgent()
        assert agent.name == "ContextAgent"

    def test_import_optimization_agent(self):
        """Test OptimizationAgent can be imported and is properly structured."""
        from agents.a_series.optimization_agent import OptimizationAgent

        assert issubclass(OptimizationAgent, RevenantAgentBase)
        assert hasattr(OptimizationAgent, 'metadata')
        assert OptimizationAgent.metadata['series'] == 'a_series'

        agent = OptimizationAgent()
        assert agent.name == "OptimizationAgent"

    def test_import_planner_agent(self):
        """Test PlannerAgent can be imported and is properly structured."""
        from agents.a_series.planner_agent import PlannerAgent

        assert issubclass(PlannerAgent, RevenantAgentBase)
        assert hasattr(PlannerAgent, 'metadata')
        assert PlannerAgent.metadata['series'] == 'a_series'

        agent = PlannerAgent()
        assert agent.name == "PlannerAgent"

    def test_import_data_miner_agent(self):
        """Test DataMinerAgent can be imported and is properly structured."""
        from agents.a_series.data_miner_agent import DataMinerAgent

        assert issubclass(DataMinerAgent, RevenantAgentBase)
        assert hasattr(DataMinerAgent, 'metadata')
        assert DataMinerAgent.metadata['series'] == 'a_series'

        agent = DataMinerAgent()
        assert agent.name == "DataMinerAgent"

    def test_import_evolution_agent(self):
        """Test EvolutionAgent can be imported and is properly structured."""
        from agents.a_series.evolution_agent import EvolutionAgent

        assert issubclass(EvolutionAgent, RevenantAgentBase)
        assert hasattr(EvolutionAgent, 'metadata')
        assert EvolutionAgent.metadata['series'] == 'a_series'

        agent = EvolutionAgent()
        assert agent.name == "EvolutionAgent"

    def test_all_agents_have_metadata(self):
        """Test that all agents have proper metadata structure."""
        from agents.a_series import (
            SecurityAgent, AnalyticsAgent, ContextAgent,
            OptimizationAgent, PlannerAgent, DataMinerAgent, EvolutionAgent
        )

        agents = [
            SecurityAgent, AnalyticsAgent, ContextAgent,
            OptimizationAgent, PlannerAgent, DataMinerAgent, EvolutionAgent
        ]

        for agent_class in agents:
            assert hasattr(agent_class, 'metadata'), f"{agent_class.__name__} missing metadata"
            metadata = agent_class.metadata

            # Check required metadata fields
            assert 'name' in metadata, f"{agent_class.__name__} missing 'name' in metadata"
            assert 'series' in metadata, f"{agent_class.__name__} missing 'series' in metadata"
            assert 'version' in metadata, f"{agent_class.__name__} missing 'version' in metadata"
            assert 'description' in metadata, f"{agent_class.__name__} missing 'description' in metadata"

            assert metadata['series'] == 'a_series', f"{agent_class.__name__} has incorrect series"

    @pytest.mark.asyncio
    async def test_all_agents_have_async_run(self):
        """Test that all agents have async run method."""
        from agents.a_series import (
            SecurityAgent, AnalyticsAgent, ContextAgent,
            OptimizationAgent, PlannerAgent, DataMinerAgent, EvolutionAgent
        )

        import inspect

        agents = [
            SecurityAgent, AnalyticsAgent, ContextAgent,
            OptimizationAgent, PlannerAgent, DataMinerAgent, EvolutionAgent
        ]

        for agent_class in agents:
            agent = agent_class()
            assert hasattr(agent, 'run'), f"{agent_class.__name__} missing run method"
            assert inspect.iscoroutinefunction(agent.run), f"{agent_class.__name__}.run is not async"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])