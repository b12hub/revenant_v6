import pytest
import inspect
from core.agent_base import RevenantAgentBase


class TestBSeriesImports:
    """Validate all B-series agents load and implement async run() correctly."""

    def test_import_data_science_agent(self):
        from agents.b_series.data_science_agent import DataScienceAgent
        assert issubclass(DataScienceAgent, RevenantAgentBase)
        assert DataScienceAgent.metadata["series"] == "b_series"
        assert hasattr(DataScienceAgent, "run")

    def test_import_legal_agent(self):
        from agents.b_series.legal_agent import LegalAgent
        assert issubclass(LegalAgent, RevenantAgentBase)
        assert LegalAgent.metadata["series"] == "b_series"
        assert hasattr(LegalAgent, "run")

    def test_import_hr_recruitment_agent(self):
        from agents.b_series.hr_recruitment_agent import HRRecruitmentAgent
        assert issubclass(HRRecruitmentAgent, RevenantAgentBase)
        assert HRRecruitmentAgent.metadata["series"] == "b_series"
        assert hasattr(HRRecruitmentAgent, "run")

    def test_import_blockchain_agent(self):
        from agents.b_series.blockchain_agent import BlockchainAgent
        assert issubclass(BlockchainAgent, RevenantAgentBase)
        assert BlockchainAgent.metadata["series"] == "b_series"
        assert hasattr(BlockchainAgent, "run")

    def test_import_iot_agent(self):
        from agents.b_series.iot_agent import IoTAgent
        assert issubclass(IoTAgent, RevenantAgentBase)
        assert IoTAgent.metadata["series"] == "b_series"
        assert hasattr(IoTAgent, "run")

    def test_import_arvr_agent(self):
        from agents.b_series.arvr_agent import ARVRAgent
        assert issubclass(ARVRAgent, RevenantAgentBase)
        assert ARVRAgent.metadata["series"] == "b_series"
        assert hasattr(ARVRAgent, "run")

    @pytest.mark.asyncio
    async def test_all_agents_have_async_run(self):
        """Ensure every B-series agent implements an async run()"""
        import agents.b_series as b_series
        for name in dir(b_series):
            if name.startswith("_"):
                continue
            obj = getattr(b_series, name)
            if inspect.isclass(obj) and issubclass(obj, RevenantAgentBase) and obj is not RevenantAgentBase:
                assert inspect.iscoroutinefunction(obj.run), f"{name}.run() must be async"