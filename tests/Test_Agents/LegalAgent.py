from agents.b_series.legal_agent import LegalAgent
import asyncio
import json


async def test_legal_agent():
    agent = LegalAgent()

    # Sample contract text
    contract_text = """
    This agreement governs data processing between Company A and Provider B.
    Provider B may process personal data on behalf of Company A for service delivery.
    Data will be retained for a period of 2 years. The parties agree to reasonable
    security measures. Liability is limited to the contract value.
    """

    result = await agent.run({
        "contract_text": contract_text,
        "analysis_type": "comprehensive"
    })

    print("Legal Agent Result:")
    print(json.dumps(result, indent=2))


asyncio.run(test_legal_agent())