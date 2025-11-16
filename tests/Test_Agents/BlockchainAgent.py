from agents.b_series.blockchain_agent import BlockchainAgent
import asyncio
import json


async def test_blockchain_agent():
    agent = BlockchainAgent()

    # Test wallet analysis
    result = await agent.run({
        "analysis_type": "wallet",
        "address": "0x742d35Cc6634C0532925a3b8Dc9F1a71acC44d6C",
        "network": "mainnet"
    })

    print("Blockchain Agent Result:")
    print(json.dumps(result, indent=2))


asyncio.run(test_blockchain_agent())