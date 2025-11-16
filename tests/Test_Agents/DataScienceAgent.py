from agents.b_series.data_science_agent import DataScienceAgent
import asyncio
import json

async def test_data_science_agent():
    agent = DataScienceAgent()

    # Sample CSV data
    csv_data = """age,income,department,satisfaction
25,50000,Engineering,8
30,75000,Marketing,9
22,45000,Sales,7
35,80000,Engineering,9
28,60000,Marketing,8"""

    result = await agent.run({
        "data": csv_data,
        "config": {"analysis_type": "comprehensive"}
    })

    print("Data Science Agent Result:")
    print(json.dumps(result, indent=2))


asyncio.run(test_data_science_agent())