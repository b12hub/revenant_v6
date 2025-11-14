from agents.security_agent import SecurityAgent
import asyncio


async def test_security_agent():
    agent = SecurityAgent()

    # Test with XSS payload
    result = await agent.run({
        "input": "<script>alert('XSS')</script>",
        "client_ip": "192.168.1.1",
        "jwt_token": "invalid.token.format"
    })

    print("Security Agent Result:")
    print(f"Status: {result['status']}")
    print(f"Threats Found: {result['threats_found']}")
    print(f"Issues: {result['issues']}")
    print(f"Recommendations: {result['recommendations']}")


# Run the test
asyncio.run(test_security_agent())