from agents.b_series.hr_recruitment_agent import HRRecruitmentAgent
import asyncio
import json


async def test_hr_agent():
    agent = HRRecruitmentAgent()

    # Create sample resumes directory
    import os
    os.makedirs('test_resumes', exist_ok=True)

    # Create sample resume
    with open('test_resumes/john_doe.txt', 'w') as f:
        f.write("""John Doe
john.doe@email.com
(555) 123-4567

SKILLS:
Python, JavaScript, React, AWS, Docker, SQL, Pandas

EXPERIENCE:
5 years as Senior Software Engineer at Tech Company

EDUCATION:
Bachelor of Science in Computer Science - University of Technology
""")

    result = await agent.run({
        "job_description": "We need a senior Python developer with AWS experience and React knowledge. 5+ years experience required.",
        "resumes_path": "./test_resumes",
        "custom_skills": ["microservices", "serverless"]
    })

    print("HR Recruitment Agent Result:")
    print(json.dumps(result, indent=2))


asyncio.run(test_hr_agent())