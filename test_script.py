import asyncio
from backend.models import AgentStartSearchRequest, AgentAnalyzeCVRequest
from backend.agent import JobSearchAgent

async def run_test():
    agent = JobSearchAgent()

    # Mock gemini client so we don't need real API key for testing structural code
    class MockGeminiClient:
        async def extract_skills_from_cv(self, cv): return ["Python"]
        async def expand_skills(self, skills): return ["Python", "Machine Learning"]
        async def generate_boolean_query(self, skills, role, loc): return "boolean query"
        async def generate_xray_queries(self, role, loc, skills): return ["xray1"]
        async def calculate_match_score(self, **kwargs): return {"score": 0.9, "reason": "good"}

    agent.gemini_client = MockGeminiClient()
    agent.skill_extractor.gemini_client = agent.gemini_client
    agent.query_engine.gemini_client = agent.gemini_client
    agent.match_scorer.gemini_client = agent.gemini_client

    print("Testing analyze_cv...")
    res = await agent.analyze_cv("user1", "CV text with Python")
    print(res)

    print("Testing start_search...")
    agent.set_user_preferences("user1", "AI", "NY", "Senior", ["Python"])
    res = await agent.start_search("user1", "AI", "NY", ["Python"])
    print(res)

    print("Testing score_match...")
    res = await agent.score_match("user1", {"job_title": "AI Eng"})
    print(res)

    print("Test finished successfully!")

if __name__ == "__main__":
    asyncio.run(run_test())
