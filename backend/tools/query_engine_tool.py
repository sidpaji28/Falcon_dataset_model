from typing import List
from backend.llm.gemini_client import GeminiClient

class QueryEngineTool:
    """
    Tool to generate boolean and X-Ray search queries based on skills.
    """
    def __init__(self, gemini_client: GeminiClient):
        self.gemini_client = gemini_client

    async def generate_queries(self, role: str, location: str, skills: List[str]) -> dict:
        """
        Generate both boolean and X-Ray queries.
        """
        # Generate Boolean Query
        boolean_query = await self.gemini_client.generate_boolean_query(skills, role, location)

        # Generate X-Ray Queries
        xray_queries = await self.gemini_client.generate_xray_queries(role, location, skills)

        return {
            "boolean_queries": [boolean_query], # wrapped in list for model schema consistency
            "xray_queries": xray_queries
        }
