from typing import List
from backend.llm.groq_client import GroqClient

class QueryEngineTool:
    """
    Tool to generate boolean and X-Ray search queries based on skills.
    """
    def __init__(self, groq_client: GroqClient):
        self.groq_client = groq_client

    async def generate_queries(self, role: str, location: str, skills: List[str]) -> dict:
        """
        Generate both boolean and X-Ray queries.
        """
        # Generate Boolean Query
        boolean_query = await self.groq_client.generate_boolean_query(skills, role, location)

        # Generate X-Ray Queries
        xray_queries = await self.groq_client.generate_xray_queries(role, location, skills)

        return {
            "boolean_queries": [boolean_query], # wrapped in list for model schema consistency
            "xray_queries": xray_queries
        }
