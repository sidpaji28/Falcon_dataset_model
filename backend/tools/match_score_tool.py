from typing import List, Dict, Any
from backend.llm.groq_client import GroqClient

class MatchScoreTool:
    """
    Tool to score job matches using LLM analysis.
    Based on weights: 0.4 skill match, 0.3 role relevance, 0.2 location, 0.1 experience.
    """
    def __init__(self, groq_client: GroqClient):
        self.groq_client = groq_client

    async def score_job(self,
                  job_title: str,
                  job_description: str,
                  job_location: str,
                  job_experience: str,
                  user_skills: List[str],
                  user_role: str,
                  user_location: str,
                  user_experience: str,
                  platform: str,
                  search_url: str) -> dict:
        """
        Calculates the score and formats the result as a JobMatchResult dict.
        """
        # Call the LLM to get semantic analysis based on weights.
        analysis = await self.groq_client.calculate_match_score(
            job_desc=job_description,
            user_skills=user_skills,
            user_role=user_role,
            user_location=user_location,
            user_experience=user_experience
        )

        match_score = analysis.get("score", 0.0)
        reason = analysis.get("reason", "Analysis failed.")

        return {
            "job_title": job_title,
            "platform": platform,
            "search_url": search_url,
            "match_score": match_score,
            "reason": reason
        }
