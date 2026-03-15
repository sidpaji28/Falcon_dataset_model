from typing import List, Dict, Any

from backend.llm.gemini_client import GeminiClient
from backend.security.mcp_guard import MCPGuard
from backend.tools.skill_extractor_tool import SkillExtractorTool
from backend.tools.query_engine_tool import QueryEngineTool
from backend.tools.platform_query_tool import PlatformQueryTool
from backend.tools.match_score_tool import MatchScoreTool

class JobSearchAgent:
    """
    Phase-2 Agentic Job Search Optimization System using Gemini LLM.
    Responsibilities:
    1. Extract skills from CV
    2. Expand related skills
    3. Generate Boolean job search queries
    4. Generate X-Ray queries
    5. Generate platform-specific search links
    6. Score job matches
    """

    def __init__(self):
        # Tools & Services
        self.mcp_guard = MCPGuard()
        self.gemini_client = GeminiClient()
        self.skill_extractor = SkillExtractorTool(self.gemini_client, self.mcp_guard)
        self.query_engine = QueryEngineTool(self.gemini_client)
        self.platform_query = PlatformQueryTool()
        self.match_scorer = MatchScoreTool(self.gemini_client)

        # State Management
        self.memory: Dict[str, Any] = {}
        self.user_preferences: Dict[str, Any] = {}

    def set_user_preferences(self, user_id: str, role: str, location: str, experience: str, skills: List[str]):
        """Store initial preferences for the user."""
        self.user_preferences[user_id] = {
            "role": role,
            "location": location,
            "experience": experience,
            "skills": skills
        }

    def update_memory(self, user_id: str, key: str, value: Any):
        """Update the agent's memory for a specific user."""
        if user_id not in self.memory:
            self.memory[user_id] = {}
        self.memory[user_id][key] = value

    async def analyze_cv(self, user_id: str, cv_text: str) -> dict:
        """
        Analyze CV: extracts, sanitizes, and expands skills.
        """
        result = await self.skill_extractor.extract_and_expand_skills(cv_text)

        # Store in memory
        self.update_memory(user_id, "cv_analysis", result)
        self.update_memory(user_id, "skills", result.get("base_skills", []))
        self.update_memory(user_id, "expanded_skills", result.get("expanded_skills", []))

        return result

    async def start_search(self, user_id: str, role: str, location: str, skills: List[str]) -> dict:
        """
        Start generating queries and links based on preferences.
        """
        # 1. Generate Queries
        queries = await self.query_engine.generate_queries(role, location, skills)
        boolean_query = queries.get("boolean_queries", [""])[0] if queries.get("boolean_queries") else ""
        xray_queries = queries.get("xray_queries", [])

        # 2. Generate Platform Links
        platform_links = self.platform_query.generate_platform_links(role, location, boolean_query)

        # 3. Compile Response
        result = {
            "boolean_queries": queries.get("boolean_queries", []),
            "xray_queries": xray_queries,
            "platform_links": platform_links,
            "skills": skills,
            "expanded_skills": self.memory.get(user_id, {}).get("expanded_skills", skills) # Fallback to passed skills if memory empty
        }

        # Store in memory
        self.update_memory(user_id, "search_result", result)

        # Also store queries in history
        history = self.memory.get(user_id, {}).get("query_history", [])
        history.append({
            "role": role,
            "location": location,
            "boolean_query": boolean_query,
            "timestamp": "now" # In real app, use datetime
        })
        self.update_memory(user_id, "query_history", history)

        return result

    async def score_match(self, user_id: str, job_data: dict) -> dict:
        """
        Score a specific job posting against the user's preferences.
        """
        prefs = self.user_preferences.get(user_id, {})
        user_skills = self.memory.get(user_id, {}).get("expanded_skills", prefs.get("skills", []))
        user_role = prefs.get("role", "Unknown")
        user_location = prefs.get("location", "Unknown")
        user_experience = prefs.get("experience", "Unknown")

        return await self.match_scorer.score_job(
            job_title=job_data.get("job_title", ""),
            job_description=job_data.get("job_description", ""),
            job_location=job_data.get("job_location", ""),
            job_experience=job_data.get("job_experience", ""),
            user_skills=user_skills,
            user_role=user_role,
            user_location=user_location,
            user_experience=user_experience,
            platform=job_data.get("platform", ""),
            search_url=job_data.get("search_url", "")
        )

    def get_query_history(self, user_id: str) -> List[Dict]:
        """Retrieve the query history for a user."""
        return self.memory.get(user_id, {}).get("query_history", [])
