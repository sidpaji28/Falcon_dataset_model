from typing import List
from backend.security.mcp_guard import MCPGuard
from backend.llm.gemini_client import GeminiClient

class SkillExtractorTool:
    """
    Tool to extract skills from CV text using Gemini LLM, guarded by MCPGuard.
    """
    def __init__(self, gemini_client: GeminiClient, mcp_guard: MCPGuard):
        self.gemini_client = gemini_client
        self.mcp_guard = mcp_guard

    async def extract_and_expand_skills(self, cv_text: str) -> dict:
        """
        Sanitizes CV, extracts skills, and expands them.
        """
        # 1. Sanitize CV (MCP Pattern)
        sanitized_cv = self.mcp_guard.sanitize_cv_text(cv_text)

        # 2. Extract Skills
        base_skills = await self.gemini_client.extract_skills_from_cv(sanitized_cv)

        # 3. Expand Skills
        expanded_skills = await self.gemini_client.expand_skills(base_skills)

        return {
            "base_skills": base_skills,
            "expanded_skills": expanded_skills
        }
