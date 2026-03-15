import os
import json
import google.generativeai as genai
from typing import List, Dict, Any

class GeminiClient:
    """
    Wrapper for Google Gemini API to provide LLM capabilities.
    """
    def __init__(self):
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        # Using gemini-1.5-flash or pro for JSON extraction
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    async def _call_llm(self, prompt: str, system_message: str = "You are a helpful AI assistant.") -> str:
        """Helper method to call Gemini LLM asynchronously."""
        full_prompt = f"System Instruction: {system_message}\n\nUser: {prompt}"
        response = await self.model.generate_content_async(full_prompt)
        return response.text

    async def extract_skills_from_cv(self, cv_text: str) -> List[str]:
        system_msg = "You are an expert HR recruiter. Extract ONLY a JSON array of skill strings from the CV text. No extra text."
        prompt = f"Extract all professional skills from the following CV text:\n\n{cv_text}"
        response_text = await self._call_llm(prompt, system_msg)
        try:
            cleaned_text = response_text.replace("```json", "").replace("```", "").strip()
            skills = json.loads(cleaned_text)
            if isinstance(skills, list):
                return skills
            return []
        except json.JSONDecodeError:
            return [s.strip('- ') for s in response_text.split('\n') if s.strip()]

    async def expand_skills(self, skills: List[str]) -> List[str]:
        system_msg = "You are an expert IT recruiter. Given a list of skills, provide an expanded list including synonyms, related tools, and standard variations. Return ONLY a JSON array of strings. No extra text."
        prompt = f"Expand the following skills: {json.dumps(skills)}"
        response_text = await self._call_llm(prompt, system_msg)
        try:
            cleaned_text = response_text.replace("```json", "").replace("```", "").strip()
            expanded = json.loads(cleaned_text)
            if isinstance(expanded, list):
                return list(set(skills + expanded))
            return skills
        except json.JSONDecodeError:
            return skills

    async def generate_boolean_query(self, skills: List[str], role: str, location: str) -> str:
        system_msg = "You are an expert at creating Boolean search strings for recruiting. Return ONLY the boolean string, nothing else."
        skills_str = " OR ".join(f'"{s}"' for s in skills[:5])
        prompt = f"Create a robust boolean search query for finding jobs for a {role} in {location} requiring these skills: {skills_str}. Format for LinkedIn or Indeed."
        response_text = await self._call_llm(prompt, system_msg)
        return response_text.strip()

    async def generate_xray_queries(self, role: str, location: str, skills: List[str]) -> List[str]:
        system_msg = "You are an expert sourcer. Generate exactly 3 distinct Google X-Ray search queries for finding job postings on LinkedIn, Indeed, and Lever/Greenhouse. Return them as a JSON array of strings. No extra text."
        skills_str = " OR ".join(f'"{s}"' for s in skills[:3])
        prompt = f"Role: {role}\nLocation: {location}\nKey Skills: {skills_str}\n\nGenerate X-Ray queries."
        response_text = await self._call_llm(prompt, system_msg)
        try:
            cleaned_text = response_text.replace("```json", "").replace("```", "").strip()
            queries = json.loads(cleaned_text)
            if isinstance(queries, list):
                return queries
            return []
        except json.JSONDecodeError:
            return []

    async def calculate_match_score(self, job_desc: str, user_skills: List[str], user_role: str, user_location: str, user_experience: str) -> Dict[str, Any]:
        system_msg = """You are an expert job matching system. Analyze the job description against the candidate profile.
        Score the match based on: 0.4 skill match, 0.3 role relevance, 0.2 location, 0.1 experience.
        Return ONLY a JSON object with two keys: "score" (a float between 0.0 and 1.0) and "reason" (a short string explaining the score)."""
        prompt = f"""
        Candidate Profile:
        Role: {user_role}
        Location: {user_location}
        Experience: {user_experience}
        Skills: {', '.join(user_skills)}

        Job Description:
        {job_desc[:2000]}

        Calculate the match score.
        """
        response_text = await self._call_llm(prompt, system_msg)
        try:
            cleaned_text = response_text.replace("```json", "").replace("```", "").strip()
            result = json.loads(cleaned_text)
            return {
                "score": float(result.get("score", 0.0)),
                "reason": result.get("reason", "Analysis failed.")
            }
        except Exception:
            return {"score": 0.0, "reason": "Failed to parse LLM response."}
