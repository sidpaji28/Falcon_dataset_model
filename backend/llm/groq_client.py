import os
import json
from typing import List, Dict, Any
from groq import AsyncGroq

class GroqClient:
    """
    Wrapper for Groq API to provide LLM capabilities.
    Recommended model: llama3-70b-8192
    """
    def __init__(self):
        # API key is automatically inferred from GROQ_API_KEY environment variable.
        self.client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama3-70b-8192"

    async def _call_llm(self, prompt: str, system_message: str = "You are a helpful AI assistant.") -> str:
        """Helper method to call Groq LLM asynchronously."""
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1 # Keep it deterministic for structured outputs
        )
        return response.choices[0].message.content

    async def extract_skills_from_cv(self, cv_text: str) -> List[str]:
        """
        Extract a list of technical and soft skills from the provided CV text.
        """
        system_msg = "You are an expert HR recruiter. Extract ONLY a JSON array of skill strings from the CV text. No extra text."
        prompt = f"Extract all professional skills from the following CV text:\n\n{cv_text}"

        response_text = await self._call_llm(prompt, system_msg)
        try:
            # Assuming the response is a JSON array: ["Python", "Machine Learning"]
            # Clean up potential markdown formatting
            cleaned_text = response_text.replace("```json", "").replace("```", "").strip()
            skills = json.loads(cleaned_text)
            if isinstance(skills, list):
                return skills
            return []
        except json.JSONDecodeError:
            # Fallback if the LLM didn't return perfect JSON
            return [s.strip('- ') for s in response_text.split('\n') if s.strip()]

    async def expand_skills(self, skills: List[str]) -> List[str]:
        """
        Expand a list of skills with synonyms, related technologies, and variations.
        """
        system_msg = "You are an expert IT recruiter. Given a list of skills, provide an expanded list including synonyms, related tools, and standard variations. Return ONLY a JSON array of strings. No extra text."
        prompt = f"Expand the following skills: {json.dumps(skills)}"

        response_text = await self._call_llm(prompt, system_msg)
        try:
            cleaned_text = response_text.replace("```json", "").replace("```", "").strip()
            expanded = json.loads(cleaned_text)
            if isinstance(expanded, list):
                return list(set(skills + expanded)) # Combine and deduplicate
            return skills
        except json.JSONDecodeError:
            return skills

    async def generate_boolean_query(self, skills: List[str], role: str, location: str) -> str:
        """
        Generate a static Boolean job search query.
        """
        system_msg = "You are an expert at creating Boolean search strings for recruiting. Return ONLY the boolean string, nothing else."
        skills_str = " OR ".join(f'"{s}"' for s in skills[:5]) # Top 5 to avoid overly long queries

        prompt = f"""
        Create a robust boolean search query for finding jobs for a {role} in {location} requiring these skills: {skills_str}.
        The query should be formatted for use on platforms like LinkedIn or Indeed.
        """

        response_text = await self._call_llm(prompt, system_msg)
        return response_text.strip()

    async def generate_xray_queries(self, role: str, location: str, skills: List[str]) -> List[str]:
        """
        Generate Google X-Ray search queries.
        """
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
        """
        Calculates match score between user profile and job description using LLM analysis.
        This provides deeper semantic analysis beyond simple string matching.
        """
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
        {job_desc[:2000]} # Truncate to save tokens

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
