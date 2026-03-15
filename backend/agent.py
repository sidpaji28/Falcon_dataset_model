from backend.tools.skill_extractor import extract_skills
from backend.tools.query_tool import generate_boolean_query
from backend.tools.platform_tool import optimize_for_platforms

class JobSearchAgent:
    """
    Agent that orchestrates the job search optimization pipeline.

    Pipeline:
    1. User Input (role, skills, experience, location)
    2. Skill Extractor Tool
    3. Query Generator Tool
    4. Platform Optimizer Tool
    """
    def __init__(self):
        # Could initialize AI models or DB connections here if needed
        pass

    def process_request(self, role: str, skills_input: str, experience: str, location: str) -> dict[str, str]:
        # Step 1 & 2: Extract skills
        extracted_skills = extract_skills(skills_input)

        # Step 3: Generate base boolean query
        base_query = generate_boolean_query(role, extracted_skills, experience, location)

        # Step 4: Optimize query for different platforms
        optimized_queries = optimize_for_platforms(base_query, role, location)

        return optimized_queries
