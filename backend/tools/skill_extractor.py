def extract_skills(skills_input: str) -> list[str]:
    """
    Extracts individual skills from a comma-separated string of skills.

    Args:
        skills_input (str): A comma-separated string of skills.

    Returns:
        list[str]: A list of cleaned, individual skills.
    """
    if not skills_input:
        return []

    # Split by comma, strip whitespace, and filter out empty strings
    skills = [skill.strip() for skill in skills_input.split(',')]
    return [skill for skill in skills if skill]
