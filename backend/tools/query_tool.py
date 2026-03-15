def generate_boolean_query(role: str, skills: list[str], experience: str, location: str) -> str:
    """
    Generates a generic base boolean query.

    Args:
        role (str): The job role.
        skills (list[str]): Extracted skills.
        experience (str): Required experience.
        location (str): Job location.

    Returns:
        str: A generic boolean search query.
    """
    query_parts = []

    if role:
        # Assuming role could be multiple words, wrap in quotes
        role_cleaned = role.strip()
        query_parts.append(f'"{role_cleaned}"')

    if skills:
        # Group skills with AND if they are required, or OR if any of them
        # Let's say we want all skills to be present for a strong match,
        # or we could group them as an AND block.
        # For a standard job search, usually they want candidates with multiple skills.
        skills_query = " AND ".join([f'"{skill}"' for skill in skills])
        query_parts.append(f'({skills_query})')

    if location:
        query_parts.append(f'"{location.strip()}"')

    # We might not explicitly use experience in the raw boolean string for all platforms,
    # but let's include it if present, e.g. "5 years" or "senior"
    if experience:
        exp_cleaned = experience.strip()
        query_parts.append(f'"{exp_cleaned}"')

    return " AND ".join(query_parts)
