def optimize_for_platforms(base_query: str, role: str, location: str) -> dict[str, str]:
    """
    Optimizes a base boolean query for different job search platforms.

    Args:
        base_query (str): The generic boolean query.
        role (str): The job role, useful for URL formulation.
        location (str): The job location, useful for URL formulation.

    Returns:
        dict[str, str]: A dictionary with platform-specific queries or URLs.
    """
    import urllib.parse

    # URL encode strings
    encoded_query = urllib.parse.quote_plus(base_query)
    encoded_role = urllib.parse.quote_plus(role) if role else ""
    encoded_location = urllib.parse.quote_plus(location) if location else ""

    # Platform-specific optimization logic
    # In a real system, these would generate exact query parameters based on how
    # each site handles boolean search. For now, we simulate platform variations.

    results = {
        "linkedin": f"site:linkedin.com/jobs {encoded_query}",
        "indeed": f"site:indeed.com/jobs {encoded_query} title:({encoded_role}) location:({encoded_location})",
        "naukri": f"site:naukri.com/job-listings {encoded_query}",
        "glassdoor": f"site:glassdoor.com/job-listing {encoded_query} jobTitle:({encoded_role})",
        "reed": f"site:reed.co.uk/jobs {encoded_query}",
        "totaljobs": f"site:totaljobs.com/job {encoded_query}"
    }

    return results
