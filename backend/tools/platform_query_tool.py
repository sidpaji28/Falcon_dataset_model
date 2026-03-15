import urllib.parse
from typing import List

class PlatformQueryTool:
    """
    Tool to generate specific job search links.
    Supported platforms: LinkedIn, Indeed, Naukri, Glassdoor, Reed, TotalJobs, Google X-Ray.
    """

    @staticmethod
    def _url_encode(query: str) -> str:
        return urllib.parse.quote(query)

    def generate_platform_links(self, role: str, location: str, boolean_query: str) -> List[str]:
        """
        Generate search links for platforms based on the boolean query.
        """
        links = []
        encoded_query = self._url_encode(boolean_query)
        encoded_role = self._url_encode(role)
        encoded_location = self._url_encode(location)

        # 1. LinkedIn
        # Example format: https://www.linkedin.com/jobs/search/?keywords=boolean_query
        linkedin_link = f"https://www.linkedin.com/jobs/search/?keywords={encoded_query}&location={encoded_location}"
        links.append(linkedin_link)

        # 2. Indeed
        # Example format: https://www.indeed.com/jobs?q=boolean_query&l=location
        indeed_link = f"https://www.indeed.com/jobs?q={encoded_query}&l={encoded_location}"
        links.append(indeed_link)

        # 3. Naukri
        # Example format: https://www.naukri.com/role-jobs-in-location
        naukri_link = f"https://www.naukri.com/{encoded_role}-jobs-in-{encoded_location}"
        links.append(naukri_link)

        # 4. Glassdoor
        # Example format: https://www.glassdoor.com/Job/jobs.htm?sc.keyword=boolean_query&locT=C&locId=1&locKeyword=location
        glassdoor_link = f"https://www.glassdoor.com/Job/jobs.htm?sc.keyword={encoded_query}&locKeyword={encoded_location}"
        links.append(glassdoor_link)

        # 5. Reed
        # Example format: https://www.reed.co.uk/jobs/role-jobs-in-location
        reed_link = f"https://www.reed.co.uk/jobs/{encoded_role}-jobs-in-{encoded_location}"
        links.append(reed_link)

        # 6. TotalJobs
        # Example format: https://www.totaljobs.com/jobs/role/in-location
        totaljobs_link = f"https://www.totaljobs.com/jobs/{encoded_role}/in-{encoded_location}"
        links.append(totaljobs_link)

        # 7. Google X-Ray (A basic example combining the role, location, and linkedin syntax)
        xray_base = f'site:linkedin.com/jobs "{role}" AND "{location}"'
        xray_encoded = self._url_encode(xray_base)
        google_xray_link = f"https://www.google.com/search?q={xray_encoded}"
        links.append(google_xray_link)

        return links
