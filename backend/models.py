from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class CVAnalysisRequest(BaseModel):
    cv_text: str = Field(..., description="The raw text extracted from the CV")

class JobMatchResult(BaseModel):
    job_title: str
    platform: str
    search_url: str
    match_score: float
    reason: str

class SearchQueryResponse(BaseModel):
    boolean_queries: List[str]
    xray_queries: List[str]
    platform_links: List[str]
    skills: List[str]
    expanded_skills: List[str]

class MatchScoreRequest(BaseModel):
    job_description: str
    job_title: str
    job_location: str
    job_experience: str
    user_skills: List[str]
    user_role: str
    user_location: str
    user_experience: str
    platform: str
    search_url: str

class AgentStartSearchRequest(BaseModel):
    user_id: str
    role: str
    location: str
    experience: str
    skills: List[str]

class AgentAnalyzeCVRequest(BaseModel):
    user_id: str
    cv_text: str

class AgentResultResponse(BaseModel):
    user_id: str
    results: List[JobMatchResult]

class AgentQueryHistoryResponse(BaseModel):
    user_id: str
    history: List[Dict[str, Any]]
