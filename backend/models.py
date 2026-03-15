from pydantic import BaseModel

class QueryRequest(BaseModel):
    role: str
    skills: str
    experience: str
    location: str

class QueryResponse(BaseModel):
    linkedin: str
    indeed: str
    naukri: str
    glassdoor: str
    reed: str
    totaljobs: str
