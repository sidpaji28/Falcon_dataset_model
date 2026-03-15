from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from backend.models import QueryRequest, QueryResponse
from backend.agent import JobSearchAgent

# Initialize FastAPI application
app = FastAPI(
    title="Agentic Job Search Optimization (JSO)",
    description="API for optimizing job search queries using AI agents.",
    version="1.0.0"
)

# Enable CORS (Cross-Origin Resource Sharing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, adjust in production
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize the job search agent
agent = JobSearchAgent()

@app.post("/generate-queries", response_model=QueryResponse)
async def generate_queries(request: QueryRequest):
    """
    Generates optimized boolean search queries for multiple job platforms.

    Expects JSON payload with:
    - role: str
    - skills: str
    - experience: str
    - location: str

    Returns JSON with optimized queries for:
    - linkedin
    - indeed
    - naukri
    - glassdoor
    - reed
    - totaljobs
    """
    try:
        # Call the agent pipeline
        result_dict = agent.process_request(
            role=request.role,
            skills_input=request.skills,
            experience=request.experience,
            location=request.location
        )

        # We need to map the returned dict to the expected QueryResponse model
        return QueryResponse(**result_dict)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
