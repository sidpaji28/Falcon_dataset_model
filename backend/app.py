import asyncio
import logging
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from backend.models import (
    AgentStartSearchRequest,
    AgentAnalyzeCVRequest,
    AgentResultResponse,
    AgentQueryHistoryResponse,
    JobMatchResult,
    MatchScoreRequest
)
from backend.agent import JobSearchAgent
from backend.services.job_monitor_service import JobMonitorService
from backend.db.supabase_client import supabase_client

app = FastAPI(
    title="Phase-2 Agentic Job Search Optimization System",
    description="Agentic AI system powered by Groq",
    version="2.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Agent
agent = JobSearchAgent()

# Database update callback for JobMonitorService (simulating Supabase update)
async def update_supabase(user_id: str, results: dict):
    # Here you would typically interact with Supabase Python client
    print(f"[{user_id}] Updating Supabase with new queries and links...")

# Initialize Background Service
monitor_service = JobMonitorService(agent, db_update_callback=update_supabase)

@app.on_event("startup")
async def startup_event():
    # Start the background job monitor
    monitor_service.start()

@app.on_event("shutdown")
async def shutdown_event():
    # Stop the background job monitor
    monitor_service.stop()

# ----- WebSocket Connections Manager -----
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

# ----- Endpoints -----

@app.post("/agent/analyze-cv", response_model=Dict[str, Any])
async def analyze_cv(request: AgentAnalyzeCVRequest):
    """
    Endpoint 1: Analyze a user's CV to extract and expand skills.
    """
    try:
        result = await agent.analyze_cv(request.user_id, request.cv_text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agent/start-search", response_model=Dict[str, Any])
async def start_search(request: AgentStartSearchRequest):
    """
    Endpoint 2: Start generating queries and platform links, save to Supabase.
    """
    try:
        # Insert preferences to Supabase
        if supabase_client:
            try:
                data = {
                    "user_id": request.user_id,
                    "role": request.role,
                    "location": request.location,
                    "experience": request.experience,
                    "skills": request.skills
                }
                supabase_client.table('user_preferences').upsert(data).execute()
            except Exception as e:
                logging.error(f"Failed to upsert preferences to Supabase: {e}")

        # Set user preferences first
        agent.set_user_preferences(
            user_id=request.user_id,
            role=request.role,
            location=request.location,
            experience=request.experience,
            skills=request.skills
        )

        # Start search
        result = await agent.start_search(
            user_id=request.user_id,
            role=request.role,
            location=request.location,
            skills=request.skills
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agent/results/{user_id}", response_model=AgentResultResponse)
async def get_results(user_id: str):
    """
    Endpoint 3: Get search results (simulating fetching saved jobs and scoring them).
    """
    try:
        # For demonstration, we'll create a dummy job posting and score it
        # based on the user's preferences stored in memory.
        prefs = agent.user_preferences.get(user_id)
        if not prefs:
            raise HTTPException(status_code=404, detail="User preferences not found.")

        dummy_job = {
            "job_title": f"Senior {prefs.get('role', 'Developer')}",
            "job_description": f"Looking for an experienced {prefs.get('role', 'Developer')} with skills in {', '.join(prefs.get('skills', [])[:3])}. Must be located in {prefs.get('location', 'Unknown')}.",
            "job_location": prefs.get('location', 'Unknown'),
            "job_experience": prefs.get('experience', 'Unknown'),
            "platform": "LinkedIn",
            "search_url": "https://linkedin.com/jobs/view/12345"
        }

        score_result = await agent.score_match(user_id, dummy_job)

        # Format the result using the Pydantic model
        match_result = JobMatchResult(
            job_title=score_result.get("job_title", ""),
            platform=score_result.get("platform", ""),
            search_url=score_result.get("search_url", ""),
            match_score=score_result.get("match_score", 0.0),
            reason=score_result.get("reason", "")
        )

        return AgentResultResponse(
            user_id=user_id,
            results=[match_result]
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agent/query-history/{user_id}", response_model=AgentQueryHistoryResponse)
async def get_query_history(user_id: str):
    """
    Endpoint 4: Retrieve the history of queries generated for the user.
    """
    try:
        history = agent.get_query_history(user_id)
        return AgentQueryHistoryResponse(
            user_id=user_id,
            history=history
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/agent-updates")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time updates.
    """
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Echo back or process commands
            await manager.broadcast(f"Agent received: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
