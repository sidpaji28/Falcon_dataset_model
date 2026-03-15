# Phase-2 Agentic Job Search Optimization System

## Required Environment Variables
Ensure you have set the Gemini API key and Supabase credentials in your environment before running the server:
```bash
export GEMINI_API_KEY="your_gemini_api_key_here"
export SUPABASE_URL="https://your_supabase_project.supabase.co"
export SUPABASE_KEY="your_supabase_anon_key"
```

## Running the Server
You can start the FastAPI server locally using Uvicorn:
```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

## Vercel Deployment
To deploy this project to Vercel, you can use the Vercel CLI:
```bash
vercel deploy
```
Make sure to add the environment variables (`GEMINI_API_KEY`, `SUPABASE_URL`, `SUPABASE_KEY`) in your Vercel project settings.

## Example API Request: Start Search
`POST /agent/start-search`

**Request Body (JSON):**
```json
{
  "user_id": "user123",
  "role": "AI Engineer",
  "location": "San Francisco, CA",
  "experience": "Senior",
  "skills": ["Python", "LangChain", "FastAPI", "LLMs"]
}
```

## Example JSON Response: Start Search

**Response:**
```json
{
  "boolean_queries": [
    "(\"Python\" OR \"LangChain\" OR \"FastAPI\" OR \"LLMs\") AND \"AI Engineer\" AND \"San Francisco, CA\""
  ],
  "xray_queries": [
    "site:linkedin.com/jobs \"AI Engineer\" \"San Francisco, CA\" (\"Python\" OR \"LangChain\")",
    "site:indeed.com/jobs \"AI Engineer\" \"San Francisco, CA\" (\"FastAPI\" OR \"LLMs\")",
    "site:lever.co OR site:greenhouse.io \"AI Engineer\" \"San Francisco, CA\" (\"Python\" OR \"LangChain\")"
  ],
  "platform_links": [
    "https://www.linkedin.com/jobs/search/?keywords=%28%22Python%22%20OR%20%22LangChain%22%20OR%20%22FastAPI%22%20OR%20%22LLMs%22%29%20AND%20%22AI%20Engineer%22%20AND%20%22San%20Francisco%2C%20CA%22&location=San%20Francisco%2C%20CA",
    "https://www.indeed.com/jobs?q=%28%22Python%22%20OR%20%22LangChain%22%20OR%20%22FastAPI%22%20OR%20%22LLMs%22%29%20AND%20%22AI%20Engineer%22%20AND%20%22San%20Francisco%2C%20CA%22&l=San%20Francisco%2C%20CA",
    "https://www.naukri.com/AI%20Engineer-jobs-in-San%20Francisco%2C%20CA",
    "https://www.glassdoor.com/Job/jobs.htm?sc.keyword=%28%22Python%22%20OR%20%22LangChain%22%20OR%20%22FastAPI%22%20OR%20%22LLMs%22%29%20AND%20%22AI%20Engineer%22%20AND%20%22San%20Francisco%2C%20CA%22&locKeyword=San%20Francisco%2C%20CA",
    "https://www.reed.co.uk/jobs/AI%20Engineer-jobs-in-San%20Francisco%2C%20CA",
    "https://www.google.com/search?q=site%3Alinkedin.com/jobs%20%22AI%20Engineer%22%20AND%20%22San%20Francisco%2C%20CA%22"
  ],
  "skills": [
    "Python",
    "LangChain",
    "FastAPI",
    "LLMs"
  ],
  "expanded_skills": [
    "Python",
    "LangChain",
    "FastAPI",
    "LLMs"
  ]
}
```
