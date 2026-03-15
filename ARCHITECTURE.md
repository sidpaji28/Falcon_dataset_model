# Phase-2 Agentic Job Search Optimization System (JSO) Architecture

This document describes how the Agentic AI system fulfills the requirements of Sections 4, 5, and 6.

---

## Section 4 - Dashboard Integration
The JSO Agent interacts with the four main dashboards by acting as a headless service available via FastAPI endpoints or WebSockets. Data persistence happens in Supabase.

**A. User Dashboard**
* **Interaction**: The user uploads their CV. The frontend calls `POST /agent/analyze-cv`.
* **Example**: A job seeker logs in, clicks "Analyze CV", and the AI agent automatically extracts their skills, expands synonyms, and begins querying Google X-Ray links and LinkedIn Boolean searches asynchronously. The user sees a real-time list of generated platform links.

**B. HR Consultant Dashboard**
* **Interaction**: HR consultants input a targeted role and location into the dashboard. The frontend calls `POST /agent/start-search` with custom skill arrays.
* **Example**: An HR consultant needs to hire a senior Python dev in NYC. They use the tool to generate 3 custom X-Ray queries that they then paste into their sourcing pipelines, leveraging the AI’s understanding of alternative skills.

**C. Super Admin Dashboard**
* **Interaction**: Reads API and system logs directly from Supabase and the background `JobMonitorService`.
* **Example**: The admin sees that the 12-hour background loop successfully triggered 500 search updates for users last night without any API rate limit failures.

**D. Licensing Dashboard**
* **Interaction**: Monitors enterprise usage statistics (API calls processed by Gemini).
* **Example**: A dashboard widget pulls the volume of `gemini-1.5-flash` API queries made by the tenant, displaying a progress bar mapping current usage against their paid tier.

---

## Section 5 - Technical Architecture
We have chosen **Google Cloud / Google** as the primary AI infrastructure provider for the Agentic layer, specifically utilizing the **Gemini API**.

**Infrastructure Design:**
1. **Frontend (Next.js/React)**: Hosted on Vercel. Connects to the backend via standard REST APIs and WebSockets.
2. **Backend (Python/FastAPI)**: Hosted as serverless functions on Vercel (via `@vercel/python`). Executes the AI orchestration, background monitoring, and MCP security layer.
3. **AI Provider (Gemini API / Vertex AI)**: Processes raw CV data, scores matches, and formulates queries using `gemini-1.5-flash` or `gemini-1.5-pro`.
4. **Database (Supabase)**: Maintains user profiles, query history, job monitoring preferences, and licensing metrics.
5. **Storage (AWS S3)**: Used for securely housing raw PDF/DOCX CV files before text is extracted.
6. **Background Task (Google Cloud Functions or Vercel Cron)**: Acts as the 12-hour trigger for the FastAPI background workers to monitor new jobs.

---

## Section 6 - Integration With Phase-1
The new Phase-2 Agent is integrated seamlessly into the existing Phase-1 tech stack.

### Components
* **Current Stack**: NextJS, React, NodeJS, Supabase, AWS S3, Google Cloud, Vercel.
* **New Agent Layer**: Python, FastAPI, Gemini API, Pydantic, MCP (Model Context Protocol) Guard.

### Connection Strategy
1. **APIs**:
   - Next.js (Phase-1) communicates with the new Python Agent (Phase-2) via FastAPI endpoints (e.g., `POST /api/agent/analyze-cv`).
   - The Python Agent exposes standard JSON responses conforming to strict Pydantic models.

2. **Event Triggers**:
   - **Upload Event**: When a user uploads a CV to AWS S3 (via the Next.js/NodeJS stack), an event trigger (or standard webhook) calls the Python Agent, passing the S3 object key or extracted text.
   - **Cron Trigger**: A Vercel Cron job or GCP Scheduler hits a secured endpoint on the Python backend every 12 hours to trigger the `JobMonitorService`.

3. **Data Flow**:
   - `User -> Next.js -> Uploads CV -> AWS S3`
   - `Next.js -> Calls Python Agent -> MCP Guard sanitizes CV`
   - `Python Agent -> Gemini API extracts skills -> Supabase stores results`
   - `Agent returns generated boolean queries -> Next.js displays links`

4. **MCP / APA for Security Protocols**:
   - We implement a security boundary via the **Model Context Protocol (MCP)** inside the Python layer.
   - `backend/security/mcp_guard.py` intercepts raw data from AWS S3 or the user before it ever reaches the Gemini API. It uses regular expressions and pattern matching to sanitize PII (Personally Identifiable Information) such as Names, Emails, Phone Numbers, and Addresses, replacing them with generic tokens like `[EMAIL MASKED]`. This ensures compliance and privacy when leveraging third-party AI models.
