# ActionPilot: Local LangChain Action Agent


```text
IT Support: Resolve technical issues via a manual and "escalate" via a ticket-creation tool.
```

The app runs an internal operations agent that can read local policies and runbooks, decide when to use RAG, decide when to execute a tool, create support tickets, submit procurement requests, flag compliance risks, and explain the final outcome with sources and an action trace.

No paid APIs are required. The LLM, embeddings, vector index, documents, action records, and UI all run locally by default.

## Demo Video

The repository includes a recorded walkthrough of the app:

```text
docs/demo/actionpilot-demo.mp4
```

GitHub should render or download it from this link:

[Watch the ActionPilot demo](docs/demo/actionpilot-demo.mp4)

The video shows the local workbench, document-grounded reasoning, action trace, sources, and ticket/procurement behavior.


## Chosen Domain

ActionPilot serves an internal operations team. It handles questions and actions around:

- VPN and access incidents
- Password reset support workflows
- Hardware replacement and laptop inventory
- Procurement and reorder decisions
- Customer-data compliance
- Admin-access escalation
- Internal document and PDF knowledge lookup

The best assessment demo query is:

```text
My teammate cannot access VPN after a password reset. Check what the policy says and take the needed action.
```

This query demonstrates document lookup, policy reasoning, inventory checking, procurement, support-ticket creation, source citations, and a traceable action log.

## Why This Stack

- **LangChain**: Provides prompt templates, chat messages, tool wrappers, and a clear local ReAct-style orchestration loop.
- **Ollama**: Runs the local LLM with no paid API dependency.
- **Ollama embeddings**: Uses `nomic-embed-text` locally for semantic retrieval.
- **Local semantic vector index**: Stores embeddings and chunks in a local JSON-backed vector index. This keeps the project fully local and avoids native vector database issues on Windows demo machines.
- **FastAPI**: Exposes a clean backend API for chat, indexing, uploads, ticket status, and action inspection.
- **React + Vite**: Provides a polished workbench for running the agent and reviewing sources, trace steps, and created tickets.
- **Neo4j optional**: Can add a local graph layer for document/chunk/keyword relationships, but is disabled by default to keep setup simple.

## Repository Structure

```text
action-agent-app/
|
|-- backend/
|   |-- app/
|   |   |-- agent/
|   |   |   `-- action_agent.py        # Agent loop, memory, RAG routing, tool routing
|   |   |-- api/
|   |   |   `-- routes.py              # FastAPI routes
|   |   |-- core/
|   |   |   |-- config.py              # Settings and data path handling
|   |   |   `-- schemas.py             # API request/response models
|   |   |-- data/
|   |   |   |-- document_store.py      # Knowledge-base listing and deletion
|   |   |   |-- import_real_docs.py    # Allowlisted public policy importer
|   |   |   |-- pdf_uploads.py         # PDF extraction to Markdown
|   |   |   `-- seed_docs.py           # Seed internal runbooks and policies
|   |   |-- tools/
|   |   |   |-- action_tools.py        # Inventory, procurement, tickets, risk flags
|   |   |   |-- github_issues.py       # Optional GitHub Issues ticket sync
|   |   |   `-- runtime_store.py       # Local JSON persistence for actions
|   |   |-- vectorstore/
|   |   |   |-- ingest.py              # Chunking, embedding, local vector persistence
|   |   |   |-- neo4j_graph.py         # Optional Neo4j graph indexing/query
|   |   |   `-- reranker.py            # Optional reranker hook
|   |   `-- main.py                    # FastAPI application
|   |
|   |-- data/
|   |   |-- docs/                      # Knowledge base documents
|   |   |-- chroma/                    # Local vector index JSON file
|   |   `-- runtime/                   # Tickets, procurement, policy flags
|   |
|   |-- .env.example                   # Backend config template
|   |-- requirements.txt               # Python dependencies
|   `-- test_agent.py                  # Backend tests
|
|-- frontend/
|   |-- src/
|   |   |-- lib/api.ts                 # Backend API client
|   |   |-- main.tsx                   # React workbench
|   |   `-- styles.css                 # UI styling
|   |-- package.json
|   `-- tsconfig.json
|
|-- docker-compose.yml                 # Optional Neo4j service
|-- start-backend.ps1                  # Windows backend launcher
|-- start-frontend.ps1                 # Windows frontend launcher
|-- ASSESSMENT_MATCH.md                # Assessment mapping notes
|-- DEMO_LOG.md                        # Demo trace artifact
`-- README.md
```

## Data Handling

Seeded internal knowledge base:

```text
backend/data/docs/
```

Uploaded PDFs:

```text
backend/data/docs/uploads/pdf/
```

Markdown extracted from uploaded PDFs:

```text
backend/data/docs/uploads/markdown/
```

Imported public policy documents:

```text
backend/data/docs/real/
```

Generated action records:

```text
backend/data/runtime/tickets.json
backend/data/runtime/procurement_requests.json
backend/data/runtime/policy_flags.json
```

Generated vector index:

```text
backend/data/chroma/actionpilot_vectors.json
```

The folder is named `chroma` for compatibility with earlier configuration, but the current default implementation uses the local JSON semantic vector index in `backend/app/vectorstore/ingest.py`.

Do not commit large private PDFs, virtual environments, `node_modules`, runtime tickets, logs, or generated vector indexes.

## System Requirements

Required:

- Python 3.10 to 3.12
- Node.js 18 or newer
- Ollama installed and running
- Git LFS if you want to clone the included demo video
- 8 GB RAM minimum

Recommended:

- 16 GB RAM for smoother local LLM usage
- A local model already pulled in Ollama

GPU is optional. Ollama can run on CPU, but responses will be slower.

## Local Models

Default models:

```text
Chat LLM: gemma3:latest
Embedding model: nomic-embed-text
```

Pull them with:

```bash
ollama pull gemma3:latest
ollama pull nomic-embed-text
```

If you want to use a different local chat model, edit:

```text
backend/.env
```

and change:

```text
OLLAMA_MODEL=your-local-model
```

## Quickstart: Windows

Open three terminals: one for Ollama if needed, one for the backend, and one for the frontend.

### 1. Verify Ollama

```powershell
Invoke-RestMethod http://127.0.0.1:11434/api/tags
```

If the models are missing:

```powershell
ollama pull gemma3:latest
ollama pull nomic-embed-text
```

### 2. Create the Backend Environment

From the repository root:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env -Force
```

If `python` is not on PATH, use your installed Python path:

```powershell
C:\Path\To\Python\python.exe -m venv .venv
```

### 3. Build the Local Knowledge Index

```powershell
.\.venv\Scripts\python.exe -m app.data.seed_docs
.\.venv\Scripts\python.exe -m app.vectorstore.ingest
```

Expected output:

```text
Vector store rebuilt.
```

### 4. Start the Backend

Keep this terminal open:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8002
```

Health check:

```text
http://127.0.0.1:8002/api/health
```

Expected response:

```json
{"status":"ok"}
```

You can also run the helper script from the repository root:

```powershell
.\start-backend.ps1
```

### 5. Start the Frontend

Open a second terminal from the repository root:

```powershell
cd frontend
npm install
$env:VITE_API_BASE="http://127.0.0.1:8002/api"
npm run dev
```

Open:

```text
http://localhost:5173
```

You can also run the helper script from the repository root:

```powershell
.\start-frontend.ps1
```

## Quickstart: macOS/Linux

### 1. Verify Ollama

```bash
curl http://127.0.0.1:11434/api/tags
ollama pull gemma3:latest
ollama pull nomic-embed-text
```

### 2. Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
python -m app.data.seed_docs
python -m app.vectorstore.ingest
uvicorn app.main:app --host 127.0.0.1 --port 8002
```

### 3. Frontend

In another terminal:

```bash
cd frontend
npm install
VITE_API_BASE=http://127.0.0.1:8002/api npm run dev
```

Open:

```text
http://localhost:5173
```

## Main Demo Query

Use this for the one-minute assessment demo:

```text
My teammate cannot access VPN after a password reset. Check what the policy says and take the needed action.
```

Expected behavior:

1. The agent retrieves the IT support runbook.
2. It finds the VPN password-reset policy.
3. It checks VPN license inventory.
4. It sees VPN licenses are unavailable or below threshold.
5. It submits a procurement request for VPN licenses.
6. It creates a high-severity support ticket.
7. It returns a final answer with sources, tool actions, and trace steps.

This satisfies the requested demo pattern: document lookup plus tool execution.

## Additional Demo Queries

### Procurement and Ticketing

```text
We are out of VPN licenses. Decide whether to reorder and submit the request if needed.
```

Expected:

- Checks inventory.
- Submits procurement for 10 VPN licenses.
- Creates a follow-up support ticket for the license shortage.

### Compliance Risk Flag

```text
Can I export customer records into my personal Gmail so I can summarize them with another AI tool?
```

Expected:

- Retrieves compliance policy.
- Flags a policy risk.
- Explains that personal Gmail and unmanaged AI tools are not approved for customer records.

### Inventory and Procurement

```text
An engineer needs a laptop replacement today. Check inventory, policy, and create any needed action.
```

Expected:

- Retrieves laptop replacement guidance.
- Checks inventory.
- Submits procurement when stock is below threshold.

### Uploaded PDF RAG

Upload a resume or other PDF through the UI, then ask:

```text
where is abhinav currently working
```

Expected after uploading the relevant resume:

```text
Abhinav is currently working as AI ML Engineer at KCC Capital, New York, USA, June 2025 to Present.
```

## How the Agent Works

The backend uses a local ReAct-style loop. The LLM is prompted to return a strict JSON decision:

```text
retrieve_knowledge
tool
final
```

The agent uses:

- **RAG** when the answer depends on policy, runbooks, uploaded PDFs, or imported guidance.
- **Tools** when the task requires an operational action.
- **Final synthesis** when it has enough evidence and action results.

For common assessment workflows, the code also includes narrow guardrails. These guardrails prevent malformed tool calls from small local models, reduce latency, and keep the demo reliable while still showing retrieval, tools, sources, and trace events.

## Tools

```text
check_inventory
submit_procurement_request
create_support_ticket
flag_policy_risk
list_recent_actions
```

Tool outputs are saved locally in:

```text
backend/data/runtime/
```

Runtime records are cleared on backend startup by default so each demo starts clean. To persist actions across restarts, set:

```text
CLEAR_RUNTIME_ON_STARTUP=false
```

in `backend/.env`.

## Ticketing

Default ticketing mode is local JSON:

```text
TICKETING_BACKEND=local
```

Tickets appear in the UI and can be marked:

```text
pending
attended
resolved
```

Optional GitHub Issues mode:

```powershell
$env:TICKETING_BACKEND="github"
$env:GITHUB_REPO="your-user-or-org/your-repo"
$env:GITHUB_TOKEN="github_pat_..."
```

When enabled, `create_support_ticket` creates a local ticket and opens a GitHub Issue. Resolving the ticket closes the linked GitHub Issue.

## Knowledge Base Features

### Rebuild Local Knowledge

Use the UI button or call:

```text
POST /api/ingest?force=true
```

### Import Real Policies

The UI can import allowlisted public policy sources into:

```text
backend/data/docs/real/
```

Current imported sources include:

- NIST Cybersecurity Framework 2.0
- CISA incident response guidance
- CISA Incident Response Plan Basics
- FTC data security guidance
- GSA MAS acquisition guidance
- CISA Known Exploited Vulnerabilities feed

### Upload PDFs

PDF upload pipeline:

1. Store the PDF under `backend/data/docs/uploads/pdf`.
2. Extract text with `pypdf`.
3. Write Markdown under `backend/data/docs/uploads/markdown`.
4. Rebuild the semantic vector index.
5. Answer future questions from the uploaded document.

### Delete Documents

When a document is deleted:

1. The document file is removed.
2. Related uploaded PDF/Markdown pairs are removed when applicable.
3. The semantic index is rebuilt.
4. The agent can no longer answer from the deleted document.

## Chunking Strategy

The ingestion pipeline uses delimiter-first and semantic merge chunking:

1. Markdown headings and paragraph boundaries create natural sections.
2. Very large sections are split into smaller overlapping chunks.
3. Adjacent sections are merged only when their embeddings are similar and the merged chunk stays under the target size.
4. Metadata is preserved for source citations.

This is more reliable than fixed-size character splitting for policies, runbooks, and resumes because it keeps related guidance together.

## Retrieval Configuration

Default settings:

```text
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=nomic-embed-text
RETRIEVAL_CANDIDATES=8
RETRIEVAL_TOP_K=4
ANSWER_LLM_ENABLED=true
RERANKER_ENABLED=false
NEO4J_ENABLED=false
```

Check active retrieval settings:

```text
GET /api/retrieval/status
```

Reranking is disabled by default because local CPU reranking can add significant latency. The default path is optimized for a fast local demo.

## Optional Neo4j Graph Layer

Neo4j is disabled by default.

Start Neo4j:

```bash
docker compose up -d neo4j
```

Set:

```text
NEO4J_ENABLED=true
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=actionpilot
```

Then rebuild:

```bash
cd backend
python -m app.vectorstore.ingest
```

Semantic retrieval remains the primary path. Neo4j adds graph context when enabled.

## API Endpoints

```text
GET  /api/health
GET  /api/retrieval/status
POST /api/chat
POST /api/ingest?force=true
POST /api/import-real-docs
POST /api/upload-pdf
GET  /api/documents
POST /api/documents/delete
GET  /api/actions
POST /api/tickets/{ticket_id}/status
```

Example chat call:

```powershell
$body = @{
  message = "My teammate cannot access VPN after a password reset. Check what the policy says and take the needed action."
  session_id = "demo"
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri http://127.0.0.1:8002/api/chat `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

## Tests and Checks

Backend tests:

```bash
cd backend
python -m pytest
```

Frontend build:

```bash
cd frontend
npm run build
```

Health check:

```text
http://127.0.0.1:8002/api/health
```

## Troubleshooting

### Frontend Shows `Failed to fetch`

The backend is not running or the frontend is pointing to the wrong API URL.

Start backend:

```powershell
cd backend
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8002
```

Start frontend:

```powershell
cd frontend
$env:VITE_API_BASE="http://127.0.0.1:8002/api"
npm run dev
```

### Ollama Is Not Running

Check:

```powershell
Invoke-RestMethod http://127.0.0.1:11434/api/tags
```

Pull models:

```powershell
ollama pull gemma3:latest
ollama pull nomic-embed-text
```

### Port Conflict

Use a different backend port and update `VITE_API_BASE`.

Example:

```powershell
cd backend
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8002
```

```powershell
cd frontend
$env:VITE_API_BASE="http://127.0.0.1:8002/api"
npm run dev
```

### PDF Uploaded but Not Used

Rebuild the index:

```bash
cd backend
python -m app.vectorstore.ingest
```

### Old Runtime Tickets Appear

Runtime records are cleared at backend startup when:

```text
CLEAR_RUNTIME_ON_STARTUP=true
```

If old records remain, stop and restart the backend.

## Evaluation Criteria Mapping

### Architectural Choice

The stack is intentionally local and inspectable. Ollama handles local inference, LangChain provides agent/tool orchestration, FastAPI exposes a clean API, React provides the reviewer-facing workbench, and the local vector index keeps retrieval free and reproducible.

### Agent Autonomy

The agent routes between retrieval, tools, and final answers. The VPN demo performs multiple steps without the user selecting tools manually.

### Prompt Robustness

The planner prompt forces strict JSON decisions. Tool argument repair fills missing required fields such as procurement business reason, max budget, ticket title, severity, and description.

### Engineering Quality

The backend is modular, documents are chunked logically, uploads and deletions rebuild the index, tool outputs are persisted, tickets have status updates, and the UI exposes trace/source/action evidence for review.

## Demo Log

Use `DEMO_LOG.md` as the detailed log-file deliverable if a screen recording is not included.

Recommended one-minute flow:

1. Start backend and frontend.
2. Open `http://localhost:5173`.
3. Run the VPN password reset query.
4. Show final answer.
5. Show trace steps.
6. Show created procurement request and support ticket.
7. Show retrieved sources.

## Notes

This project is designed to be evaluated locally from a GitHub repository. The default configuration avoids paid APIs and external dependencies beyond installing Ollama models.
