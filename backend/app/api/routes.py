import gc
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.agent.action_agent import LocalActionAgent
from app.core.config import get_settings
from app.core.schemas import ChatRequest, ChatResponse, DeleteDocumentRequest, TicketStatusRequest
from app.data.document_store import delete_document, list_documents
from app.data.import_real_docs import import_real_docs
from app.data.pdf_uploads import process_pdf_upload
from app.tools.github_issues import sync_github_issue_status
from app.tools.runtime_store import read_records, update_record
from app.vectorstore.ingest import build_vectorstore
from app.vectorstore.ingest import get_embeddings

router = APIRouter()
_agent: LocalActionAgent | None = None


def get_agent() -> LocalActionAgent:
    global _agent
    if _agent is None:
        _agent = LocalActionAgent()
    return _agent


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/retrieval/status")
async def retrieval_status() -> dict[str, object]:
    settings = get_settings()
    embeddings = get_embeddings()
    active_embedding = getattr(embeddings, "model", getattr(embeddings, "model_name", settings.embedding_model))
    return {
        "embedding_provider": settings.embedding_provider,
        "embedding_model_configured": settings.embedding_model,
        "embedding_model_active": active_embedding,
        "fallback_embedding_model": settings.fallback_embedding_model,
        "reranker_enabled": settings.reranker_enabled,
        "reranker_model": settings.reranker_model,
        "retrieval_candidates": settings.retrieval_candidates,
        "retrieval_top_k": settings.retrieval_top_k,
    }


@router.post("/ingest")
async def ingest(force: bool = True) -> dict[str, str]:
    global _agent
    _agent = None
    gc.collect()
    build_vectorstore(force=force)
    _agent = None
    return {"status": "indexed"}


@router.post("/import-real-docs")
async def import_real_documents() -> dict[str, object]:
    global _agent
    try:
        imported = import_real_docs()
        _agent = None
        gc.collect()
        build_vectorstore(force=True)
        _agent = None
        return {"status": "imported", "count": len(imported), "sources": imported}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)) -> dict[str, object]:
    global _agent
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    if file.content_type not in {None, "", "application/pdf", "application/octet-stream"}:
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
            temp_path = Path(temp.name)
            while chunk := file.file.read(1024 * 1024):
                temp.write(chunk)
        uploaded = process_pdf_upload(temp_path, file.filename)
        _agent = None
        gc.collect()
        build_vectorstore(force=True)
        _agent = None
        return {"status": "processed", **uploaded}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()


@router.get("/documents")
async def documents() -> dict[str, object]:
    return {"documents": list_documents()}


@router.post("/documents/delete")
async def remove_document(payload: DeleteDocumentRequest) -> dict[str, object]:
    global _agent
    try:
        deleted = delete_document(payload.document_id)
        _agent = None
        gc.collect()
        build_vectorstore(force=True)
        _agent = None
        return {"status": "deleted", "document": deleted}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Document not found.") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest) -> ChatResponse:
    try:
        result = get_agent().invoke(payload.message, session_id=payload.session_id)
        return ChatResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/actions")
async def actions() -> dict[str, list[dict]]:
    return {
        "tickets": read_records("tickets.json"),
        "policy_flags": read_records("policy_flags.json"),
        "procurement_requests": read_records("procurement_requests.json"),
    }


@router.post("/tickets/{ticket_id}/status")
async def update_ticket_status(ticket_id: str, payload: TicketStatusRequest) -> dict[str, object]:
    try:
        existing_ticket = next((ticket for ticket in read_records("tickets.json") if ticket.get("id") == ticket_id), None)
        if existing_ticket is None:
            raise KeyError(ticket_id)
        github_sync = sync_github_issue_status(existing_ticket, payload.status)
        updates: dict[str, object] = {"status": payload.status}
        if github_sync.get("enabled"):
            updates["github_status_sync"] = github_sync
            if github_sync.get("state"):
                updates["github_issue_state"] = github_sync["state"]
        ticket = update_record("tickets.json", ticket_id, updates)
        return {"status": "updated", "ticket": ticket}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Ticket not found.") from exc
