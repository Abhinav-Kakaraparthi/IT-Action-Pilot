import json
from pathlib import Path

from fastapi.testclient import TestClient
from langchain_core.documents import Document
from pypdf import PdfWriter

from app.agent.action_agent import AgentState, LocalActionAgent
from app.core.schemas import Source
from app.data.document_store import find_documents, list_documents
from app.data.import_real_docs import TextExtractor, _kev_to_markdown
from app.data.pdf_uploads import process_pdf_upload
from app.main import app
from app.tools.runtime_store import append_record, clear_runtime_records, read_records
from app.tools.action_tools import check_inventory, submit_procurement_request
from app.tools.github_issues import create_github_issue
from app.vectorstore.ingest import _load_chunks
from app.vectorstore import ingest
from app.vectorstore.neo4j_graph import index_chunks_in_neo4j, query_neo4j_context
from app.vectorstore.reranker import rerank_documents


def test_inventory_tool_returns_json():
    result = check_inventory.invoke({"item": "vpn license"})
    assert "needs_reorder" in result


def test_procurement_tool_creates_record(tmp_path, monkeypatch):
    from app.core.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "runtime_dir", str(tmp_path))
    result = submit_procurement_request.invoke(
        {"item": "vpn license", "quantity": 10, "business_reason": "Access blocker", "max_budget_usd": 120}
    )
    assert "submitted" in result


def test_inventory_tool_accepts_plural_vpn_license():
    result = json.loads(check_inventory.invoke({"item": "VPN licenses"}))
    assert result["item"] == "vpn license"
    assert result["needs_reorder"] is True


def test_agent_repairs_missing_procurement_args(tmp_path, monkeypatch):
    from app.core.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "runtime_dir", str(tmp_path))
    agent = LocalActionAgent()
    state = AgentState(query="My teammate cannot access VPN after a password reset.")

    result = agent._run_tool(state, "submit_procurement_request", {"item": "VPN licenses", "quantity": 1}, step=1)

    assert "submitted" in result
    assert state.actions[0]["args"]["business_reason"]
    assert state.actions[0]["args"]["quantity"] == 10
    assert state.actions[0]["args"]["max_budget_usd"] == 120.0


def test_agent_repairs_missing_support_ticket_title(tmp_path, monkeypatch):
    from app.core.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "runtime_dir", str(tmp_path))
    agent = LocalActionAgent()
    state = AgentState(query="My teammate cannot access VPN after a password reset and is blocked.")

    result = agent._run_tool(
        state,
        "create_support_ticket",
        {"description": "Blocked from VPN", "severity": "high", "priority": "high"},
        step=1,
    )

    assert "User cannot access VPN after password reset" in result
    assert state.actions[0]["args"]["title"] == "User cannot access VPN after password reset"
    assert "priority" not in state.actions[0]["args"]


def test_ingest_endpoint_rebuilds_vectorstore():
    response = TestClient(app).post("/api/ingest?force=true")

    assert response.status_code == 200
    assert response.json() == {"status": "indexed"}


def test_vpn_query_uses_fast_path_without_tool_errors():
    response = TestClient(app).post(
        "/api/chat",
        json={
            "message": "My teammate cannot access VPN after a password reset. Check what the policy says and take the needed action.",
            "session_id": "pytest-fast-path",
        },
    )
    data = response.json()

    assert response.status_code == 200
    assert [event for event in data["trace"] if event["kind"] == "error"] == []
    assert [action["tool"] for action in data["actions"]] == [
        "check_inventory",
        "submit_procurement_request",
        "create_support_ticket",
    ]


def test_out_of_scope_movie_query_runs_no_tools_or_retrieval():
    response = TestClient(app).post(
        "/api/chat",
        json={"message": "what are the movies releasing this week", "session_id": "pytest-movie"},
    )
    data = response.json()

    assert response.status_code == 200
    assert data["actions"] == []
    assert data["sources"] == []
    assert data["trace"][0]["title"] == "Guardrail: out of scope"
    assert "Movie releases and similar entertainment/news questions require current external data" in data["answer"]


def test_cisa_real_policy_question_is_in_scope():
    agent = LocalActionAgent()

    assert agent._scope_reason("What will CISA do working with partners") is None


def test_cisa_real_policy_question_returns_source_answer():
    response = TestClient(app).post(
        "/api/chat",
        json={"message": "What will CISA do working with partners", "session_id": "pytest-cisa-answer"},
    )
    data = response.json()

    assert response.status_code == 200
    assert data["actions"] == []
    assert data["sources"]
    assert data["trace"][-1]["detail"] == "Synthesized answer from indexed public guidance."
    assert "CISA" in data["answer"]
    assert "Source" in data["answer"] or "Sources" in data["answer"]


def test_out_of_scope_movie_query_does_not_use_prior_vpn_context():
    client = TestClient(app)
    client.post(
        "/api/chat",
        json={
            "message": "My teammate cannot access VPN after a password reset. Check what the policy says and take the needed action.",
            "session_id": "pytest-movie-after-vpn",
        },
    )

    response = client.post(
        "/api/chat",
        json={"message": "what are the movies releasing this week", "session_id": "pytest-movie-after-vpn"},
    )
    data = response.json()

    assert response.status_code == 200
    assert data["actions"] == []
    assert data["sources"] == []
    assert "VPN" not in data["answer"]
    assert data["trace"][0]["title"] == "Guardrail: out of scope"


def test_memory_can_recall_out_of_scope_topic_after_in_scope_query():
    client = TestClient(app)
    session_id = "pytest-sachin-memory"
    first = client.post(
        "/api/chat",
        json={"message": "tell me about Sachin Tendulkar", "session_id": session_id},
    )
    assert first.status_code == 200
    assert first.json()["actions"] == []

    client.post(
        "/api/chat",
        json={
            "message": "My teammate cannot access VPN after a password reset. Check what the policy says and take the needed action.",
            "session_id": session_id,
        },
    )

    recall = client.post(
        "/api/chat",
        json={"message": "no i asked some thing like sachin", "session_id": session_id},
    )
    data = recall.json()

    assert recall.status_code == 200
    assert data["actions"] == []
    assert data["sources"] == []
    assert "Sachin Tendulkar" in data["answer"]
    assert data["trace"][0]["detail"] == "Answered from session memory."


def test_memory_recall_uses_summary_after_compaction(monkeypatch):
    agent = LocalActionAgent()
    agent.settings.memory_context_tokens = 20
    agent.settings.memory_recent_tokens = 8
    monkeypatch.setattr(
        agent,
        "_summarize_messages",
        lambda summary, messages: "User previously asked about Sachin Tendulkar.",
    )

    agent._remember("pytest-summary-memory", "tell me about Sachin Tendulkar", "Out of scope.")
    response = agent._memory_followup_response("no i asked some thing like sachin", "pytest-summary-memory")

    assert response is not None
    assert "Sachin Tendulkar" in response["answer"]
    assert response["actions"] == []


def test_delimiter_semantic_chunking_adds_metadata():
    chunks = _load_chunks()

    assert chunks
    assert all(chunk.metadata["chunking"] == "delimiter+semantic" for chunk in chunks)
    assert all("heading" in chunk.metadata for chunk in chunks)


def test_neo4j_helpers_noop_when_disabled(monkeypatch):
    from app.core.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "neo4j_enabled", False)

    assert index_chunks_in_neo4j([], force=True) is False
    assert query_neo4j_context("vpn license") == []


def test_delete_uploaded_pdf_removes_extracted_markdown(tmp_path, monkeypatch):
    from app.core.config import get_settings
    from app.data.document_store import delete_document, list_documents

    docs_dir = tmp_path / "docs"
    pdf_dir = docs_dir / "uploads" / "pdf"
    md_dir = docs_dir / "uploads" / "markdown"
    pdf_dir.mkdir(parents=True)
    md_dir.mkdir(parents=True)
    (pdf_dir / "sample.pdf").write_bytes(b"%PDF-1.4")
    (md_dir / "sample.md").write_text("# Sample", encoding="utf-8")

    settings = get_settings()
    monkeypatch.setattr(settings, "docs_dir", str(docs_dir))
    doc_id = next(doc["id"] for doc in list_documents() if doc["path"] == "uploads/pdf/sample.pdf")

    delete_document(str(doc_id))

    assert not (pdf_dir / "sample.pdf").exists()
    assert not (md_dir / "sample.md").exists()


def test_reranker_skips_small_candidate_sets(monkeypatch):
    from app.core.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "reranker_enabled", True)
    monkeypatch.setattr(settings, "reranker_min_candidates", 4)
    monkeypatch.setattr(settings, "retrieval_top_k", 4)
    docs = [Document(page_content="alpha"), Document(page_content="beta")]

    assert rerank_documents("alpha", docs) == docs


def test_reranker_falls_back_when_model_unavailable(monkeypatch):
    from app.core.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "reranker_enabled", True)
    monkeypatch.setattr(settings, "reranker_min_candidates", 2)
    monkeypatch.setattr(settings, "retrieval_top_k", 1)
    monkeypatch.setattr("app.vectorstore.reranker._get_cross_encoder", lambda: (_ for _ in ()).throw(RuntimeError("missing")))
    docs = [Document(page_content="alpha"), Document(page_content="beta")]

    assert rerank_documents("alpha", docs) == docs[:1]


def test_embedding_model_falls_back_when_primary_unavailable(monkeypatch):
    from app.core.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "embedding_model", "missing-primary-model")
    monkeypatch.setattr(settings, "fallback_embedding_model", "all-MiniLM-L6-v2")
    ingest.get_embeddings.cache_clear()
    try:
        embeddings = ingest.get_embeddings()
        assert embeddings is not None
    finally:
        ingest.get_embeddings.cache_clear()


def test_real_doc_html_extractor_ignores_scripts():
    extractor = TextExtractor()
    extractor.feed("<html><script>bad()</script><h1>Policy</h1><p>Use approved systems.</p></html>")

    text = extractor.text()

    assert "bad" not in text
    assert "Policy" in text
    assert "Use approved systems." in text


def test_kev_json_converts_to_markdown():
    markdown = _kev_to_markdown(
        "CISA KEV",
        "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json",
        {
            "vulnerabilities": [
                {
                    "cveID": "CVE-2024-0001",
                    "vendorProject": "Example",
                    "product": "Gateway",
                    "vulnerabilityName": "Example Vulnerability",
                    "requiredAction": "Apply updates.",
                    "dueDate": "2026-01-01",
                }
            ]
        },
        "2026-05-13T00:00:00+00:00",
    )

    assert "CVE-2024-0001" in markdown
    assert "Apply updates." in markdown


def test_pdf_upload_processor_rejects_empty_pdf(tmp_path):
    pdf_path = tmp_path / "empty.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with pdf_path.open("wb") as handle:
        writer.write(handle)

    try:
        process_pdf_upload(pdf_path, "empty.pdf")
    except ValueError as exc:
        assert "No extractable text" in str(exc)
    else:
        raise AssertionError("Expected empty PDF to be rejected")


def test_upload_pdf_endpoint_rejects_non_pdf():
    response = TestClient(app).post(
        "/api/upload-pdf",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 400


def test_document_registry_finds_uploaded_resume():
    from app.core.config import get_settings

    settings = get_settings()
    docs_dir = Path(settings.docs_dir)
    resume_dir = docs_dir / "uploads" / "markdown"
    resume_dir.mkdir(parents=True, exist_ok=True)
    resume_path = resume_dir / "pytest-resume.md"
    resume_path.write_text("# Resume\nPytest resume fixture.", encoding="utf-8")

    matches = find_documents("resume")

    try:
        assert any("pytest-resume" in str(doc["path"]).lower() for doc in matches)
    finally:
        resume_path.unlink(missing_ok=True)


def test_resume_availability_question_uses_document_inventory():
    response = TestClient(app).post(
        "/api/chat",
        json={"message": "is any file called resume available in the knowledge base", "session_id": "pytest-doc-inventory"},
    )
    data = response.json()

    assert response.status_code == 200
    assert data["actions"] == []
    assert data["sources"] == []
    assert "resume" in data["answer"].lower()
    assert data["trace"][0]["title"] == "Document inventory"


def test_resume_current_work_question_extracts_current_role():
    class FakeAnswerChain:
        def invoke(self, payload):
            class Result:
                content = "Abhinav is currently working as an Applied AI Engineer at KCC Capital. Sources: Resume."

            assert "Applied AI Engineer" in payload["sources"]
            return Result()

    agent = LocalActionAgent()
    agent.answer_chain = FakeAnswerChain()
    state = AgentState(
        query="where is abhinav currently working",
        sources=[
            Source(
                title="Resume",
                snippet="EXPERIENCE Applied AI Engineer | KCC Capital, New York, USA June 2025 to Present",
            )
        ],
    )

    answer = agent._source_answer("where is abhinav currently working", state)

    assert "Applied AI Engineer" in answer
    assert "KCC Capital" in answer


def test_resume_education_question_is_synthesized_from_pdf():
    class FakeAnswerChain:
        def invoke(self, payload):
            class Result:
                content = "Abhinav studied at Columbia University and New York Institute of Technology. Sources: Resume."

            assert "Columbia University" in payload["sources"]
            return Result()

    agent = LocalActionAgent()
    agent.answer_chain = FakeAnswerChain()
    state = AgentState(
        query="where did abhinav study",
        sources=[
            Source(
                title="Resume",
                snippet="EDUCATION Columbia University, New York, NY Machine Learning Graduate Certification. New York Institute of Technology, New York, NY Master of Science in Computer Science.",
            )
        ],
    )

    answer = agent._source_answer("where did abhinav study", state)

    assert "Columbia" in answer
    assert "New York Institute of Technology" in answer
    assert "Uploaded PDF" not in answer


def test_ticket_status_can_be_updated(tmp_path, monkeypatch):
    from app.core.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "runtime_dir", str(tmp_path))
    ticket = append_record(
        "tickets.json",
        {
            "title": "Admin access request",
            "severity": "medium",
            "description": "Needs admin review",
            "owner_team": "Platform Operations",
            "status": "open",
        },
    )

    response = TestClient(app).post(f"/api/tickets/{ticket['id']}/status", json={"status": "resolved"})

    assert response.status_code == 200
    assert response.json()["ticket"]["status"] == "resolved"


def test_runtime_records_can_be_cleared_on_startup(tmp_path, monkeypatch):
    from app.core.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "runtime_dir", str(tmp_path))
    append_record("tickets.json", {"title": "Old ticket", "severity": "low", "description": "stale"})
    append_record("policy_flags.json", {"concern": "stale"})
    append_record("procurement_requests.json", {"item": "vpn license"})

    clear_runtime_records()

    assert read_records("tickets.json") == []
    assert read_records("policy_flags.json") == []
    assert read_records("procurement_requests.json") == []


def test_github_issue_creation_is_optional_when_unconfigured(monkeypatch):
    from app.core.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "ticketing_backend", "local")
    monkeypatch.setattr(settings, "github_token", "")
    monkeypatch.setattr(settings, "github_repo", "")

    result = create_github_issue("Title", "medium", "Description", "Platform Operations")

    assert result["enabled"] is False


def test_support_ticket_can_create_linked_github_issue(tmp_path, monkeypatch):
    from app.core.config import get_settings
    from app.tools.action_tools import create_support_ticket

    settings = get_settings()
    monkeypatch.setattr(settings, "runtime_dir", str(tmp_path))
    monkeypatch.setattr(settings, "ticketing_backend", "github")
    monkeypatch.setattr(settings, "github_token", "token")
    monkeypatch.setattr(settings, "github_repo", "owner/repo")
    monkeypatch.setattr(
        "app.tools.github_issues._github_request",
        lambda method, path, payload=None: {
            "number": 42,
            "html_url": "https://github.com/owner/repo/issues/42",
            "state": "open",
        },
    )

    ticket = json.loads(
        create_support_ticket.invoke(
            {
                "title": "Escalation",
                "severity": "high",
                "description": "Needs external ticket.",
                "owner_team": "Platform Operations",
            }
        )
    )

    assert ticket["github_issue_number"] == 42
    assert ticket["github_issue_url"].endswith("/42")


def test_admin_access_query_creates_ticket():
    response = TestClient(app).post(
        "/api/chat",
        json={"message": "Please grant admin access to production", "session_id": "pytest-admin-ticket"},
    )
    data = response.json()

    assert response.status_code == 200
    assert [action["tool"] for action in data["actions"]] == ["create_support_ticket"]
