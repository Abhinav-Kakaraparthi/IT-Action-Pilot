from __future__ import annotations

from pathlib import Path
from urllib.parse import quote, unquote

from app.core.config import get_settings


def _docs_root() -> Path:
    return Path(get_settings().docs_dir).resolve()


def _is_doc_file(path: Path) -> bool:
    return path.suffix.lower() in {".md", ".pdf"} and path.name != "manifest.json"


def _doc_id(path: Path) -> str:
    return quote(path.relative_to(_docs_root()).as_posix(), safe="")


def list_documents() -> list[dict[str, str | int]]:
    root = _docs_root()
    docs: list[dict[str, str | int]] = []
    if not root.exists():
        return docs

    for path in sorted(root.rglob("*")):
        if not path.is_file() or not _is_doc_file(path):
            continue
        relative = path.relative_to(root).as_posix()
        docs.append(
            {
                "id": _doc_id(path),
                "name": path.name,
                "path": relative,
                "type": path.suffix.lower().lstrip("."),
                "size": path.stat().st_size,
            }
        )
    return docs


def find_documents(term: str) -> list[dict[str, str | int]]:
    term_l = term.lower()
    return [doc for doc in list_documents() if term_l in str(doc["name"]).lower() or term_l in str(doc["path"]).lower()]


def delete_document(document_id: str) -> dict[str, str | int]:
    root = _docs_root()
    relative = unquote(document_id)
    target = (root / relative).resolve()
    if root not in target.parents or not target.exists() or not target.is_file() or not _is_doc_file(target):
        raise FileNotFoundError(relative)

    deleted = {
        "id": document_id,
        "name": target.name,
        "path": target.relative_to(root).as_posix(),
        "type": target.suffix.lower().lstrip("."),
        "size": target.stat().st_size,
    }
    peers = [target]
    path_text = f"/{deleted['path']}"
    if "/uploads/markdown/" in path_text and target.suffix.lower() == ".md":
        peers.append(root / str(deleted["path"]).replace("uploads/markdown/", "uploads/pdf/").replace(".md", ".pdf"))
    if "/uploads/pdf/" in path_text and target.suffix.lower() == ".pdf":
        peers.append(root / str(deleted["path"]).replace("uploads/pdf/", "uploads/markdown/").replace(".pdf", ".md"))

    for peer in peers:
        resolved_peer = peer.resolve()
        if resolved_peer.exists() and root in resolved_peer.parents and resolved_peer.is_file():
            resolved_peer.unlink()
    return deleted
