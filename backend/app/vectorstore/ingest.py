from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import re
from typing import Iterable

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings

from app.core.config import get_settings
from app.vectorstore.neo4j_graph import index_chunks_in_neo4j


class _CollectionInfo:
    def __init__(self, vectorstore: "SimpleVectorStore") -> None:
        self._vectorstore = vectorstore

    def count(self) -> int:
        return len(self._vectorstore.documents)


class SimpleVectorStore:
    """Small local vector store used when native Chroma is unstable on Windows."""

    def __init__(self, embedding_function: Embeddings, persist_directory: str) -> None:
        self.embedding_function = embedding_function
        self.persist_directory = Path(persist_directory)
        self.index_path = self.persist_directory / "actionpilot_vectors.json"
        self.documents: list[Document] = []
        self.vectors: list[list[float]] = []
        self._collection = _CollectionInfo(self)
        self._load()

    def _load(self) -> None:
        if not self.index_path.exists():
            return
        payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        self.documents = [
            Document(page_content=item["page_content"], metadata=item.get("metadata") or {})
            for item in payload.get("documents", [])
        ]
        self.vectors = payload.get("vectors", [])

    def _persist(self) -> None:
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "documents": [
                {"page_content": document.page_content, "metadata": document.metadata}
                for document in self.documents
            ],
            "vectors": self.vectors,
        }
        self.index_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def delete_collection(self) -> None:
        self.documents = []
        self.vectors = []
        if self.index_path.exists():
            self.index_path.unlink()

    def add_documents(self, documents: list[Document], ids: list[str] | None = None) -> None:
        del ids
        if not documents:
            return
        self.documents.extend(documents)
        self.vectors.extend(self.embedding_function.embed_documents([document.page_content for document in documents]))
        self._persist()

    def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        if not self.documents:
            return []
        query_vector = self.embedding_function.embed_query(query)
        scored = [
            (_cosine_similarity(query_vector, vector), document)
            for vector, document in zip(self.vectors, self.documents)
        ]
        return [document for _, document in sorted(scored, key=lambda item: item[0], reverse=True)[:k]]


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    settings = get_settings()
    if settings.embedding_provider.lower() == "ollama":
        return OllamaEmbeddings(model=settings.embedding_model, base_url=settings.ollama_base_url)

    from langchain_huggingface import HuggingFaceEmbeddings

    try:
        return HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"local_files_only": settings.embedding_local_files_only},
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception:
        if settings.embedding_model == settings.fallback_embedding_model:
            raise
        return HuggingFaceEmbeddings(
            model_name=settings.fallback_embedding_model,
            model_kwargs={"local_files_only": True},
            encode_kwargs={"normalize_embeddings": True},
        )


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def _split_markdown_by_delimiter(text: str, source: Path) -> list[Document]:
    sections: list[Document] = []
    current_heading = source.stem.replace("_", " ").title()
    current_lines: list[str] = []

    for line in text.splitlines():
        if re.match(r"^#{1,6}\s+", line):
            if current_lines:
                sections.append(
                    Document(
                        page_content="\n".join(current_lines).strip(),
                        metadata={"source": str(source), "heading": current_heading},
                    )
                )
            current_heading = re.sub(r"^#{1,6}\s+", "", line).strip()
            current_lines = [line]
        elif line.strip() == "" and current_lines and _estimate_tokens("\n".join(current_lines)) > 160:
            sections.append(
                Document(
                    page_content="\n".join(current_lines).strip(),
                    metadata={"source": str(source), "heading": current_heading},
                )
            )
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append(
            Document(
                page_content="\n".join(current_lines).strip(),
                metadata={"source": str(source), "heading": current_heading},
            )
        )

    return [section for section in sections if section.page_content]


def _split_large_section(section: Document, max_tokens: int = 360, overlap_tokens: int = 45) -> list[Document]:
    if _estimate_tokens(section.page_content) <= max_tokens:
        return [section]

    words = section.page_content.split()
    words_per_chunk = max_tokens * 2
    overlap_words = overlap_tokens * 2
    chunks: list[Document] = []
    start = 0
    part = 1
    while start < len(words):
        end = min(len(words), start + words_per_chunk)
        text = " ".join(words[start:end]).strip()
        if text:
            metadata = dict(section.metadata)
            metadata["heading"] = f"{metadata.get('heading', 'Section')} part {part}"
            chunks.append(Document(page_content=text, metadata=metadata))
            part += 1
        if end >= len(words):
            break
        start = max(end - overlap_words, start + 1)
    return chunks


def _split_large_sections(sections: list[Document]) -> list[Document]:
    chunks: list[Document] = []
    for section in sections:
        chunks.extend(_split_large_section(section))
    return chunks


def _merge_semantic_sections(
    sections: list[Document],
    embeddings: Embeddings,
    max_tokens: int = 520,
    similarity_threshold: float = 0.58,
) -> list[Document]:
    if not sections:
        return []

    vectors = embeddings.embed_documents([section.page_content for section in sections])
    chunks: list[Document] = []
    current = sections[0].page_content
    current_meta = dict(sections[0].metadata)

    for index, section in enumerate(sections[1:], start=1):
        proposed = f"{current}\n\n{section.page_content}".strip()
        similarity = _cosine_similarity(vectors[index - 1], vectors[index])
        if _estimate_tokens(proposed) <= max_tokens and similarity >= similarity_threshold:
            current = proposed
            if section.metadata.get("heading") and section.metadata["heading"] != current_meta.get("heading"):
                current_meta["heading"] = f"{current_meta.get('heading')} / {section.metadata['heading']}"
        else:
            chunks.append(Document(page_content=current, metadata=current_meta))
            current = section.page_content
            current_meta = dict(section.metadata)

    chunks.append(Document(page_content=current, metadata=current_meta))
    return chunks


def _load_chunks(embeddings: Embeddings | None = None) -> list[Document]:
    settings = get_settings()
    docs_dir = Path(settings.docs_dir)
    sources = sorted(docs_dir.glob("**/*.md"))
    if not sources:
        return []

    embeddings = embeddings or get_embeddings()
    chunks: list[Document] = []
    for source in sources:
        sections = _split_markdown_by_delimiter(source.read_text(encoding="utf-8"), source)
        sections = [
            section
            for section in sections
            if not (
                "Source:" in section.page_content
                and "Dataset:" in section.page_content
                and "## Extracted guidance" not in section.page_content
                and "## Recent known exploited vulnerabilities" not in section.page_content
            )
        ]
        sections = _split_large_sections(sections)
        chunks.extend(_merge_semantic_sections(sections, embeddings))

    for chunk in chunks:
        source = Path(chunk.metadata.get("source", "unknown")).name
        chunk.metadata["title"] = source.replace(".md", "").replace("_", " ").title()
        chunk.metadata["chunking"] = "delimiter+semantic"
    return chunks


def build_vectorstore(force: bool = False) -> SimpleVectorStore:
    settings = get_settings()
    persist_dir = Path(settings.chroma_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)
    embeddings = get_embeddings()

    existing = SimpleVectorStore(embedding_function=embeddings, persist_directory=str(persist_dir))
    try:
        if not force and existing._collection.count() > 0:
            return existing
    except Exception:
        pass

    if force:
        try:
            existing.delete_collection()
        except Exception:
            pass

    chunks = _load_chunks(embeddings)
    if not chunks:
        return SimpleVectorStore(embedding_function=embeddings, persist_directory=str(persist_dir))

    index_chunks_in_neo4j(chunks, force=force)

    db = SimpleVectorStore(embedding_function=embeddings, persist_directory=str(persist_dir))
    for index in range(0, len(chunks), 16):
        batch = chunks[index : index + 16]
        ids = [f"chunk-{index + offset}" for offset in range(len(batch))]
        db.add_documents(batch, ids=ids)
    return db


def get_vectorstore() -> SimpleVectorStore:
    settings = get_settings()
    return SimpleVectorStore(embedding_function=get_embeddings(), persist_directory=settings.chroma_dir)


if __name__ == "__main__":
    build_vectorstore(force=True)
    print("Vector store rebuilt.")
