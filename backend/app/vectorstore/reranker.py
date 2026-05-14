from __future__ import annotations

from functools import lru_cache

from langchain_core.documents import Document

from app.core.config import get_settings

_RERANKER_UNAVAILABLE = False


@lru_cache(maxsize=1)
def _get_cross_encoder():
    settings = get_settings()
    from sentence_transformers import CrossEncoder

    try:
        return CrossEncoder(settings.reranker_model, local_files_only=settings.reranker_local_files_only)
    except Exception:
        if not settings.fallback_reranker_model:
            raise
        return CrossEncoder(settings.fallback_reranker_model, local_files_only=True)


def rerank_documents(query: str, docs: list[Document]) -> list[Document]:
    global _RERANKER_UNAVAILABLE
    settings = get_settings()
    if not settings.reranker_enabled or len(docs) < settings.reranker_min_candidates:
        return docs[: settings.retrieval_top_k]
    if settings.reranker_min_candidates > settings.retrieval_candidates:
        return docs[: settings.retrieval_top_k]
    if _RERANKER_UNAVAILABLE:
        return docs[: settings.retrieval_top_k]

    try:
        model = _get_cross_encoder()
        pairs = [(query, doc.page_content[:1600]) for doc in docs]
        scores = model.predict(pairs)
        ranked = sorted(zip(docs, scores), key=lambda item: float(item[1]), reverse=True)
        return [doc for doc, _ in ranked[: settings.retrieval_top_k]]
    except Exception:
        _RERANKER_UNAVAILABLE = True
        return docs[: settings.retrieval_top_k]
