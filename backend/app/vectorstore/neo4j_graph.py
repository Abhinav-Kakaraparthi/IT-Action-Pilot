from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from app.core.config import get_settings

_NEO4J_UNAVAILABLE = False


def _driver() -> Any | None:
    global _NEO4J_UNAVAILABLE
    settings = get_settings()
    if not settings.neo4j_enabled or _NEO4J_UNAVAILABLE:
        return None
    try:
        from neo4j import GraphDatabase
    except Exception:
        _NEO4J_UNAVAILABLE = True
        return None
    try:
        return GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
            connection_timeout=settings.neo4j_timeout_seconds,
            max_transaction_retry_time=settings.neo4j_timeout_seconds,
        )
    except Exception:
        _NEO4J_UNAVAILABLE = True
        return None


def _chunk_id(doc: Document) -> str:
    source = str(doc.metadata.get("source", "unknown"))
    digest = hashlib.sha1(f"{source}\n{doc.page_content}".encode("utf-8")).hexdigest()
    return digest[:16]


def _keywords(text: str, limit: int = 10) -> list[str]:
    stopwords = {
        "about",
        "after",
        "again",
        "available",
        "before",
        "below",
        "customer",
        "every",
        "first",
        "from",
        "have",
        "include",
        "internal",
        "should",
        "that",
        "the",
        "their",
        "this",
        "when",
        "with",
    }
    words = [word.lower() for word in re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}", text)]
    counts: dict[str, int] = {}
    for word in words:
        if word not in stopwords:
            counts[word] = counts.get(word, 0) + 1
    return [word for word, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]]


def index_chunks_in_neo4j(chunks: list[Document], force: bool = False) -> bool:
    driver = _driver()
    if driver is None:
        return False

    try:
        with driver.session() as session:
            session.run("CREATE CONSTRAINT actionpilot_doc_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE")
            session.run("CREATE CONSTRAINT actionpilot_chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE")
            session.run("CREATE CONSTRAINT actionpilot_keyword_name IF NOT EXISTS FOR (k:Keyword) REQUIRE k.name IS UNIQUE")
            if force:
                session.run("MATCH (n:ActionPilot) DETACH DELETE n")

            for doc in chunks:
                source = str(doc.metadata.get("source", "unknown"))
                doc_id = Path(source).stem
                chunk_id = _chunk_id(doc)
                title = str(doc.metadata.get("title", doc_id))
                heading = str(doc.metadata.get("heading", title))
                keywords = _keywords(f"{heading}\n{doc.page_content}")
                session.run(
                    """
                    MERGE (d:Document:ActionPilot {id: $doc_id})
                    SET d.title = $title, d.source = $source
                    MERGE (c:Chunk:ActionPilot {id: $chunk_id})
                    SET c.text = $text, c.heading = $heading, c.title = $title
                    MERGE (d)-[:HAS_CHUNK]->(c)
                    WITH c
                    UNWIND $keywords AS keyword
                    MERGE (k:Keyword:ActionPilot {name: keyword})
                    MERGE (c)-[:MENTIONS]->(k)
                    """,
                    doc_id=doc_id,
                    title=title,
                    source=source,
                    chunk_id=chunk_id,
                    text=doc.page_content,
                    heading=heading,
                    keywords=keywords,
                )
        return True
    except Exception:
        global _NEO4J_UNAVAILABLE
        _NEO4J_UNAVAILABLE = True
        return False
    finally:
        driver.close()


def query_neo4j_context(query: str, limit: int = 4) -> list[dict[str, str]]:
    driver = _driver()
    if driver is None:
        return []

    keywords = _keywords(query, limit=8)
    if not keywords:
        return []

    try:
        with driver.session() as session:
            result = session.run(
                """
                MATCH (k:Keyword)<-[:MENTIONS]-(c:Chunk)<-[:HAS_CHUNK]-(d:Document)
                WHERE k.name IN $keywords
                WITH c, d, count(k) AS score
                RETURN d.title AS title, c.heading AS heading, c.text AS text, score
                ORDER BY score DESC
                LIMIT $limit
                """,
                keywords=keywords,
                limit=limit,
            )
            return [
                {
                    "title": record["title"],
                    "heading": record["heading"],
                    "text": record["text"],
                }
                for record in result
            ]
    except Exception:
        global _NEO4J_UNAVAILABLE
        _NEO4J_UNAVAILABLE = True
        return []
    finally:
        driver.close()
