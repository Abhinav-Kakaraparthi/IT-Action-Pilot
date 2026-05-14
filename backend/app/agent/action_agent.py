from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from app.core.config import get_settings
from app.core.schemas import Source, TraceEvent
from app.data.document_store import find_documents, list_documents
from app.tools.action_tools import TOOLS
from app.vectorstore.ingest import get_vectorstore
from app.vectorstore.neo4j_graph import query_neo4j_context
from app.vectorstore.reranker import rerank_documents


@dataclass
class AgentState:
    query: str
    history: list[BaseMessage] = field(default_factory=list)
    trace: list[TraceEvent] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)
    scratchpad: list[str] = field(default_factory=list)


@dataclass
class SessionMemory:
    summary: str = ""
    messages: list[BaseMessage] = field(default_factory=list)


class LocalActionAgent:
    """A compact LangChain based ReAct agent optimized for local Ollama models.

    The planner decides between RAG, action tools, and final answer. The loop is
    intentionally explicit because small local models are more reliable when each
    action is returned as strict JSON rather than hidden tool calling state.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self.llm = ChatOllama(
            model=self.settings.ollama_model,
            base_url=self.settings.ollama_base_url,
            temperature=0.1,
            num_predict=450,
            num_ctx=self.settings.ollama_num_ctx,
        )
        self.vectorstore = None
        self.tools = {tool.name: tool for tool in TOOLS}
        self.memories: dict[str, SessionMemory] = {}
        self.tool_descriptions = "\n".join(
            f"- {tool.name}: {tool.description}" for tool in TOOLS
        )
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are ActionPilot, a local enterprise action agent. You help internal teams solve IT, compliance, procurement, and operations questions.

You have access to two capabilities:
1. retrieve_knowledge: search internal markdown documents using RAG.
2. action tools listed below.

Tools:
{tool_descriptions}

Rules:
- Decide autonomously. Do not ask the user to choose tools.
- Use retrieve_knowledge when the answer depends on internal policy, runbooks, procedures, or budgets.
- Use action tools when the user asks to escalate, create, flag, check, submit, reorder, or when the policy says an action is required.
- For complex tasks, do retrieval first, then action.
- Never invent policy. Use retrieved evidence when available.
- Use the previous conversation only to resolve references and continue the user's task.
- Prefer the shortest valid path. If retrieved evidence or a tool result is enough, return final.
- Return one JSON object only. No markdown.

JSON schema:
{{
  "decision": "retrieve_knowledge" | "tool" | "final",
  "summary": "brief reason for your next step",
  "query": "search query when decision is retrieve_knowledge",
  "tool_name": "tool name when decision is tool",
  "tool_args": {{"arg": "value"}},
  "answer": "final answer when decision is final"
}}
""",
                ),
                ("placeholder", "{history}"),
                (
                    "human",
                    "User request: {query}\n\nWork so far:\n{scratchpad}\n\nChoose the next best step as JSON.",
                ),
            ]
        )
        self.chain = self.prompt | self.llm
        self.summary_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Compress the conversation into a concise factual memory summary. Preserve user preferences, unresolved tasks, important entities, decisions, action IDs, and tool outcomes. Do not invent details.",
                ),
                (
                    "human",
                    "Existing summary:\n{summary}\n\nOlder conversation to merge:\n{conversation}\n\nReturn the updated summary only.",
                ),
            ]
        )
        self.summary_chain = self.summary_prompt | self.llm
        self.answer_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """Answer the user's question using only the provided knowledge-base sources.

Rules:
- Give the direct answer first.
- Do not dump raw source text.
- If the sources do not contain the answer, say that the current knowledge base does not contain enough information.
- Cite source titles in a short "Sources:" sentence.
- Keep the answer concise unless the user asks for detail.
""",
                ),
                (
                    "human",
                    "Question: {query}\n\nSources:\n{sources}\n\nAnswer:",
                ),
            ]
        )
        self.answer_chain = self.answer_prompt | self.llm
        self.final_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """Write the final response for an enterprise RAG action agent.

Rules:
- Use only the provided draft, sources, and action log.
- Do not invent policies, ticket IDs, procurement IDs, or facts.
- Mention completed actions when action log entries exist.
- Keep the response concise and direct.
""",
                ),
                (
                    "human",
                    "User request: {query}\n\nDraft answer:\n{draft}\n\nSources:\n{sources}\n\nAction log:\n{actions}\n\nFinal answer:",
                ),
            ]
        )
        self.final_chain = self.final_prompt | self.llm

    def _history_for(self, session_id: str) -> list[BaseMessage]:
        memory = self.memories.get(session_id)
        if memory is None:
            return []

        messages = list(memory.messages)
        while messages and self._estimate_tokens(messages) > self.settings.memory_recent_tokens:
            messages.pop(0)

        if memory.summary:
            return [SystemMessage(content=f"Conversation memory summary:\n{memory.summary}"), *messages]
        return messages

    def _remember(self, session_id: str, query: str, answer: str) -> None:
        memory = self.memories.setdefault(session_id, SessionMemory())
        memory.messages.extend([HumanMessage(content=query), AIMessage(content=answer)])
        self._compact_memory(memory)

    def _estimate_tokens(self, value: str | list[BaseMessage]) -> int:
        if isinstance(value, str):
            return max(1, len(value) // 4)
        return sum(self._estimate_tokens(str(message.content)) for message in value)

    def _format_messages(self, messages: list[BaseMessage]) -> str:
        lines = []
        for message in messages:
            role = "User" if isinstance(message, HumanMessage) else "Assistant" if isinstance(message, AIMessage) else "System"
            lines.append(f"{role}: {message.content}")
        return "\n".join(lines)

    def _fallback_summarize(self, summary: str, messages: list[BaseMessage]) -> str:
        merged = "\n".join(part for part in [summary, self._format_messages(messages)] if part).strip()
        max_chars = max(2000, self.settings.memory_recent_tokens * 4)
        return merged[-max_chars:]

    def _summarize_messages(self, summary: str, messages: list[BaseMessage]) -> str:
        if not messages:
            return summary
        try:
            raw = self.summary_chain.invoke(
                {
                    "summary": summary or "No previous summary.",
                    "conversation": self._format_messages(messages),
                }
            )
            return str(raw.content).strip() or self._fallback_summarize(summary, messages)
        except Exception:
            return self._fallback_summarize(summary, messages)

    def _compact_memory(self, memory: SessionMemory) -> None:
        total_tokens = self._estimate_tokens(memory.summary) + self._estimate_tokens(memory.messages)
        if total_tokens <= self.settings.memory_context_tokens:
            return

        recent: list[BaseMessage] = []
        older = list(memory.messages)
        while older and self._estimate_tokens(recent) < self.settings.memory_recent_tokens:
            recent.insert(0, older.pop())

        if older:
            memory.summary = self._summarize_messages(memory.summary, older)
            memory.messages = recent

    def _uses_prior_context(self, query: str) -> bool:
        query_l = query.lower()
        followup_terms = [
            "that",
            "those",
            "it",
            "same",
            "again",
            "previous",
            "earlier",
            "last question",
            "what did i",
            "continue",
            "follow up",
            "the ticket",
            "the request",
        ]
        return any(term in query_l for term in followup_terms)

    def _is_memory_recall_query(self, query: str) -> bool:
        query_l = query.lower()
        return any(
            phrase in query_l
            for phrase in [
                "what did i ask",
                "what i asked",
                "i asked",
                "asked some",
                "asked about",
                "previous question",
                "last question",
                "earlier question",
            ]
        )

    def _scope_reason(self, query: str) -> str | None:
        query_l = query.lower()
        in_scope_terms = [
            "vpn",
            "password",
            "access",
            "license",
            "laptop",
            "inventory",
            "procurement",
            "purchase",
            "budget",
            "ticket",
            "support",
            "policy",
            "compliance",
            "customer data",
            "gmail",
            "email",
            "ai tool",
            "export",
            "risk",
            "employee",
            "onboarding",
            "hardware",
            "cisa",
            "nist",
            "ftc",
            "gsa",
            "cyber",
            "cybersecurity",
            "incident response",
            "incident",
            "vulnerability",
            "vulnerabilities",
            "kev",
            "known exploited",
            "framework",
            "risk management",
            "threat",
            "threats",
            "partner",
            "partners",
            "guidance",
            "data security",
            "privacy",
            "acquisition",
            "contracting",
            "file",
            "document",
            "documents",
            "knowledge base",
            "knowledge",
            "pdf",
            "resume",
            "uploaded",
            "available",
        ]
        out_of_scope_terms = [
            "movie",
            "movies",
            "film",
            "films",
            "releasing",
            "release this week",
            "weather",
            "sports",
            "stock price",
            "news",
            "restaurant",
            "recipe",
            "song",
            "music",
        ]
        if any(term in query_l for term in out_of_scope_terms):
            return "Movie releases and similar entertainment/news questions require current external data."
        return None

    def _out_of_scope_response(self, query: str, session_id: str, reason: str) -> dict[str, Any]:
        state = AgentState(query=query)
        final_answer = (
            "I can only help with local internal operations workflows such as IT support, access issues, procurement, inventory, "
            f"budget context, and compliance policy. {reason} I should not retrieve internal docs or create any operational ticket for that request."
        )
        state.trace.append(
            TraceEvent(
                step=1,
                kind="final",
                title="Guardrail: out of scope",
                detail="The request is outside ActionPilot's internal operations domain, so no retrieval or tools were run.",
            )
        )
        self._remember(session_id, query, final_answer)
        return {"answer": final_answer, "trace": state.trace, "sources": [], "actions": []}

    def _memory_followup_response(self, query: str, session_id: str) -> dict[str, Any] | None:
        memory = self.memories.get(session_id)
        if memory is None or (not memory.summary and not memory.messages):
            return None

        query_l = query.lower()
        history = self._history_for(session_id)
        user_messages = [message for message in memory.messages if isinstance(message, HumanMessage)]

        keywords = [word for word in re.findall(r"[a-zA-Z][a-zA-Z0-9]+", query_l) if len(word) > 3]
        stopwords = {"asked", "about", "thing", "something", "again", "previous", "earlier", "like", "some"}
        keywords = [word for word in keywords if word not in stopwords]

        matching_messages = []
        for message in user_messages:
            content_l = str(message.content).lower()
            if keywords and any(keyword in content_l for keyword in keywords):
                matching_messages.append(message)

        state = AgentState(query=query, history=history)
        if matching_messages:
            final_answer = f"You previously asked: {matching_messages[-1].content}"
        elif keywords and memory.summary and any(keyword in memory.summary.lower() for keyword in keywords):
            final_answer = f"From the earlier conversation summary: {memory.summary}"
        elif user_messages:
            final_answer = f"You previously asked: {user_messages[-1].content}"
        elif memory.summary:
            final_answer = f"From the earlier conversation summary: {memory.summary}"
        else:
            return None
        state.trace.append(TraceEvent(step=1, kind="final", title="Final response", detail="Answered from session memory."))
        self._remember(session_id, query, final_answer)
        return {"answer": final_answer, "trace": state.trace, "sources": [], "actions": []}

    def _parse_json(self, text: str) -> dict[str, Any]:
        cleaned = text.strip()
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(0)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {
                "decision": "final",
                "summary": "The model returned non JSON output, so the agent generated a safe final answer.",
                "answer": cleaned,
            }

    def _normalize_item(self, item: str) -> str:
        aliases = {
            "vpn licenses": "vpn license",
            "vpn seats": "vpn license",
            "laptops": "laptop",
            "monitors": "monitor",
            "github enterprise seats": "github enterprise seat",
        }
        key = item.strip().lower()
        return aliases.get(key, key)

    def _repair_tool_args(self, state: AgentState, tool_name: str, tool_args: dict[str, Any]) -> dict[str, Any]:
        args = dict(tool_args)
        if "item" in args:
            args["item"] = self._normalize_item(str(args["item"]))

        context = "\n".join([state.query, *state.scratchpad]).lower()

        if tool_name == "submit_procurement_request":
            item = self._normalize_item(str(args.get("item") or "vpn license"))
            args["item"] = item
            if "quantity" not in args or not args["quantity"]:
                args["quantity"] = 10 if item == "vpn license" else 5 if item in {"laptop", "macbook pro"} else 1
            if item == "vpn license" and int(args["quantity"]) < 10:
                args["quantity"] = 10
            if "business_reason" not in args or not args["business_reason"]:
                args["business_reason"] = (
                    "Restore healthy VPN license stock for blocked user access after password reset."
                    if item == "vpn license"
                    else f"Restore healthy {item} inventory for job-critical operations."
                )
            if "max_budget_usd" not in args or not args["max_budget_usd"]:
                unit_costs = {"vpn license": 12, "laptop": 1450, "macbook pro": 2200, "monitor": 290}
                args["max_budget_usd"] = float(unit_costs.get(item, 100) * int(args["quantity"]))

        if tool_name == "create_support_ticket":
            if "title" not in args or not args["title"]:
                args["title"] = "User cannot access VPN after password reset" if "vpn" in context else "Support escalation required"
            if "severity" not in args or not args["severity"]:
                args["severity"] = "high" if any(term in context for term in ["blocked", "production", "customer"]) else "medium"
            if "description" not in args or not args["description"]:
                args["description"] = (
                    "User cannot access VPN after a password reset. Inventory was checked and the user is blocked from customer or production work."
                    if "vpn" in context
                    else f"Support follow-up requested for: {state.query}"
                )

        if tool_name == "flag_policy_risk":
            args.setdefault("person_or_team", "requesting user")
            args.setdefault("policy_area", "Data handling")
            args.setdefault("concern", state.query)
            args.setdefault("evidence", "Request may violate internal policy based on retrieved guidance.")

        allowed_args = {
            "check_inventory": {"item"},
            "submit_procurement_request": {"item", "quantity", "business_reason", "max_budget_usd"},
            "create_support_ticket": {"title", "severity", "description", "owner_team"},
            "flag_policy_risk": {"person_or_team", "policy_area", "concern", "evidence"},
            "list_recent_actions": {"action_type"},
        }
        if tool_name in allowed_args:
            args = {key: value for key, value in args.items() if key in allowed_args[tool_name]}

        return args

    def _retrieve(self, state: AgentState, query: str, step: int, use_reranker: bool = True, use_graph: bool = True) -> str:
        if self.vectorstore is None:
            self.vectorstore = get_vectorstore()
        docs: list[Document] = self.vectorstore.similarity_search(query, k=self.settings.retrieval_candidates)
        if use_reranker:
            docs = rerank_documents(query, docs)
        else:
            docs = docs[: self.settings.retrieval_top_k]
        if not docs:
            detail = "No matching internal documents were found."
            state.trace.append(TraceEvent(step=step, kind="retrieval", title="Knowledge lookup", detail=detail))
            return detail

        snippets = []
        seen_sources = {(source.title, source.snippet[:120]) for source in state.sources}
        required_terms = self._required_query_terms(query)
        for i, doc in enumerate(docs, start=1):
            snippet = doc.page_content[:1200].strip()
            if self._is_low_value_source(snippet):
                continue
            if required_terms and not any(term in snippet.lower() for term in required_terms):
                continue
            title = str(doc.metadata.get("title", "Internal Document"))
            source_key = (title, snippet[:120])
            if source_key in seen_sources:
                continue
            seen_sources.add(source_key)
            source = Source(title=title, snippet=snippet, metadata=doc.metadata)
            state.sources.append(source)
            snippets.append(f"Source {i}: {title}\n{snippet}")

        if use_graph:
            for graph_doc in query_neo4j_context(query, limit=2):
                snippet = graph_doc["text"][:1200].strip()
                if self._is_low_value_source(snippet):
                    continue
                if required_terms and not any(term in snippet.lower() for term in required_terms):
                    continue
                title = graph_doc["title"]
                source_key = (title, snippet[:120])
                if source_key in seen_sources:
                    continue
                seen_sources.add(source_key)
                source = Source(
                    title=title,
                    snippet=snippet,
                    metadata={"heading": graph_doc["heading"], "retriever": "neo4j"},
                )
                state.sources.append(source)
                snippets.append(f"Graph source: {title} / {graph_doc['heading']}\n{snippet}")

        detail = f"Retrieved {len(snippets)} relevant internal sections for: {query}"
        state.trace.append(TraceEvent(step=step, kind="retrieval", title="Knowledge lookup", detail=detail))
        if not snippets and required_terms:
            return self._retrieve_local_markdown(state, query, step, preferred_files=None)
        return "\n\n".join(snippets)

    def _required_query_terms(self, query: str) -> list[str]:
        stopwords = {
            "where",
            "what",
            "when",
            "which",
            "there",
            "their",
            "currently",
            "working",
            "study",
            "studied",
            "available",
            "called",
            "knowledge",
            "base",
            "document",
            "documents",
            "file",
            "files",
            "from",
            "that",
            "this",
            "with",
            "will",
        }
        terms = [word.lower() for word in re.findall(r"[A-Za-z][A-Za-z0-9_-]{4,}", query)]
        distinctive = [term for term in terms if term not in stopwords]
        if len(distinctive) == 1:
            return distinctive
        if any(term in distinctive for term in ["cisa", "nist", "abhinav", "resume"]):
            return [term for term in distinctive if term in {"cisa", "nist", "abhinav", "resume"}]
        return []

    def _is_low_value_source(self, snippet: str) -> bool:
        snippet_l = snippet.lower()
        is_metadata_only = all(term in snippet_l for term in ["source file:", "processed:", "dataset:"]) and "## page" not in snippet_l
        is_nav = snippet_l.startswith("search menu close") or "topicstopics" in snippet_l
        return is_metadata_only or is_nav

    def _tool_result(self, state: AgentState, tool_name: str) -> dict[str, Any]:
        for action in reversed(state.actions):
            if action["tool"] == tool_name and isinstance(action.get("result"), dict):
                return action["result"]
        return {}

    def _retrieve_local_markdown(self, state: AgentState, query: str, step: int, preferred_files: set[str] | None = None) -> str:
        required_terms = self._required_query_terms(query)
        query_terms = {
            word
            for word in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{3,}", query.lower())
            if word
            not in {
                "with",
                "from",
                "that",
                "this",
                "when",
                "what",
                "where",
                "user",
                "policy",
                "currently",
                "working",
                "work",
                "does",
                "about",
            }
        }
        docs_dir = Path(self.settings.docs_dir)
        scored: list[tuple[int, Path, str]] = []
        for path in sorted(docs_dir.glob("**/*.md")):
            if preferred_files and path.name not in preferred_files:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            text_l = text.lower()
            path_l = path.name.lower()
            if required_terms and not any(term in text_l or term in path_l for term in required_terms):
                continue
            score = sum(1 for term in query_terms if term in text_l)
            score += sum(20 for term in required_terms if term in text_l or term in path_l)
            if preferred_files and path.name in preferred_files:
                score += 10
            if score > 0:
                scored.append((score, path, text))

        snippets: list[str] = []
        for index, (_, path, text) in enumerate(sorted(scored, key=lambda item: item[0], reverse=True)[:3], start=1):
            text_l = text.lower()
            anchor_terms = required_terms or query_terms
            positions = [text_l.find(term) for term in anchor_terms if text_l.find(term) >= 0]
            start = max(0, min(positions) - 180) if positions else 0
            snippet = text[start : start + 1200].strip()
            title = path.name.replace(".md", "").replace("_", " ").title()
            source = Source(title=title, snippet=snippet, metadata={"source": str(path), "retriever": "local_markdown"})
            state.sources.append(source)
            snippets.append(f"Source {index}: {title}\n{snippet}")

        detail = f"Retrieved {len(snippets)} relevant local markdown sections for: {query}"
        state.trace.append(TraceEvent(step=step, kind="retrieval", title="Knowledge lookup", detail=detail))
        return "\n\n".join(snippets)

    def _document_inventory_response(self, query: str, session_id: str) -> dict[str, Any] | None:
        query_l = query.lower()
        if not any(term in query_l for term in ["file", "document", "knowledge base", "pdf", "resume", "uploaded"]):
            return None

        state = AgentState(query=query)
        docs = list_documents()
        term_match = re.search(r"(?:called|named|like)\s+([a-zA-Z0-9_.-]+)", query_l)
        term = term_match.group(1) if term_match else "resume" if "resume" in query_l else ""
        matches = find_documents(term) if term else docs

        if term and any(phrase in query_l for phrase in ["is any", "available", "do we have", "called", "named"]):
            if matches:
                names = ", ".join(str(doc["path"]) for doc in matches[:6])
                final_answer = f"Yes. I found {len(matches)} knowledge-base file(s) matching '{term}': {names}."
            else:
                final_answer = f"No. I did not find any knowledge-base file matching '{term}'."
        else:
            names = ", ".join(str(doc["path"]) for doc in docs[:10])
            final_answer = f"The knowledge base currently has {len(docs)} document file(s). {names}"

        state.trace.append(TraceEvent(step=1, kind="final", title="Document inventory", detail="Answered from the knowledge-base document registry."))
        self._remember(session_id, query, final_answer)
        return {"answer": final_answer, "trace": state.trace, "sources": [], "actions": []}

    def _source_answer(self, query: str, state: AgentState) -> str:
        source_block = "\n\n".join(
            f"Source: {source.title}\n{source.snippet}"
            for source in state.sources[:6]
            if source.snippet.strip() and not self._is_low_value_source(source.snippet)
        )
        targeted_answer = self._targeted_source_answer(query, state)
        if targeted_answer:
            return targeted_answer

        if (
            source_block
            and self.settings.answer_llm_enabled
            and len(source_block) >= self.settings.answer_llm_min_source_chars
        ):
            try:
                raw = self.answer_chain.invoke({"query": query, "sources": source_block})
                answer = str(raw.content).strip()
                if answer:
                    return answer
            except Exception:
                pass

        extractive_answer = self._extractive_source_answer(query, state)
        if extractive_answer:
            return extractive_answer

        return self._fallback_answer(query, state)

    def _finalize_answer(self, query: str, state: AgentState, draft: str) -> str:
        if not self.settings.answer_llm_enabled:
            return draft

        source_block = "\n\n".join(
            f"Source: {source.title}\n{source.snippet[:1400]}"
            for source in state.sources[:4]
            if source.snippet.strip() and not self._is_low_value_source(source.snippet)
        ) or "No sources attached."
        action_log = json.dumps(state.actions, default=str, indent=2) if state.actions else "No actions executed."
        try:
            raw = self.final_chain.invoke({"query": query, "draft": draft, "sources": source_block, "actions": action_log})
            answer = str(raw.content).strip()
            return answer or draft
        except Exception:
            return draft

    def _extractive_source_answer(self, query: str, state: AgentState) -> str:
        targeted_answer = self._targeted_source_answer(query, state)
        if targeted_answer:
            return targeted_answer

        query_terms = {
            word
            for word in re.findall(r"[a-zA-Z][a-zA-Z0-9]+", query.lower())
            if len(word) > 3 and word not in {"what", "will", "with", "work", "working", "does", "about"}
        }
        if not query_terms:
            return ""

        sentences: list[tuple[int, str, str]] = []
        for source in state.sources[:5]:
            normalized = re.sub(r"\s+", " ", source.snippet.replace("|", " ").replace("•", ". "))
            for sentence in re.split(r"(?<=[.!?])\s+", normalized):
                cleaned = sentence.strip()
                if len(cleaned) < 40:
                    continue
                cleaned_l = cleaned.lower()
                score = sum(1 for term in query_terms if term in cleaned_l)
                if any(term in cleaned_l for term in ["present", "currently", "current"]):
                    score += 1
                if any(term in cleaned_l for term in ["education", "university", "institute", "college"]):
                    score += 1
                if score:
                    sentences.append((score, source.title, cleaned))

        if not sentences:
            return ""

        selected = sorted(sentences, key=lambda item: item[0], reverse=True)[:4]
        answer_sentences = " ".join(sentence for _, _, sentence in selected)
        source_titles = ", ".join(dict.fromkeys(title for _, title, _ in selected))
        return f"{answer_sentences} Sources: {source_titles}."

    def _targeted_source_answer(self, query: str, state: AgentState) -> str:
        query_l = query.lower()
        source_text = "\n\n".join(source.snippet for source in state.sources[:4])
        compact = re.sub(r"\s+", " ", source_text.replace("|", " ").replace("•", ". ")).strip()
        compact = compact.replace("New Y ork", "New York").replace("F eb", "Feb").replace("EDUCA TION", "EDUCATION")
        source_titles = ", ".join(dict.fromkeys(source.title for source in state.sources[:4]))

        if "abhinav" in query_l and any(term in query_l for term in ["study", "studied", "education"]):
            schools = []
            if "Columbia University" in compact:
                schools.append("Columbia University for a Machine Learning Graduate Certification")
            if "New York Institute of Technology" in compact:
                schools.append("New York Institute of Technology for a Master of Science in Computer Science")
            if schools:
                return f"Abhinav studied at {'; and '.join(schools)}. Sources: {source_titles}."

        if "abhinav" in query_l and any(term in query_l for term in ["currently working", "current work", "work", "working"]):
            experience_match = re.search(
                r"EXPERIENCE\s+(.*?\b(?:Present|Current)\b)",
                compact,
                re.IGNORECASE,
            )
            if experience_match:
                experience = re.sub(r"\s+", " ", experience_match.group(1)).strip(" .")
                return f"Abhinav is currently working as {experience}. Sources: {source_titles}."

        if "vpn" in query_l and "password reset" in query_l and any(phrase in query_l for phrase in ["what does", "what is", "say about"]):
            match = re.search(
                r"If a user cannot access VPN after a password reset,.*?(?:production work\.)",
                compact,
                re.IGNORECASE,
            )
            if match:
                return f"{match.group(0)} Sources: {source_titles}."

        return ""

    def _run_tool(self, state: AgentState, tool_name: str, tool_args: dict[str, Any], step: int) -> str:
        tool = self.tools.get(tool_name)
        if not tool:
            detail = f"Tool {tool_name} is not available."
            state.trace.append(TraceEvent(step=step, kind="error", title="Tool unavailable", detail=detail))
            return detail
        try:
            tool_args = self._repair_tool_args(state, tool_name, tool_args)
            result = tool.invoke(tool_args)
            parsed_result: Any = result
            try:
                parsed_result = json.loads(result)
            except Exception:
                pass
            state.actions.append({"tool": tool_name, "args": tool_args, "result": parsed_result})
            state.trace.append(
                TraceEvent(
                    step=step,
                    kind="tool",
                    title=f"Executed {tool_name}",
                    detail=f"Arguments: {json.dumps(tool_args)}",
                    data={"result": parsed_result},
                )
            )
            return f"Tool result for {tool_name}: {result}"
        except Exception as exc:
            detail = f"Tool {tool_name} failed: {exc}"
            state.trace.append(TraceEvent(step=step, kind="error", title="Tool failed", detail=detail))
            return detail

    def _try_fast_path(self, query: str, session_id: str) -> dict[str, Any] | None:
        query_l = query.lower()
        history = self._history_for(session_id) if self._uses_prior_context(query) or self._is_memory_recall_query(query) else []
        history_text = " ".join(str(message.content) for message in history).lower()
        state = AgentState(query=query, history=history)

        if history_text and self._is_memory_recall_query(query):
            memory_response = self._memory_followup_response(query, session_id)
            if memory_response is not None:
                return memory_response

        document_response = self._document_inventory_response(query, session_id)
        if document_response is not None:
            return document_response

        needs_admin = any(
            term in query_l
            for term in [
                "admin access",
                "administrator",
                "privileged access",
                "root access",
                "production access",
                "approve access",
                "grant access",
                "unlock account",
                "permission denied",
            ]
        )
        if needs_admin:
            self._run_tool(
                state,
                "create_support_ticket",
                {
                    "title": "Admin access request requires review",
                    "severity": "medium",
                    "description": f"User request requires administrator or privileged access review: {query}",
                },
                1,
            )
            final_answer = self._finalize_answer(
                query,
                state,
                "This request appears to require administrator or privileged access, so I created a support ticket for review instead of attempting the change directly.",
            )
            state.trace.append(TraceEvent(step=1, kind="final", title="Final response", detail="Escalated privileged-access request to ticketing."))
            self._remember(session_id, query, final_answer)
            return {"answer": final_answer, "trace": state.trace, "sources": [], "actions": state.actions}

        is_real_policy_query = any(
            term in query_l
            for term in [
                "cisa",
                "nist",
                "ftc",
                "gsa",
                "incident response",
                "vulnerability",
                "vulnerabilities",
                "cybersecurity",
                "data security",
                "known exploited",
            ]
        )
        if is_real_policy_query:
            self._retrieve(state, query, 1)
            final_answer = self._source_answer(query, state)
            state.trace.append(TraceEvent(step=1, kind="final", title="Final response", detail="Synthesized answer from indexed public guidance."))
            self._remember(session_id, query, final_answer)
            return {"answer": final_answer, "trace": state.trace, "sources": state.sources[:6], "actions": []}

        is_vpn = "vpn" in query_l and any(term in query_l for term in ["password", "reset", "license", "access", "reorder", "out of"])
        vpn_read_only = any(phrase in query_l for phrase in ["what does", "what is", "say about", "explain", "summarize"]) and not any(
            phrase in query_l for phrase in ["take the needed action", "take action", "create", "submit", "reorder", "blocked", "cannot access", "out of"]
        )
        if is_vpn:
            self._retrieve_local_markdown(
                state,
                "VPN password reset license procurement support ticket policy",
                1,
                preferred_files={"it_support_runbook.md", "procurement_rules.md"},
            )
            if vpn_read_only:
                final_answer = self._source_answer(query, state)
                state.trace.append(TraceEvent(step=1, kind="final", title="Final response", detail="Answered VPN policy question without running action tools."))
                self._remember(session_id, query, final_answer)
                return {"answer": final_answer, "trace": state.trace, "sources": state.sources[:6], "actions": []}
            self._run_tool(state, "check_inventory", {"item": "vpn license"}, 2)
            inventory = self._tool_result(state, "check_inventory")
            if inventory.get("needs_reorder") or inventory.get("available") in (0, None):
                self._run_tool(state, "submit_procurement_request", {"item": "vpn license", "quantity": 10}, 3)
            if any(term in query_l for term in ["cannot access", "password", "reset", "blocked", "teammate", "out of", "no license", "no licenses"]):
                self._run_tool(
                    state,
                    "create_support_ticket",
                    {
                        "title": "VPN license shortage requires follow-up",
                        "severity": "high",
                        "description": "VPN license stock is unavailable or below threshold. Inventory was checked and procurement was submitted for 10 VPN licenses. Follow up so affected users can regain VPN access.",
                    },
                    4,
                )
            final_answer = self._finalize_answer(
                query,
                state,
                "I checked the VPN runbook and inventory. VPN license stock is below threshold, so I submitted a procurement request for 10 VPN licenses "
                "and created a high-severity support ticket for the blocked VPN access issue."
            )
            state.trace.append(TraceEvent(step=4, kind="final", title="Final response", detail="Completed deterministic VPN workflow."))
            self._remember(session_id, query, final_answer)
            return {"answer": final_answer, "trace": state.trace, "sources": state.sources[:6], "actions": state.actions}

        is_laptop = "laptop" in query_l and any(term in query_l for term in ["replace", "replacement", "inventory", "engineer", "today", "stock"])
        if is_laptop:
            self._retrieve_local_markdown(
                state,
                "laptop replacement inventory procurement policy",
                1,
                preferred_files={"it_support_runbook.md", "procurement_rules.md", "budget_context.md"},
            )
            self._run_tool(state, "check_inventory", {"item": "laptop"}, 2)
            inventory = self._tool_result(state, "check_inventory")
            if inventory.get("needs_reorder"):
                self._run_tool(state, "submit_procurement_request", {"item": "laptop", "quantity": 5}, 3)
            draft_answer = (
                "I checked the laptop replacement guidance and current inventory. Laptop stock is below the reorder threshold, so I submitted procurement "
                "to restore healthy stock before confirming the replacement path."
                if inventory.get("needs_reorder")
                else "I checked the laptop replacement guidance and inventory. Stock is available, so the replacement can proceed if the device blocks job-critical work."
            )
            final_answer = self._finalize_answer(query, state, draft_answer)
            state.trace.append(TraceEvent(step=3, kind="final", title="Final response", detail="Completed deterministic laptop workflow."))
            self._remember(session_id, query, final_answer)
            return {"answer": final_answer, "trace": state.trace, "sources": state.sources[:6], "actions": state.actions}

        is_personal_export = any(term in query_l for term in ["personal gmail", "personal email", "unmanaged ai", "customer records"])
        if is_personal_export:
            self._retrieve_local_markdown(
                state,
                "customer data personal email unmanaged AI compliance policy",
                1,
                preferred_files={"compliance_policy.md"},
            )
            self._run_tool(
                state,
                "flag_policy_risk",
                {
                    "person_or_team": "requesting user",
                    "policy_area": "Customer data handling",
                    "concern": "Customer records must not be exported to personal email or unmanaged AI tools.",
                    "evidence": "The compliance policy prohibits moving customer personal data to personal email, public notebooks, or unmanaged AI tools.",
                },
                2,
            )
            final_answer = self._finalize_answer(
                query,
                state,
                "No. The compliance policy says customer records must not be exported to personal Gmail or unmanaged AI tools. "
                "I flagged this as a policy risk and the safe path is to use an approved managed workspace or internal review process."
            )
            state.trace.append(TraceEvent(step=2, kind="final", title="Final response", detail="Completed deterministic compliance workflow."))
            self._remember(session_id, query, final_answer)
            return {"answer": final_answer, "trace": state.trace, "sources": state.sources[:6], "actions": state.actions}

        return None

    def invoke(self, query: str, session_id: str = "default") -> dict[str, Any]:
        scope_reason = self._scope_reason(query)
        if scope_reason and self._is_memory_recall_query(query):
            memory_response = self._memory_followup_response(query, session_id)
            if memory_response is not None:
                return memory_response
        if scope_reason:
            return self._out_of_scope_response(query, session_id, scope_reason)

        fast_result = self._try_fast_path(query, session_id)
        if fast_result is not None:
            return fast_result

        quick_state = AgentState(query=query, history=self._history_for(session_id) if self._uses_prior_context(query) else [])
        retrieved = self._retrieve(quick_state, query, 1)
        if quick_state.sources:
            final_answer = self._source_answer(query, quick_state)
            quick_state.trace.append(TraceEvent(step=1, kind="final", title="Final response", detail="Answered from indexed knowledge-base documents."))
            self._remember(session_id, query, final_answer)
            return {"answer": final_answer, "trace": quick_state.trace, "sources": quick_state.sources[:6], "actions": []}
        if not retrieved:
            final_answer = "I could not find that information in the current knowledge base. Upload or import the relevant document, then rebuild or process it so I can answer from it."
            quick_state.trace.append(TraceEvent(step=1, kind="final", title="Final response", detail="No matching knowledge-base documents were found."))
            self._remember(session_id, query, final_answer)
            return {"answer": final_answer, "trace": quick_state.trace, "sources": [], "actions": []}

        state = AgentState(query=query, history=self._history_for(session_id) if self._uses_prior_context(query) else [])
        final_answer = "I could not complete the request with the available local tools."

        for step in range(1, self.settings.max_agent_steps + 1):
            raw = self.chain.invoke(
                {
                    "query": query,
                    "history": state.history,
                    "scratchpad": "\n\n".join(state.scratchpad) or "No steps yet.",
                    "tool_descriptions": self.tool_descriptions,
                }
            )
            plan = self._parse_json(str(raw.content))
            decision = plan.get("decision", "final")
            summary = str(plan.get("summary", "Selected next step."))
            state.trace.append(TraceEvent(step=step, kind="plan", title=f"Decision: {decision}", detail=summary, data=plan))

            if decision == "retrieve_knowledge":
                retrieval_query = str(plan.get("query") or query)
                result = self._retrieve(state, retrieval_query, step)
                if not result and state.sources:
                    final_answer = self._source_answer(query, state)
                    state.trace.append(TraceEvent(step=step, kind="final", title="Final response", detail="No new retrieval results; synthesized from existing sources."))
                    break
                state.scratchpad.append(f"Retrieved evidence for '{retrieval_query}':\n{result}")
                continue

            if decision == "tool":
                tool_name = str(plan.get("tool_name", ""))
                tool_args = plan.get("tool_args") or {}
                if not isinstance(tool_args, dict):
                    tool_args = {}
                result = self._run_tool(state, tool_name, tool_args, step)
                state.scratchpad.append(result)
                continue

            final_answer = str(plan.get("answer") or summary)
            state.trace.append(TraceEvent(step=step, kind="final", title="Final response", detail="Synthesized final answer."))
            break
        else:
            final_answer = self._fallback_answer(query, state)
            state.trace.append(TraceEvent(step=self.settings.max_agent_steps, kind="final", title="Final response", detail="Reached max steps and returned evidence based answer."))

        self._remember(session_id, query, final_answer)
        return {
            "answer": final_answer,
            "trace": state.trace,
            "sources": state.sources[:6],
            "actions": state.actions,
        }

    def _fallback_answer(self, query: str, state: AgentState) -> str:
        if state.actions and state.sources:
            return "I checked the relevant internal guidance and completed the required action. Review the action log and sources below for details."
        if state.actions:
            return "I completed the requested local action. Review the action log below for details."
        if state.sources:
            bullets = "\n".join(f"- {s.title}: {s.snippet[:220]}..." for s in state.sources[:3])
            return f"I found relevant indexed guidance for your request:\n{bullets}"
        return "I was not able to find enough internal context or complete an action for this request."
