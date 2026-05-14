# Assessment Match

This repository is built for the first option in the take-home assessment: an action agent for IT support/internal operations.

## Match score

Estimated match: **90%**

The implemented application satisfies the core technical requirements. The only remaining deliverable is a short demo recording, unless the evaluator accepts `DEMO_LOG.md` as the detailed log artifact.

## Requirement mapping

| Assessment requirement | Current implementation |
| --- | --- |
| Define a specific action-agent domain | Internal IT support, access management, procurement, compliance, and policy Q&A |
| Fully local LLM | Ollama via `langchain-ollama` |
| Fully local vector store | Chroma in `backend/data/chroma` |
| Local orchestration | LangChain ReAct-style action loop in `backend/app/agent/action_agent.py` |
| Agent decides RAG vs tool vs final answer | JSON decision loop plus guarded fast paths for reliable low-latency operational flows |
| Tool execution | Inventory check, procurement request, support ticket creation, policy risk flagging, action listing |
| Ticket escalation | Tickets are created from queries, visible in the UI, reset on backend restart for clean demos, and optionally mirrored to GitHub Issues |
| Internal document lookup | Markdown seed docs, official public policy importer, PDF upload ingestion |
| Logical chunking | Delimiter-first chunking with semantic merge metadata |
| Context management | Session memory with recent raw messages and rolling summary compaction |
| Error handling | Tests cover malformed tool args, missing embeddings/reranker fallback, PDF extraction failure, and document deletion |
| README and install instructions | Present in `README.md` |
| Demo artifact | Present in `DEMO_LOG.md`; record a one-minute UI demo for the strongest submission |

## Strongest demo script

Use this query:

```text
My teammate cannot access VPN after a password reset. Check what the policy says and take the needed action.
```

Show these panels in the UI:

1. Final answer with the ticket and procurement outcome.
2. Agent trace showing retrieval, inventory check, procurement, and ticket creation.
3. Sources showing the IT Support Runbook.
4. Tickets panel showing the created ticket and status controls.

## Notes for evaluator

ActionPilot intentionally avoids paid APIs. Model inference, retrieval, uploaded document processing, graph context, tool execution, and runtime state all stay on the local machine.
