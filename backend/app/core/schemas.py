from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=2)
    session_id: str = "default"


class TraceEvent(BaseModel):
    step: int
    kind: Literal["plan", "retrieval", "tool", "final", "error"]
    title: str
    detail: str
    data: dict[str, Any] = Field(default_factory=dict)


class Source(BaseModel):
    title: str
    snippet: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    answer: str
    trace: list[TraceEvent]
    sources: list[Source]
    actions: list[dict[str, Any]]


class DeleteDocumentRequest(BaseModel):
    document_id: str


class TicketStatusRequest(BaseModel):
    status: Literal["pending", "attended", "resolved"]
