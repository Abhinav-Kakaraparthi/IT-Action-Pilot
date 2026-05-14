export type TraceEvent = {
  step: number;
  kind: "plan" | "retrieval" | "tool" | "final" | "error";
  title: string;
  detail: string;
  data?: Record<string, unknown>;
};

export type Source = {
  title: string;
  snippet: string;
  metadata: Record<string, unknown>;
};

export type ChatResponse = {
  answer: string;
  trace: TraceEvent[];
  sources: Source[];
  actions: Array<Record<string, unknown>>;
};

export type KnowledgeDocument = {
  id: string;
  name: string;
  path: string;
  type: string;
  size: number;
};

export type Ticket = {
  id: string;
  title: string;
  severity: string;
  description: string;
  owner_team: string;
  status: string;
  created_at: string;
  updated_at?: string;
};

const API_BASE = "http://127.0.0.1:8002/api";

export async function sendMessage(message: string): Promise<ChatResponse> {
  const res = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, session_id: "web" }),
  });
  if (!res.ok) {
    const error = await res.text();
    throw new Error(error || "Request failed");
  }
  return res.json();
}

export async function ingestDocs(): Promise<void> {
  const res = await fetch(`${API_BASE}/ingest?force=true`, { method: "POST" });
  if (!res.ok) throw new Error("Failed to index documents");
}

export async function importRealDocs(): Promise<void> {
  const res = await fetch(`${API_BASE}/import-real-docs`, { method: "POST" });
  if (!res.ok) {
    const error = await res.text();
    throw new Error(error || "Failed to import real documents");
  }
}

export async function uploadPdf(file: File): Promise<void> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/upload-pdf`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const error = await res.text();
    throw new Error(error || "Failed to process PDF");
  }
}

export async function getDocuments(): Promise<KnowledgeDocument[]> {
  const res = await fetch(`${API_BASE}/documents`);
  if (!res.ok) throw new Error("Failed to load documents");
  const data = await res.json();
  return data.documents;
}

export async function deleteDocument(documentId: string): Promise<void> {
  const res = await fetch(`${API_BASE}/documents/delete`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ document_id: documentId }),
  });
  if (!res.ok) {
    const error = await res.text();
    throw new Error(error || "Failed to delete document");
  }
}

export async function getTickets(): Promise<Ticket[]> {
  const res = await fetch(`${API_BASE}/actions`);
  if (!res.ok) throw new Error("Failed to load tickets");
  const data = await res.json();
  return data.tickets;
}

export async function updateTicketStatus(ticketId: string, status: "pending" | "attended" | "resolved"): Promise<void> {
  const res = await fetch(`${API_BASE}/tickets/${encodeURIComponent(ticketId)}/status`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ status }),
  });
  if (!res.ok) {
    const error = await res.text();
    throw new Error(error || "Failed to update ticket");
  }
}
