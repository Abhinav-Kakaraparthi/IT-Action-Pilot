import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { motion } from "framer-motion";
import { ArrowUpRight, Bot, Boxes, BrainCircuit, CheckCircle2, Database, FileText, Loader2, ShieldAlert, Sparkles, Trash2, Upload, Wrench } from "lucide-react";
import { ChatResponse, KnowledgeDocument, Ticket, deleteDocument, getDocuments, getTickets, importRealDocs, ingestDocs, sendMessage, updateTicketStatus, uploadPdf } from "./lib/api";
import "./styles.css";

const samples = [
  "My teammate cannot access VPN after a password reset. Check what the policy says and take the needed action.",
  "We are out of VPN licenses. Decide whether to reorder and submit the request if needed.",
  "Can I export customer records into my personal Gmail so I can summarize them with another AI tool?",
  "An engineer needs a laptop replacement today. Check inventory, policy, and create any needed action."
];

function App() {
  const [message, setMessage] = useState(samples[0]);
  const [loading, setLoading] = useState(false);
  const [indexing, setIndexing] = useState(false);
  const [importing, setImporting] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [documents, setDocuments] = useState<KnowledgeDocument[]>([]);
  const [tickets, setTickets] = useState<Ticket[]>([]);
  const [adminBusy, setAdminBusy] = useState<string | null>(null);
  const [response, setResponse] = useState<ChatResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    refreshAdminData();
  }, []);

  async function refreshAdminData() {
    try {
      const [docs, ticketRows] = await Promise.all([getDocuments(), getTickets()]);
      setDocuments(docs);
      setTickets(ticketRows);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load admin data");
    }
  }

  async function onSubmit(e?: React.FormEvent) {
    e?.preventDefault();
    if (!message.trim()) return;
    setLoading(true);
    setError(null);
    setResponse(null);
    try {
      const data = await sendMessage(message);
      setResponse(data);
      await refreshAdminData();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  }

  async function onIndex() {
    setIndexing(true);
    setError(null);
    try {
      await ingestDocs();
      await refreshAdminData();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Indexing failed");
    } finally {
      setIndexing(false);
    }
  }

  async function onImportRealDocs() {
    setImporting(true);
    setError(null);
    try {
      await importRealDocs();
      await refreshAdminData();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Import failed");
    } finally {
      setImporting(false);
    }
  }

  async function onPdfUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;
    if (file.type !== "application/pdf" && !file.name.toLowerCase().endsWith(".pdf")) {
      setError("Upload a PDF file.");
      return;
    }
    setUploading(true);
    setError(null);
    try {
      await uploadPdf(file);
      setMessage(`Summarize the uploaded PDF: ${file.name}`);
      await refreshAdminData();
    } catch (err) {
      setError(err instanceof Error ? err.message : "PDF processing failed");
    } finally {
      setUploading(false);
    }
  }

  async function onDeleteDocument(documentId: string) {
    setAdminBusy(documentId);
    setError(null);
    try {
      await deleteDocument(documentId);
      await refreshAdminData();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Delete failed");
    } finally {
      setAdminBusy(null);
    }
  }

  async function onTicketStatus(ticketId: string, status: "pending" | "attended" | "resolved") {
    setAdminBusy(`${ticketId}-${status}`);
    setError(null);
    try {
      await updateTicketStatus(ticketId, status);
      await refreshAdminData();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Status update failed");
    } finally {
      setAdminBusy(null);
    }
  }

  return (
    <main className="page-shell">
      <nav className="nav">
        <div className="brand">ActionPilot</div>
        <div className="nav-links"><a href="#workbench">Workbench</a><a href="#trace">Trace</a><a href="#sources">Sources</a></div>
      </nav>

      <section className="hero">
        <motion.div initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6 }}>
          <div className="eyebrow"><Sparkles size={16}/> local LangChain action agent</div>
          <h1>Enterprise work gets messy. This agent turns ambiguity into action.</h1>
          <p className="hero-copy">A fully local AI agent that reads internal documents, reasons through the next step, executes mock business tools, and explains the outcome through a polished product style interface.</p>
          <div className="hero-tags"><span>0 to 1 Agent</span><span>RAG</span><span>Tool Use</span><span>Ollama</span><span>Local vectors</span></div>
        </motion.div>
        <motion.div className="hero-card" initial={{ opacity: 0, scale: 0.96 }} animate={{ opacity: 1, scale: 1 }} transition={{ duration: 0.6, delay: 0.1 }}>
          <div className="hero-card-top"><BrainCircuit/><span>Autonomy score</span></div>
          <strong>RAG + Action Loop</strong>
          <p>Retrieves only when policy context is needed, calls tools when an operational action is required, and synthesizes a grounded final answer.</p>
          <button onClick={onIndex} disabled={indexing}>{indexing ? <Loader2 className="spin"/> : <Database/>}{indexing ? "Indexing" : "Rebuild local knowledge"}</button>
          <button onClick={onImportRealDocs} disabled={importing}>{importing ? <Loader2 className="spin"/> : <ShieldAlert/>}{importing ? "Importing" : "Import real policies"}</button>
          <label className={`upload-button ${uploading ? "disabled" : ""}`}>
            {uploading ? <Loader2 className="spin"/> : <Upload/>}
            {uploading ? "Processing PDF" : "Upload PDF"}
            <input type="file" accept="application/pdf,.pdf" onChange={onPdfUpload} disabled={uploading} />
          </label>
        </motion.div>
      </section>

      <section className="metric-grid">
        <Metric icon={<Database/>} label="Vector store" value="Local semantic index" />
        <Metric icon={<Bot/>} label="LLM runner" value="Ollama" />
        <Metric icon={<Wrench/>} label="Tools" value="Tickets, risk flags, procurement" />
      </section>

      <section id="workbench" className="workbench">
        <div className="section-title"><span>Agent workbench</span><h2>Ask something that needs both judgment and execution.</h2></div>
        <form onSubmit={onSubmit} className="composer">
          <textarea value={message} onChange={(e) => setMessage(e.target.value)} placeholder="Ask the agent to solve an internal operation task..." />
          <button disabled={loading}>{loading ? <Loader2 className="spin"/> : <ArrowUpRight/>}{loading ? "Thinking" : "Run agent"}</button>
        </form>
        <div className="samples">{samples.map((s) => <button key={s} onClick={() => setMessage(s)}>{s}</button>)}</div>
        {error && <div className="error">{error}</div>}
      </section>

      <section className="admin-grid">
        <div className="panel">
          <div className="panel-heading"><FileText/> Knowledge documents</div>
          {documents.length === 0 ? <p className="muted">No documents indexed.</p> : documents.slice(0, 12).map((doc) => (
            <div className="admin-row" key={doc.id}>
              <div><strong>{doc.name}</strong><span>{doc.path}</span></div>
              <button title="Delete document" onClick={() => onDeleteDocument(doc.id)} disabled={adminBusy === doc.id}>{adminBusy === doc.id ? <Loader2 className="spin"/> : <Trash2/>}</button>
            </div>
          ))}
        </div>
        <div className="panel">
          <div className="panel-heading"><Boxes/> Tickets</div>
          {tickets.length === 0 ? <p className="muted">No tickets raised.</p> : tickets.slice().reverse().slice(0, 8).map((ticket) => (
            <div className="ticket-row" key={ticket.id}>
              <div><strong>{ticket.title}</strong><span>{ticket.id} · {ticket.severity} · {ticket.status}</span></div>
              <div className="status-actions">
                {(["pending", "attended", "resolved"] as const).map((status) => (
                  <button key={status} disabled={ticket.status === status || adminBusy === `${ticket.id}-${status}`} onClick={() => onTicketStatus(ticket.id, status)}>{status}</button>
                ))}
              </div>
            </div>
          ))}
        </div>
      </section>

      {response && <Results response={response} />}
    </main>
  );
}

function Metric({ icon, label, value }: { icon: React.ReactNode; label: string; value: string }) {
  return <div className="metric-card"><div>{icon}</div><p>{label}</p><strong>{value}</strong></div>;
}

function Results({ response }: { response: ChatResponse }) {
  return (
    <section className="results">
      <div className="answer-card">
        <div className="card-label"><CheckCircle2/> Final answer</div>
        <p>{response.answer}</p>
      </div>
      <div className="two-column">
        <div id="trace" className="panel">
          <div className="panel-heading"><BrainCircuit/> Agent trace</div>
          {response.trace.map((event, idx) => <div className={`trace trace-${event.kind}`} key={`${event.step}-${idx}`}><span>Step {event.step}</span><strong>{event.title}</strong><p>{event.detail}</p></div>)}
        </div>
        <div className="panel">
          <div className="panel-heading"><Boxes/> Actions</div>
          {response.actions.length === 0 ? <p className="muted">No action tool was needed.</p> : response.actions.map((a, i) => <pre key={i}>{JSON.stringify(a, null, 2)}</pre>)}
        </div>
      </div>
      <div id="sources" className="source-grid">
        {response.sources.map((source, i) => <article className="source-card" key={`${source.title}-${i}`}><div><FileText size={18}/><strong>{source.title}</strong></div><p>{source.snippet}</p></article>)}
      </div>
    </section>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
