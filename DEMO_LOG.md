# Demo Log: First Assessment Challenge

## Query

My teammate cannot access VPN after a password reset. Check what the policy says and take the needed action.

## Expected trace in the UI

1. Decision: retrieve internal knowledge.
2. Retrieval: IT Support Runbook section about VPN incidents after password resets.
3. Decision: check inventory because the runbook says to verify VPN license availability first.
4. Tool: `check_inventory` returns low or unavailable VPN license stock and indicates reorder is needed.
5. Decision: submit procurement request because the policy says to reorder when no license is available.
6. Tool: `submit_procurement_request` creates a local procurement action for a batch of VPN licenses.
7. Decision: create support ticket because the teammate is blocked from customer or production work.
8. Tool: `create_support_ticket` creates a high severity Platform Operations ticket.
9. Final response: explains the policy basis, inventory result, procurement action, ticket ID/status, and source document.

## Example action records

```json
{
  "tool": "check_inventory",
  "args": {
    "item": "VPN licenses"
  },
  "result": {
    "item": "vpn license",
    "needs_reorder": true
  }
}
```

```json
{
  "tool": "submit_procurement_request",
  "args": {
    "item": "VPN licenses",
    "quantity": 10,
    "business_reason": "Production access blocker after password reset",
    "max_budget_usd": 120
  }
}
```

```json
{
  "tool": "create_support_ticket",
  "args": {
    "title": "User cannot access VPN after password reset",
    "severity": "high",
    "description": "User is unable to access VPN after a password reset and is blocked from customer or production work."
  }
}
```

## Why this satisfies the assessment

The query requires both document lookup and tool execution. The agent reads the internal runbook, decides that inventory has to be checked, executes operational tools, repairs missing tool arguments when needed, and synthesizes the final answer with evidence and action logs. Everything runs locally through Ollama, Chroma, LangChain, and local JSON persistence.
