from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import tool

from app.tools.github_issues import create_github_issue
from app.tools.runtime_store import append_record, read_records


@tool
def create_support_ticket(title: str, severity: str, description: str, owner_team: str = "Platform Operations") -> str:
    """Create a support ticket when the user needs escalation, engineering follow up, or incident handling."""
    github_issue = create_github_issue(title, severity, description, owner_team)
    external_fields: dict[str, Any] = {}
    if github_issue.get("enabled"):
        external_fields["external_ticketing_backend"] = "github"
        if github_issue.get("number"):
            external_fields["github_issue_number"] = github_issue["number"]
        if github_issue.get("url"):
            external_fields["github_issue_url"] = github_issue["url"]
        if github_issue.get("state"):
            external_fields["github_issue_state"] = github_issue["state"]
        if github_issue.get("error"):
            external_fields["external_sync_error"] = github_issue["error"]

    record = append_record(
        "tickets.json",
        {
            "title": title,
            "severity": severity,
            "description": description,
            "owner_team": owner_team,
            "status": "open",
            **external_fields,
        },
    )
    return json.dumps(record)


@tool
def flag_policy_risk(person_or_team: str, policy_area: str, concern: str, evidence: str) -> str:
    """Create a local compliance risk flag when a request may violate policy or requires review."""
    record = append_record(
        "policy_flags.json",
        {
            "person_or_team": person_or_team,
            "policy_area": policy_area,
            "concern": concern,
            "evidence": evidence,
            "status": "needs_review",
        },
    )
    return json.dumps(record)


@tool
def submit_procurement_request(item: str, quantity: int, business_reason: str, max_budget_usd: float) -> str:
    """Submit a mock procurement request for hardware, SaaS licenses, or replacement equipment."""
    record = append_record(
        "procurement_requests.json",
        {
            "item": item,
            "quantity": quantity,
            "business_reason": business_reason,
            "max_budget_usd": max_budget_usd,
            "status": "submitted",
        },
    )
    return json.dumps(record)


@tool
def check_inventory(item: str) -> str:
    """Check a local inventory snapshot for availability of hardware, licenses, and accessories."""
    inventory = {
        "laptop": {"available": 2, "reorder_threshold": 3, "unit_cost": 1450},
        "macbook pro": {"available": 1, "reorder_threshold": 2, "unit_cost": 2200},
        "monitor": {"available": 8, "reorder_threshold": 5, "unit_cost": 290},
        "vpn license": {"available": 0, "reorder_threshold": 5, "unit_cost": 12},
        "github enterprise seat": {"available": 4, "reorder_threshold": 10, "unit_cost": 21},
    }
    aliases = {
        "laptops": "laptop",
        "vpn licenses": "vpn license",
        "vpn seats": "vpn license",
        "github enterprise seats": "github enterprise seat",
        "monitors": "monitor",
    }
    key = aliases.get(item.strip().lower(), item.strip().lower())
    match = inventory.get(key)
    if not match:
        return json.dumps({"item": item, "available": None, "message": "Item not found in inventory snapshot."})
    return json.dumps({"item": key, **match, "needs_reorder": match["available"] < match["reorder_threshold"]})


@tool
def list_recent_actions(action_type: str = "all") -> str:
    """List previously created local actions such as tickets, policy flags, and procurement requests."""
    mapping = {
        "tickets": "tickets.json",
        "policy_flags": "policy_flags.json",
        "procurement": "procurement_requests.json",
    }
    if action_type == "all":
        data: dict[str, Any] = {key: read_records(file) for key, file in mapping.items()}
    else:
        data = {action_type: read_records(mapping.get(action_type, "tickets.json"))}
    return json.dumps(data)


TOOLS = [create_support_ticket, flag_policy_risk, submit_procurement_request, check_inventory, list_recent_actions]
