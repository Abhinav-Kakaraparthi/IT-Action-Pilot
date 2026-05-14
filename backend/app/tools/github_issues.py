from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.core.config import get_settings


def github_ticketing_enabled() -> bool:
    settings = get_settings()
    return (
        settings.ticketing_backend.lower() in {"github", "both"}
        and bool(settings.github_token.strip())
        and bool(settings.github_repo.strip())
    )


def _github_request(method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    settings = get_settings()
    base_url = settings.github_api_url.rstrip("/")
    url = f"{base_url}{path}"
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = Request(
        url,
        data=body,
        method=method,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {settings.github_token}",
            "Content-Type": "application/json",
            "User-Agent": "ActionPilot",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urlopen(request, timeout=12) as response:
        response_body = response.read().decode("utf-8")
    return json.loads(response_body) if response_body else {}


def create_github_issue(title: str, severity: str, description: str, owner_team: str) -> dict[str, Any]:
    settings = get_settings()
    if not github_ticketing_enabled():
        return {"enabled": False, "message": "GitHub ticketing is not configured."}

    labels = ["actionpilot", f"severity:{severity.lower()}"]
    body = (
        f"## Escalation\n\n{description}\n\n"
        f"## Metadata\n\n"
        f"- Severity: {severity}\n"
        f"- Owner team: {owner_team}\n"
        f"- Created by: ActionPilot local agent\n"
    )
    try:
        issue = _github_request(
            "POST",
            f"/repos/{settings.github_repo}/issues",
            {"title": title, "body": body, "labels": labels},
        )
        return {
            "enabled": True,
            "number": issue.get("number"),
            "url": issue.get("html_url"),
            "state": issue.get("state"),
        }
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        return {"enabled": True, "error": str(exc)}


def sync_github_issue_status(ticket: dict[str, Any], status: str) -> dict[str, Any]:
    if not github_ticketing_enabled():
        return {"enabled": False, "message": "GitHub ticketing is not configured."}

    issue_number = ticket.get("github_issue_number")
    if not issue_number:
        return {"enabled": True, "message": "Ticket is not linked to a GitHub issue."}

    settings = get_settings()
    try:
        _github_request(
            "POST",
            f"/repos/{settings.github_repo}/issues/{issue_number}/comments",
            {"body": f"ActionPilot ticket status updated to `{status}`."},
        )
        if status == "resolved":
            issue = _github_request(
                "PATCH",
                f"/repos/{settings.github_repo}/issues/{issue_number}",
                {"state": "closed", "state_reason": "completed"},
            )
            return {"enabled": True, "state": issue.get("state"), "url": issue.get("html_url")}
        return {"enabled": True, "status_comment": "created"}
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        return {"enabled": True, "error": str(exc)}
