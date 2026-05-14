from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.config import get_settings

RUNTIME_RECORD_FILES = ("tickets.json", "policy_flags.json", "procurement_requests.json")


def append_record(filename: str, record: dict[str, Any]) -> dict[str, Any]:
    settings = get_settings()
    path = settings.runtime_path / filename
    existing: list[dict[str, Any]] = []
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = []
    record = {
        "id": f"{filename.replace('.json','')}-{len(existing)+1:04d}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        **record,
    }
    existing.append(record)
    path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    return record


def read_records(filename: str) -> list[dict[str, Any]]:
    path = get_settings().runtime_path / filename
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def update_record(filename: str, record_id: str, updates: dict[str, Any]) -> dict[str, Any]:
    path = get_settings().runtime_path / filename
    records = read_records(filename)
    for record in records:
        if record.get("id") == record_id:
            record.update(updates)
            record["updated_at"] = datetime.now(timezone.utc).isoformat()
            path.write_text(json.dumps(records, indent=2), encoding="utf-8")
            return record
    raise KeyError(record_id)


def clear_runtime_records(filenames: tuple[str, ...] = RUNTIME_RECORD_FILES) -> None:
    settings = get_settings()
    settings.runtime_path.mkdir(parents=True, exist_ok=True)
    for filename in filenames:
        path = settings.runtime_path / filename
        path.write_text("[]", encoding="utf-8")
