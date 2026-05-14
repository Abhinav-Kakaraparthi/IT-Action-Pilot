from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

import httpx

from app.core.config import get_settings


@dataclass(frozen=True)
class RealSource:
    filename: str
    title: str
    url: str
    source_type: str = "html"


REAL_SOURCES = [
    RealSource(
        filename="nist_csf_2_0.md",
        title="NIST Cybersecurity Framework 2.0",
        url="https://www.nist.gov/cyberframework",
    ),
    RealSource(
        filename="cisa_incident_response.md",
        title="CISA Incident Response Guidance",
        url="https://www.cisa.gov/topics/cybersecurity-best-practices/organizations-and-cyber-safety/cybersecurity-incident-response",
    ),
    RealSource(
        filename="cisa_irp_basics.md",
        title="CISA Incident Response Plan Basics",
        url="https://www.cisa.gov/resources-tools/resources/incident-response-plan-irp-basics",
    ),
    RealSource(
        filename="ftc_data_security.md",
        title="FTC Data Security Guidance",
        url="https://www.ftc.gov/business-guidance/privacy-security/data-security",
    ),
    RealSource(
        filename="gsa_mas_acquisition.md",
        title="GSA MAS Acquisition Guidance",
        url="https://www.gsa.gov/buying-selling/purchasing-programs/gsa-schedules/schedule-buyers/contracting-officer-guidance-schedule-ordering-procedures",
    ),
    RealSource(
        filename="cisa_kev_catalog.md",
        title="CISA Known Exploited Vulnerabilities Catalog",
        url="https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json",
        source_type="kev_json",
    ),
]

ALLOWED_DOMAINS = {"www.nist.gov", "www.cisa.gov", "www.ftc.gov", "www.gsa.gov"}


class TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1
        if tag in {"h1", "h2", "h3"}:
            self.parts.append("\n\n")
        if tag in {"p", "li", "tr", "section", "article"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"} and self._skip_depth:
            self._skip_depth -= 1
        if tag in {"h1", "h2", "h3", "p", "li"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        cleaned = re.sub(r"\s+", " ", data).strip()
        if cleaned:
            self.parts.append(cleaned)

    def text(self) -> str:
        text = "".join(self.parts)
        text = re.sub(r"[ \t]{2,}", " ", text)
        lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
        boilerplate = (
            "skip to main content",
            "an official website",
            "here's how you know",
            "here’s how you know",
            "official websites",
            "secure .gov websites",
            "share sensitive information only",
            "no-cost cyber services",
            "secure by design",
            "report a cyber issue",
            "topics topics",
            "resources & tools",
            "news & events",
            "careers careers",
            "about about",
            "breadcrumb",
            "search menu close",
            "topicstopics",
            "cybersecurity best practicescyber threats",
            "careersbenefits",
            "return to top",
            "facebook",
            " cisa.gov ",
            " dhs.gov ",
            "linkedin",
            "youtube",
            "instagram",
            "website feedback",
            "privacy policy",
            "foia requests",
        )
        cleaned: list[str] = []
        for line in lines:
            if not line:
                continue
            line_l = line.lower()
            if any(term in line_l for term in boilerplate):
                continue
            cleaned.append(line)
        return "\n".join(cleaned).strip()


def _assert_allowed(url: str) -> None:
    domain = urlparse(url).netloc.lower()
    if domain not in ALLOWED_DOMAINS:
        raise ValueError(f"Refusing to fetch non-allowlisted domain: {domain}")


def _html_to_markdown(title: str, url: str, html: str, retrieved_at: str) -> str:
    extractor = TextExtractor()
    extractor.feed(html)
    text = extractor.text()
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("Share:"):
            text = "\n".join(lines[index + 1 :]).strip()
            break
    return f"""# {title}

Source: {url}
Retrieved: {retrieved_at}
Dataset: public official guidance

## Extracted guidance
{text}
"""


def _kev_to_markdown(title: str, url: str, payload: dict, retrieved_at: str, limit: int = 50) -> str:
    vulnerabilities = payload.get("vulnerabilities", [])[:limit]
    lines = [
        f"# {title}",
        "",
        f"Source: {url}",
        f"Retrieved: {retrieved_at}",
        "Dataset: public official CISA KEV JSON feed",
        "",
        "## Recent known exploited vulnerabilities",
    ]
    for item in vulnerabilities:
        cve = item.get("cveID", "Unknown CVE")
        vendor = item.get("vendorProject", "Unknown vendor")
        product = item.get("product", "Unknown product")
        name = item.get("vulnerabilityName", "Known exploited vulnerability")
        action = item.get("requiredAction", "Follow vendor mitigation guidance.")
        due_date = item.get("dueDate", "No due date listed")
        lines.extend(
            [
                "",
                f"### {cve}: {name}",
                f"- Vendor/project: {vendor}",
                f"- Product: {product}",
                f"- Due date: {due_date}",
                f"- Required action: {action}",
            ]
        )
    return "\n".join(lines)


def import_real_docs() -> list[dict[str, str]]:
    settings = get_settings()
    output_dir = Path(settings.docs_dir) / "real"
    output_dir.mkdir(parents=True, exist_ok=True)
    retrieved_at = datetime.now(timezone.utc).isoformat()
    imported: list[dict[str, str]] = []

    with httpx.Client(timeout=30.0, follow_redirects=True, headers={"User-Agent": "ActionPilot local demo importer"}) as client:
        for source in REAL_SOURCES:
            _assert_allowed(source.url)
            response = client.get(source.url)
            response.raise_for_status()
            if source.source_type == "kev_json":
                content = _kev_to_markdown(source.title, source.url, response.json(), retrieved_at)
            else:
                content = _html_to_markdown(source.title, source.url, response.text, retrieved_at)
            path = output_dir / source.filename
            path.write_text(content, encoding="utf-8")
            imported.append({"title": source.title, "url": source.url, "path": str(path)})

    manifest = {
        "retrieved_at": retrieved_at,
        "sources": imported,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return imported


if __name__ == "__main__":
    for item in import_real_docs():
        print(f"Imported {item['title']} -> {item['path']}")
