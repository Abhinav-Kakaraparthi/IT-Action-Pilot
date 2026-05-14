from __future__ import annotations

import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

from pypdf import PdfReader

from app.core.config import get_settings


def _safe_stem(filename: str) -> str:
    stem = Path(filename).stem.lower()
    stem = re.sub(r"[^a-z0-9_-]+", "-", stem).strip("-")
    return stem or "uploaded-document"


def _extract_pdf_text(path: Path) -> list[tuple[int, str]]:
    reader = PdfReader(str(path))
    pages: list[tuple[int, str]] = []
    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if text:
            pages.append((index, text))
    return pages


def process_pdf_upload(source_path: Path, original_filename: str) -> dict[str, str | int]:
    settings = get_settings()
    upload_root = Path(settings.docs_dir) / "uploads"
    pdf_dir = upload_root / "pdf"
    markdown_dir = upload_root / "markdown"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    markdown_dir.mkdir(parents=True, exist_ok=True)

    safe_stem = _safe_stem(original_filename)
    pdf_path = pdf_dir / f"{safe_stem}.pdf"
    markdown_path = markdown_dir / f"{safe_stem}.md"
    shutil.copyfile(source_path, pdf_path)

    pages = _extract_pdf_text(pdf_path)
    if not pages:
        raise ValueError("No extractable text was found in the PDF.")

    retrieved_at = datetime.now(timezone.utc).isoformat()
    sections = [
        f"# Uploaded PDF: {Path(original_filename).stem}",
        "",
        f"Source file: {original_filename}",
        f"Stored PDF: {pdf_path}",
        f"Processed: {retrieved_at}",
        "Dataset: user uploaded PDF",
        "",
    ]
    for page_number, text in pages:
        sections.extend([f"## Page {page_number}", text, ""])

    markdown_path.write_text("\n".join(sections), encoding="utf-8")
    return {
        "filename": original_filename,
        "pdf_path": str(pdf_path),
        "markdown_path": str(markdown_path),
        "pages": len(pages),
    }
