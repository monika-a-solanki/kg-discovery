"""Shared helpers for the discovery prototype.

Discovery is intentionally INDEPENDENT of any prior schema: no gazetteers,
no hand-picked label lists, no predicate vocabulary. Everything here works
from raw corpus text + a general-purpose English NLP model only.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "corpus"
OUTPUT = ROOT / "output"

# Strip trailing reference/bibliography sections — common in papers, reports,
# and articles — so models see prose, not citation lists. Light-touch.
_REF_HEAD = re.compile(r"\n\s*(references|bibliography|acknowledg)", re.I)


def _extract_text_from_section(section) -> list[str]:
    """Recursively extract paragraph text from a section dict."""
    if isinstance(section, str):
        return [section]
    parts = []
    if section.get("title"):
        parts.append(section["title"])
    for item in section.get("content", []):
        if "paragraph" in item:
            para = item["paragraph"]
            parts.extend(para.get("content", []))
        if "subsections" in item:
            sub = item["subsections"]
            if isinstance(sub, dict):
                parts.extend(_extract_text_from_section(sub))
            elif isinstance(sub, list):
                for s in sub:
                    parts.extend(_extract_text_from_section(s))
    return parts


def _extract_text_from_sections(sections: list) -> str:
    """Extract all paragraph text from a list of section objects."""
    parts = []
    for section in sections:
        parts.extend(_extract_text_from_section(section))
    return "\n".join(parts)


def read_body(path: Path) -> str:
    text = path.read_text(errors="ignore")
    m = _REF_HEAD.search(text)
    if m:
        text = text[: m.start()]
    return text


def read_json_body(path: Path) -> str:
    """Extract text from a processed document JSON file's content field."""
    data = json.loads(path.read_text(errors="ignore"))
    content = data.get("content", {})
    sections = content.get("sections", [])
    return _extract_text_from_sections(sections)


def iter_docs(limit: int | None = None):
    """Yield (doc_id, body_text) from corpus .json or .txt files."""
    json_files = sorted(CORPUS.glob("*.json"))
    txt_files = sorted(CORPUS.glob("*.txt"))

    if json_files:
        files = json_files
        if limit:
            files = files[:limit]
        for f in files:
            text = read_json_body(f)
            if text.strip():
                yield f.stem, text
    else:
        if limit:
            txt_files = txt_files[:limit]
        for f in txt_files:
            yield f.stem, read_body(f)


SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
