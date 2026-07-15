"""Shared helpers for the discovery prototype.

Discovery is intentionally INDEPENDENT of any prior schema: no gazetteers,
no hand-picked label lists, no predicate vocabulary. Everything here works
from raw corpus text + a general-purpose English NLP model only.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "corpus"
OUTPUT = ROOT / "output"

# Strip trailing reference/bibliography sections — common in papers, reports,
# and articles — so models see prose, not citation lists. Light-touch.
_REF_HEAD = re.compile(r"\n\s*(references|bibliography|acknowledg)", re.I)


def read_body(path: Path) -> str:
    text = path.read_text(errors="ignore")
    m = _REF_HEAD.search(text)
    if m:
        text = text[: m.start()]
    return text


def iter_docs(limit: int | None = None):
    """Yield (doc_id, body_text) for every .txt file in the corpus."""
    files = sorted(CORPUS.glob("*.txt"))
    if limit:
        files = files[:limit]
    for f in files:
        yield f.stem, read_body(f)


SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
