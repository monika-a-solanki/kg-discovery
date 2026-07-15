"""Shared helpers for the discovery prototype.

Discovery is intentionally INDEPENDENT of any prior schema: no gazetteers,
no hand-picked label lists, no predicate vocabulary from the json-to-graph
project. Everything here works from the raw corpus text + general/biomedical
NLP models only.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "corpus"
OUTPUT = ROOT / "output"

# Strip the PMC reference/boilerplate tail and obvious non-body lines so the
# linguistic models see prose, not citation lists. Deliberately light-touch.
_REF_HEAD = re.compile(r"\n\s*(references|bibliography|acknowledg)", re.I)


def read_body(path: Path) -> str:
    text = path.read_text(errors="ignore")
    m = _REF_HEAD.search(text)
    if m:
        text = text[: m.start()]
    return text


def iter_docs(limit: int | None = None):
    files = sorted(CORPUS.glob("PMC*.txt"))
    if limit:
        files = files[:limit]
    for f in files:
        yield f.stem, read_body(f)


SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
