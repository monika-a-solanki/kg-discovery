# KG Schema Discovery

Discovers candidate **entity types** and **relation types** from a corpus of
biomedical full-text articles, *without any pre-supplied label set*. The goal is
a corpus-grounded schema you can then hand to an extractor (e.g. GLiNER-Relex).

Pipeline (no generative LLM involved):
- **01_characterize** — corpus stats + TF-IDF salient vocabulary
- **02_discover_entities** — scispaCy biomedical NER + noun-chunk embedding
  clusters → candidate entity types
- **03_discover_relations** — dependency-path OpenIE + predicate clustering →
  candidate relation types

---

## 1. Prerequisites

- **Python 3.12** (scispaCy model wheels don't build on 3.13)
- ~4 GB free disk (torch + models), ~8 GB RAM
- Either [`uv`](https://github.com/astral-sh/uv) (fast, recommended) or plain
  `pip` + `venv`

---

## 2. Setup

The corpus must live in `corpus/` as `PMC*.txt` files (already included here).

Dependencies are declared in `pyproject.toml` and fully pinned in `uv.lock`
(the lockfile is the source of truth — every transitive package is pinned, and
`requires-python` enforces 3.12). `requirements.txt` is a generated, pinned
export of the same lock for non-uv users.

### Option A — uv (recommended)
```bash
cd kg-discovery
uv sync --frozen        # builds .venv from uv.lock — exact, reproducible
```

### Option B — plain pip
```bash
cd kg-discovery
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The three scispaCy models are pinned as direct URLs, so no separate
`spacy download` step is needed.

> Maintainers: after editing deps in `pyproject.toml`, run `uv lock` to refresh
> the lockfile, then `uv export --no-hashes --no-emit-project -o requirements.txt`
> to keep the pip export in sync.

---

## 3. Run

Scripts read from `corpus/` and write to `output/`. **Run them from the
`scripts/` directory** (they import `common.py` and resolve paths relative to
the workspace root).

```bash
cd scripts
PY=../.venv/bin/python          # or: source ../.venv/bin/activate

# optional: silence harmless scispaCy regex warnings
export PYTHONWARNINGS=ignore

# each script takes an optional doc-limit arg; omit it to run the FULL corpus
$PY 01_characterize.py            # all docs   (or: 01_characterize.py 150)
$PY 02_discover_entities.py       # all docs
$PY 03_discover_relations.py      # all docs
```

To run everything and capture a log:
```bash
cd scripts
{ ../.venv/bin/python 01_characterize.py
  ../.venv/bin/python 02_discover_entities.py
  ../.venv/bin/python 03_discover_relations.py
} > ../output/discovery.log 2>&1
```

### Quick smoke test
Pass a small number to process only the first N docs (seconds, not hours):
```bash
$PY 02_discover_entities.py 10
$PY 03_discover_relations.py 10
```

---

## 4. Outputs (`output/`)

| File | Produced by | Contents |
|------|-------------|----------|
| `01_characterization.txt` | 01 | corpus size, TF-IDF terms, noun-chunk heads |
| `02_entities.{txt,json}`  | 02 | NER type inventory + emergent entity clusters |
| `02_counts.json`          | 02 | raw NER/chunk counts (checkpoint) |
| `03_relations.{txt,json}` | 03 | candidate relation types + top typed triples |
| `03_counts.json`          | 03 | raw triple/predicate counts (checkpoint) |
| `proposed_schema.md`      | (manual curation of the above) | the hand-off schema |

---

## 5. Runtime & resources

Full 1000-doc corpus on a laptop (Apple Silicon, CPU):
- 01 ≈ 3 min · 02 ≈ 1 hr · 03 ≈ 1 hr → **~2 hr end-to-end**
- Memory stays ~5–6% (scripts stream docs one at a time; per-doc text capped at
  120k chars). They are safe to leave running unattended.

## 6. Crash recovery

02 and 03 **checkpoint** raw counts to `output/0{2,3}_counts.json` every 100
docs. If a run is interrupted *after* accumulation, re-run just the
clustering/report step (no re-parsing) with:
```bash
$PY 02_discover_entities.py --cluster-only
$PY 03_discover_relations.py --cluster-only
```

---

## 7. Notes for the next person

- **No generative LLM** is used. The only neural language model is a small
  embedding encoder (`all-MiniLM-L6-v2`) used purely to cluster near-synonyms.
- **Discovery uses no prior schema** (no gazetteers, no hand-picked labels) —
  that independence is the whole point.
- Relation **direction is not resolved** by discovery (both orderings are
  enumerated); treat it as a curation decision.
- Tunables live at the top of each script (`MAX_DOC_CHARS`, `N_CLUSTERS`,
  `MIN_*_FREQ`, `EMBED_MODEL`).
