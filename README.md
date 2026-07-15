# KG Schema Discovery

Discovers candidate **entity types** and **relation types** from any corpus of
plain-text documents — with no pre-supplied label set and no generative LLM.
The output is a corpus-grounded schema ready for downstream extraction.

**Works with any domain.** Drop `.txt` files into `corpus/` and run.

---

## How it works

Four scripts, run in order:

| Script | What it does | Models used |
|--------|-------------|-------------|
| `00_ingest_s3.py` | Download JSON/JSONL documents from S3 → `corpus/*.txt` | boto3 |
| `01_characterize.py` | Corpus statistics, TF-IDF vocabulary, noun-chunk frequencies | spaCy `en_core_web_md` |
| `02_discover_entities.py` | Zero-shot NER (configurable labels) + unsupervised noun-chunk clustering | GLiNER + `all-MiniLM-L6-v2` + HDBSCAN |
| `03_discover_relations.py` | Coreference resolution → sentence-level triple extraction → relation clustering + entity typing | fastcoref + REBEL + GLiNER + HDBSCAN |

**No domain-specific models or gazetteers are required.** Relation direction is
not resolved during discovery — treat that as a curation decision.

---

## Prerequisites

- Python ≥ 3.10
- [`uv`](https://github.com/astral-sh/uv) (recommended) or plain `pip`
- ~8 GB free disk (REBEL ~1.5 GB, fastcoref ~400 MB, GLiNER ~500 MB, torch)
- ~8 GB RAM

---

## Setup

```bash
cd kg-discovery
uv sync          # creates .venv and installs all dependencies from uv.lock
```

For non-uv users:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Quick Start

**1. Add your documents**

From local `.txt` files:
```bash
cp /path/to/your/documents/*.txt corpus/
```

From S3 (JSON or JSONL format):
```bash
# One .json file per document — specify which field holds the text
python scripts/00_ingest_s3.py s3://my-bucket/path/to/docs/ --text-field body

# Single .jsonl file (one document per line)
python scripts/00_ingest_s3.py s3://my-bucket/corpus.jsonl --text-field content --id-field doc_id

# Quick test: first 50 docs only
python scripts/00_ingest_s3.py s3://my-bucket/path/ --text-field body --limit 50 --clear
```

AWS credentials are read from the environment in the usual order:
`AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`, `~/.aws/credentials`, or an IAM role.

**2. Run the full pipeline**

```bash
./run_pipeline.sh                    # full corpus
./run_pipeline.sh 50                 # quick test: first 50 docs only
REBEL_DEVICE=0 ./run_pipeline.sh     # use GPU for REBEL (script 03 — recommended)
```

The pipeline pauses briefly after step 1 and prints the characterization
summary so you can optionally set entity type labels before steps 2 and 3
proceed. It then continues automatically.

**3. Read the results**

```
output/01_characterization.txt   corpus vocabulary and entity anchors
output/02_entities.txt           discovered entity types
output/03_relations.txt          discovered relation types with examples
```

---

## Running scripts individually

Scripts read from `corpus/` and write to `output/`. Run from the repo root.

```bash
PY=.venv/bin/python
```

### Step 1 — Characterize (always run first)

```bash
$PY scripts/01_characterize.py            # full corpus
$PY scripts/01_characterize.py 50         # quick test: first 50 docs
```

Open `output/01_characterization.txt` and look at:
- **Top TF-IDF terms** — corpus-salient vocabulary; tells you what domain you're in
- **Noun-chunk heads** — the most frequent entity anchors

Use these to decide what entity type labels to supply in step 2.

---

### Step 2 — Discover entity types

**Optionally** open `scripts/02_discover_entities.py` and set `ENTITY_LABELS`
based on what you saw in step 1. Leave the list empty to run fully unsupervised
(noun-chunk clustering only — no GLiNER NER pass).

```python
# Example labels for a biomedical corpus (from script 01 output):
ENTITY_LABELS = [
    "disease or condition",
    "pathogen or virus",
    "drug or therapeutic antibody",
    "protein or gene",
    "cell type or cell line",
    "organism or animal model",
    "tissue or organ",
    "biological process",
    "laboratory method or assay",
    "treatment or therapy",
]
```

```bash
$PY scripts/02_discover_entities.py            # full corpus
$PY scripts/02_discover_entities.py 50         # quick test
```

**Resume after a crash** (re-runs clustering without the slow NER pass):
```bash
$PY scripts/02_discover_entities.py --cluster-only
```

---

### Step 3 — Discover relation types

Mirror the same `ENTITY_LABELS` into `scripts/03_discover_relations.py` (used
for domain→range typing). If left empty, REBEL still extracts triples in full
— only the typing step is skipped.

```bash
$PY scripts/03_discover_relations.py            # full corpus
$PY scripts/03_discover_relations.py 50         # quick test
```

**GPU acceleration (recommended — script 03 is slow on CPU):**

```bash
REBEL_DEVICE=0   $PY scripts/03_discover_relations.py   # first CUDA GPU
REBEL_DEVICE=mps $PY scripts/03_discover_relations.py   # Apple Silicon
```

**Resume after a crash:**
```bash
$PY scripts/03_discover_relations.py --cluster-only
```

---

## Outputs

All files are written to `output/`.

| File | Produced by | Contents |
|------|-------------|----------|
| `01_characterization.txt` | 01 | Corpus size stats, TF-IDF terms, noun-chunk heads |
| `02_entities.txt` | 02 | Human-readable entity type inventory + emergent clusters |
| `02_entities.json` | 02 | Structured version of the above |
| `02_counts.json` | 02 | Raw NER/chunk counts — checkpoint for `--cluster-only` |
| `03_relations.txt` | 03 | Human-readable relation clusters with domain→range + examples |
| `03_relations.json` | 03 | Structured version of the above |
| `03_counts.json` | 03 | Raw triple/predicate counts — checkpoint for `--cluster-only` |

---

## Runtime estimates

| Script | 1,000 docs (CPU) | 4,000 docs (CPU) | 4,000 docs (GPU) |
|--------|-----------------|-----------------|-----------------|
| `01_characterize.py` | ~3 min | ~12 min | — |
| `02_discover_entities.py` | ~1–2 hr | ~5–8 hr | — |
| `03_discover_relations.py` | ~5–8 hr | ~20–30 hr | ~4–6 hr |

For large corpora (4k+ docs), running script 03 on a GPU is strongly recommended.
Set `REBEL_DEVICE=0` for CUDA or `REBEL_DEVICE=mps` for Apple Silicon.

Scripts stream documents one at a time and checkpoint every 100 docs, so they
are safe to leave running unattended and can be resumed if interrupted.

---

## Configuration

All tunables are constants at the top of each script:

| Constant | Script | Purpose |
|----------|--------|---------|
| `ENTITY_LABELS` | 02, 03 | Entity type hints for GLiNER zero-shot NER |
| `GLINER_THRESHOLD` | 02, 03 | Confidence threshold for GLiNER (default 0.5) |
| `MIN_CHUNK_FREQ` | 02 | Min occurrences for a noun-chunk head to enter clustering |
| `MIN_CLUSTER_SIZE` | 02, 03 | HDBSCAN minimum cluster size |
| `MAX_DOC_CHARS` | 02, 03 | Characters of each document passed to models |
| `MAX_SENTS_PER_DOC` | 03 | Sentences per document fed to REBEL (speed cap) |
| `REBEL_BATCH_SIZE` | 03 | Sentences per REBEL forward pass |
| `LIMIT` | all | Set to N to process only the first N docs |
