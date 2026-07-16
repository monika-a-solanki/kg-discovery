"""Step 2 - Entity-type DISCOVERY (GLiNER2 zero-shot NER → HDBSCAN clustering).

Reads output/01_characterization.json to derive entity labels from the corpus
vocabulary — no hardcoded domain assumptions.

Two signal sources, fused:
  A) GLiNER2 (zero-shot NER): entity labels are derived from top TF-IDF terms
     and noun-chunk heads found during characterization.
  B) Frequent noun-chunk spans embedded and clustered with HDBSCAN: surfaces
     emergent concepts that fall outside the label set. Cluster count is
     discovered automatically — no fixed k.

Hardened for large-corpus scale:
  - streams docs one at a time
  - hard per-doc char cap to bound memory
  - flushed progress every PROGRESS_EVERY docs
  - checkpoints raw counts to output/02_counts.json every CKPT_EVERY docs
    (re-run clustering only via --cluster-only)

Outputs:
  output/02_counts.json    - raw aggregates (checkpoint, resumable)
  output/02_entities.json  - structured candidates (after clustering)
  output/02_entities.txt   - human-readable
"""

import json
import re
import sys
from collections import Counter, defaultdict

import numpy as np
import torch
import spacy
from gliner2 import GLiNER2
from sentence_transformers import SentenceTransformer
from sklearn.cluster import HDBSCAN


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

from common import iter_docs, OUTPUT

# ── Configuration ─────────────────────────────────────────────────────────────

LIMIT = None           # None = whole corpus; set e.g. 100 for a quick test run
MAX_DOC_CHARS = 120_000
CHUNK_SIZE = 2000      # chars per GLiNER2 call (model encoder max is ~512 tokens)
PROGRESS_EVERY = 25
CKPT_EVERY = 100
MIN_CHUNK_FREQ = 8     # min occurrences for a noun-chunk head to enter clustering
MIN_CLUSTER_SIZE = 5   # HDBSCAN: fewest items that form a cluster
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GLINER2_MODEL = "fastino/gliner2-large-v1"
CHARACT_PATH = OUTPUT / "01_characterization.json"
MAX_LABELS = 30        # max entity labels to derive from characterization

COUNTS = OUTPUT / "02_counts.json"


def derive_entity_labels() -> dict[str, str]:
    """Derive entity labels from script 01's characterization output.

    Groups top TF-IDF terms and noun-chunk heads into candidate entity types.
    Each label is a term from the corpus; the description helps GLiNER2 precision.
    """
    if not CHARACT_PATH.exists():
        raise FileNotFoundError(
            f"{CHARACT_PATH} not found — run script 01_characterize.py first"
        )
    data = json.loads(CHARACT_PATH.read_text())

    # Combine top TF-IDF terms and noun-chunk heads as candidate labels
    seen = set()
    candidates = []

    for item in data.get("top_noun_chunks", []):
        term = item["term"].lower().strip()
        if term not in seen and len(term) > 3 and term.isalpha():
            seen.add(term)
            candidates.append(term)

    for item in data.get("top_tfidf", []):
        term = item["term"].lower().strip()
        if term not in seen and len(term) > 3 and " " not in term and term.isalpha():
            seen.add(term)
            candidates.append(term)

    # Use top candidates as entity labels with auto-generated descriptions
    labels = {}
    for term in candidates[:MAX_LABELS]:
        labels[term] = f"Instances, mentions, or references to {term}"

    return labels

# ── Helpers ───────────────────────────────────────────────────────────────────


def clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def log(msg: str) -> None:
    print(msg, flush=True)


# ── Accumulation pass ─────────────────────────────────────────────────────────


def accumulate(limit, entity_labels):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"loading GLiNER2 on {device}…")
    extractor = GLiNER2.from_pretrained(GLINER2_MODEL).to(device)

    log("loading spaCy for noun chunks…")
    nlp = spacy.load("en_core_web_md", disable=["ner"])
    nlp.max_length = 2_000_000

    log(f"entity labels ({len(entity_labels)}): {', '.join(entity_labels.keys())}")

    ner_types: Counter = Counter()
    ner_examples: dict[str, Counter] = defaultdict(Counter)
    chunk_freq: Counter = Counter()

    def checkpoint(n: int) -> None:
        COUNTS.write_text(json.dumps({
            "docs_processed": n,
            "ner_types": dict(ner_types),
            "ner_examples": {k: dict(v.most_common(40)) for k, v in ner_examples.items()},
            "chunk_freq": dict(chunk_freq.most_common(6000)),
        }))

    i = 0
    for doc_id, text in iter_docs(limit):
        i += 1
        t = text[:MAX_DOC_CHARS]

        # A) GLiNER2 zero-shot NER (chunked — model encoder is limited to ~512 tokens)
        for start in range(0, len(t), CHUNK_SIZE):
            chunk = t[start:start + CHUNK_SIZE]
            if len(chunk.strip()) < 20:
                continue
            try:
                result = extractor.extract_entities(chunk, entity_labels)
                for label, mentions in result.get("entities", {}).items():
                    for mention in mentions:
                        txt = clean(mention).lower()
                        if 2 < len(txt) < 80:
                            ner_types[label] += 1
                            ner_examples[label][txt] += 1
            except Exception:
                pass

        # B) Noun-chunk heads for emergent types
        doc = nlp(t)
        for ch in doc.noun_chunks:
            head = clean(ch.root.text).lower()
            if head.isalpha() and 3 < len(head) < 30:
                chunk_freq[head] += 1
        del doc

        if i % PROGRESS_EVERY == 0:
            log(f"  …{i} docs | ner_types={len(ner_types)} chunks={len(chunk_freq)}")
        if i % CKPT_EVERY == 0:
            checkpoint(i)

    checkpoint(i)
    log(f"accumulation done: {i} docs")
    return ner_types, ner_examples, chunk_freq


# ── Clustering + report ───────────────────────────────────────────────────────


def cluster_and_report(ner_types, ner_examples, chunk_freq, entity_labels):
    candidates = [h for h, c in chunk_freq.items() if c >= MIN_CHUNK_FREQ]
    log(f"embedding {len(candidates)} noun-chunk heads…")

    encoder = SentenceTransformer(EMBED_MODEL)
    emb = encoder.encode(candidates, show_progress_bar=False, normalize_embeddings=True, batch_size=256)

    log("clustering with HDBSCAN (auto-discovers k)…")
    raw_labels = HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        metric="euclidean",
        cluster_selection_method="eom",
    ).fit_predict(emb)

    clusters: dict[int, list] = defaultdict(list)
    for term, lab in zip(candidates, raw_labels):
        if lab != -1:
            clusters[lab].append((term, chunk_freq[term]))

    cluster_summaries = []
    for lab, members in clusters.items():
        members.sort(key=lambda x: -x[1])
        cluster_summaries.append({
            "cluster": lab,
            "total_freq": sum(c for _, c in members),
            "label_hint": members[0][0],
            "members": [{"term": t, "freq": c} for t, c in members[:12]],
        })
    cluster_summaries.sort(key=lambda x: -x["total_freq"])

    result = {
        "ner_source": "GLiNER2 (fastino/gliner2-large-v1)",
        "entity_labels": list(entity_labels.keys()),
        "ner_types": [
            {
                "label": lab,
                "count": cnt,
                "examples": [w for w, _ in Counter(ner_examples[lab]).most_common(10)],
            }
            for lab, cnt in Counter(ner_types).most_common()
        ],
        "emergent_clusters": cluster_summaries,
    }
    (OUTPUT / "02_entities.json").write_text(json.dumps(result, indent=2, cls=_NumpyEncoder))

    n_clusters = len(clusters)
    lines = ["ENTITY-TYPE DISCOVERY", "=" * 60, ""]
    if result["ner_types"]:
        lines.append(f"A) GLiNER2 zero-shot NER  ({len(entity_labels)} labels):")
        for r in result["ner_types"]:
            lines.append(f"  {r['count']:>6}  {r['label']:<25}  e.g. {', '.join(r['examples'][:5])}")
        lines.append("")
    lines.append(f"B) Emergent types — noun-chunk HDBSCAN  ({n_clusters} clusters discovered):")
    for c in cluster_summaries:
        terms = ", ".join(m["term"] for m in c["members"][:8])
        lines.append(f"  [{c['total_freq']:>5}]  ~{c['label_hint']}: {terms}")

    (OUTPUT / "02_entities.txt").write_text("\n".join(lines))
    log("\n".join(lines))
    log(f"\n→ {OUTPUT / '02_entities.json'}")


# ── Entry point ───────────────────────────────────────────────────────────────


def main():
    OUTPUT.mkdir(exist_ok=True)
    log("deriving entity labels from characterization…")
    entity_labels = derive_entity_labels()

    if "--cluster-only" in sys.argv:
        data = json.loads(COUNTS.read_text())
        log(f"loaded checkpoint: {data['docs_processed']} docs")
        cluster_and_report(
            Counter(data["ner_types"]),
            {k: Counter(v) for k, v in data["ner_examples"].items()},
            Counter(data["chunk_freq"]),
            entity_labels,
        )
        return
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    limit = int(args[0]) if args else LIMIT
    ner_types, ner_examples, chunk_freq = accumulate(limit, entity_labels)
    cluster_and_report(ner_types, ner_examples, chunk_freq, entity_labels)


if __name__ == "__main__":
    main()
