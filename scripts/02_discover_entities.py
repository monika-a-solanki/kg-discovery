"""Step 2 - Entity-type DISCOVERY (zero-shot NER → HDBSCAN clustering).

Two signal sources, fused:
  A) GLiNER (zero-shot NER): extracts entity spans for any set of type labels
     without domain-specific training. Labels are configurable at the top —
     adjust them to match the concepts you expect in your corpus.
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

import spacy
from gliner import GLiNER
from sentence_transformers import SentenceTransformer
from sklearn.cluster import HDBSCAN

from common import iter_docs, OUTPUT

# ── Configuration ─────────────────────────────────────────────────────────────

LIMIT = None           # None = whole corpus; set e.g. 100 for a quick test run
MAX_DOC_CHARS = 120_000
PROGRESS_EVERY = 25
CKPT_EVERY = 100
MIN_CHUNK_FREQ = 8     # min occurrences for a noun-chunk head to enter clustering
MIN_CLUSTER_SIZE = 5   # HDBSCAN: fewest items that form a cluster
EMBED_MODEL = "all-MiniLM-L6-v2"
GLINER_MODEL = "urchade/gliner_medium-v2.1"
GLINER_THRESHOLD = 0.5

# Seed labels for GLiNER zero-shot NER.
# These drive extraction across any domain without fine-tuning.
# Add, remove, or rename to match your corpus.
ENTITY_LABELS = [
    "person",
    "organization",
    "location",
    "product",
    "technology",
    "event",
    "concept",
    "law or regulation",
    "date or time period",
    "financial value",
    "quantity or measurement",
    "role or title",
]

COUNTS = OUTPUT / "02_counts.json"

# ── Helpers ───────────────────────────────────────────────────────────────────


def clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def log(msg: str) -> None:
    print(msg, flush=True)


# ── Accumulation pass ─────────────────────────────────────────────────────────


def accumulate(limit):
    log("loading GLiNER…")
    gliner = GLiNER.from_pretrained(GLINER_MODEL)
    log("loading spaCy for noun chunks…")
    nlp = spacy.load("en_core_web_md", disable=["ner"])
    nlp.max_length = 2_000_000

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

        # A) GLiNER zero-shot NER
        for ent in gliner.predict_entities(t, ENTITY_LABELS, threshold=GLINER_THRESHOLD):
            txt = clean(ent["text"]).lower()
            if 2 < len(txt) < 80:
                ner_types[ent["label"]] += 1
                ner_examples[ent["label"]][txt] += 1

        # B) Noun-chunk heads for emergent types beyond the label set
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


def cluster_and_report(ner_types, ner_examples, chunk_freq):
    candidates = [h for h, c in chunk_freq.items() if c >= MIN_CHUNK_FREQ]
    log(f"embedding {len(candidates)} noun-chunk heads…")

    encoder = SentenceTransformer(EMBED_MODEL)
    emb = encoder.encode(candidates, show_progress_bar=False, normalize_embeddings=True, batch_size=256)

    log("clustering with HDBSCAN (auto-discovers k)…")
    raw_labels = HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        metric="euclidean",           # normalized vectors → cosine-equivalent
        cluster_selection_method="eom",
    ).fit_predict(emb)

    clusters: dict[int, list] = defaultdict(list)
    for term, lab in zip(candidates, raw_labels):
        if lab != -1:  # -1 = noise point, not assigned to any cluster
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
        "gliner_labels": ENTITY_LABELS,
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
    (OUTPUT / "02_entities.json").write_text(json.dumps(result, indent=2))

    n_clusters = len(clusters)
    lines = [
        "ENTITY-TYPE DISCOVERY", "=" * 60, "",
        f"A) GLiNER zero-shot NER  ({len(ENTITY_LABELS)} labels, threshold={GLINER_THRESHOLD}):",
    ]
    for r in result["ner_types"]:
        lines.append(f"  {r['count']:>6}  {r['label']:<25}  e.g. {', '.join(r['examples'][:5])}")
    lines += ["", f"B) Emergent types — noun-chunk HDBSCAN  ({n_clusters} clusters discovered):"]
    for c in cluster_summaries:
        terms = ", ".join(m["term"] for m in c["members"][:8])
        lines.append(f"  [{c['total_freq']:>5}]  ~{c['label_hint']}: {terms}")

    (OUTPUT / "02_entities.txt").write_text("\n".join(lines))
    log("\n".join(lines))
    log(f"\n→ {OUTPUT / '02_entities.json'}")


# ── Entry point ───────────────────────────────────────────────────────────────


def main():
    if "--cluster-only" in sys.argv:
        data = json.loads(COUNTS.read_text())
        log(f"loaded checkpoint: {data['docs_processed']} docs")
        cluster_and_report(
            Counter(data["ner_types"]),
            {k: Counter(v) for k, v in data["ner_examples"].items()},
            Counter(data["chunk_freq"]),
        )
        return
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    limit = int(args[0]) if args else LIMIT
    ner_types, ner_examples, chunk_freq = accumulate(limit)
    cluster_and_report(ner_types, ner_examples, chunk_freq)


if __name__ == "__main__":
    main()
