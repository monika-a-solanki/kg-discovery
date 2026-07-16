"""Step 3 - Relation-type DISCOVERY (GLiREL zero-shot + HDBSCAN clustering).

Pipeline per document:
  1. spaCy: tokenization, sentence splitting, noun-chunk detection
  2. GLiREL (zero-shot relation extraction): extracts (head, relation, tail)
     triples sentence by sentence using entity spans from spaCy.
  3. Relation strings are embedded and clustered with HDBSCAN: near-synonyms
     merge automatically. Cluster count is discovered, not specified.

Hardened for large-corpus scale: streams docs, caps per-doc length and
sentence count, prints flushed progress, checkpoints every CKPT_EVERY docs.
Use --cluster-only to re-run clustering/typing from the checkpoint.

Outputs:
  output/03_counts.json     - raw aggregates (checkpoint, resumable)
  output/03_relations.json
  output/03_relations.txt
"""

import json
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np
import torch

import spacy
from glirel import GLiREL
from glirel.model import load_config_as_namespace
from sentence_transformers import SentenceTransformer
from sklearn.cluster import HDBSCAN

from common import iter_docs, OUTPUT


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# ── Configuration ─────────────────────────────────────────────────────────────

LIMIT = None
MAX_DOC_CHARS = 120_000
MAX_SENTS_PER_DOC = 60
MIN_SENT_CHARS = 30
PROGRESS_EVERY = 25
CKPT_EVERY = 100
MIN_REL_FREQ = 3
MIN_CLUSTER_SIZE = 3
GLIREL_THRESHOLD = 0.3
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_HF_HOME = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
GLIREL_MODEL_PATH = os.path.join(
    _HF_HOME, "hub", "models--jackboyla--glirel-large-v0", "snapshots",
    "14ffca16a521322d6f2fbc52d2cf371d2175b309541bc9b74a003b46d909d941"
)

RELATION_LABELS = [
    "located in",
    "part of",
    "operates at",
    "uses",
    "qualified by",
    "manufactured by",
    "measured by",
    "described in",
    "contains",
    "connected to",
    "requires",
    "produces",
    "controls",
    "monitors",
    "complies with",
    "associated with",
]

COUNTS = OUTPUT / "03_counts.json"


# ── Helpers ──────────────────────────────────────────────────────────────────


def log(msg: str) -> None:
    print(msg, flush=True)


def load_glirel():
    """Load GLiREL model manually (workaround for huggingface_hub version mismatch)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = load_config_as_namespace(GLIREL_MODEL_PATH + '/glirel_config.json')
    model = GLiREL(config)
    state = torch.load(GLIREL_MODEL_PATH + '/pytorch_model.bin', map_location='cpu')
    model.load_state_dict(state, strict=False)
    model.eval()
    model.to(device)
    return model


def get_entity_spans(doc):
    """Get entity spans from a spaCy doc (NER + noun chunks, deduplicated)."""
    spans = []
    seen = set()

    for ent in doc.ents:
        key = (ent.start, ent.end)
        if key not in seen and len(ent.text.strip()) > 2:
            seen.add(key)
            spans.append([ent.start, ent.end, ent.label_, ent.text])

    for chunk in doc.noun_chunks:
        # Skip chunks that are just pronouns, determiners, or very short
        if chunk.root.pos_ in ("PRON", "DET") or len(chunk.text.strip()) <= 3:
            continue
        key = (chunk.start, chunk.end)
        if key not in seen:
            seen.add(key)
            spans.append([chunk.start, chunk.end, "ENTITY", chunk.text])

    return spans


# ── Accumulation pass ────────────────────────────────────────────────────────


def accumulate(limit):
    log("loading spaCy…")
    nlp = spacy.load("en_core_web_md")
    nlp.max_length = 2_000_000

    log("loading GLiREL…")
    glirel = load_glirel()

    rel_freq: Counter = Counter()
    rel_examples: dict[str, list] = defaultdict(list)
    raw_triples: Counter = Counter()

    def checkpoint(n: int) -> None:
        COUNTS.write_text(json.dumps({
            "docs_processed": n,
            "rel_freq": dict(rel_freq),
            "rel_examples": {r: ex for r, ex in rel_examples.items()},
            "raw_triples": {
                f"{h}\t{r}\t{t}": c
                for (h, r, t), c in raw_triples.most_common(1000)
            },
        }))

    i = 0
    for doc_id, text in iter_docs(limit):
        i += 1
        t = text[:MAX_DOC_CHARS]
        doc = nlp(t)

        sentences = [
            sent for sent in doc.sents
            if len(sent.text.strip()) >= MIN_SENT_CHARS
        ][:MAX_SENTS_PER_DOC]

        for sent in sentences:
            sent_doc = sent.as_doc()
            tokens = [token.text for token in sent_doc]
            ner_spans = get_entity_spans(nlp(sent.text))

            if len(ner_spans) < 2:
                continue

            try:
                relations = glirel.predict_relations(
                    tokens, RELATION_LABELS,
                    threshold=GLIREL_THRESHOLD,
                    ner=ner_spans, top_k=5
                )
            except Exception:
                continue

            for rel in relations:
                h = rel.get("head_text", "")
                r = rel.get("label", "")
                tl = rel.get("tail_text", "")
                score = rel.get("score", 0)

                if isinstance(h, list):
                    h = " ".join(h)
                if isinstance(tl, list):
                    tl = " ".join(tl)

                h = h.strip()
                tl = tl.strip()

                if h and r and tl and len(r) >= 2 and score >= GLIREL_THRESHOLD:
                    raw_triples[(h, r, tl)] += 1
                    rel_freq[r] += 1
                    if len(rel_examples[r]) < 5:
                        rel_examples[r].append({
                            "sentence": sent.text[:200],
                            "head": h,
                            "tail": tl,
                            "score": round(score, 3),
                        })

        del doc

        if i % PROGRESS_EVERY == 0:
            log(f"  …{i} docs | relations={len(rel_freq)} triples={len(raw_triples)}")
        if i % CKPT_EVERY == 0:
            checkpoint(i)

    checkpoint(i)
    log(f"accumulation done: {i} docs")
    return rel_freq, rel_examples, raw_triples


# ── Clustering + report ──────────────────────────────────────────────────────


def cluster_and_report(rel_freq, rel_examples, raw_triples):
    preds = [r for r, c in rel_freq.items() if c >= MIN_REL_FREQ]

    if not preds:
        log("No relations met the minimum frequency threshold.")
        result = {"top_triples": [], "relation_clusters": []}
        (OUTPUT / "03_relations.json").write_text(json.dumps(result, indent=2))
        (OUTPUT / "03_relations.txt").write_text("No relations discovered.\n")
        return

    log(f"embedding {len(preds)} relation strings…")
    encoder = SentenceTransformer(EMBED_MODEL)
    emb = encoder.encode(preds, show_progress_bar=False, normalize_embeddings=True, batch_size=256)

    log("clustering with HDBSCAN…")
    raw_labels = HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        metric="euclidean",
        cluster_selection_method="eom",
    ).fit_predict(emb)

    groups: dict[int, list[str]] = defaultdict(list)
    for rel, lab in zip(preds, raw_labels):
        if lab != -1:
            groups[lab].append(rel)

    top_triples = [
        {"head": h, "relation": r, "tail": tl, "count": count}
        for (h, r, tl), count in raw_triples.most_common(200)
    ]

    clusters_out = []
    for lab, members in groups.items():
        members.sort(key=lambda r: -rel_freq[r])
        clusters_out.append({
            "label_hint": members[0],
            "total_freq": sum(rel_freq[r] for r in members),
            "relations": members[:10],
            "examples": rel_examples.get(members[0], [])[:3],
        })
    clusters_out.sort(key=lambda x: -x["total_freq"])

    result = {
        "top_triples": top_triples[:50],
        "relation_clusters": clusters_out,
    }
    (OUTPUT / "03_relations.json").write_text(json.dumps(result, indent=2, cls=_NumpyEncoder))

    lines = [
        "RELATION-TYPE DISCOVERY", "=" * 60, "",
        f"Candidate relation types (GLiREL + HDBSCAN, {len(clusters_out)} clusters):",
    ]
    for c in clusters_out:
        lines.append(f"\n  ~{c['label_hint'].upper()}  [freq {c['total_freq']}]")
        lines.append(f"     relations: {', '.join(c['relations'])}")
        for ex in c["examples"]:
            lines.append(f"     e.g. \"{ex['head']}\" → \"{ex['tail']}\"  ({ex['sentence'][:80]}…)")
    lines += ["", "Top extracted triples:"]
    for tri in top_triples[:30]:
        lines.append(f"  {tri['count']:>4}  {tri['head']} -[{tri['relation']}]→ {tri['tail']}")

    (OUTPUT / "03_relations.txt").write_text("\n".join(lines))
    log("\n".join(lines))
    log(f"\n→ {OUTPUT / '03_relations.json'}")


# ── Entry point ──────────────────────────────────────────────────────────────


def main():
    OUTPUT.mkdir(exist_ok=True)
    if "--cluster-only" in sys.argv:
        d = json.loads(COUNTS.read_text())
        log(f"loaded checkpoint: {d['docs_processed']} docs")
        raw_triples: Counter = Counter({
            tuple(k.split("\t")): v for k, v in d["raw_triples"].items()
        })
        cluster_and_report(Counter(d["rel_freq"]), d["rel_examples"], raw_triples)
        return
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    limit = int(args[0]) if args else LIMIT
    rel_freq, rel_examples, raw_triples = accumulate(limit)
    cluster_and_report(rel_freq, rel_examples, raw_triples)


if __name__ == "__main__":
    main()
