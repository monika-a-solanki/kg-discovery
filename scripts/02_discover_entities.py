"""Step 2 - Entity-type DISCOVERY (open -> cluster -> name).

Independent of any prior schema. Two signal sources, fused:
  A) biomedical NER (scispaCy en_ner_bc5cdr_md + en_ner_jnlpba_md) -> what
     recognized biomedical entity TYPES occur and how often.
  B) frequent noun-chunk spans (en_core_sci_md) embedded with a sentence
     encoder and clustered (KMeans) -> emergent candidate types the fixed
     NER label set does not cover (buffers, process steps, glycoforms...).

Hardened for full-corpus (1000-doc) scale:
  - streams docs one at a time (no giant in-memory list)
  - hard per-doc char cap to bound memory
  - prints flushed progress every PROGRESS_EVERY docs
  - checkpoints raw counts to output/02_counts.json every CKPT_EVERY docs,
    so the expensive NER pass survives a crash (clustering can be re-run
    from the checkpoint via --cluster-only)

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
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

from common import iter_docs, OUTPUT

LIMIT = None
MAX_DOC_CHARS = 120_000   # bound per-doc memory
PROGRESS_EVERY = 25
CKPT_EVERY = 100
N_CLUSTERS = 25
MIN_CHUNK_FREQ = 8
EMBED_MODEL = "all-MiniLM-L6-v2"

COUNTS = OUTPUT / "02_counts.json"


def clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def log(msg):
    print(msg, flush=True)


def accumulate(limit):
    bc5 = spacy.load("en_ner_bc5cdr_md")
    jnl = spacy.load("en_ner_jnlpba_md")
    sci = spacy.load("en_core_sci_md", disable=["ner"])
    for n in (bc5, jnl, sci):
        n.max_length = 2_000_000

    ner_types = Counter()
    ner_examples = defaultdict(Counter)
    chunk_freq = Counter()

    def checkpoint(i):
        COUNTS.write_text(json.dumps({
            "docs_processed": i,
            "ner_types": ner_types,
            "ner_examples": {k: dict(v.most_common(40)) for k, v in ner_examples.items()},
            "chunk_freq": dict(chunk_freq.most_common(6000)),
        }))

    i = 0
    for pmcid, text in iter_docs(limit):
        i += 1
        t = text[:MAX_DOC_CHARS]
        for nlp in (bc5, jnl):
            with nlp.select_pipes(enable=["tok2vec", "ner"]):
                d = nlp(t)
            for e in d.ents:
                txt = clean(e.text).lower()
                if 2 < len(txt) < 40:
                    ner_types[e.label_] += 1
                    ner_examples[e.label_][txt] += 1
            del d
        d = sci(t)
        for ch in d.noun_chunks:
            head = clean(ch.root.text).lower()
            if head.isalpha() and 3 < len(head) < 30:
                chunk_freq[head] += 1
        del d

        if i % PROGRESS_EVERY == 0:
            log(f"  ...{i} docs  | ner_types={len(ner_types)} chunks={len(chunk_freq)}")
        if i % CKPT_EVERY == 0:
            checkpoint(i)
    checkpoint(i)
    log(f"accumulation done: {i} docs")
    return ner_types, ner_examples, chunk_freq


def cluster_and_report(ner_types, ner_examples, chunk_freq):
    candidates = [h for h, c in chunk_freq.items() if c >= MIN_CHUNK_FREQ]
    log(f"clustering {len(candidates)} candidate noun-chunk heads...")
    encoder = SentenceTransformer(EMBED_MODEL)
    emb = encoder.encode(candidates, show_progress_bar=False, normalize_embeddings=True,
                         batch_size=256)
    k = min(N_CLUSTERS, len(candidates))
    km = KMeans(n_clusters=k, random_state=0, n_init=10).fit(emb)

    clusters = defaultdict(list)
    for term, lab in zip(candidates, km.labels_):
        clusters[int(lab)].append((term, chunk_freq[term]))
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
        "ner_types": [
            {"label": lab, "count": cnt,
             "examples": [w for w, _ in Counter(ner_examples[lab]).most_common(10)]}
            for lab, cnt in Counter(ner_types).most_common()
        ],
        "emergent_clusters": cluster_summaries,
    }
    (OUTPUT / "02_entities.json").write_text(json.dumps(result, indent=2))

    lines = ["ENTITY-TYPE DISCOVERY", "=" * 60, "",
             "A) Biomedical NER label inventory (scispaCy):"]
    for r in result["ner_types"]:
        lines.append(f"  {r['count']:>6}  {r['label']:<12}  e.g. {', '.join(r['examples'][:6])}")
    lines += ["", f"B) Emergent candidate types (noun-chunk clusters, k={k}):"]
    for c in cluster_summaries:
        terms = ", ".join(m["term"] for m in c["members"][:8])
        lines.append(f"  [{c['total_freq']:>5}] ~{c['label_hint']}: {terms}")
    (OUTPUT / "02_entities.txt").write_text("\n".join(lines))
    log("\n".join(lines))
    log(f"\n-> {OUTPUT/'02_entities.json'}")


def main():
    if "--cluster-only" in sys.argv:
        data = json.loads(COUNTS.read_text())
        log(f"loaded checkpoint: {data['docs_processed']} docs")
        cluster_and_report(Counter(data["ner_types"]),
                           {k: Counter(v) for k, v in data["ner_examples"].items()},
                           Counter(data["chunk_freq"]))
        return
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    limit = int(args[0]) if args else LIMIT
    ner_types, ner_examples, chunk_freq = accumulate(limit)
    cluster_and_report(ner_types, ner_examples, chunk_freq)


if __name__ == "__main__":
    main()
