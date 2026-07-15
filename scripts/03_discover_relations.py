"""Step 3 - Relation-type DISCOVERY (fastcoref → REBEL → HDBSCAN → GLiNER typing).

Pipeline per document:
  1. fastcoref (transformer-based): resolves pronoun/reference mentions at
     document level. Pronouns in each sentence are replaced with the named
     entity they refer to, so REBEL sees clean entity text across sentences.
  2. REBEL (seq2seq, Babelscape/rebel-large): extracts (head, relation, tail)
     triples sentence by sentence. No fixed predicate vocabulary — relations
     emerge directly from the text.
  3. Relation strings are embedded and clustered with HDBSCAN: near-synonyms
     ("founded by", "was founded by", "co-founded by") merge automatically.
     Cluster count is discovered, not specified.
  4. GLiNER types the head/tail entity texts in the top triples to give each
     relation cluster a domain → range signature.

Hardened for large-corpus scale: streams docs, caps per-doc length and
sentence count, prints flushed progress, checkpoints every CKPT_EVERY docs.
Use --cluster-only to re-run clustering/typing from the checkpoint.

GPU note: REBEL on CPU is ~1 s/sentence. Set REBEL_DEVICE=0 to use a GPU.

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

import spacy
from fastcoref import FCoref
from gliner import GLiNER
from sentence_transformers import SentenceTransformer
from sklearn.cluster import HDBSCAN
from transformers import pipeline as hf_pipeline

from common import iter_docs, OUTPUT

# ── Configuration ─────────────────────────────────────────────────────────────

LIMIT = None
MAX_DOC_CHARS = 120_000
MAX_SENTS_PER_DOC = 60      # cap sentences fed to REBEL per doc (speed)
MIN_SENT_CHARS = 20         # skip very short / boilerplate sentences
PROGRESS_EVERY = 25
CKPT_EVERY = 100
MIN_REL_FREQ = 3            # min occurrences for a relation string to cluster
MIN_CLUSTER_SIZE = 3        # HDBSCAN: fewest items that form a cluster
REBEL_BATCH_SIZE = 8        # sentences per REBEL forward pass
REBEL_DEVICE = int(os.getenv("REBEL_DEVICE", "-1"))  # -1 = CPU; 0 = first GPU
EMBED_MODEL = "all-MiniLM-L6-v2"
GLINER_MODEL = "urchade/gliner_medium-v2.1"
GLINER_THRESHOLD = 0.4

# Mirror the entity labels from script 02 for domain/range typing.
# Leave empty to skip GLiNER typing — REBEL triple extraction still runs in full.
ENTITY_LABELS: list[str] = [
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

COUNTS = OUTPUT / "03_counts.json"

# ── Coreference helpers ───────────────────────────────────────────────────────


def build_coref_map(text: str, doc, coref_model: FCoref) -> dict[int, str | None]:
    """Map spaCy token indices of pronoun mentions to their resolved entity text.

    fastcoref returns character-level (start, end) spans; we reconcile them
    with the spaCy token index space so resolve_sentence can iterate tokens.

    Returns {token_idx: replacement_text} where None means "skip this token"
    (it is a non-head token of a multi-token pronoun span).
    """
    preds = coref_model.predict(texts=[text])
    clusters_spans = preds[0].get_clusters()           # list[list[(start_char, end_char)]]
    clusters_texts = preds[0].get_clusters(as_strings=True)  # same shape, as strings

    coref_map: dict[int, str | None] = {}
    for span_group, text_group in zip(clusters_spans, clusters_texts):
        # Pick the most informative mention — prefer a span with a proper noun.
        best_text: str | None = None
        for (cs, ce), mention_text in zip(span_group, text_group):
            spacy_span = doc.char_span(cs, ce, alignment_mode="expand")
            if spacy_span and any(t.pos_ == "PROPN" for t in spacy_span):
                best_text = mention_text
                break
        if best_text is None and text_group:
            best_text = text_group[0]

        # Replace pronoun-only mentions with the best representative text.
        for (cs, ce), mention_text in zip(span_group, text_group):
            if mention_text == best_text:
                continue
            spacy_span = doc.char_span(cs, ce, alignment_mode="expand")
            if spacy_span and all(t.pos_ == "PRON" for t in spacy_span):
                coref_map[spacy_span[0].i] = best_text
                for tok in spacy_span[1:]:
                    coref_map[tok.i] = None  # secondary tokens → drop
    return coref_map


def resolve_sentence(sent, coref_map: dict[int, str | None]) -> str:
    """Return sentence text with pronoun spans replaced by their referents."""
    parts: list[str] = []
    for tok in sent:
        if tok.i not in coref_map:
            parts.append(tok.text_with_ws)
        elif coref_map[tok.i] is not None:
            parts.append(coref_map[tok.i] + " ")
        # else: secondary token of a pronoun span — omit
    return "".join(parts).strip()


# ── REBEL helpers ─────────────────────────────────────────────────────────────


def extract_rebel_triplets(generated_text: str) -> list[dict]:
    """Parse REBEL's linearized output: <triplet> SUBJ <subj> OBJ <obj> REL …"""
    triplets: list[dict] = []
    subject = object_ = relation = ""
    current: str | None = None
    for token in (
        generated_text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split()
    ):
        if token == "<triplet>":
            if subject and relation and object_:
                triplets.append({
                    "head": subject.strip(),
                    "relation": relation.strip(),
                    "tail": object_.strip(),
                })
            subject = object_ = relation = ""
            current = "subject"
        elif token == "<subj>":
            if relation:  # second <subj> within same triplet group
                triplets.append({
                    "head": subject.strip(),
                    "relation": relation.strip(),
                    "tail": object_.strip(),
                })
            object_ = ""
            current = "object"
        elif token == "<obj>":
            relation = ""
            current = "relation"
        elif current == "subject":
            subject += " " + token
        elif current == "object":
            object_ += " " + token
        elif current == "relation":
            relation += " " + token
    if subject and relation and object_:
        triplets.append({
            "head": subject.strip(),
            "relation": relation.strip(),
            "tail": object_.strip(),
        })
    return triplets


def log(msg: str) -> None:
    print(msg, flush=True)


# ── Accumulation pass ─────────────────────────────────────────────────────────


def accumulate(limit):
    log("loading spaCy…")
    nlp = spacy.load("en_core_web_md")
    nlp.max_length = 2_000_000

    log("loading fastcoref…")
    coref_model = FCoref(device="cpu")

    log("loading REBEL…")
    rebel = hf_pipeline(
        "text2text-generation",
        model="Babelscape/rebel-large",
        device=REBEL_DEVICE,
    )

    rel_freq: Counter = Counter()
    rel_examples: dict[str, list] = defaultdict(list)
    raw_triples: Counter = Counter()  # (head, relation, tail) → count

    def checkpoint(n: int) -> None:
        COUNTS.write_text(json.dumps({
            "docs_processed": n,
            "rel_freq": dict(rel_freq),
            "rel_examples": {r: ex for r, ex in rel_examples.items()},
            # Tab-separated key so head/tail text can contain | safely.
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
        coref_map = build_coref_map(t, doc, coref_model)

        sentences = [
            resolve_sentence(sent, coref_map)
            for sent in doc.sents
            if len(sent.text.strip()) >= MIN_SENT_CHARS
        ][:MAX_SENTS_PER_DOC]
        del doc

        if sentences:
            outputs = rebel(sentences, max_length=512, num_beams=3, batch_size=REBEL_BATCH_SIZE)
            for sent_text, out in zip(sentences, outputs):
                for tri in extract_rebel_triplets(out["generated_text"]):
                    h, r, tl = tri["head"], tri["relation"], tri["tail"]
                    if h and r and tl and len(r) >= 2:
                        raw_triples[(h, r, tl)] += 1
                        rel_freq[r] += 1
                        if len(rel_examples[r]) < 5:
                            rel_examples[r].append({
                                "sentence": sent_text[:200],
                                "head": h,
                                "tail": tl,
                            })

        if i % PROGRESS_EVERY == 0:
            log(f"  …{i} docs | relations={len(rel_freq)} triples={len(raw_triples)}")
        if i % CKPT_EVERY == 0:
            checkpoint(i)

    checkpoint(i)
    log(f"accumulation done: {i} docs")
    return rel_freq, rel_examples, raw_triples


# ── Clustering + report ───────────────────────────────────────────────────────


def type_entity(gliner: GLiNER, text: str) -> str:
    """Return the best GLiNER label for an entity string, or 'unknown'."""
    hits = gliner.predict_entities(text, ENTITY_LABELS, threshold=GLINER_THRESHOLD)
    return hits[0]["label"] if hits else "unknown"


def cluster_and_report(rel_freq, rel_examples, raw_triples):
    preds = [r for r, c in rel_freq.items() if c >= MIN_REL_FREQ]
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

    if ENTITY_LABELS:
        log("typing head/tail entities in top triples with GLiNER…")
        gliner = GLiNER.from_pretrained(GLINER_MODEL)
        typed_triples = [
            {
                "head": h,
                "head_type": type_entity(gliner, h),
                "relation": r,
                "tail": tl,
                "tail_type": type_entity(gliner, tl),
                "count": count,
            }
            for (h, r, tl), count in raw_triples.most_common(200)
        ]
    else:
        log("ENTITY_LABELS is empty — skipping domain/range typing; set labels in the script to enable.")
        typed_triples = [
            {"head": h, "head_type": "?", "relation": r, "tail": tl, "tail_type": "?", "count": count}
            for (h, r, tl), count in raw_triples.most_common(200)
        ]

    clusters_out = []
    for lab, members in groups.items():
        members.sort(key=lambda r: -rel_freq[r])
        dr: Counter = Counter()
        for tri in typed_triples:
            if tri["relation"] in members:
                dr[(tri["head_type"], tri["tail_type"])] += tri["count"]
        clusters_out.append({
            "label_hint": members[0],
            "total_freq": sum(rel_freq[r] for r in members),
            "relations": members[:10],
            "top_domain_range": [
                {"domain": a, "range": b, "count": c}
                for (a, b), c in dr.most_common(4)
            ],
            "examples": rel_examples.get(members[0], [])[:2],
        })
    clusters_out.sort(key=lambda x: -x["total_freq"])

    result = {
        "top_typed_triples": typed_triples[:40],
        "relation_clusters": clusters_out,
    }
    (OUTPUT / "03_relations.json").write_text(json.dumps(result, indent=2))

    lines = [
        "RELATION-TYPE DISCOVERY", "=" * 60, "",
        "Candidate relation types (REBEL + HDBSCAN):",
    ]
    for c in clusters_out:
        dr = "; ".join(
            f"{d['domain']}→{d['range']}({d['count']})"
            for d in c["top_domain_range"]
        )
        lines.append(f"\n  ~{c['label_hint'].upper()}  [freq {c['total_freq']}]")
        lines.append(f"     relations: {', '.join(c['relations'])}")
        if dr:
            lines.append(f"     domain→range: {dr}")
        for ex in c["examples"]:
            lines.append(f"     e.g. {ex['sentence']}")
    lines += ["", "Top typed (head_type) -[relation]→ (tail_type) triples:"]
    for tri in typed_triples[:25]:
        lines.append(
            f"  {tri['count']:>4}  ({tri['head_type']}) -[{tri['relation']}]→ ({tri['tail_type']})"
            f"   [{tri['head']} → {tri['tail']}]"
        )

    (OUTPUT / "03_relations.txt").write_text("\n".join(lines))
    log("\n".join(lines))
    log(f"\n→ {OUTPUT / '03_relations.json'}")


# ── Entry point ───────────────────────────────────────────────────────────────


def main():
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
