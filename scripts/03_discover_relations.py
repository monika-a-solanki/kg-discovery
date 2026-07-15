"""Step 3 - Relation-type DISCOVERY (OpenIE -> cluster -> name).

No fixed predicate list. For each sentence:
  1. Tag entity spans using the biomedical NER models (NOT gazetteers) -> this
     is what keeps discovery independent of the prior schema.
  2. For co-occurring entity pairs, extract the connecting predicate from the
     dependency parse (verb lemma on the path between the two entity heads,
     plus any governing preposition). Real OpenIE, not a verb list.
  3. Record (subjType, predicateLemma, objType) triples.

Then cluster predicate lemmas by embedding so near-synonyms collapse into a
candidate relation type, with dominant domain/range + examples.

Hardened for full-corpus scale: streams docs, caps per-doc length, prints
flushed progress, checkpoints raw triple counts every CKPT_EVERY docs so the
expensive parse survives a crash (--cluster-only re-runs clustering only).

Outputs:
  output/03_counts.json     - raw aggregates (checkpoint, resumable)
  output/03_relations.json
  output/03_relations.txt
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
MAX_DOC_CHARS = 120_000
PROGRESS_EVERY = 25
CKPT_EVERY = 100
N_PRED_CLUSTERS = 18
MIN_PRED_FREQ = 5
EMBED_MODEL = "all-MiniLM-L6-v2"
ENT_LABELS = {"DISEASE", "CHEMICAL", "PROTEIN", "CELL_LINE", "CELL_TYPE", "DNA", "RNA"}

COUNTS = OUTPUT / "03_counts.json"


def log(msg):
    print(msg, flush=True)


def dep_predicate(a_head, b_head):
    a_anc = list(a_head.ancestors)
    b_anc = {t.i for t in b_head.ancestors}
    lca = next((t for t in a_anc if t.i in b_anc), None)
    cand = lca if lca is not None else a_head.head
    verb = None
    if cand.pos_ in ("VERB", "AUX"):
        verb = cand.lemma_.lower()
    else:
        for t in [cand] + list(cand.ancestors):
            if t.pos_ in ("VERB", "AUX"):
                verb = t.lemma_.lower()
                break
    prep = None
    for t in (a_head, b_head):
        if t.dep_ == "pobj" and t.head.pos_ == "ADP":
            prep = t.head.lemma_.lower()
    if verb and prep:
        return f"{verb}_{prep}"
    return verb


def accumulate(limit):
    base = spacy.load("en_core_sci_md", disable=["ner"])
    bc5 = spacy.load("en_ner_bc5cdr_md")
    jnl = spacy.load("en_ner_jnlpba_md")
    for n in (base, bc5, jnl):
        n.max_length = 2_000_000

    triples = Counter()
    pred_freq = Counter()
    pred_pair = defaultdict(Counter)
    examples = defaultdict(list)

    def checkpoint(i):
        COUNTS.write_text(json.dumps({
            "docs_processed": i,
            "triples": {f"{s}|{p}|{o}": c for (s, p, o), c in triples.items()},
            "pred_freq": dict(pred_freq),
            "pred_pair": {p: {f"{a}|{b}": c for (a, b), c in pp.items()}
                          for p, pp in pred_pair.items()},
            "examples": {p: ex for p, ex in examples.items()},
        }))

    i = 0
    for pmcid, text in iter_docs(limit):
        i += 1
        t = text[:MAX_DOC_CHARS]
        doc = base(t)
        spans = []
        for nlp in (bc5, jnl):
            with nlp.select_pipes(enable=["tok2vec", "ner"]):
                nd = nlp(t)
            for e in nd.ents:
                if e.label_ in ENT_LABELS and 2 < len(e.text) < 40:
                    spans.append((e.start_char, e.end_char, e.label_))
            del nd
        if len(spans) >= 2:
            for sent in doc.sents:
                ss = [(s, e, l) for s, e, l in spans
                      if s >= sent.start_char and e <= sent.end_char]
                if len(ss) < 2:
                    continue
                heads = []
                for s, e, l in ss:
                    span = doc.char_span(s, e, alignment_mode="expand")
                    if span is not None:
                        heads.append((span.root, l))
                for a in range(len(heads)):
                    for b in range(len(heads)):
                        if a == b:
                            continue
                        (ha, la), (hb, lb) = heads[a], heads[b]
                        if la == lb or abs(ha.i - hb.i) > 25:
                            continue
                        pred = dep_predicate(ha, hb)
                        if not pred or len(pred) < 2:
                            continue
                        triples[(la, pred, lb)] += 1
                        pred_freq[pred] += 1
                        pred_pair[pred][(la, lb)] += 1
                        if len(examples[pred]) < 3:
                            examples[pred].append(re.sub(r"\s+", " ", sent.text).strip()[:200])
        del doc

        if i % PROGRESS_EVERY == 0:
            log(f"  ...{i} docs | preds={len(pred_freq)} triples={len(triples)}")
        if i % CKPT_EVERY == 0:
            checkpoint(i)
    checkpoint(i)
    log(f"accumulation done: {i} docs")
    return triples, pred_freq, pred_pair, examples


def cluster_and_report(triples, pred_freq, pred_pair, examples):
    preds = [p for p, c in pred_freq.items() if c >= MIN_PRED_FREQ]
    log(f"clustering {len(preds)} predicate lemmas...")
    clusters_out = []
    if preds:
        enc = SentenceTransformer(EMBED_MODEL)
        emb = enc.encode(preds, show_progress_bar=False, normalize_embeddings=True, batch_size=256)
        k = min(N_PRED_CLUSTERS, len(preds))
        labels = KMeans(n_clusters=k, random_state=0, n_init=10).fit_predict(emb)
        groups = defaultdict(list)
        for p, lab in zip(preds, labels):
            groups[int(lab)].append(p)
        for lab, members in groups.items():
            members.sort(key=lambda p: -pred_freq[p])
            dr = Counter()
            for p in members:
                dr.update(pred_pair[p])
            clusters_out.append({
                "label_hint": members[0],
                "total_freq": sum(pred_freq[p] for p in members),
                "predicates": members[:10],
                "top_domain_range": [{"domain": a, "range": b, "count": c}
                                     for (a, b), c in dr.most_common(4)],
                "examples": examples.get(members[0], [])[:2],
            })
        clusters_out.sort(key=lambda x: -x["total_freq"])

    result = {
        "top_raw_triples": [
            {"domain": s, "predicate": p, "range": o, "count": c}
            for (s, p, o), c in triples.most_common(40)
        ],
        "relation_clusters": clusters_out,
    }
    (OUTPUT / "03_relations.json").write_text(json.dumps(result, indent=2))

    lines = ["RELATION-TYPE DISCOVERY", "=" * 60, "",
             "Candidate relation types (clustered predicate lemmas):"]
    for c in clusters_out:
        dr = "; ".join(f"{d['domain']}->{d['range']}({d['count']})" for d in c["top_domain_range"])
        lines.append(f"\n  ~{c['label_hint'].upper()}  [freq {c['total_freq']}]")
        lines.append(f"     predicates: {', '.join(c['predicates'])}")
        lines.append(f"     domain->range: {dr}")
        for ex in c["examples"]:
            lines.append(f"     e.g. {ex}")
    lines += ["", "Top raw (type, predicate, type) triples:"]
    for r in result["top_raw_triples"][:25]:
        lines.append(f"  {r['count']:>4}  ({r['domain']}) -[{r['predicate']}]-> ({r['range']})")
    (OUTPUT / "03_relations.txt").write_text("\n".join(lines))
    log("\n".join(lines))
    log(f"\n-> {OUTPUT/'03_relations.json'}")


def main():
    if "--cluster-only" in sys.argv:
        d = json.loads(COUNTS.read_text())
        log(f"loaded checkpoint: {d['docs_processed']} docs")
        triples = Counter({tuple(k.split("|")): v for k, v in d["triples"].items()})
        pred_pair = defaultdict(Counter)
        for p, pp in d["pred_pair"].items():
            for ab, c in pp.items():
                a, b = ab.split("|")
                pred_pair[p][(a, b)] = c
        cluster_and_report(triples, Counter(d["pred_freq"]), pred_pair, d["examples"])
        return
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    limit = int(args[0]) if args else LIMIT
    triples, pred_freq, pred_pair, examples = accumulate(limit)
    cluster_and_report(triples, pred_freq, pred_pair, examples)


if __name__ == "__main__":
    main()
