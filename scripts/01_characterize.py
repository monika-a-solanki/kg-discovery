"""Step 1 - Corpus characterization.

Answers "what domain shape am I dealing with?" before any extractor is built.
Uses only general NLP (spaCy en_core_web_md) + TF-IDF. No domain priors.

Outputs:
  output/01_characterization.txt  - human-readable summary
"""

from collections import Counter

import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

from common import iter_docs, OUTPUT, SENT_SPLIT

LIMIT = None  # set e.g. 200 for a quick pass


def main(limit=LIMIT):
    docs = list(iter_docs(limit))
    n = len(docs)
    lengths = [len(t) for _, t in docs]
    n_sents = [len(SENT_SPLIT.split(t)) for _, t in docs]

    # TF-IDF salient unigrams/bigrams across the corpus (domain vocabulary).
    texts = [t for _, t in docs]
    vec = TfidfVectorizer(
        ngram_range=(1, 2), min_df=5, max_df=0.5,
        stop_words="english", max_features=4000,
    )
    X = vec.fit_transform(texts)
    vocab = vec.get_feature_names_out()
    means = X.mean(axis=0).A1  # mean tf-idf per term = corpus-wide salience
    top = sorted(zip(vocab, means), key=lambda x: -x[1])[:60]

    # Coarse keyphrase noun-chunk frequency via the general English model.
    nlp = spacy.load("en_core_web_md", disable=["ner"])
    nlp.max_length = 2_000_000
    chunk_freq = Counter()
    for _, t in docs[: min(n, 150)]:  # cap for speed
        doc = nlp(t[:200_000])
        for ch in doc.noun_chunks:
            h = ch.root.text.lower()
            if h.isalpha() and len(h) > 3:
                chunk_freq[h] += 1

    lines = []
    lines.append(f"CORPUS CHARACTERIZATION  ({n} documents)")
    lines.append("=" * 60)
    lines.append(f"chars/doc:  min={min(lengths)}  median={sorted(lengths)[n//2]}  max={max(lengths)}")
    lines.append(f"sents/doc:  median={sorted(n_sents)[n//2]}")
    lines.append("")
    lines.append("Top TF-IDF terms (corpus-salient vocabulary):")
    for term, sc in top:
        lines.append(f"  {sc:.4f}  {term}")
    lines.append("")
    lines.append("Most frequent noun-chunk heads (candidate entity anchors):")
    for h, c in chunk_freq.most_common(50):
        lines.append(f"  {c:>5}  {h}")

    out = OUTPUT / "01_characterization.txt"
    out.write_text("\n".join(lines))
    print("\n".join(lines[:25]))
    print(f"\n... full report -> {out}")


if __name__ == "__main__":
    import sys
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else LIMIT
    main(lim)
