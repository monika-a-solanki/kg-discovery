# Corpus-grounded KG schema (proposed)

Derived from the discovery run over all 1000 PMC documents
(`02_entities.*`, `03_relations.*`). No prior/inherited labels were used —
entities come from scispaCy biomedical NER, relations from dependency-path
OpenIE. This document is the **curation** of that raw output: keep the
biological signal, drop research-process noise, name predicates, assign
domain/range + direction.

Status: PROPOSED — needs human (domain-expert) confirmation before extraction.

---

## Entity types

### Core (high-confidence, from NER frequency)
| Type | Corpus freq | Notes |
|------|------------:|-------|
| `Protein`  | 187,462 | incl. antibodies, cytokines, receptors (NER lumps these together) |
| `Chemical` | 100,163 | small molecules, salts, media components (glucose, NaCl, sucrose) |
| `Disease`  |  65,810 | conditions, infections, tumors |
| `CellType` |  45,277 | macrophages, T cells, tumor cells |
| `Gene`     |  44,867 | (NER label DNA) genes, cDNA, plasmids, loci |
| `CellLine` |  36,948 | HEK293T, hiPSCs, Vero E6 |
| `RNA`      |   4,237 | mRNA, viral RNA |

### Emergent candidates (from noun-chunk clusters — need validation)
| Candidate type | Evidence | Decision |
|----------------|----------|----------|
| `Assay/Technique` | assay, ELISA, qPCR, chromatography, blot | PROMOTE — recurring, domain-meaningful |
| `Formulation/Nanocarrier` | nanoparticles, nanobodies, nanocarriers | PROMOTE — distinct mAb-delivery type |
| `Buffer/Excipient` | buffer, NaCl, EDTA, DMSO | MERGE into `Chemical` (overlaps) |
| `Vendor/Reagent-brand` | Sigma, Gibco, Invitrogen, ThermoFisher | DROP — metadata, not domain content |

> Note: the generic biomedical NER does **not** isolate `Antibody` as its own
> type — antibodies sit inside `Protein`. An antibody-focused KG would need a
> sub-typing rule (suffix `*mab`, "monoclonal antibody") layered on `Protein`.

---

## Relation types (curated)

Raw OpenIE produced ~11 predicate clusters. Below: the **kept** biological
relations with canonical names, then the **dropped** clusters.

### KEEP — biological / causal relations
| Canonical predicate | From cluster(s) | Domain → Range | Direction |
|---------------------|-----------------|----------------|-----------|
| `BINDS` | LEAD (bind, target, fuse) | Protein/Chemical → Protein | ligand → target |
| `INHIBITS` | INHIBIT + SUPPRESS (inhibit, suppress, block, prevent) | Chemical/Protein → Protein/Disease | agent → target |
| `ACTIVATES` | INDUCE + part of UPREGULATE (induce, promote, activate, upregulate) | Chemical/Protein → Protein/Disease | agent → target |
| `REGULATES_EXPRESSION` | INCREASE + UPREGULATE (increase, decrease, up/downregulate) | Protein/Chemical → Protein/Gene | regulator → regulated; valence (+/-) as edge property |
| `ASSOCIATED_WITH` | (PROTEIN↔DISEASE co-occurrence) | Protein → Disease | implicated-in |
| `HAS_COMPONENT` | INCLUDE (include, comprise, consist) | Protein-complex/Formulation → Protein/Chemical | whole → part |

### DROP — research-process / epistemic verbs (not domain relations)
| Cluster | Why dropped |
|---------|-------------|
| `SHOW` (91k) | show, find, suggest, report, confirm — "the paper states X" |
| `OBSERVE` (75k) | observe, detect, reveal, investigate, analyze — measurement framing |
| `USE` (75k) | use, obtain, purchase, represent — methodological |
| measurement verbs in `REDUCE` | determine, evaluate, calculate, quantify — assay reporting |

---

## Key findings driving this schema

1. **Center of mass is `Chemical ↔ Protein` then `Protein ↔ Disease`/`Protein ↔ Gene`.**
   The corpus is broadly biomolecular — much wider than the antibody-centric
   `Antibody → {Target, CellLine, Disease}` schema used previously.

2. **~40% of surface predicate volume is epistemic** (SHOW/OBSERVE/USE).
   Any extractor MUST filter these or the KG fills with "(paper) shows (protein)".

3. **Direction is unresolved by discovery.** Symmetric cluster counts
   (CHEMICAL→PROTEIN ≈ PROTEIN→CHEMICAL) are an artifact of enumerating both
   orderings. Direction above is assigned by biological convention, not measured.

4. **Clusters are imperfect.** Some mix valences/senses (REDUCE holds both
   "reduce" and "calculate"). A second curation/split pass would tighten them.

---

## Proposed schema summary (for extraction)

```
ENTITIES:  Protein, Chemical, Disease, CellType, Gene, CellLine, RNA,
           Assay, Formulation            (+ Antibody as Protein subtype)

RELATIONS: (Protein|Chemical) -BINDS-> Protein
           (Chemical|Protein) -INHIBITS-> (Protein|Disease)
           (Chemical|Protein) -ACTIVATES-> (Protein|Disease)
           (Protein|Chemical) -REGULATES_EXPRESSION[valence]-> (Protein|Gene)
           Protein -ASSOCIATED_WITH-> Disease
           (Complex|Formulation) -HAS_COMPONENT-> (Protein|Chemical)
```

Next step: confirm/edit, then feed these labels to GLiNER-Relex for extraction
— this time the label set is corpus-grounded, not inherited.
