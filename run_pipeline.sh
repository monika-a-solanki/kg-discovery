#!/usr/bin/env bash
# run_pipeline.sh — run all three discovery scripts in sequence.
#
# Usage:
#   ./run_pipeline.sh            # full corpus
#   ./run_pipeline.sh 50         # first 50 docs (quick test)
#   REBEL_DEVICE=0 ./run_pipeline.sh   # use GPU for REBEL (script 03)
#
# Each script writes a checkpoint file in output/ every 100 docs,
# so you can safely kill and resume mid-run using --cluster-only.

set -euo pipefail

LIMIT=${1:-}       # optional: number of docs to process
PY="${PY:-.venv/bin/python}"

if [[ ! -f "$PY" ]]; then
    echo "ERROR: Python not found at '$PY'."
    echo "  Run 'uv sync' first, or set PY=/path/to/python."
    exit 1
fi

if [[ -z "$(ls corpus/*.txt 2>/dev/null)" ]] && [[ -z "$(ls corpus/*.json 2>/dev/null)" ]]; then
    echo "ERROR: No .txt or .json files found in corpus/."
    echo "  Place your documents there and re-run."
    exit 1
fi

echo "========================================"
echo " KG Discovery Pipeline"
echo " Python: $PY"
[[ -n "$LIMIT" ]] && echo " Limit: $LIMIT docs" || echo " Limit: full corpus"
[[ -n "${REBEL_DEVICE:-}" ]] && echo " REBEL_DEVICE: $REBEL_DEVICE"
echo "========================================"
echo

echo "--- Step 1: Characterize corpus ---"
"$PY" scripts/01_characterize.py ${LIMIT:+$LIMIT}
echo
echo "Output written to output/01_characterization.txt"
echo
echo "-----------------------------------------------------------"
echo " Review output/01_characterization.txt."
echo " If you want GLiNER NER (entity typing), edit:"
echo "   ENTITY_LABELS in scripts/02_discover_entities.py"
echo "   ENTITY_LABELS in scripts/03_discover_relations.py"
echo " Leave lists empty to run fully unsupervised."
echo "-----------------------------------------------------------"
echo

echo "--- Step 2: Discover entity types ---"
"$PY" scripts/02_discover_entities.py ${LIMIT:+$LIMIT}
echo
echo "Output written to output/02_entities.txt and output/02_entities.json"
echo

echo "--- Step 3: Discover relation types ---"
"$PY" scripts/03_discover_relations.py ${LIMIT:+$LIMIT}
echo
echo "Output written to output/03_relations.txt and output/03_relations.json"
echo

echo "========================================"
echo " Pipeline complete."
echo " Results are in the output/ directory."
echo "========================================"
