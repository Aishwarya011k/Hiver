#!/usr/bin/env bash
set -euo pipefail

LIMIT="${LIMIT:-}"
DRY="${DRY_RUN:-}"
ARGS=()
if [[ -n "$LIMIT" ]]; then ARGS+=(--limit "$LIMIT"); fi
if [[ -n "$DRY" ]]; then ARGS+=(--dry-run); fi

python3 data_prep.py --input data/archive/twcs/twcs.csv --brand AmazonHelp "${ARGS[@]}"
python3 intent_taxonomy.py "${ARGS[@]}"
python3 retrieval.py "${ARGS[@]}"
python3 eval/build_golden_set.py "${ARGS[@]}"
python3 -c 'import json, os, pandas as pd; from pipeline import run; from pathlib import Path; d = pd.read_csv("data/pairs.csv").head(30); results = run(d["customer_message"].tolist(), "data/pairs.csv", "data/intents.json", "data/retrieval_index", dry_run=bool(os.getenv("DRY_RUN"))); Path("data/pipeline_results.json").write_text(json.dumps(results, indent=2))'
python3 -m py_compile data_prep.py intent_taxonomy.py classifier.py baselines.py retrieval.py reply_generator.py escalation_policy.py pipeline.py eval/*.py utils/*.py
printf '\nPipeline code validation passed.\n'
printf 'API estimate: 30-example smoke path is about 27 calls; full 150-example reply evaluation is about 55 uncached calls.\n'
printf 'Warm-cache reruns should make zero calls for identical prompts.\n'
