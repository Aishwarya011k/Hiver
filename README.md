# AmazonHelp Support Agent

A take-home implementation for the Kaggle Customer Support on Twitter dataset. It reconstructs AmazonHelp customer/support pairs, discovers intents with local embeddings and KMeans, classifies in batched `gpt-4o-mini` requests, retrieves historical resolutions with local cosine similarity, drafts grounded replies, and applies a deterministic escalation policy.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# put your OPENAI_API_KEY in .env
mkdir -p data
# put Kaggle twcs.csv at data/archive/twcs/twcs.csv
```

The first `sentence-transformers` run downloads `all-MiniLM-L6-v2`; all later embedding work is local. OpenAI calls use only `gpt-4o-mini`, are cached in `.cache/llm`, retried on rate limits, and separated by a conservative two-second interval. The first run may therefore be slow; warm-cache reruns reuse responses.

## Exact reproduction

```bash
./run_all.sh
```

For a no-API smoke test on ten rows:

```bash
LIMIT=10 DRY_RUN=1 ./run_all.sh
printf '%s\n' 'My package is late' 'I need a refund' > /tmp/messages.txt
python3 pipeline.py --input /tmp/messages.txt --limit 2 --dry-run
```

The normal pipeline sequence is:

```bash
python3 data_prep.py --brand AmazonHelp
python3 intent_taxonomy.py
python3 retrieval.py
python3 eval/build_golden_set.py
python3 pipeline.py --input messages.txt --limit 5
```

After generating `data/golden_set.csv`, hand-label `true_intent`, then train/evaluate local baselines. Copy pipeline outputs into an evaluation table with the required ground-truth columns before running `eval/metrics.py`.

## Scope and budget

Default scope is 700 linked threads, a 150-example golden set, and a 30-example judge/human subset. This deliberate scope-down reflects the rate-limited API constraint. A cold 30-example smoke/evaluation path is approximately 27 calls: 1 taxonomy naming call, 10 classifier batches for 150 examples, 2 classifier batches for the smoke run, 8 reply batches for 30 examples, and 6 judge batches. A full 150-example reply evaluation is approximately 55 calls because it needs 38 reply batches instead of 8. Processing replies for all 700 threads would be approximately 230 calls overall, so it should be done only after the smaller evaluation is accepted. Cache hits do not call the API. The wrapper prints calls, cache-hit rate, token counts, and approximate spend.

## Manual inputs

You must choose/confirm the brand handle, provide `twcs.csv`, review and rename `data/intents.json`, hand-label the golden set, fill the 30-row human template, and complete the report's TODO-ME sections with actual metrics and examples. The report intentionally does not fabricate evaluation results.
