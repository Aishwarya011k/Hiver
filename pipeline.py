"""End-to-end AmazonHelp support-agent CLI."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from classifier import classify, load_intents
from escalation_policy import decide
from reply_generator import generate
from retrieval import load_index
from utils.llm_client import print_stats, stats


def run(messages: list[str], pairs_path: str, intents_path: str, index_dir: str, dry_run: bool = False) -> list[dict]:
    pairs = pd.read_csv(pairs_path)
    intents = load_intents(intents_path)
    retriever = load_index(index_dir) if Path(index_dir).exists() else None
    classifications = classify(messages, intents, dry_run=dry_run)
    reply_requests = []
    for message, classification in zip(messages, classifications):
        evidence = retriever.search(message, 3) if retriever else []
        reply_requests.append({"message": message, "intent": classification["intent"], "evidence": evidence})
    replies = generate(reply_requests, dry_run=dry_run)
    outputs = []
    for message, classification, reply in zip(messages, classifications, replies):
        policy = decide(message, classification["intent"], classification["confidence"], reply["reply_confidence"])
        outputs.append({"message": message, **classification, **reply, **policy})
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Text file, one customer message per line")
    parser.add_argument("--pairs", default="data/pairs.csv")
    parser.add_argument("--intents", default="data/intents.json")
    parser.add_argument("--index-dir", default="data/retrieval_index")
    parser.add_argument("--output", default="data/pipeline_results.json")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    messages = [line.strip() for line in Path(args.input).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        messages = messages[:args.limit]
    results = run(messages, args.pairs, args.intents, args.index_dir, args.dry_run)
    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print_stats("Pipeline")
    print(f"Estimated total API calls made this process: {stats()['api_calls']}; cache hit rate: {stats()['cache_hit_rate']}")


if __name__ == "__main__":
    main()
