"""Batched few-shot intent classification through the shared LLM client."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from utils.llm_client import call_json, print_stats


def load_intents(path: str) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))["intents"]


def classify(messages: list[str], intents: list[dict], batch_size: int = 15, dry_run: bool = False) -> list[dict]:
    catalog = [{"id": x["id"], "name": x["name"], "definition": x.get("definition", "")} for x in intents]
    results = []
    for start in range(0, len(messages), batch_size):
        batch = messages[start:start + batch_size]
        if dry_run:
            results.extend({"index": i, "intent_id": catalog[0]["id"], "intent": catalog[0]["name"], "confidence": 0.2, "reason": "dry-run placeholder"} for i in range(len(batch)))
            continue
        prompt = ("Classify each message into exactly one catalog intent. Return JSON with key predictions, "
                  "an array of {index, intent_id, confidence, reason}; confidence is 0-1.\n"
                  f"Catalog: {json.dumps(catalog)}\nMessages: " + json.dumps([{"index": i, "message": m} for i, m in enumerate(batch)]))
        response = call_json(prompt, max_tokens=max(800, 100 * len(batch)))
        for item in response.get("predictions", []):
            item["index"] = int(item["index"])
            item["intent_id"] = int(item["intent_id"])
            item["confidence"] = float(item.get("confidence", 0.0))
            item["intent"] = next((x["name"] for x in intents if x["id"] == item["intent_id"]), "unknown")
            results.append(item)
    results.sort(key=lambda x: x["index"])
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/pairs.csv")
    parser.add_argument("--intents", default="data/intents.json")
    parser.add_argument("--output", default="data/classifications.csv")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=15)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    data = pd.read_csv(args.input).head(args.limit)
    intents = load_intents(args.intents)
    predictions = classify(data["customer_message"].tolist(), intents, args.batch_size, args.dry_run)
    output = data.reset_index(drop=True).copy()
    pred = pd.DataFrame(predictions).rename(columns={"intent": "predicted_intent"})
    output = pd.concat([output, pred], axis=1)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print_stats("Classifier")


if __name__ == "__main__":
    main()
