"""Grounded reply drafting using retrieved historical resolutions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from utils.llm_client import call_json, print_stats


def generate(requests: list[dict], batch_size: int = 4, dry_run: bool = False) -> list[dict]:
    results = []
    for start in range(0, len(requests), batch_size):
        batch = requests[start:start + batch_size]
        if dry_run:
            for i, request in enumerate(batch):
                evidence = request.get("evidence", [])
                reply = evidence[0]["brand_resolution"] if evidence else "A support specialist will review this request."
                results.append({"index": i, "draft_reply": reply, "reply_confidence": 0.35, "grounding_note": "dry-run historical reply"})
            continue
        prompt = ("Draft concise, empathetic customer-support replies. Use only facts supported by the "
                  "historical resolutions; do not invent order details, refunds, dates, or promises. "
                  "Return JSON key replies, array of {index, draft_reply, reply_confidence, grounding_note}; "
                  "confidence is 0-1.\n" + json.dumps([{"index": i, **item} for i, item in enumerate(batch)]))
        response = call_json(prompt, max_tokens=max(900, 180 * len(batch)))
        for item in response.get("replies", []):
            item["index"] = int(item["index"])
            item["reply_confidence"] = float(item.get("reply_confidence", 0.0))
            results.append(item)
    results.sort(key=lambda x: x["index"])
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/reply_requests.json")
    parser.add_argument("--output", default="data/replies.json")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    requests = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if args.limit:
        requests = requests[:args.limit]
    replies = generate(requests, dry_run=args.dry_run)
    Path(args.output).write_text(json.dumps(replies, indent=2), encoding="utf-8")
    print_stats("Replies")


if __name__ == "__main__":
    main()
