"""Batched rubric judge and agreement helper."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from utils.llm_client import call_json, print_stats


def judge(rows: list[dict], batch_size: int = 5, dry_run: bool = False) -> list[dict]:
    output = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        if dry_run:
            output.extend({"index": i, "correctness": 3, "tone": 3, "groundedness": 3, "escalation_appropriateness": 3, "overall": 3, "reason": "dry-run placeholder"} for i in range(len(batch)))
            continue
        prompt = ("Judge each support response from 1-5 for correctness, tone, groundedness, and "
                  "escalation-decision appropriateness. Return JSON key scores, each with index, "
                  "four named scores, overall, and reason.\n" + json.dumps([{"index": i, **row} for i, row in enumerate(batch)]))
        response = call_json(prompt, max_tokens=180 * len(batch))
        output.extend(response.get("scores", []))
    return output


def agreement(judge_df: pd.DataFrame, human_df: pd.DataFrame) -> dict:
    joined = judge_df.merge(human_df, on="message", suffixes=("_judge", "_human"))
    if joined.empty:
        return {"n": 0, "percent_agreement": None, "cohen_kappa": None}
    judge_decision = joined["escalation_decision_judge"]
    human_decision = joined["ground_truth_escalate"]
    return {"n": len(joined), "percent_agreement": float((judge_decision == human_decision).mean()), "cohen_kappa": float(cohen_kappa_score(human_decision, judge_decision)) if joined["ground_truth_escalate"].nunique() > 1 else None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/judge_inputs.json")
    parser.add_argument("--output", default="data/judge_scores.json")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    rows = json.loads(Path(args.input).read_text(encoding="utf-8"))[:args.limit]
    scores = judge(rows, dry_run=args.dry_run)
    Path(args.output).write_text(json.dumps(scores, indent=2), encoding="utf-8")
    print_stats("Judge")


if __name__ == "__main__":
    main()
