"""No-API evaluation metrics for classifier, replies, and escalation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support


def classifier_metrics(data: pd.DataFrame) -> dict:
    data = data.dropna(subset=["true_intent", "predicted_intent"])
    labels = sorted(set(data["true_intent"]) | set(data["predicted_intent"]))
    return {"accuracy": float(accuracy_score(data["true_intent"], data["predicted_intent"])), "per_intent": classification_report(data["true_intent"], data["predicted_intent"], labels=labels, output_dict=True, zero_division=0), "confusion_matrix": confusion_matrix(data["true_intent"], data["predicted_intent"], labels=labels).tolist(), "labels": labels}


def escalation_metrics(data: pd.DataFrame) -> dict:
    data = data.dropna(subset=["ground_truth_escalate", "decision"])
    if data.empty:
        return {"precision": None, "recall": None, "n": 0}
    truth = data["ground_truth_escalate"].astype(str).str.lower().isin(["true", "yes", "escalate", "escalate-to-human"])
    pred = data["decision"].eq("escalate-to-human")
    precision, recall, _, _ = precision_recall_fscore_support(truth, pred, average="binary", zero_division=0)
    return {"precision": float(precision), "recall": float(recall), "n": len(data)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/evaluation.csv")
    parser.add_argument("--output", default="data/metrics.json")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    data = pd.read_csv(args.input).head(args.limit)
    result = {"classifier": classifier_metrics(data) if {"true_intent", "predicted_intent"} <= set(data) else {}, "escalation": escalation_metrics(data)}
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
