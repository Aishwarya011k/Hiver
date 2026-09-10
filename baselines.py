"""Local majority and TF-IDF/logistic-regression baselines."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def evaluate(golden_csv: str, output_json: str, limit: int | None = None) -> dict:
    data = pd.read_csv(golden_csv)
    data = data.dropna(subset=["true_intent"])
    if limit:
        data = data.head(limit)
    if data.empty:
        raise ValueError("Golden set has no hand-labeled true_intent values yet.")
    majority = data["true_intent"].mode().iloc[0]
    majority_accuracy = float((data["true_intent"] == majority).mean())
    model = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)), ("lr", LogisticRegression(max_iter=1000))])
    model.fit(data["message"], data["true_intent"])
    logistic_accuracy = float((model.predict(data["message"]) == data["true_intent"]).mean())
    result = {"n": len(data), "majority_intent": majority, "majority_accuracy": majority_accuracy, "tfidf_logreg_accuracy": logistic_accuracy}
    Path(output_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/golden_set.csv")
    parser.add_argument("--output", default="data/baselines.json")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    evaluate(args.input, args.output, args.limit)


if __name__ == "__main__":
    main()
