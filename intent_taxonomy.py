"""Bottom-up intent discovery with local embeddings, KMeans, and one naming call."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from utils.embeddings import embed
from utils.llm_client import call_json, print_stats


def choose_k(vectors: np.ndarray, low: int = 6, high: int = 10) -> tuple[int, dict[int, float]]:
    scores = {}
    upper = min(high, len(vectors) - 1)
    for k in range(low, upper + 1):
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(vectors)
        scores[k] = float(silhouette_score(vectors, labels))
    return max(scores, key=scores.get), scores


def discover(input_csv: str, output_json: str, limit: int | None = None, dry_run: bool = False) -> None:
    data = pd.read_csv(input_csv).dropna(subset=["customer_message"])
    if limit:
        data = data.head(limit)
    if len(data) < 8:
        raise ValueError("Need at least 8 prepared pairs for clustering.")
    vectors = embed(data["customer_message"].tolist())
    k, scores = choose_k(vectors, low=min(6, len(data) - 2), high=min(10, len(data) - 1))
    model = KMeans(n_clusters=k, random_state=42, n_init=10).fit(vectors)
    examples = []
    for cluster_id in range(k):
        member_idx = np.where(model.labels_ == cluster_id)[0]
        distances = np.linalg.norm(vectors[member_idx] - model.cluster_centers_[cluster_id], axis=1)
        nearest = member_idx[np.argsort(distances)[:5]]
        examples.append({"cluster_id": cluster_id, "examples": data.iloc[nearest]["customer_message"].tolist()})
    if dry_run:
        names = [{"cluster_id": x["cluster_id"], "name": f"cluster_{x['cluster_id']}", "definition": "Manual review needed."} for x in examples]
    else:
        prompt = ("Name all customer-support intent clusters below. Return JSON object with key intents, "
                  "an array of {cluster_id, name, definition}; use 6-10 concise, distinct names.\n" + json.dumps(examples))
        names = call_json(prompt, max_tokens=1400).get("intents", [])
    by_id = {int(item["cluster_id"]): item for item in names}
    intents = []
    for cluster_id in range(k):
        item = by_id.get(cluster_id, {"name": f"cluster_{cluster_id}", "definition": "Manual review needed."})
        intents.append({"id": cluster_id, "name": item["name"], "definition": item.get("definition", ""), "examples": examples[cluster_id]["examples"]})
    Path(output_json).write_text(json.dumps({"k": k, "silhouette_scores": scores, "intents": intents}, indent=2), encoding="utf-8")
    print(json.dumps({"k": k, "silhouette_scores": scores, "output": output_json}, indent=2))
    print_stats("Taxonomy")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/pairs.csv")
    parser.add_argument("--output", default="data/intents.json")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    discover(args.input, args.output, args.limit, args.dry_run)


if __name__ == "__main__":
    main()
