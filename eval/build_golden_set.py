"""Create a deliberately small, stratified, hand-label-ready golden set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from utils.embeddings import embed


def build(input_csv: str, intents_json: str, output_csv: str, size: int = 150, seed: int = 42, limit: int | None = None) -> None:
    pairs = pd.read_csv(input_csv).drop_duplicates("customer_message")
    if limit:
        pairs = pairs.head(limit)
    vectors = embed(pairs["customer_message"].tolist())
    k = min(json.loads(Path(intents_json).read_text(encoding="utf-8"))["k"], len(pairs))
    labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(vectors)
    pairs = pairs.assign(_cluster=labels)
    per_cluster = max(1, size // k)
    selected = pairs.groupby("_cluster", group_keys=False).apply(lambda part: part.sample(min(per_cluster, len(part)), random_state=seed))
    selected = selected.sample(min(size, len(selected)), random_state=seed)
    golden = pd.DataFrame({"message": selected["customer_message"], "resolution": selected["brand_resolution"], "true_intent": "", "notes": "Hand-label; edge/ambiguous cases included where available."})
    golden.to_csv(output_csv, index=False)
    log = {"requested_size": size, "actual_size": len(golden), "seed": seed, "stratification": "KMeans cluster over local sentence-transformer message embeddings; approximately equal allocation per cluster; final seeded sample", "clusters": k, "manual_fields": ["true_intent", "notes"]}
    Path(output_csv).with_suffix(".log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(json.dumps(log, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/pairs.csv")
    parser.add_argument("--intents", default="data/intents.json")
    parser.add_argument("--output", default="data/golden_set.csv")
    parser.add_argument("--size", type=int, default=150)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    build(args.input, args.intents, args.output, min(args.size, args.limit) if args.dry_run and args.limit else args.size, limit=args.limit if args.dry_run else None)


if __name__ == "__main__":
    main()
