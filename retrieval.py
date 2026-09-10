"""Local normalized-vector retrieval over historical support resolutions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from utils.embeddings import embed


class ResolutionRetriever:
    def __init__(self, pairs: pd.DataFrame):
        self.pairs = pairs.reset_index(drop=True)
        self.vectors = embed(self.pairs["customer_message"].tolist())

    def search(self, message: str, top_k: int = 3) -> list[dict]:
        query = embed([message])[0]
        scores = self.vectors @ query
        indices = np.argsort(scores)[::-1][:top_k]
        return [{"customer_message": self.pairs.iloc[i]["customer_message"],
                 "brand_resolution": self.pairs.iloc[i]["brand_resolution"],
                 "similarity": round(float(scores[i]), 4)} for i in indices]


def build_index(input_csv: str, output_dir: str, limit: int | None = None) -> None:
    pairs = pd.read_csv(input_csv).head(limit)
    vectors = embed(pairs["customer_message"].tolist())
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / "vectors.npy", vectors)
    pairs.to_json(path / "pairs.json", orient="records", indent=2)
    print(json.dumps({"rows": len(pairs), "output": str(path)}))


def load_index(output_dir: str) -> ResolutionRetriever:
    path = Path(output_dir)
    pairs = pd.read_json(path / "pairs.json")
    retriever = ResolutionRetriever.__new__(ResolutionRetriever)
    retriever.pairs = pairs
    retriever.vectors = np.load(path / "vectors.npy")
    return retriever


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/pairs.csv")
    parser.add_argument("--output-dir", default="data/retrieval_index")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    build_index(args.input, args.output_dir, args.limit)


if __name__ == "__main__":
    main()
