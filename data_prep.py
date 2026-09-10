"""Create clean customer/support pairs from Kaggle's twcs.csv."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+")


def clean_text(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_pairs(input_csv: str, brand: str, sample_size: int = 700, seed: int = 42) -> tuple[pd.DataFrame, dict]:
    frame = pd.read_csv(input_csv)
    required = {"tweet_id", "author_id", "inbound", "text", "response_tweet_id", "in_response_to_tweet_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"twcs.csv is missing columns: {sorted(missing)}")
    frame["tweet_id"] = frame["tweet_id"].astype(str)
    frame["in_response_to_tweet_id"] = frame["in_response_to_tweet_id"].fillna("").astype(str)
    frame["response_tweet_id"] = frame["response_tweet_id"].fillna("").astype(str)
    inbound = frame["inbound"].astype(str).str.lower().isin(["true", "1"])
    brand_mask = frame["text"].fillna("").str.contains("@" + brand.lstrip("@"), case=False, regex=False)
    replies = frame.loc[~inbound].set_index("tweet_id")
    rows = []
    for _, customer in frame.loc[inbound & brand_mask].iterrows():
        reply_id = customer["response_tweet_id"]
        reply = replies.loc[reply_id] if reply_id in replies.index else None
        if reply is None:
            reverse = frame.loc[frame["in_response_to_tweet_id"] == customer["tweet_id"]]
            reverse = reverse.loc[~inbound]
            reply = reverse.iloc[0] if not reverse.empty else None
        if reply is None:
            continue
        message, resolution = clean_text(customer["text"]), clean_text(reply["text"])
        if len(message) < 8 or len(resolution) < 8:
            continue
        rows.append({"tweet_id": customer["tweet_id"], "customer_message": message, "brand_resolution": resolution})
    pairs = pd.DataFrame(rows).drop_duplicates(subset=["customer_message", "brand_resolution"])
    if len(pairs) > sample_size:
        pairs = pairs.sample(n=sample_size, random_state=seed)
    pairs = pairs.reset_index(drop=True)
    log = {
        "input_rows": len(frame), "matched_pairs": len(rows), "output_pairs": len(pairs),
        "brand": brand.lstrip("@"), "random_seed": seed, "sample_size": sample_size,
        "filters": ["inbound customer rows", "message contains chosen @brand", "linked non-inbound reply",
                    "strip URLs and mentions", "drop messages/resolutions shorter than 8 chars", "deduplicate"],
    }
    return pairs, log


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/archive/twcs/twcs.csv")
    parser.add_argument("--brand", default="AmazonHelp")
    parser.add_argument("--output", default="data/pairs.csv")
    parser.add_argument("--limit", type=int, default=None, help="Debug cap after preparation")
    parser.add_argument("--sample-size", type=int, default=700)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="Prepare at most --limit rows")
    args = parser.parse_args()
    size = min(args.sample_size, args.limit) if args.dry_run and args.limit else args.sample_size
    pairs, log = build_pairs(args.input, args.brand, size, args.seed)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(args.output, index=False)
    Path(args.output).with_suffix(".log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
