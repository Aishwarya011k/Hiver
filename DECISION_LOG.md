# Decision Log

1. **AmazonHelp as the initial brand.** The CLI keeps `--brand` configurable, but a concrete default makes the assignment reproducible.
2. **Linked tweet IDs define a usable thread.** We require an inbound customer tweet and a non-inbound response linked by `response_tweet_id` or the reverse relation instead of guessing from nearby timestamps.
3. **Local embeddings instead of OpenAI embeddings.** `all-MiniLM-L6-v2` removes the largest avoidable API cost and makes retrieval reproducible offline.
4. **KMeans is bottom-up taxonomy discovery.** Clustering precedes naming, so intent categories come from this brand's language rather than a generic support ontology.
5. **One naming call for all clusters.** Cluster examples are sent in one structured prompt; naming does not become one API call per cluster.
6. **700-thread working sample.** A 500-800 sample is enough for a take-home while limiting embedding and labeling time.
7. **150-example golden set and 30-example agreement subset.** These are the low ends requested because the API key is rate-limited; this is a deliberate measurement tradeoff.
8. **Batched classifier requests.** Fifteen messages per call reduces request count while retaining per-message confidence and reasons.
9. **Batched reply and judge requests.** Four replies or five judgments per call balances prompt size, parseability, and call count.
10. **Disk cache keyed by full prompt parameters.** Debugging and warm reruns never repeat an identical paid request.
11. **Two-second global interval plus exponential jittered retry.** The wrapper is conservative before a 429 and resilient after one.
12. **No LLM call in escalation policy.** Risk tiers, confidence thresholds, and explainable rules are more auditable than asking a model to decide safety.
13. **Grounding confidence is an explicit signal.** A polished response is not enough if retrieval evidence is weak; weak evidence routes to a human.
14. **Dry-run placeholders.** Every CLI can exercise file paths and output schemas without an API key, preserving budget during debugging.
