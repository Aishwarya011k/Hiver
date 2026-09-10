# Support Agent Report

## 1. Problem framing — what "good" means for this brand, what I chose NOT to build.

For AmazonHelp, good means a response is relevant to the customer's intent, grounded in a historically used resolution, transparent about uncertainty, and conservative when billing, privacy, fraud, legal, or other high-risk issues appear. The agent is designed to save support time on repeatable low-risk issues, not to replace human judgment.

I chose not to build live Twitter ingestion, account/order lookup, transactional refunds, authentication, sentiment as a separate product feature, or autonomous action-taking. The system only drafts a response from the supplied historical dataset and emits an auditable policy decision.

## 2. Results vs. both baselines (table).

| System | Accuracy | Per-intent F1 | Notes |
|---|---:|---:|---|
| Majority intent | TODO-ME | TODO-ME | Local baseline |
| TF-IDF + Logistic Regression | TODO-ME | TODO-ME | Local baseline |
| Batched gpt-4o-mini classifier | TODO-ME | TODO-ME | Requires hand-labeled golden set |

Reply mean scores and escalation precision/recall: **TODO-ME after hand-labeling and judge run**.

## 3. Failure analysis — top 5 failure modes with REAL examples from my eval run + hypotheses.

1. **TODO-ME: ambiguous delivery vs. missing-item language.** Example: `TODO-ME`. Hypothesis: overlapping intent definitions and insufficient edge examples.
2. **TODO-ME: billing/refund risk boundary.** Example: `TODO-ME`. Hypothesis: policy keywords are conservative and intent confidence is miscalibrated.
3. **TODO-ME: weak retrieval evidence.** Example: `TODO-ME`. Hypothesis: short tweets lose useful context after cleaning and cosine similarity over-indexes shared words.
4. **TODO-ME: historical reply does not fit current wording.** Example: `TODO-ME`. Hypothesis: the dataset contains templated resolutions and incomplete thread context.
5. **TODO-ME: judge disagreement.** Example: `TODO-ME`. Hypothesis: rubric interpretation and judge blind spots differ from human support quality standards.

## 4. "What is misleading about my headline number?"

The headline classifier number is not a production accuracy claim. The working sample is deliberately only about 700 threads, the golden set is 150 examples, and the judge-agreement subset is 30 because the API key is rate-limited. The sample can be biased toward AmazonHelp threads with explicit brand mentions and linked replies. The golden labels are human-entered and may have taxonomy ambiguity. The LLM judge can reward fluent but incorrect replies, and confidence scores have not been calibrated. A warm-cache run also measures a fixed prompt distribution rather than API variability. These limitations should accompany any headline result.

## 5. What I'd do next with one more week (mention: larger sample if rate limits allow).

I would label a larger and more deliberately balanced sample if rate limits allow, add multiple human labelers and adjudication, calibrate confidence on a held-out set, improve thread reconstruction, test retrieval ablations, and measure escalation cost asymmetrically. I would also add red-team cases for account security, privacy, legal threats, and requests that historical replies cannot safely answer.
