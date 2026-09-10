"""Centralized, cached, rate-limited OpenAI access."""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential

load_dotenv()

CACHE_DIR = Path(os.getenv("LLM_CACHE_DIR", ".cache/llm"))
MODEL = "gpt-4o-mini"
_CALLS = 0
_CACHE_HITS = 0
_INPUT_TOKENS = 0
_OUTPUT_TOKENS = 0
_LAST_CALL = 0.0
_LOCK = threading.Lock()


def _cache_key(prompt: str, model: str, temperature: float, max_tokens: int) -> str:
    payload = json.dumps(
        {"prompt": prompt, "model": model, "temperature": temperature, "max_tokens": max_tokens},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_path(key: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{key}.json"


def _wait_for_slot() -> None:
    global _LAST_CALL
    interval = float(os.getenv("OPENAI_CALL_INTERVAL_SECONDS", "2.0"))
    with _LOCK:
        delay = interval - (time.monotonic() - _LAST_CALL)
        if delay > 0:
            time.sleep(delay)
        _LAST_CALL = time.monotonic()


@retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_random_exponential(multiplier=1, max=30),
    stop=stop_after_attempt(5),
    reraise=True,
)
def _request(client: OpenAI, prompt: str, model: str, temperature: float, max_tokens: int):
    _wait_for_slot()
    return client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Return only valid JSON. Do not use markdown fences."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )


def call_json(
    prompt: str,
    *,
    model: str = MODEL,
    temperature: float = 0.0,
    max_tokens: int = 1200,
) -> dict[str, Any]:
    """Make one cached JSON call. The process-level stats are intentionally visible."""
    global _CALLS, _CACHE_HITS, _INPUT_TOKENS, _OUTPUT_TOKENS
    key = _cache_key(prompt, model, temperature, max_tokens)
    path = _cache_path(key)
    if path.exists():
        _CACHE_HITS += 1
        return json.loads(path.read_text(encoding="utf-8"))

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing. Put it in a .env file.")
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = _request(client, prompt, model, temperature, max_tokens)
    _CALLS += 1
    if response.usage:
        _INPUT_TOKENS += response.usage.prompt_tokens or 0
        _OUTPUT_TOKENS += response.usage.completion_tokens or 0
    content = response.choices[0].message.content or "{}"
    try:
        result = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"OpenAI returned invalid JSON: {content[:300]}") from exc
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def stats() -> dict[str, Any]:
    """Return conservative approximate spend; exact billing depends on account pricing."""
    input_cost = _INPUT_TOKENS / 1_000_000 * 0.15
    output_cost = _OUTPUT_TOKENS / 1_000_000 * 0.60
    total = _CALLS + _CACHE_HITS
    return {
        "api_calls": _CALLS,
        "cache_hits": _CACHE_HITS,
        "cache_hit_rate": round(_CACHE_HITS / total, 3) if total else 0.0,
        "input_tokens": _INPUT_TOKENS,
        "output_tokens": _OUTPUT_TOKENS,
        "estimated_cost_usd": round(input_cost + output_cost, 4),
    }


def print_stats(prefix: str = "LLM") -> None:
    print(f"{prefix} stats: {json.dumps(stats(), sort_keys=True)}")
