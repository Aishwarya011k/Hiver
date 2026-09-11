"""Centralized, cached, rate-limited access to Claude, OpenAI, or Ollama."""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential

load_dotenv()

CACHE_DIR = Path(os.getenv("LLM_CACHE_DIR", ".cache/llm"))
MODEL = "gpt-4o-mini"
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-20241022")
PROVIDER = os.getenv("LLM_PROVIDER", "claude").lower()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
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


def _request_ollama(prompt: str, model: str, temperature: float, max_tokens: int) -> str:
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": "Return only valid JSON. Do not use markdown fences."},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "format": "json",
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }).encode("utf-8")
    request = urllib.request.Request(
        f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not reach Ollama at {OLLAMA_BASE_URL}. Start Ollama and pull {model}."
        ) from exc
    content = body.get("message", {}).get("content", "")
    if not content:
        raise ValueError(f"Ollama returned no message content: {body}")
    return content


def _request_claude(prompt: str, model: str, temperature: float, max_tokens: int) -> str:
    api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is missing. Put it in a .env file.")

    try:
        from anthropic import Anthropic
    except ImportError as exc:
        raise RuntimeError("anthropic package is required when LLM_PROVIDER=claude. Install it with pip install anthropic.") from exc

    try:
        client = Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system="Return only valid JSON. Do not use markdown fences.",
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as exc:
        message = str(exc)
        lowered = message.lower()
        if "credit balance" in lowered or "billing" in lowered or "insufficient credits" in lowered:
            raise RuntimeError(
                "Anthropic rejected the request because the credit balance is too low to access the API. "
                "Please upgrade, purchase credits, or switch LLM_PROVIDER to openai or ollama."
            ) from exc
        raise

    text_parts = []
    for block in getattr(response, "content", []) or []:
        if getattr(block, "type", None) == "text":
            text_parts.append(block.text)
    content = "".join(text_parts)
    if not content:
        raise ValueError(f"Claude returned no message content: {response}")
    return content


def call_json(
    prompt: str,
    *,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 1200,
) -> dict[str, Any]:
    """Make one cached JSON call. The process-level stats are intentionally visible."""
    global _CALLS, _CACHE_HITS, _INPUT_TOKENS, _OUTPUT_TOKENS
    active_provider = PROVIDER
    if active_provider in {"claude", "anthropic"}:
        active_model = model or CLAUDE_MODEL
    elif active_provider == "ollama":
        active_model = model or OLLAMA_MODEL
    else:
        active_model = model or MODEL

    key = _cache_key(f"{active_provider}:{prompt}", active_model, temperature, max_tokens)
    path = _cache_path(key)
    if path.exists():
        _CACHE_HITS += 1
        return json.loads(path.read_text(encoding="utf-8"))

    if active_provider in {"claude", "anthropic"}:
        content = _request_claude(prompt, active_model, temperature, max_tokens)
    elif active_provider == "ollama":
        content = _request_ollama(prompt, active_model, temperature, max_tokens)
    elif active_provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is missing. Put it in a .env file.")
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        response = _request(client, prompt, active_model, temperature, max_tokens)
        if response.usage:
            _INPUT_TOKENS += response.usage.prompt_tokens or 0
            _OUTPUT_TOKENS += response.usage.completion_tokens or 0
        content = response.choices[0].message.content or "{}"
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {active_provider}. Use claude, openai, or ollama.")
    _CALLS += 1
    try:
        result = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{active_provider} returned invalid JSON: {content[:300]}") from exc
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
