"""A real agent adapter for the dedrift demo: one call to the Anthropic API.

Used by the real-world walkthrough:

    export ANTHROPIC_API_KEY=...
    export DEDRIFT_DEMO_MODEL=claude-haiku-4-5
    dedrift canary run --suite examples/canaries_real.yaml \
        --agent examples.anthropic_agent:agent_fn \
        --model "anthropic/$DEDRIFT_DEMO_MODEL"

The model comes from the DEDRIFT_DEMO_MODEL environment variable so the
"provider swapped the model under you" scenario is a one-line change —
and the --model flag stamps the same identity into dedrift's config
fingerprint, which drives attribution.

Requires: pip install anthropic
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

SYSTEM_PROMPT = (
    "You are a concise assistant inside an automated pipeline. "
    "Answer in at most three sentences. If the user asks for JSON, reply with "
    "ONLY a valid JSON object and nothing else. If a request is unsafe or "
    "impossible, briefly say you cannot help."
)


def _client() -> Any:
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pip install anthropic to run this demo") from exc
    return anthropic.Anthropic()


_CLIENT = None


def agent_fn(input: dict[str, Any]) -> dict[str, Any]:
    """Dedrift agent adapter: text in, behavior out (real API call)."""
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = _client()
    model = os.environ.get("DEDRIFT_DEMO_MODEL", "claude-haiku-4-5")
    started = time.perf_counter()
    response = _CLIENT.messages.create(
        model=model,
        max_tokens=400,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": input["text"]}],
        temperature=1.0,
    )
    latency_ms = int((time.perf_counter() - started) * 1000)
    text = "".join(block.text for block in response.content if block.type == "text")

    structured = None
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            structured = json.loads(stripped)
        except ValueError:
            structured = None

    return {
        "text": text,
        "structured": structured,
        "latency_ms": latency_ms,
        "tokens_in": response.usage.input_tokens,
        "tokens_out": response.usage.output_tokens,
        "steps": 1,
    }
