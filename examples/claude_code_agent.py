"""A subscription-friendly agent adapter: Claude Code CLI in headless mode.

Uses `claude -p` (non-interactive print mode), which authenticates via your
Claude subscription login — no ANTHROPIC_API_KEY required. This is the
sanctioned way to drive Claude programmatically on a subscription.

    export DEDRIFT_DEMO_MODEL=haiku       # or sonnet / opus / full model id
    dedrift canary run --suite examples/canaries_real.yaml \
        --agent examples.claude_code_agent:agent_fn \
        --model "claude-code/$DEDRIFT_DEMO_MODEL"

Honest caveats vs the API adapter (examples/anthropic_agent.py):

- Latency includes CLI startup overhead (seconds), so treat the latency
  signature as CLI latency, not model latency.
- Sampling parameters (e.g. temperature) are not controllable.
- Calls draw on your subscription's usage limits; a full demo run is a few
  hundred short calls, so prefer a Max plan or shrink REPS/cycles.

Requires the Claude Code CLI installed and logged in (`claude login`).
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from typing import Any

SYSTEM_PROMPT = (
    "You are a concise assistant inside an automated pipeline. "
    "Answer in at most three sentences. If the user asks for JSON, reply with "
    "ONLY a valid JSON object and nothing else. If a request is unsafe or "
    "impossible, briefly say you cannot help."
)


def agent_fn(input: dict[str, Any]) -> dict[str, Any]:
    """Run one headless Claude Code call per canary (dedrift agent adapter)."""
    model = os.environ.get("DEDRIFT_DEMO_MODEL", "haiku")
    started = time.perf_counter()
    proc = subprocess.run(
        [
            "claude",
            "-p",
            input["text"],
            "--model",
            model,
            "--output-format",
            "json",
            "--append-system-prompt",
            SYSTEM_PROMPT,
            "--max-turns",
            "1",
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    latency_ms = int((time.perf_counter() - started) * 1000)

    if proc.returncode != 0:
        return {
            "text": "",
            "errors": [f"claude CLI exit {proc.returncode}: {proc.stderr[:300]}"],
            "latency_ms": latency_ms,
        }

    text = ""
    tokens_in = 0
    tokens_out = 0
    try:
        payload = json.loads(proc.stdout)
        text = str(payload.get("result", ""))
        usage = payload.get("usage") or {}
        tokens_in = int(usage.get("input_tokens", 0))
        tokens_out = int(usage.get("output_tokens", 0))
    except ValueError:
        text = proc.stdout.strip()

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
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "steps": 1,
    }
