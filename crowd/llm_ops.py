"""
crowd/llm_ops.py
─────────────────
LLM operator assistant for the CIC dashboard.

Takes the current platform state (zone densities, risk levels, recent alerts)
and streams a situational-awareness response + SOP guidance via Claude API.

The system is constrained to advisory-only mode — all decisions remain
with human operators. Responses cite only the provided crowd data.

Usage:
    for chunk in ask("Which zones need attention?", state, api_key):
        print(chunk, end="", flush=True)
"""

import json
import logging
from typing import Iterator

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an ICCC (Integrated Command and Control Centre) operator \
assistant supporting crowd management at a large public gathering.

You have access to real-time crowd sensor data provided with each query.
Your role is strictly ADVISORY — all deployment decisions remain with human operators.

Guidelines:
- Answer concisely and precisely based ONLY on the data provided
- Lead with the highest-risk zones
- Suggest specific SOP actions for elevated risk levels:
  * CAUTION: Deploy additional stewards to zone entrance
  * HIGH RISK: Close one entry gate, open alternate exit, alert sector commander
  * CRITICAL: Activate SOP-3 (crowd pressure protocol): halt new entries,
    open all exits, broadcast PA announcement, alert medical team
- For lost-person queries, refer to Khoya-Paya desk at command tent
- If data shows no elevated risk, confirm current safe status
- Keep response under 200 words"""


def _build_context(state: dict) -> str:
    zones = state.get("zones", {})
    total = state.get("total_count", 0)
    alerts = state.get("alerts", [])

    lines = [f"Total persons detected across all cameras: {total}"]
    lines.append("\nZone Status:")
    for zid, z in zones.items():
        lines.append(
            f"  {z['name']}: {z['count']} persons, "
            f"density={z['density']:.3f} p/m², risk={z['risk'].upper()}"
        )

    if alerts:
        lines.append("\nRecent Alerts (last 5):")
        for a in alerts[:5]:
            lines.append(f"  [{a['severity'].upper()}] {a['zone']}: {a['message']}")
    else:
        lines.append("\nNo active alerts.")

    return "\n".join(lines)


def ask(question: str, state: dict, api_key: str) -> Iterator[str]:
    """
    Stream operator response as text chunks.
    Yields empty string on error with [ERROR] prefix.
    """
    if not api_key:
        yield "[LLM not configured — add ANTHROPIC_API_KEY to your .env file]"
        return

    try:
        import anthropic
    except ImportError:
        yield "[anthropic SDK not installed — run: pip install anthropic]"
        return

    context = _build_context(state)
    prompt  = f"Current crowd data:\n{context}\n\nOperator question: {question}"

    try:
        client = anthropic.Anthropic(api_key=api_key)
        with client.messages.stream(
            model="claude-sonnet-4-6",
            max_tokens=400,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            for text in stream.text_stream:
                yield text
    except Exception as e:
        logger.warning(f"LLM ask error: {e}")
        yield f"[Error contacting Claude API: {e}]"
