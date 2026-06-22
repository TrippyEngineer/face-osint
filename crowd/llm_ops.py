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
    zones  = state.get("zones", {})
    alerts = state.get("alerts", [])
    # Describe ONLY zones with a live camera feed — not the static zones.json
    # config. Otherwise the assistant always talks about all 4 zones (3 sitting
    # at 0/SAFE) no matter what's actually running, giving generic, identical answers.
    active     = {s.get("slot") for s in state.get("slots", []) if s.get("active")}
    live_zones = {zid: z for zid, z in zones.items() if z.get("slot") in active}

    if not live_zones:
        return ("No camera feeds are currently active, so there is no live crowd "
                "data to analyze. Advise the operator to start a camera in the "
                "Cameras tab before requesting situational guidance.")

    total = sum(z.get("count", 0) for z in live_zones.values())
    lines = [
        f"Active camera feeds: {len(live_zones)}",
        f"Total persons detected (live feeds): {total}",
        "",
        "Zone Status:",
    ]
    for zid, z in live_zones.items():
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

    try:
        context = _build_context(state)
        prompt  = f"Current crowd data:\n{context}\n\nOperator question: {question}"
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


def ask_chat(history: list, state: dict, api_key: str) -> Iterator[str]:
    """
    Multi-turn variant: stream a reply given the full conversation history
    (list of {role, content}) plus the current crowd state. Live crowd data is
    injected into the system prompt so every turn sees the latest numbers.
    """
    if not api_key:
        yield "[LLM not configured — add ANTHROPIC_API_KEY to your .env file]"
        return
    try:
        import anthropic
    except ImportError:
        yield "[anthropic SDK not installed — run: pip install anthropic]"
        return

    messages = [{"role": m["role"], "content": m["content"]}
                for m in history
                if m.get("role") in ("user", "assistant") and (m.get("content") or "").strip()]
    if not messages:
        return

    try:
        system = SYSTEM_PROMPT + "\n\n--- LIVE CROWD DATA (current) ---\n" + _build_context(state)
        client = anthropic.Anthropic(api_key=api_key)
        with client.messages.stream(
            model="claude-sonnet-4-6",
            max_tokens=600,
            system=system,
            messages=messages,
        ) as stream:
            for text in stream.text_stream:
                yield text
    except Exception as e:
        logger.warning(f"LLM ask_chat error: {e}")
        yield f"[Error contacting Claude API: {e}]"
