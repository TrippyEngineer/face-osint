"""crowd/notifier.py — pluggable outbound alert notifications.

Only WebhookNotifier ships now; the Notifier interface lets Telegram/email/SMS
drop in later without touching platform.py. All channels are config-gated and
absent config → no notifier is built (silent no-op)."""
import json
import logging
import time

import requests

import config

logger = logging.getLogger(__name__)

_SEVERITY_RANK = {"warning": 1, "caution": 1, "high": 2, "critical": 3}


def should_notify(severity: str, min_severity: str) -> bool:
    return _SEVERITY_RANK.get(severity, 0) >= _SEVERITY_RANK.get(min_severity, 99)


def build_payload(alert: dict, venue: dict, zone: dict) -> dict:
    return {"alert": alert, "venue": venue or {}, "zone": zone or {}, "ts": time.time()}


class Notifier:
    def send(self, alert: dict, context: dict) -> bool:
        raise NotImplementedError


class WebhookNotifier(Notifier):
    def __init__(self, url: str, headers: dict, timeout: int):
        self.url = url
        self.headers = headers or {}
        self.timeout = timeout

    def send(self, alert: dict, context: dict) -> bool:
        payload = build_payload(alert, context.get("venue", {}), context.get("zone", {}))
        try:
            r = requests.post(self.url, json=payload, headers=self.headers,
                              timeout=self.timeout)
            if not r.ok:
                logger.warning(f"webhook HTTP {r.status_code}")
                return False
            return True
        except Exception as e:
            logger.warning(f"webhook send failed: {e}")
            return False


_DISCORD_COLORS = {"warning": 0xF1C40F, "caution": 0xF1C40F,
                   "high": 0xE67E22, "critical": 0xE74C3C}


def build_discord_payload(alert: dict, context: dict) -> dict:
    """Format an alert into Discord's webhook schema (content + rich embed)."""
    sev  = (alert.get("severity") or "warning").lower()
    zone = (context.get("zone") or {}).get("name") or alert.get("zone") or "?"
    return {
        "content": f"[{sev.upper()}] {zone} - {alert.get('count', '?')} persons",
        "embeds": [{
            "title":       f"CIC Alert - {sev.upper()}",
            "description": alert.get("message", ""),
            "color":       _DISCORD_COLORS.get(sev, 0xF1C40F),
            "fields": [
                {"name": "Zone",    "value": str(zone),                          "inline": True},
                {"name": "Count",   "value": str(alert.get("count", "?")),       "inline": True},
                {"name": "Density", "value": f"{alert.get('density', '?')} p/m2", "inline": True},
            ],
        }],
    }


class DiscordNotifier(Notifier):
    def __init__(self, url: str, timeout: int):
        self.url = url
        self.timeout = timeout

    def send(self, alert: dict, context: dict) -> bool:
        try:
            r = requests.post(self.url, json=build_discord_payload(alert, context),
                              timeout=self.timeout)
            if not r.ok:
                logger.warning(f"discord webhook HTTP {r.status_code}")
                return False
            return True
        except Exception as e:
            logger.warning(f"discord send failed: {e}")
            return False


def build_notifiers_from_config() -> list:
    notifiers = []
    url = getattr(config, "CIC_WEBHOOK_URL", "")
    if url:
        raw = getattr(config, "CIC_WEBHOOK_HEADERS", "")
        try:
            headers = json.loads(raw) if raw else {}
        except Exception:
            headers = {}
        notifiers.append(WebhookNotifier(
            url, headers, getattr(config, "CIC_WEBHOOK_TIMEOUT_S", 6)))
    discord_url = getattr(config, "CIC_DISCORD_WEBHOOK_URL", "")
    if discord_url:
        notifiers.append(DiscordNotifier(
            discord_url, getattr(config, "CIC_WEBHOOK_TIMEOUT_S", 6)))
    return notifiers
