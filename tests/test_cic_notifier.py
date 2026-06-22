from unittest.mock import patch, MagicMock

import crowd.notifier as N


def test_should_notify_severity_gate():
    assert N.should_notify("critical", "high") is True
    assert N.should_notify("high", "high") is True
    assert N.should_notify("warning", "high") is False
    assert N.should_notify("warning", "warning") is True


def test_build_payload_shape():
    alert = {"id": "a1", "severity": "high", "zone_id": "z0", "zone": "Gate", "count": 42}
    p = N.build_payload(alert, venue={"name": "Stadium"}, zone={"id": "z0", "name": "Gate"})
    assert p["alert"]["id"] == "a1"
    assert p["venue"]["name"] == "Stadium"
    assert p["zone"]["id"] == "z0"
    assert "ts" in p


def test_webhook_send_posts_json():
    with patch("crowd.notifier.requests.post") as post:
        post.return_value = MagicMock(ok=True, status_code=200)
        wh = N.WebhookNotifier(url="https://example.test/hook", headers={}, timeout=6)
        ok = wh.send({"severity": "high", "id": "a1"}, {"venue": {}, "zone": {}})
        assert ok is True
        assert post.call_args.kwargs["json"]["alert"]["id"] == "a1"


def test_factory_empty_when_no_url():
    with patch.object(N.config, "CIC_WEBHOOK_URL", ""), \
         patch.object(N.config, "CIC_DISCORD_WEBHOOK_URL", ""):
        assert N.build_notifiers_from_config() == []


def test_build_discord_payload_embed():
    alert = {"id": "a1", "severity": "critical", "zone": "Gate",
             "message": "CRITICAL crush risk", "count": 150, "density": 6.2}
    p = N.build_discord_payload(alert, {"zone": {"name": "Gate"}})
    assert p["embeds"] and isinstance(p["embeds"], list)
    emb = p["embeds"][0]
    assert "CRITICAL" in (emb["title"] + emb.get("description", "")).upper()
    assert isinstance(emb["color"], int)
    blob = str(p)
    assert "Gate" in blob and "150" in blob


def test_factory_includes_discord_when_url_set():
    with patch.object(N.config, "CIC_WEBHOOK_URL", ""), \
         patch.object(N.config, "CIC_DISCORD_WEBHOOK_URL", "https://discord.test/hook"):
        ns = N.build_notifiers_from_config()
        assert any(isinstance(n, N.DiscordNotifier) for n in ns)


def test_discord_send_posts_embed():
    with patch("crowd.notifier.requests.post") as post:
        post.return_value = MagicMock(ok=True, status_code=204)
        d = N.DiscordNotifier(url="https://discord.test/hook", timeout=6)
        ok = d.send({"severity": "high", "zone": "Gate", "message": "HIGH",
                     "count": 42, "density": 3.1}, {"zone": {"name": "Gate"}})
        assert ok is True
        assert "embeds" in post.call_args.kwargs["json"]
