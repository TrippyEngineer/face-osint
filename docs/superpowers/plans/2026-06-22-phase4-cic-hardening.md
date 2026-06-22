# Phase 4 — CIC Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Crowd Intelligence Center deployment-grade: persist alerts + zone history to SQLite, push alerts to a webhook, record incident clips, and improve dense-crowd counting with YOLO tiling.

**Architecture:** Pure logic (DB CRUD, reading-cadence decision, webhook payload/gate, tile generation + NMS merge, clip encoding) lives in small testable units. The threaded glue in `crowd/platform.py` (1 Hz aggregator) and `crowd/analyzer.py` (per-slot YOLO loop) calls those units; thread/camera wiring is verified manually against `data/crowd.mp4`.

**Tech Stack:** Python 3.11 (Windows interpreter), SQLite (WAL), OpenCV, ultralytics YOLOv8, `requests`, pytest.

## Global Constraints

- **Python 3.10/3.11 only; runs on Windows Python 3.11**, not WSL. Run everything (incl. pytest) with `"/mnt/c/Program Files/Python311/python.exe"`.
- **`config.py` is the single source of truth** — every new threshold/path/toggle goes there, no magic numbers elsewhere.
- **Zero-key/free path must keep working** — every feature config-gated, **off by default** (except clips, default on), absent config → silent no-op, never an error.
- **SQLite is WAL on `data/`**, only openable from Windows python on `/mnt/d`.
- **No auto-reload** — restart `app.py` after edits to verify.
- No Docker / no Redis.
- Branch: `cic-fixes`. One commit per task.

**Test runner (use verbatim):**
```bash
PY="/mnt/c/Program Files/Python311/python.exe"
"$PY" -m pytest tests/ -v
```

## File Structure

- `config.py` (modify) — all new `CIC_*` constants.
- `storage/database.py` (modify) — `cic_alerts` + `cic_zone_readings` tables, CRUD, prune.
- `crowd/persistence.py` (create) — pure `should_persist_reading()` cadence decision.
- `crowd/notifier.py` (create) — `should_notify()`, `build_payload()`, `WebhookNotifier`, `build_notifiers_from_config()`.
- `crowd/clip.py` (create) — pure `incident_clip_path()` + `write_clip(frames, path, fps)`.
- `crowd/tiling.py` (create) — pure `generate_tiles()` + `merge_boxes()` (NMS).
- `crowd/analyzer.py` (modify) — raw-frame ring buffer, `record_incident()`, tiled detection.
- `crowd/platform.py` (modify) — persist hooks, reading cadence, notifier dispatch, clip trigger, startup prune.
- `app.py` (modify) — `/crowd/api/alerts/history`, `/crowd/api/zones/history`.
- `docs/cic-calibration.md` (create) — persons/m² calibration method.
- Tests: `tests/test_cic_database.py`, `tests/test_cic_persistence.py`, `tests/test_cic_notifier.py`, `tests/test_cic_clip.py`, `tests/test_cic_tiling.py`.

---

### Task 1: CIC persistence — DB tables + CRUD

**Files:**
- Modify: `config.py` (append CIC persistence config)
- Modify: `storage/database.py` (extend `_SCHEMA`; add methods)
- Test: `tests/test_cic_database.py`

**Interfaces:**
- Produces: `Database.insert_cic_alert(alert: dict) -> None`, `Database.update_cic_alert_clip(alert_uid: str, clip_path: str) -> None`, `Database.get_cic_alerts(limit: int = 100, since: str = "") -> list[dict]`, `Database.insert_cic_reading(reading: dict) -> None`, `Database.get_cic_readings(zone_id: str = "", since: str = "") -> list[dict]`, `Database.prune_cic_data(ttl_days: int) -> int`

- [ ] **Step 1: Add config constants**

In `config.py`, after the CIC detection block (around the `CIC_FACE_MIN_CROP_PX` line), add:

```python
# ── CIC Phase 4: persistence / retention ────────────────────────────────────
CIC_READING_PERSIST_S = 10     # min seconds between persisted zone-reading snapshots
CIC_DATA_TTL_DAYS     = 30     # prune cic_alerts / cic_zone_readings older than this
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_cic_database.py`:

```python
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from storage.database import Database


@pytest.fixture
def db(tmp_path):
    return Database(path=tmp_path / "test.db")


def _alert(uid="a1", zone="zone_0", sev="high"):
    return {"id": uid, "zone_id": zone, "zone": "Main Gate", "type": "density_high",
            "severity": sev, "message": "HIGH", "density": 3.1, "count": 42, "acked": False}


def _reading(zone="zone_0"):
    return {"zone_id": zone, "zone_name": "Main Gate", "count": 42,
            "density": 3.1, "risk": "high", "n_suspicious": 0}


def test_insert_and_get_alert(db):
    db.insert_cic_alert(_alert())
    rows = db.get_cic_alerts()
    assert len(rows) == 1
    assert rows[0]["alert_uid"] == "a1"
    assert rows[0]["severity"] == "high"
    assert rows[0]["clip_path"] is None


def test_update_alert_clip(db):
    db.insert_cic_alert(_alert("a2"))
    db.update_cic_alert_clip("a2", "/data/output/cic_incidents/x.mp4")
    rows = db.get_cic_alerts()
    assert rows[0]["clip_path"].endswith("x.mp4")


def test_insert_and_get_reading(db):
    db.insert_cic_reading(_reading())
    rows = db.get_cic_readings(zone_id="zone_0")
    assert len(rows) == 1
    assert rows[0]["count"] == 42


def test_prune_removes_old_rows(db):
    db.insert_cic_alert(_alert("new"))
    # backdate one alert 40 days
    old_ts = (datetime.now() - timedelta(days=40)).strftime("%Y-%m-%d %H:%M:%S")
    with db._connect() as conn:
        conn.execute(
            "INSERT INTO cic_alerts (alert_uid, zone_id, zone_name, type, severity, created_at)"
            " VALUES ('old','z','Z','density_high','high',?)", (old_ts,))
    removed = db.prune_cic_data(ttl_days=30)
    uids = [r["alert_uid"] for r in db.get_cic_alerts()]
    assert "old" not in uids and "new" in uids
    assert removed >= 1
```

- [ ] **Step 3: Run test to verify it fails**

```bash
"$PY" -m pytest tests/test_cic_database.py -v
```
Expected: FAIL — `AttributeError: 'Database' object has no attribute 'insert_cic_alert'`.

- [ ] **Step 4: Extend the schema**

In `storage/database.py`, inside the `_SCHEMA` string, after the `cic_face_captures` block, add:

```sql
-- CIC Phase 4: persisted alerts + zone readings (survive restart)
CREATE TABLE IF NOT EXISTS cic_alerts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_uid   TEXT,
    zone_id     TEXT NOT NULL,
    zone_name   TEXT NOT NULL,
    type        TEXT NOT NULL,
    severity    TEXT NOT NULL,
    message     TEXT,
    density     REAL,
    count       INTEGER,
    clip_path   TEXT,
    acked       INTEGER DEFAULT 0,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cic_alerts_zone ON cic_alerts(zone_id);
CREATE INDEX IF NOT EXISTS idx_cic_alerts_time ON cic_alerts(created_at);

CREATE TABLE IF NOT EXISTS cic_zone_readings (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    zone_id      TEXT NOT NULL,
    zone_name    TEXT NOT NULL,
    count        INTEGER,
    density      REAL,
    risk         TEXT,
    n_suspicious INTEGER,
    created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cic_readings_zone ON cic_zone_readings(zone_id, created_at);
```

- [ ] **Step 5: Add CRUD methods**

In `storage/database.py`, before the module-level `def _now()`, add these methods to the `Database` class:

```python
    # ── CIC Phase 4: alerts + readings ────────────────────────────────────
    def insert_cic_alert(self, alert: dict):
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO cic_alerts
                   (alert_uid, zone_id, zone_name, type, severity, message,
                    density, count, clip_path, acked, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (alert.get("id"), alert.get("zone_id"),
                 alert.get("zone") or alert.get("zone_name"),
                 alert.get("type"), alert.get("severity"), alert.get("message"),
                 alert.get("density"), alert.get("count"),
                 alert.get("clip_path"), 1 if alert.get("acked") else 0, _now()),
            )

    def update_cic_alert_clip(self, alert_uid: str, clip_path: str):
        with self._connect() as conn:
            conn.execute("UPDATE cic_alerts SET clip_path=? WHERE alert_uid=?",
                         (clip_path, alert_uid))

    def get_cic_alerts(self, limit: int = 100, since: str = "") -> list:
        q = "SELECT * FROM cic_alerts"
        args: tuple = ()
        if since:
            q += " WHERE created_at >= ?"; args = (since,)
        q += " ORDER BY created_at DESC LIMIT ?"; args = args + (limit,)
        with self._connect() as conn:
            return [dict(r) for r in conn.execute(q, args).fetchall()]

    def insert_cic_reading(self, reading: dict):
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO cic_zone_readings
                   (zone_id, zone_name, count, density, risk, n_suspicious, created_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (reading.get("zone_id"), reading.get("zone_name"),
                 reading.get("count"), reading.get("density"),
                 reading.get("risk"), reading.get("n_suspicious", 0), _now()),
            )

    def get_cic_readings(self, zone_id: str = "", since: str = "") -> list:
        q = "SELECT * FROM cic_zone_readings"
        clauses, args = [], []
        if zone_id:
            clauses.append("zone_id=?"); args.append(zone_id)
        if since:
            clauses.append("created_at >= ?"); args.append(since)
        if clauses:
            q += " WHERE " + " AND ".join(clauses)
        q += " ORDER BY created_at ASC"
        with self._connect() as conn:
            return [dict(r) for r in conn.execute(q, tuple(args)).fetchall()]

    def prune_cic_data(self, ttl_days: int) -> int:
        from datetime import datetime, timedelta
        cutoff = (datetime.now() - timedelta(days=ttl_days)).strftime("%Y-%m-%d %H:%M:%S")
        with self._connect() as conn:
            c1 = conn.execute("DELETE FROM cic_alerts WHERE created_at < ?", (cutoff,)).rowcount
            c2 = conn.execute("DELETE FROM cic_zone_readings WHERE created_at < ?", (cutoff,)).rowcount
        return (c1 or 0) + (c2 or 0)
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
"$PY" -m pytest tests/test_cic_database.py -v
```
Expected: 4 passed.

- [ ] **Step 7: Commit**

```bash
git add config.py storage/database.py tests/test_cic_database.py
git commit -m "feat(cic): persist alerts + zone readings to SQLite with TTL prune"
```

---

### Task 2: Reading-cadence decision (pure)

**Files:**
- Create: `crowd/persistence.py`
- Test: `tests/test_cic_persistence.py`

**Interfaces:**
- Produces: `should_persist_reading(now: float, last_ts: float, risk: str, prev_risk: str, interval_s: float) -> bool`

- [ ] **Step 1: Write the failing test**

Create `tests/test_cic_persistence.py`:

```python
from crowd.persistence import should_persist_reading


def test_persists_on_interval_boundary():
    assert should_persist_reading(now=100.0, last_ts=89.0, risk="safe", prev_risk="safe", interval_s=10) is True


def test_skips_within_interval():
    assert should_persist_reading(now=95.0, last_ts=89.0, risk="safe", prev_risk="safe", interval_s=10) is False


def test_persists_on_risk_transition_even_within_interval():
    assert should_persist_reading(now=95.0, last_ts=89.0, risk="high", prev_risk="safe", interval_s=10) is True


def test_first_reading_persists():
    assert should_persist_reading(now=1.0, last_ts=0.0, risk="safe", prev_risk="safe", interval_s=10) is False
    assert should_persist_reading(now=11.0, last_ts=0.0, risk="safe", prev_risk="safe", interval_s=10) is True
```

- [ ] **Step 2: Run test to verify it fails**

```bash
"$PY" -m pytest tests/test_cic_persistence.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'crowd.persistence'`.

- [ ] **Step 3: Implement**

Create `crowd/persistence.py`:

```python
"""crowd/persistence.py — pure helpers for CIC persistence cadence."""


def should_persist_reading(now: float, last_ts: float, risk: str,
                           prev_risk: str, interval_s: float) -> bool:
    """Persist a zone reading when the interval has elapsed OR the risk band
    changed since the last reading (so transitions are never lost to downsampling)."""
    if risk != prev_risk:
        return True
    return (now - last_ts) >= interval_s
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
"$PY" -m pytest tests/test_cic_persistence.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add crowd/persistence.py tests/test_cic_persistence.py
git commit -m "feat(cic): pure reading-cadence decision helper"
```

---

### Task 3: Wire persistence into platform + history routes

**Files:**
- Modify: `crowd/platform.py` (imports, `__init__`, `_run`, startup prune)
- Modify: `app.py` (two history routes)

**Interfaces:**
- Consumes: `Database.insert_cic_alert`, `Database.insert_cic_reading`, `Database.prune_cic_data`, `Database.get_cic_alerts`, `Database.get_cic_readings`, `should_persist_reading`

- [ ] **Step 1: Add imports + state in `platform.py`**

At the top of `crowd/platform.py`, after `from crowd.analyzer import CameraAnalyzer`, add:

```python
import config
from storage.database import Database
from crowd.persistence import should_persist_reading
```

In `Platform.__init__`, after `self._alert_ts = {}`, add:

```python
        self._db = Database()
        self._reading_last_ts: dict = {}   # zone_id → last persisted reading time
        try:
            removed = self._db.prune_cic_data(getattr(config, "CIC_DATA_TTL_DAYS", 30))
            if removed:
                logger.info(f"CIC startup prune: removed {removed} expired alert/reading rows")
        except Exception as e:
            logger.warning(f"CIC startup prune failed: {e}")
```

- [ ] **Step 2: Persist alerts + readings in `_run`**

In `crowd/platform.py`, inside `_run`, locate the alert block:

```python
                    alert = self._maybe_alert(zid, zs["name"], risk, prev_risk, meta)
                    if alert:
                        self._alerts.insert(0, alert)
                        self._alerts = self._alerts[:100]
                        new_alerts.append(alert)
```

Replace it with:

```python
                    alert = self._maybe_alert(zid, zs["name"], risk, prev_risk, meta)
                    if alert:
                        self._alerts.insert(0, alert)
                        self._alerts = self._alerts[:100]
                        new_alerts.append(alert)
                        try:
                            self._db.insert_cic_alert(alert)
                        except Exception as e:
                            logger.warning(f"persist alert failed: {e}")

                    # Downsampled zone-reading persistence (+ on risk transition)
                    now_ts = time.time()
                    if should_persist_reading(
                        now_ts, self._reading_last_ts.get(zid, 0.0),
                        risk, prev_risk,
                        getattr(config, "CIC_READING_PERSIST_S", 10),
                    ):
                        self._reading_last_ts[zid] = now_ts
                        try:
                            self._db.insert_cic_reading({
                                "zone_id": zid, "zone_name": zs["name"],
                                "count": meta["count"], "density": meta["density"],
                                "risk": risk, "n_suspicious": meta.get("n_suspicious", 0),
                            })
                        except Exception as e:
                            logger.warning(f"persist reading failed: {e}")
```

- [ ] **Step 3: Add accessor methods on `Platform`**

In `crowd/platform.py`, after `get_alerts`, add:

```python
    def get_alert_history(self, limit: int = 100, since: str = "") -> list:
        return self._db.get_cic_alerts(limit=limit, since=since)

    def get_zone_history(self, zone_id: str = "", since: str = "") -> list:
        return self._db.get_cic_readings(zone_id=zone_id, since=since)
```

- [ ] **Step 4: Add Flask routes in `app.py`**

In `app.py`, find an existing CIC route (search `@app.route("/crowd/api/`) and add nearby:

```python
@app.route("/crowd/api/alerts/history")
def crowd_alerts_history():
    from crowd.platform import get_platform
    limit = int(request.args.get("limit", 100))
    since = request.args.get("since", "")
    return jsonify(alerts=get_platform().get_alert_history(limit=limit, since=since))


@app.route("/crowd/api/zones/history")
def crowd_zones_history():
    from crowd.platform import get_platform
    zone_id = request.args.get("zone_id", "")
    since   = request.args.get("since", "")
    return jsonify(readings=get_platform().get_zone_history(zone_id=zone_id, since=since))
```

- [ ] **Step 5: Verify import + syntax**

```bash
"$PY" -m py_compile crowd/platform.py app.py && echo "COMPILE OK"
"$PY" -m pytest tests/ -v
```
Expected: COMPILE OK; all prior tests still pass.

- [ ] **Step 6: Manual verification**

Restart the server (`"$PY" -u app.py`), run `data/crowd.mp4` on a CIC slot until a `high`/`critical` alert fires. Then:
```bash
"$PY" -c "import urllib.request,json; print(json.load(urllib.request.urlopen('http://127.0.0.1:5000/crowd/api/alerts/history?limit=5')))"
```
Expected: at least one persisted alert. Restart the server and re-run the command — the alert is **still there** (survived restart).

- [ ] **Step 7: Commit**

```bash
git add crowd/platform.py app.py
git commit -m "feat(cic): wire alert/reading persistence + history routes"
```

---

### Task 4: Webhook notifier (pure logic + impl)

**Files:**
- Modify: `config.py` (webhook config)
- Create: `crowd/notifier.py`
- Test: `tests/test_cic_notifier.py`

**Interfaces:**
- Produces: `should_notify(severity: str, min_severity: str) -> bool`, `build_payload(alert: dict, venue: dict, zone: dict) -> dict`, `WebhookNotifier(url, headers, timeout).send(alert, context) -> bool`, `build_notifiers_from_config() -> list`

- [ ] **Step 1: Add config**

In `config.py`, after the CIC persistence block, add:

```python
# ── CIC Phase 4: outbound webhook notifications ─────────────────────────────
CIC_WEBHOOK_URL          = os.getenv("CIC_WEBHOOK_URL", "")        # empty → disabled
CIC_WEBHOOK_MIN_SEVERITY = os.getenv("CIC_WEBHOOK_MIN_SEVERITY", "high")  # warning|high|critical
CIC_WEBHOOK_HEADERS      = os.getenv("CIC_WEBHOOK_HEADERS", "")    # optional JSON string
CIC_WEBHOOK_TIMEOUT_S    = int(os.getenv("CIC_WEBHOOK_TIMEOUT_S", "6"))
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_cic_notifier.py`:

```python
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
    with patch.object(N.config, "CIC_WEBHOOK_URL", ""):
        assert N.build_notifiers_from_config() == []
```

- [ ] **Step 3: Run test to verify it fails**

```bash
"$PY" -m pytest tests/test_cic_notifier.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'crowd.notifier'`.

- [ ] **Step 4: Implement**

Create `crowd/notifier.py`:

```python
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
    return notifiers
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
"$PY" -m pytest tests/test_cic_notifier.py -v
```
Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add config.py crowd/notifier.py tests/test_cic_notifier.py
git commit -m "feat(cic): webhook notifier with severity gate + payload builder"
```

---

### Task 5: Wire notifier into platform

**Files:**
- Modify: `crowd/platform.py` (`__init__`, `_run` alert block)

**Interfaces:**
- Consumes: `build_notifiers_from_config`, `should_notify`

- [ ] **Step 1: Construct notifiers**

In `crowd/platform.py`, add to the imports added in Task 3:

```python
from crowd.notifier import build_notifiers_from_config, should_notify
```

In `Platform.__init__`, after the prune block, add:

```python
        self._notifiers = build_notifiers_from_config()
        if self._notifiers:
            logger.info(f"CIC notifiers active: {len(self._notifiers)}")
```

- [ ] **Step 2: Dispatch on qualifying alert (async)**

In `crowd/platform.py`, inside the `if alert:` block in `_run` (added in Task 3), after the `self._db.insert_cic_alert(alert)` try/except, add:

```python
                        if self._notifiers:
                            min_sev = getattr(config, "CIC_WEBHOOK_MIN_SEVERITY", "high")
                            if should_notify(alert.get("severity", ""), min_sev):
                                ctx = {"venue": self._zones_raw.get("venue", {}),
                                       "zone": {"id": zid, "name": zs["name"]}}
                                for n in self._notifiers:
                                    threading.Thread(target=n.send, args=(alert, ctx),
                                                     daemon=True,
                                                     name="CIC-notify").start()
```

- [ ] **Step 3: Verify syntax + tests**

```bash
"$PY" -m py_compile crowd/platform.py && echo "COMPILE OK"
"$PY" -m pytest tests/ -v
```
Expected: COMPILE OK; all tests pass.

- [ ] **Step 4: Manual verification**

Get a test URL from https://webhook.site. In `.env` set `CIC_WEBHOOK_URL=<that url>` and `CIC_WEBHOOK_MIN_SEVERITY=caution` (so it fires sooner). Restart, run `data/crowd.mp4`, drive a `caution`+ alert, confirm the JSON POST appears on webhook.site. Reset `.env` afterward.

- [ ] **Step 5: Commit**

```bash
git add crowd/platform.py
git commit -m "feat(cic): dispatch alerts to webhook notifiers asynchronously"
```

---

### Task 6: Incident clip recording

**Files:**
- Modify: `config.py` (clip config)
- Create: `crowd/clip.py`
- Modify: `crowd/analyzer.py` (ring buffer + `record_incident`)
- Modify: `crowd/platform.py` (trigger + link clip)
- Test: `tests/test_cic_clip.py`

**Interfaces:**
- Produces: `incident_clip_path(out_dir, zone_id, ts=None) -> Path`, `write_clip(frames: list, path, fps: int) -> bool`, `CameraAnalyzer.record_incident(on_done) -> bool`

- [ ] **Step 1: Add config**

In `config.py`, after the webhook block, add:

```python
# ── CIC Phase 4: incident clips ─────────────────────────────────────────────
CIC_CLIPS_ENABLED = os.getenv("CIC_CLIPS_ENABLED", "1") not in ("0", "false", "False")
CIC_CLIP_PRE_S    = int(os.getenv("CIC_CLIP_PRE_S", "10"))
CIC_CLIP_POST_S   = int(os.getenv("CIC_CLIP_POST_S", "10"))
CIC_INCIDENT_DIR  = OUTPUT_DIR / "cic_incidents"
CIC_INCIDENT_DIR.mkdir(parents=True, exist_ok=True)
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_cic_clip.py`:

```python
import numpy as np
import cv2

from crowd.clip import incident_clip_path, write_clip


def test_incident_clip_path_shape(tmp_path):
    p = incident_clip_path(tmp_path, "zone_0", ts=1750000000)
    assert p.parent == tmp_path
    assert p.name.startswith("zone_0_") and p.suffix == ".mp4"


def test_write_clip_creates_readable_mp4(tmp_path):
    frames = [np.full((120, 160, 3), i, np.uint8) for i in range(0, 30)]
    out = tmp_path / "clip.mp4"
    ok = write_clip(frames, out, fps=5)
    assert ok is True
    assert out.exists() and out.stat().st_size > 0
    cap = cv2.VideoCapture(str(out))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert n >= 1


def test_write_clip_empty_returns_false(tmp_path):
    assert write_clip([], tmp_path / "x.mp4", fps=5) is False
```

- [ ] **Step 3: Run test to verify it fails**

```bash
"$PY" -m pytest tests/test_cic_clip.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'crowd.clip'`.

- [ ] **Step 4: Implement `crowd/clip.py`**

```python
"""crowd/clip.py — pure incident-clip path + encoder."""
import logging
import time
from pathlib import Path

import cv2

logger = logging.getLogger(__name__)


def incident_clip_path(out_dir, zone_id: str, ts=None) -> Path:
    ts = ts or time.time()
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(ts))
    return Path(out_dir) / f"{zone_id}_{stamp}.mp4"


def write_clip(frames: list, path, fps: int) -> bool:
    if not frames:
        return False
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (w, h))
    if not writer.isOpened():
        logger.warning(f"VideoWriter failed to open for {path}")
        return False
    try:
        for f in frames:
            if f.shape[:2] != (h, w):
                f = cv2.resize(f, (w, h))
            writer.write(f)
    finally:
        writer.release()
    return True
```

- [ ] **Step 5: Run clip tests to verify they pass**

```bash
"$PY" -m pytest tests/test_cic_clip.py -v
```
Expected: 3 passed.

- [ ] **Step 6: Add ring buffer + `record_incident` to `analyzer.py`**

In `crowd/analyzer.py`, add `from collections import deque` to the imports. In `CameraAnalyzer.__init__`, after `self._last_frame = None`, add:

```python
        import config as _cfg
        self._clip_fps   = INFERENCE_FPS
        self._clip_buf   = deque(maxlen=getattr(_cfg, "CIC_CLIP_PRE_S", 10) * INFERENCE_FPS)
        self._recording  = False
```

In `_run`, immediately after the inference-downscale block (right after `self._frame_count += 1`), add:

```python
            # Buffer raw (pre-overlay) frames for incident clips
            try:
                self._clip_buf.append(frame.copy())
            except Exception:
                pass
```

Add this method to `CameraAnalyzer` (after `get_meta`):

```python
    def record_incident(self, on_done) -> bool:
        """Snapshot the pre-buffer, keep appending for CIC_CLIP_POST_S, encode an
        mp4 in a background thread, then call on_done(path:str). Returns False if a
        recording is already in flight or clips disabled."""
        import config as _cfg
        if not getattr(_cfg, "CIC_CLIPS_ENABLED", True):
            return False
        with self._lock:
            if self._recording:
                return False
            self._recording = True
            pre = list(self._clip_buf)

        def _worker():
            from crowd.clip import incident_clip_path, write_clip
            post_n = getattr(_cfg, "CIC_CLIP_POST_S", 10) * self._clip_fps
            collected = []
            for _ in range(post_n):
                with self._lock:
                    if not self._active:
                        break
                    lf = self._last_frame
                if lf is not None:
                    collected.append(lf.copy())
                time.sleep(1.0 / self._clip_fps)
            frames = pre + collected
            path = incident_clip_path(getattr(_cfg, "CIC_INCIDENT_DIR"),
                                      self.zone_cfg.get("id", f"zone_{self.slot_id}"))
            ok = write_clip(frames, path, self._clip_fps)
            with self._lock:
                self._recording = False
            if ok:
                try:
                    on_done(str(path))
                except Exception:
                    pass

        threading.Thread(target=_worker, daemon=True,
                         name=f"CIC-clip{self.slot_id}").start()
        return True
```

- [ ] **Step 7: Trigger clip from `platform.py`**

In `crowd/platform.py`, inside the `if alert:` block in `_run`, after the notifier dispatch, add:

```python
                        if (getattr(config, "CIC_CLIPS_ENABLED", True)
                                and alert.get("severity") in ("high", "critical")):
                            a = self._analyzers.get(meta.get("slot"))
                            if a is not None:
                                uid = alert.get("id")
                                a.record_incident(
                                    lambda p, _uid=uid: self._db.update_cic_alert_clip(_uid, p))
```

- [ ] **Step 8: Verify syntax + full suite**

```bash
"$PY" -m py_compile crowd/clip.py crowd/analyzer.py crowd/platform.py && echo "COMPILE OK"
"$PY" -m pytest tests/ -v
```
Expected: COMPILE OK; all tests pass.

- [ ] **Step 9: Manual verification**

Restart, run `data/crowd.mp4`, drive a `high`/`critical` alert, then check:
```bash
ls -la data/output/cic_incidents/
```
Expected: an `.mp4` exists; `get_cic_alerts` shows its `clip_path` populated.

- [ ] **Step 10: Commit**

```bash
git add config.py crowd/clip.py crowd/analyzer.py crowd/platform.py tests/test_cic_clip.py
git commit -m "feat(cic): record incident mp4 clips around high/critical alerts"
```

---

### Task 7: YOLO tiling for dense-crowd counting

**Files:**
- Modify: `config.py` (tiling config)
- Create: `crowd/tiling.py`
- Modify: `crowd/analyzer.py` (`_analyze` uses tiled count when enabled)
- Test: `tests/test_cic_tiling.py`

**Interfaces:**
- Produces: `generate_tiles(w, h, grid="2x2", overlap=0.2) -> list[tuple]`, `merge_boxes(boxes: list, iou_thresh: float) -> list` (boxes are `(x1,y1,x2,y2,conf)`)

- [ ] **Step 1: Add config**

In `config.py`, after the clip block, add:

```python
# ── CIC Phase 4: dense-crowd tiling ─────────────────────────────────────────
CIC_TILING       = os.getenv("CIC_TILING", "0") in ("1", "true", "True")
CIC_TILE_GRID    = os.getenv("CIC_TILE_GRID", "2x2")
CIC_TILE_OVERLAP = float(os.getenv("CIC_TILE_OVERLAP", "0.2"))
CIC_TILE_NMS_IOU = float(os.getenv("CIC_TILE_NMS_IOU", "0.5"))
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_cic_tiling.py`:

```python
from crowd.tiling import generate_tiles, merge_boxes


def test_generate_tiles_2x2_covers_frame():
    tiles = generate_tiles(1000, 800, grid="2x2", overlap=0.0)
    assert len(tiles) == 4
    # union covers corners
    assert any(t[0] == 0 and t[1] == 0 for t in tiles)
    assert any(t[2] == 1000 and t[3] == 800 for t in tiles)


def test_generate_tiles_overlap_widens_tiles():
    no = generate_tiles(1000, 800, grid="2x2", overlap=0.0)
    ov = generate_tiles(1000, 800, grid="2x2", overlap=0.2)
    # an overlapped interior tile is wider than the non-overlapped one
    assert (ov[0][2] - ov[0][0]) > (no[0][2] - no[0][0])


def test_merge_boxes_dedups_seam_duplicates():
    # same person detected in two tiles → near-identical boxes
    boxes = [(100, 100, 140, 200, 0.9), (102, 101, 141, 201, 0.8),
             (500, 300, 540, 400, 0.95)]
    merged = merge_boxes(boxes, iou_thresh=0.5)
    assert len(merged) == 2


def test_merge_boxes_keeps_distinct():
    boxes = [(0, 0, 40, 100, 0.9), (500, 500, 540, 600, 0.9)]
    assert len(merge_boxes(boxes, iou_thresh=0.5)) == 2
```

- [ ] **Step 3: Run test to verify it fails**

```bash
"$PY" -m pytest tests/test_cic_tiling.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'crowd.tiling'`.

- [ ] **Step 4: Implement `crowd/tiling.py`**

```python
"""crowd/tiling.py — pure tile generation + box merge (NMS) for dense crowds."""


def generate_tiles(w: int, h: int, grid: str = "2x2", overlap: float = 0.2):
    """Return a list of (x1, y1, x2, y2) tile rects covering a w×h frame.
    `overlap` widens each tile by that fraction so people on seams are caught."""
    try:
        gx, gy = (int(p) for p in grid.lower().split("x"))
    except Exception:
        gx, gy = 2, 2
    gx, gy = max(1, gx), max(1, gy)
    tw, th = w / gx, h / gy
    ox, oy = tw * overlap, th * overlap
    tiles = []
    for j in range(gy):
        for i in range(gx):
            x1 = max(0, int(i * tw - ox)); y1 = max(0, int(j * th - oy))
            x2 = min(w, int((i + 1) * tw + ox)); y2 = min(h, int((j + 1) * th + oy))
            tiles.append((x1, y1, x2, y2))
    return tiles


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a[:4]
    bx1, by1, bx2, by2 = b[:4]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / float(area_a + area_b - inter)


def merge_boxes(boxes: list, iou_thresh: float = 0.5) -> list:
    """Greedy NMS: drop lower-confidence boxes that overlap a kept box ≥ iou_thresh."""
    kept = []
    for box in sorted(boxes, key=lambda b: b[4], reverse=True):
        if all(_iou(box, k) < iou_thresh for k in kept):
            kept.append(box)
    return kept
```

- [ ] **Step 5: Run tiling tests to verify they pass**

```bash
"$PY" -m pytest tests/test_cic_tiling.py -v
```
Expected: 4 passed.

- [ ] **Step 6: Use tiling in `analyzer._analyze`**

In `crowd/analyzer.py`, in `_analyze`, after the existing full-frame detection block (after the `for box in boxes:` loop populates `detections`) and **before** the `count = len(detections)` line, add:

```python
        # Dense-crowd tiling: recount via overlapping tiles (config-gated).
        # Full-frame track() above still drives behavior flags + track IDs;
        # tiling only produces a more accurate count/density/heatmap.
        tiled_centers = None
        if model is not None and getattr(config, "CIC_TILING", False):
            try:
                from crowd.tiling import generate_tiles, merge_boxes
                tiles = generate_tiles(w, h,
                                       getattr(config, "CIC_TILE_GRID", "2x2"),
                                       getattr(config, "CIC_TILE_OVERLAP", 0.2))
                tboxes = []
                for (tx1, ty1, tx2, ty2) in tiles:
                    sub = frame[ty1:ty2, tx1:tx2]
                    if sub.size == 0:
                        continue
                    res = model(sub, classes=[0], verbose=False,
                                conf=getattr(config, "CIC_YOLO_CONF", 0.25),
                                max_det=getattr(config, "CIC_MAX_DET", 1000))
                    if res and res[0].boxes is not None:
                        for b in res[0].boxes:
                            x1, y1, x2, y2 = b.xyxy[0].tolist()
                            tboxes.append((x1 + tx1, y1 + ty1, x2 + tx1, y2 + ty1,
                                           float(b.conf[0])))
                merged = merge_boxes(tboxes, getattr(config, "CIC_TILE_NMS_IOU", 0.5))
                tiled_centers = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in merged]
            except Exception as e:
                logger.debug(f"Slot {self.slot_id} tiling error: {e}")
```

Then change the `count` and `heatmap_pts` so tiling (when present) is authoritative. Replace:

```python
        count    = len(detections)
```
with:
```python
        count    = len(tiled_centers) if tiled_centers is not None else len(detections)
```

And in the `meta = {...}` dict, replace the `heatmap_pts` value with:

```python
            "heatmap_pts": [
                [round(cx / w, 3), round(cy / h, 3)]
                for (cx, cy) in (tiled_centers if tiled_centers is not None
                                 else [d["center"] for d in detections])
            ],
```

- [ ] **Step 7: Verify syntax + full suite**

```bash
"$PY" -m py_compile crowd/analyzer.py && echo "COMPILE OK"
"$PY" -m pytest tests/ -v
```
Expected: COMPILE OK; all tests pass.

- [ ] **Step 8: Manual verification**

With `CIC_TILING=0`, run `data/crowd.mp4`, note the Count badge. Restart with `CIC_TILING=1` (set in `.env`) and re-run — the dense-crowd Count should be **higher** (catches small/distant heads), no crash, FPS still usable.

- [ ] **Step 9: Commit**

```bash
git add config.py crowd/tiling.py crowd/analyzer.py tests/test_cic_tiling.py
git commit -m "feat(cic): YOLO tiling for accurate dense-crowd counting"
```

---

### Task 8: persons/m² calibration documentation

**Files:**
- Create: `docs/cic-calibration.md`
- Modify: `crowd/zones.json` (fill `fov_area_m2` per zone)

- [ ] **Step 1: Write the calibration doc**

Create `docs/cic-calibration.md`:

```markdown
# CIC persons/m² Calibration

Density = persons ÷ **camera-visible area (m²)**, NOT the whole zone area — otherwise
risk never escalates. Set `fov_area_m2` per zone in `crowd/zones.json`.

## Method A — quick (rectangular ground patch)
1. Identify the ground area the camera actually sees (the visible footprint).
2. Measure its real width and depth in metres (site plan, pacing, or known landmarks
   like a 1.8 m doorway).
3. `fov_area_m2 = width_m × depth_m`. Enter it on the zone in `zones.json`.

## Method B — homography (accurate, perspective-correct)
1. Pick 4 ground points visible in frame with known real-world metres (corners of a
   marked rectangle, paving grid, court lines).
2. Compute the image→ground homography (`cv2.findHomography`).
3. Project the frame's ground-visible polygon to metres; its area is `fov_area_m2`.
   For finer density, project each person's foot point and bin into m² cells.

## Thresholds
Per-zone `thresholds` (`caution`/`high`/`critical`, persons/m²) override
`config.CIC_DENSITY_*`. Crowd-safety references: ~4 p/m² comfortable upper bound,
≥5–6 p/m² crush risk. Tune per venue.

## With tiling
`CIC_TILING=1` raises recall (more true heads) so counts rise; re-check thresholds
against the calibrated area after enabling it.
```

- [ ] **Step 2: Fill `fov_area_m2` per zone**

In `crowd/zones.json`, add a realistic `"fov_area_m2": <number>` to each zone object that lacks one (use Method A estimates; document assumptions in the doc above).

- [ ] **Step 3: Commit**

```bash
git add docs/cic-calibration.md crowd/zones.json
git commit -m "docs(cic): persons/m2 calibration method + per-zone fov areas"
```

---

## Self-Review

**Spec coverage:**
- Item 1 (persist alerts+history) → Tasks 1, 2, 3 ✓
- Item 2 (webhook) → Tasks 4, 5 ✓
- Item 3 (incident clips) → Task 6 ✓
- Item 4 (tiling + calibration) → Tasks 7, 8 ✓
- History routes ✓ (Task 3); TTL prune ✓ (Tasks 1, 3); config-gated/off-by-default ✓ (all).

**Type consistency:** `insert_cic_alert(alert)` / `update_cic_alert_clip(uid, path)` / `get_cic_alerts` / `insert_cic_reading` / `get_cic_readings` / `prune_cic_data` consistent across Tasks 1, 3, 6. `should_persist_reading` signature consistent (Tasks 2, 3). `should_notify`/`build_payload`/`WebhookNotifier.send`/`build_notifiers_from_config` consistent (Tasks 4, 5). `generate_tiles`/`merge_boxes` consistent (Tasks 7). `record_incident(on_done)` + `incident_clip_path`/`write_clip` consistent (Task 6).

**Notes for the implementer:**
- `_maybe_alert` returns the alert dict with key `"id"` (8-char uid) and `"zone"` (name) — Task 1's `insert_cic_alert` reads both `zone`/`zone_name` and `id`.
- `meta["slot"]` is the slot id used to find the analyzer for clip recording (Task 6).
- Tiling runs YOLO per tile via `model(sub, ...)` (detection), independent of the full-frame `model.track(...)` that still drives behavior — expect higher CPU when `CIC_TILING=1`.
