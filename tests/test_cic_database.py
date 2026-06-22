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
