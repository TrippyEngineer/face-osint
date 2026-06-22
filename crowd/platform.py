"""
crowd/platform.py
──────────────────
Central CIC platform manager (singleton).

Responsibilities:
  - Manage 4 camera slots (CameraAnalyzer instances)
  - Aggregate per-slot metrics into zone states every second
  - Deduplicated alert generation with 60s cooldown per (zone, type)
  - SSE broadcast to all connected dashboard browsers
  - Rolling 5-min history per zone (300 readings @ 1 Hz)

Usage:
    plat = get_platform()           # get singleton
    plat.start_slot(0, 0)           # webcam on slot 0
    plat.start_slot(2, "/path/to/crowd.mp4")
    state = plat.get_state()
    q = plat.subscribe()            # SSE queue
    plat.unsubscribe(q)
"""

import json
import logging
import queue
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Optional

from crowd.analyzer import CameraAnalyzer

import config
from storage.database import Database
from crowd.persistence import should_persist_reading
from crowd.notifier import build_notifiers_from_config, should_notify

logger = logging.getLogger(__name__)

_ZONES_PATH = Path(__file__).parent / "zones.json"
ALERT_COOLDOWN_S = 60   # minimum seconds between same (zone, type) alerts
HISTORY_LEN      = 300  # 5 min @ 1 Hz


class Platform:
    def __init__(self):
        self._lock        = threading.RLock()   # RLock: get_state() calls get_active_slots() which re-acquires
        self._zones_raw   = self._load_zones()
        self._analyzers   = {}          # slot_id → CameraAnalyzer
        self._zone_states = self._init_zone_states()
        self._alerts      = []          # list of alert dicts, newest first
        self._alert_ts    = {}          # (zone_id, alert_type) → last_alert_time
        self._db          = Database()
        self._reading_last_ts: dict = {}   # zone_id → last persisted reading time
        try:
            removed = self._db.prune_cic_data(getattr(config, "CIC_DATA_TTL_DAYS", 30))
            if removed:
                logger.info(f"CIC startup prune: removed {removed} expired alert/reading rows")
        except Exception as e:
            logger.warning(f"CIC startup prune failed: {e}")
        self._notifiers = build_notifiers_from_config()
        if self._notifiers:
            logger.info(f"CIC notifiers active: {len(self._notifiers)}")
        self._subscribers: list = []
        self._running     = False
        self._thread: Optional[threading.Thread] = None

    # ── Zone loading ──────────────────────────────────────────────────────

    def _load_zones(self) -> dict:
        try:
            return json.loads(_ZONES_PATH.read_text(encoding="utf-8"))
        except Exception as e:
            logger.error(f"Failed to load zones.json: {e}")
            return {"venue": {}, "zones": []}

    def _init_zone_states(self) -> dict:
        states = {}
        for z in self._zones_raw.get("zones", []):
            states[z["id"]] = {
                "id":       z["id"],
                "name":     z["name"],
                "count":    0,
                "density":  0.0,
                "risk":     "safe",
                "flow":     {"dx": 0.0, "dy": 0.0, "speed": 0.0},
                "history":  deque([0] * HISTORY_LEN, maxlen=HISTORY_LEN),
                "capacity": z.get("capacity", 10000),
                "area_m2":  z.get("area_m2", 5000),
                "slot":     z.get("camera_slot", -1),
                "lat_lon":  z.get("lat_lon", []),
                "color":    z.get("color", "#6366f1"),
            }
        return states

    def get_zones_raw(self) -> dict:
        return self._zones_raw

    # ── Slot management ───────────────────────────────────────────────────

    def _zone_for_slot(self, slot_id: int) -> dict:
        for z in self._zones_raw.get("zones", []):
            if z.get("camera_slot") == slot_id:
                return z
        return {"id": f"zone_{slot_id}", "name": f"Zone {slot_id}",
                "area_m2": 5000, "capacity": 10000,
                "thresholds": {"caution": 1.5, "high": 3.0, "critical": 6.0}}

    def start_slot(self, slot_id: int, source) -> bool:
        # Resolve IP camera shorthand (192.168.x.x:8080 → probe for stream URL)
        if isinstance(source, str) and _looks_like_ip_cam(source):
            probed = _probe_ip_cam(source)
            if probed:
                logger.info(f"Slot {slot_id}: IP cam probed → {probed}")
                source = probed
            else:
                logger.warning(f"Slot {slot_id}: IP cam probe failed for {source}, trying as-is")

        with self._lock:
            if slot_id in self._analyzers:
                self._analyzers[slot_id].stop()
            zone_cfg = self._zone_for_slot(slot_id)
            analyzer = CameraAnalyzer(slot_id, zone_cfg)
            ok = analyzer.start(source)
            if ok:
                self._analyzers[slot_id] = analyzer
                self._ensure_running()
            return ok

    def set_toggle(self, slot_id: int, name: str, value: bool):
        with self._lock:
            a = self._analyzers.get(slot_id)
        if a:
            a.toggles[name] = value

    def stop_slot(self, slot_id: int):
        with self._lock:
            a = self._analyzers.pop(slot_id, None)
        if a:
            a.stop()

    def get_slot_frame_b64(self, slot_id: int) -> Optional[str]:
        with self._lock:
            a = self._analyzers.get(slot_id)
        return a.get_overlay_frame_b64() if a else None

    def get_active_slots(self) -> list:
        with self._lock:
            return [
                {"slot": sid, "source": str(a._source or ""), "active": a.is_active()}
                for sid, a in self._analyzers.items()
            ]

    # ── State access ──────────────────────────────────────────────────────

    def get_state(self) -> dict:
        with self._lock:
            zones_snap = {}
            for zid, zs in self._zone_states.items():
                zones_snap[zid] = {
                    k: (list(v) if isinstance(v, deque) else v)
                    for k, v in zs.items()
                    if k != "history"
                }
                zones_snap[zid]["trend"] = list(zs["history"])[-30:]

            return {
                "zones":        zones_snap,
                "alerts":       list(self._alerts[:50]),
                "slots":        self.get_active_slots(),
                "total_count":  sum(z.get("count", 0) for z in zones_snap.values()),
                "venue":        self._zones_raw.get("venue", {}),
            }

    def get_heatmap(self) -> list:
        """Return [lat, lng, intensity] points for Leaflet.heat from active slots."""
        points = []
        with self._lock:
            analyzers = dict(self._analyzers)
        for slot_id, analyzer in analyzers.items():
            meta = analyzer.get_meta()
            if not meta:
                continue
            zone_cfg = self._zone_for_slot(slot_id)
            lat_lon  = zone_cfg.get("lat_lon", [])
            if not lat_lon:
                continue
            # Bounding box of zone polygon
            lats = [p[0] for p in lat_lon]
            lons = [p[1] for p in lat_lon]
            lat_min, lat_max = min(lats), max(lats)
            lon_min, lon_max = min(lons), max(lons)
            # Map normalized heatmap_pts (0-1) to zone lat/lon
            for px, py in meta.get("heatmap_pts", []):
                lat = lat_max - py * (lat_max - lat_min)
                lon = lon_min + px * (lon_max - lon_min)
                points.append([round(lat, 6), round(lon, 6), 0.8])
        return points

    def get_alerts(self, limit: int = 50) -> list:
        with self._lock:
            return list(self._alerts[:limit])

    def get_alert_history(self, limit: int = 100, since: str = "") -> list:
        return self._db.get_cic_alerts(limit=limit, since=since)

    def get_zone_history(self, zone_id: str = "", since: str = "") -> list:
        return self._db.get_cic_readings(zone_id=zone_id, since=since)

    # ── SSE pub-sub ───────────────────────────────────────────────────────

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=200)   # bounded: drop on slow client, don't leak
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue):
        with self._lock:
            try:
                self._subscribers.remove(q)
            except ValueError:
                pass

    def _broadcast(self, **msg):
        with self._lock:
            subs = list(self._subscribers)
        for q in subs:
            try:
                q.put_nowait(msg)
            except queue.Full:
                pass

    # ── Background aggregation loop ───────────────────────────────────────

    def _ensure_running(self):
        if not self._running:
            self._running = True
            self._thread  = threading.Thread(
                target=self._run, daemon=True, name="CIC-platform"
            )
            self._thread.start()

    def _run(self):
        logger.info("CIC platform aggregator started")
        while True:
            time.sleep(1.0)

            with self._lock:
                analyzers = dict(self._analyzers)
                if not analyzers:
                    self._running = False
                    break

            updated = {}
            for slot_id, analyzer in analyzers.items():
                meta = analyzer.get_meta()
                if not meta:
                    continue
                zone_id = meta.get("zone_id")
                if not zone_id:
                    continue
                updated[zone_id] = meta

            new_alerts = []
            with self._lock:
                for zid, meta in updated.items():
                    if zid not in self._zone_states:
                        continue
                    zs = self._zone_states[zid]
                    prev_risk = zs["risk"]
                    zs["count"]        = meta["count"]
                    zs["density"]      = meta["density"]
                    zs["risk"]         = meta["risk"]
                    zs["flow"]         = meta.get("flow", {})
                    zs["n_suspicious"] = meta.get("n_suspicious", 0)
                    zs["n_running"]    = meta.get("n_running", 0)
                    zs["n_children"]   = meta.get("n_children", 0)
                    zs["history"].append(meta["count"])

                    # Alert on risk escalation
                    risk = meta["risk"]
                    alert = self._maybe_alert(zid, zs["name"], risk, prev_risk, meta)
                    if alert:
                        self._alerts.insert(0, alert)
                        self._alerts = self._alerts[:100]
                        new_alerts.append(alert)
                        try:
                            self._db.insert_cic_alert(alert)
                        except Exception as e:
                            logger.warning(f"persist alert failed: {e}")
                        if self._notifiers:
                            min_sev = getattr(config, "CIC_WEBHOOK_MIN_SEVERITY", "high")
                            if should_notify(alert.get("severity", ""), min_sev):
                                ctx = {"venue": self._zones_raw.get("venue", {}),
                                       "zone": {"id": zid, "name": zs["name"]}}
                                for n in self._notifiers:
                                    threading.Thread(target=n.send, args=(alert, ctx),
                                                     daemon=True, name="CIC-notify").start()
                        if (getattr(config, "CIC_CLIPS_ENABLED", True)
                                and alert.get("severity") in ("high", "critical")):
                            a = self._analyzers.get(meta.get("slot"))
                            if a is not None:
                                _uid = alert.get("id")
                                a.record_incident(
                                    lambda p, u=_uid: self._db.update_cic_alert_clip(u, p))

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

            # Broadcast update
            state_snap = self.get_state()
            self._broadcast(
                type="update",
                zones={zid: {
                    "count":       zs["count"],
                    "density":     zs["density"],
                    "risk":        zs["risk"],
                    "trend":       list(zs["history"])[-30:],
                    "n_suspicious": zs.get("n_suspicious", 0),
                    "n_running":   zs.get("n_running", 0),
                    "n_children":  zs.get("n_children", 0),
                } for zid, zs in self._zone_states.items()},
                total_count=state_snap["total_count"],
                alerts=new_alerts,
                ts=time.time(),
            )

        logger.info("CIC platform aggregator stopped (no active slots)")

    def _maybe_alert(self, zone_id: str, zone_name: str,
                     risk: str, prev_risk: str, meta: dict) -> Optional[dict]:
        if risk == "safe":
            return None
        if risk == prev_risk:
            # Only re-alert critical after cooldown
            if risk != "critical":
                return None

        alert_type = f"density_{risk}"
        now        = time.time()
        key        = (zone_id, alert_type)
        last       = self._alert_ts.get(key, 0)

        if now - last < ALERT_COOLDOWN_S:
            return None

        self._alert_ts[key] = now

        messages = {
            "caution":  f"Density approaching safe limit — {meta['count']} persons detected",
            "high":     f"HIGH density — {meta['count']} persons. Deploy stewards to {zone_name}.",
            "critical": f"CRITICAL — {meta['count']} persons. Possible crush risk. ACTIVATE SOP-3.",
        }
        severity_map = {"caution": "warning", "high": "high", "critical": "critical"}

        return {
            "id":        str(uuid.uuid4())[:8],
            "timestamp": time.strftime("%H:%M:%S"),
            "zone_id":   zone_id,
            "zone":      zone_name,
            "type":      alert_type,
            "severity":  severity_map.get(risk, "warning"),
            "message":   messages.get(risk, ""),
            "density":   meta["density"],
            "count":     meta["count"],
            "acked":     False,
        }


# ── IP camera probe helpers ───────────────────────────────────────────────────
_IP_CAM_ENDPOINTS = ["/video", "/videofeed", "/video?submenu=mjpg", "/mjpeg.cgi"]

def _looks_like_ip_cam(source: str) -> bool:
    """True if source looks like 192.168.x.x:port or hostname:port (not http/rtsp URL)."""
    import re
    return bool(re.match(r'^[\d]{1,3}\.[\d]{1,3}\.[\d]{1,3}\.[\d]{1,3}(:\d+)?$', source)
                or re.match(r'^[a-zA-Z0-9._-]+:\d+$', source))

def _probe_ip_cam(addr: str) -> Optional[str]:
    """Try common IP camera endpoints; return working MJPEG URL or None."""
    import urllib.request
    if not addr.startswith("http"):
        addr = "http://" + addr
    for ep in _IP_CAM_ENDPOINTS:
        url = addr.rstrip("/") + ep
        try:
            r = urllib.request.urlopen(url, timeout=2)
            ct = r.headers.get("Content-Type", "")
            if "image" in ct or "video" in ct or "multipart" in ct:
                return url
        except Exception:
            pass
    return None


# ── Module-level singleton ────────────────────────────────────────────────────
_platform: Optional[Platform] = None
_platform_lock = threading.Lock()


def get_platform() -> Platform:
    global _platform
    with _platform_lock:
        if _platform is None:
            _platform = Platform()
    return _platform
