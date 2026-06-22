# Phase 4 — CIC Hardening — Design Spec

**Date:** 2026-06-22
**Owner:** TrippyEngineer
**Branch:** `cic-fixes` (commit Phase 4 in logical per-item chunks)
**Status:** approved design → implementation plan next
**Roadmap ref:** `PHASES.md` → Phase 4

> Goal: make the Crowd Intelligence Center deployment-grade. Today alerts and zone
> history are **in-memory only** (`crowd/platform.py`: `_alerts` list capped at 100,
> per-zone `deque` of 300 readings), alerts surface **SSE-only**, and YOLOv8n head
> counting **saturates under occlusion** in dense crowds.

## Scope (this round)

In: Phase 4 items **1, 2, 3, 4**. Out: item 5 (cross-camera re-ID, stretch).

Decisions locked with the user:
- **Notifications:** generic **webhook only** (Telegram/email/SMS deferred; built behind an
  interface so they can drop in later).
- **Dense-crowd accuracy:** **YOLO tiling + persons/m² calibration** — stay in the current
  `ultralytics` stack, **no** new ML framework (no CSRNet/onnxruntime).

## Global constraints (project invariants — must hold)

- Python 3.10/3.11 only; runs on **Windows Python 3.11**, not WSL (TF/DeepFace/OpenCV/
  ultralytics live there).
- `config.py` is the **single source of truth** for every new threshold/path/toggle. No
  magic numbers elsewhere.
- **Free / zero-key path must keep working.** Every new feature is config-gated and
  **off by default**; absent config → silent no-op, never an error.
- SQLite is WAL on `data/`; only openable from Windows python on `/mnt/d`.
- **No auto-reload** — restart `app.py` after edits.
- No Docker / no Redis.

## Architecture — integration points

- **`crowd/platform.py`** — the 1 Hz aggregator + alert engine (`_run`, `_maybe_alert`).
  Spine for **persistence**, **webhook**, and **clip triggering**.
- **`crowd/analyzer.py`** — per-slot YOLO loop (`_run`, `_analyze`). Owns frames →
  **clip frame-buffer** and **tiling** live here.
- **`storage/database.py`** — new tables + CRUD, mirroring the existing `cic_face_captures`
  pattern (per-call connections, `_now()` timestamps).
- **`crowd/notifier.py`** (new) — pluggable notification layer.
- **`config.py`** — all new toggles/thresholds.
- **`app.py`** — new `/crowd/api/*` read routes for history.

Alert path (after Phase 4):
`analyzer._analyze` → `platform._run` aggregates → `_maybe_alert` (60 s cooldown) →
**persist alert** + **persist reading** + **webhook (async)** + **trigger clip** + SSE
broadcast (unchanged).

---

## Item 1 — Persist alerts + zone history (SQLite)

**New tables** (append to `_SCHEMA` in `storage/database.py`):

```sql
CREATE TABLE IF NOT EXISTS cic_alerts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_uid   TEXT,                 -- the 8-char id used in the dashboard
    zone_id     TEXT NOT NULL,
    zone_name   TEXT NOT NULL,
    type        TEXT NOT NULL,        -- e.g. density_high
    severity    TEXT NOT NULL,        -- warning|high|critical
    message     TEXT,
    density     REAL,
    count       INTEGER,
    clip_path   TEXT,                 -- set when an incident clip is recorded
    acked       INTEGER DEFAULT 0,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cic_alerts_zone ON cic_alerts(zone_id);
CREATE INDEX IF NOT EXISTS idx_cic_alerts_time ON cic_alerts(created_at);

CREATE TABLE IF NOT EXISTS cic_zone_readings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    zone_id     TEXT NOT NULL,
    zone_name   TEXT NOT NULL,
    count       INTEGER,
    density     REAL,
    risk        TEXT,
    n_suspicious INTEGER,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cic_readings_zone ON cic_zone_readings(zone_id, created_at);
```

**CRUD** (new `Database` methods): `insert_cic_alert`, `update_cic_alert_clip`,
`get_cic_alerts(limit, since)`, `insert_cic_reading`, `get_cic_readings(zone_id, since)`,
`prune_cic_data(ttl_days)`.

**Hooks (`platform._run`):**
- On a new alert from `_maybe_alert` → `db.insert_cic_alert(...)`.
- **Reading cadence (downsampled):** persist a per-zone snapshot every
  `CIC_READING_PERSIST_S` (default 10 s) **and** always on a risk transition
  (`risk != prev_risk`). Avoids ~86k rows/zone/day at 1 Hz.

**Retention:** `prune_cic_data(CIC_DATA_TTL_DAYS)` runs once at platform startup; deletes
alerts/readings older than the TTL and their orphaned clip files.

**New routes (`app.py`):**
- `GET /crowd/api/alerts/history?limit=&since=` → past alerts (survives restart).
- `GET /crowd/api/zones/history?zone_id=&since=` → reading time-series for charts.

## Item 2 — Outbound webhook notifications

**New module `crowd/notifier.py`:**
- `Notifier` base (`send(alert: dict, context: dict)`), `WebhookNotifier` impl.
- `build_notifiers_from_config()` returns the enabled notifiers (empty list when none).
- Platform holds the notifier list; on a qualifying alert, dispatches in a **daemon
  thread** (never blocks the 1 Hz loop). Failures are logged, never raised.

**Config:**
- `CIC_WEBHOOK_URL` (default `""` → disabled).
- `CIC_WEBHOOK_MIN_SEVERITY` (default `"high"` → no webhook for `caution`/`warning`).
- `CIC_WEBHOOK_HEADERS` (optional JSON for auth headers).
- `CIC_WEBHOOK_TIMEOUT_S` (default 6).

**Payload:** `{ alert: {...full alert...}, venue: {...}, zone: {id,name}, ts }` as JSON POST.
The existing 60 s per-`(zone, type)` cooldown already throttles frequency.

## Item 3 — Incident clip recording

**Frame buffer (`CameraAnalyzer`):** a `deque(maxlen = CIC_CLIP_PRE_S * INFERENCE_FPS)` of
recent **raw** (pre-overlay) frames, appended in `_run`.

**Trigger:** on a `high`/`critical` alert, `platform` calls
`analyzer.record_incident(meta)` on the alert's slot. The analyzer snapshots the pre-buffer,
keeps appending for `CIC_CLIP_POST_S`, then encodes an `.mp4` (`cv2.VideoWriter`, `mp4v`) in a
background thread to `data/output/cic_incidents/<zone_id>_<YYYYMMDD_HHMMSS>.mp4`. On finish →
`db.update_cic_alert_clip(alert_uid, path)`.

**Guards:** at most one in-flight clip per slot; clips only for `high`/`critical`; pruned by
the same TTL. Disabled when `CIC_CLIPS_ENABLED = False`.

**Config:** `CIC_CLIPS_ENABLED` (default True), `CIC_CLIP_PRE_S` (10), `CIC_CLIP_POST_S` (10).

## Item 4 — Dense-crowd accuracy (YOLO tiling + calibration)

**Tiling (`analyzer._analyze`):** when `CIC_TILING` is on, split the frame into a
`CIC_TILE_GRID` (default `2x2`) of overlapping tiles (`CIC_TILE_OVERLAP`, default 0.2), run
YOLO **detection** per tile, map boxes back to full-frame coords, and merge with a global
NMS (`CIC_TILE_NMS_IOU`, default 0.5). This drives **count / density / heatmap** with far
better recall on small/distant heads.

**ByteTrack nuance:** tiling and `model.track(persist=True)` don't compose (track IDs are
per-tile). Resolution: tiling mode drives **counting only**; the existing **full-frame
`track()`** continues to drive **behavior flags** (loiter/run) + track-IDs, running
independently. If `CIC_TILING` is off, behavior is unchanged.

**Calibration:** density already supports per-zone `fov_area_m2` (`zones.json` →
`self.zone_cfg.get("fov_area_m2")`, fallback `CIC_FOV_AREA_M2=100`). Deliverable: a documented
homography/visible-area method in `docs/cic-calibration.md` so persons/m² is real, plus
`fov_area_m2` filled per zone in `zones.json`.

**Config:** `CIC_TILING` (default False), `CIC_TILE_GRID` ("2x2"), `CIC_TILE_OVERLAP` (0.2),
`CIC_TILE_NMS_IOU` (0.5).

---

## Config additions (all in `config.py`)

```
# Persistence / retention
CIC_READING_PERSIST_S = 10
CIC_DATA_TTL_DAYS     = 30
# Webhook
CIC_WEBHOOK_URL          = ""
CIC_WEBHOOK_MIN_SEVERITY = "high"
CIC_WEBHOOK_HEADERS      = ""     # JSON
CIC_WEBHOOK_TIMEOUT_S    = 6
# Incident clips
CIC_CLIPS_ENABLED = True
CIC_CLIP_PRE_S    = 10
CIC_CLIP_POST_S   = 10
# Dense-crowd tiling
CIC_TILING        = False
CIC_TILE_GRID     = "2x2"
CIC_TILE_OVERLAP  = 0.2
CIC_TILE_NMS_IOU  = 0.5
```

## Testing

**Unit (pure, no network/GPU/camera) — extend `tests/`:**
- Tile-merge NMS: synthetic overlapping boxes across tile seams → correct merged count.
- Reading-downsample cadence: emits on interval boundary + on risk transition only.
- Retention prune: selects only rows older than TTL.
- Webhook payload builder + min-severity gate (mock HTTP; assert no-op when URL empty).
- DB round-trip: insert/get alerts + readings against a temp SQLite file.

**Manual CIC verification (Windows python, the `crowd.mp4` crowd video):**
- Alert now appears in `Alerts & SOP` under load; **survives restart** via history route.
- Webhook hits a test endpoint (webhook.site) on a `high`/`critical` alert.
- An `.mp4` lands in `data/output/cic_incidents/` and is linked from the alert.
- With `CIC_TILING=True`, dense-crowd count rises vs. baseline (no crash, acceptable FPS).

## Out of scope
- Cross-camera re-ID (Phase 4 stretch).
- Telegram/email/SMS channels (interface ready; not built).
- Density-map (CSRNet) model.
- Auth/CSRF/retention-policy from Phase 1 (soft dep; webhook + persistence stand alone here).

## Build/test sequence (tight loops, one at a time)
1. DB tables + CRUD + persistence hooks + history routes → restart, verify alerts persist.
2. `notifier.py` + webhook dispatch → restart, verify webhook.site receives an alert.
3. Incident clip buffer + writer → restart, verify a clip is saved + linked.
4. YOLO tiling + calibration doc → restart, verify denser counts.
