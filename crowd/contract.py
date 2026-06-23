"""
crowd/contract.py
─────────────────
THE stable per-camera analysis contract — the single source of truth for the
`meta` dict that an analyzer (edge worker) PRODUCES and the aggregation layer
(Platform) CONSUMES.

This is the keystone of the scale path (see docs/CIC_MAHAKUMBH_ARCHITECTURE.md):
the *interface* never changes whether `meta` is produced by `CameraAnalyzer` on a
laptop CPU or a DeepStream pipeline on a GPU, and whether it travels by in-process
call, Redis Stream, or Kafka topic. Everything downstream imports the schema from
here.

Deliberately STDLIB-ONLY (no cv2/numpy/config) so it is import-clean on an edge
worker and runnable under any Python — including WSL python where the CV stack
isn't installed.
"""
from typing import Any, TypedDict

# Canonical risk bands, ascending severity. `risk` MUST be one of these.
RISK_LEVELS: tuple = ("safe", "caution", "high", "critical")
RISK_RANK: dict = {r: i for i, r in enumerate(RISK_LEVELS)}

# Crowd-pressure states (crowd/pressure.py), ascending severity.
CROWD_STATES: tuple = ("normal", "dense", "risky", "critical")


class Flow(TypedDict, total=False):
    dx: float
    dy: float
    speed: float


class Meta(TypedDict, total=False):
    """Per-camera analysis result for one cycle. `total=False`: optional fields
    (detections, behavior counts, and the P2 pressure fields) may be absent."""
    # ── identity / required core ──
    slot:        int
    zone_id:     str
    zone_name:   str
    count:        int
    count_method: str           # detector | occlusion | tiling (crowd/counting.py)
    density:      float          # persons / m² of the camera FOV
    risk:         str            # one of RISK_LEVELS
    # ── motion / context ──
    flow:        Flow
    timestamp:   float
    detections:  list
    heatmap_pts: list
    # ── behavioral counts ──
    n_suspicious: int
    n_running:    int
    n_loitering:  int
    n_children:   int
    # ── P2 crowd-pressure early-warning (added by crowd/pressure.py) ──
    pressure:     float         # Helbing crowd pressure = density × Var(velocity)
    pressure_cv:  float         # velocity coefficient of variation (turbulence proxy)
    los:          str           # Fruin Level-of-Service band A..F+
    crowd_state:  str           # one of CROWD_STATES
    turbulence:   bool          # stop-and-go / turbulent-flow precursor detected


# The fields every consumer (Platform aggregation) relies on existing.
REQUIRED_FIELDS: tuple = ("zone_id", "count", "density", "risk")


def empty_meta(slot: int = -1, zone_id: str = "", zone_name: str = "") -> dict:
    """A valid 'no data' meta — what an analyzer returns before its first frame."""
    return {
        "slot": slot,
        "zone_id": zone_id or f"zone_{slot}",
        "zone_name": zone_name or f"Zone {slot}",
        "count": 0,
        "density": 0.0,
        "risk": "safe",
        "flow": {"dx": 0.0, "dy": 0.0, "speed": 0.0},
        "detections": [],
        "heatmap_pts": [],
    }


def validate_meta(meta: Any) -> list:
    """Return a list of contract violations (empty list == valid). Cheap enough to
    run on every cycle in debug; the aggregation layer can log/skip on violations
    instead of crashing on a malformed producer."""
    if not isinstance(meta, dict):
        return ["meta is not a dict"]
    problems = []
    for f in REQUIRED_FIELDS:
        if f not in meta:
            problems.append(f"missing required field: {f}")
    r = meta.get("risk")
    if "risk" in meta and r not in RISK_LEVELS:
        problems.append(f"risk {r!r} not in {RISK_LEVELS}")
    c = meta.get("count")
    if "count" in meta and (not isinstance(c, int) or isinstance(c, bool) or c < 0):
        problems.append("count must be a non-negative int")
    d = meta.get("density")
    if "density" in meta and (not isinstance(d, (int, float)) or isinstance(d, bool) or d < 0):
        problems.append("density must be a non-negative number")
    cs = meta.get("crowd_state")
    if cs is not None and cs not in CROWD_STATES:
        problems.append(f"crowd_state {cs!r} not in {CROWD_STATES}")
    return problems


def is_valid_meta(meta: Any) -> bool:
    return not validate_meta(meta)
