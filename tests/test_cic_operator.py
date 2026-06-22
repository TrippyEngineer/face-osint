"""CIC operator context: must describe only zones with a LIVE camera feed,
not the static 4-zone zones.json config (which made answers generic/identical)."""
from crowd.llm_ops import _build_context


def _zones():
    return {
        "zone_a": {"name": "Sangam Ghat",    "count": 0,  "density": 0.0,  "risk": "safe",    "slot": 0},
        "zone_b": {"name": "Pontoon Bridge", "count": 40, "density": 0.57, "risk": "caution", "slot": 1},
        "zone_c": {"name": "Sector 4",       "count": 0,  "density": 0.0,  "risk": "safe",    "slot": 2},
        "zone_d": {"name": "Approach Road",  "count": 0,  "density": 0.0,  "risk": "safe",    "slot": 3},
    }


def test_context_only_includes_active_feed():
    state = {"slots": [{"slot": 1, "active": True}], "zones": _zones(), "alerts": []}
    ctx = _build_context(state)
    assert "Pontoon Bridge" in ctx          # the one active zone
    assert "Sangam Ghat" not in ctx         # inactive zones excluded
    assert "Sector 4" not in ctx
    assert "Approach Road" not in ctx
    assert "Active camera feeds: 1" in ctx


def test_context_no_active_feeds():
    state = {"slots": [], "zones": _zones(), "alerts": []}
    ctx = _build_context(state)
    assert "No camera feeds are currently active" in ctx
