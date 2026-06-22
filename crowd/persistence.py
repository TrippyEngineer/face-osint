"""crowd/persistence.py — pure helpers for CIC persistence cadence."""


def should_persist_reading(now: float, last_ts: float, risk: str,
                           prev_risk: str, interval_s: float) -> bool:
    """Persist a zone reading when the interval has elapsed OR the risk band
    changed since the last reading (so transitions are never lost to downsampling)."""
    if risk != prev_risk:
        return True
    return (now - last_ts) >= interval_s
