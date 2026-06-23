"""
crowd/pressure.py
─────────────────
Crowd-PRESSURE early-warning — the subsystem the Mahakumbh/Hajj architecture
identifies as the decisive stampede precursor (docs/CIC_MAHAKUMBH_ARCHITECTURE.md).

Why this exists: a density THRESHOLD alarm is NOT stampede prediction. The 29 Jan
2025 Sangam crush happened with the AI density alarm in place. The quantity that
actually spikes ~10 minutes before a crush is Helbing's **crowd pressure**:

        P = ρ · Var(v)        (local density × variance of pedestrian velocities)

When a dense crowd's motion turns turbulent (stop-and-go: some people jammed, some
surging → high velocity variance) at high density, pressure spikes BEFORE the
density alarm would. We classify density into Fruin Level-of-Service bands and a
coarse crowd_state, and let pressure/turbulence ESCALATE the risk so the existing
alert/SOP pipeline fires earlier.

Deliberately STDLIB-ONLY (statistics) — no cv2/numpy/config — so it is import-clean
on an edge worker and runs unchanged at any scale. Thresholds are injected by the
caller (analyzer reads them from config); velocity units are caller-defined
(px/frame here) so deployments calibrate to their cameras.

Refs: Helbing & Johansson, "Dynamics of crowd disasters" (arXiv:0708.3339);
Fruin Level-of-Service; Itaewon 2022 calibration (PLoS ONE 10.1371/journal.pone.0306764).
"""
import statistics

from crowd.contract import RISK_LEVELS, RISK_RANK

# Fruin-style Level-of-Service density bands (persons/m²), ascending crowding.
# A = free flow … F = jammed … F+ = dangerous compression. Calibration anchors:
# compression begins ~5–8 ppl/m², turbulence ~7–12 ppl/m².
_LOS_BANDS = (
    (0.31, "A"), (0.43, "B"), (0.72, "C"),
    (1.08, "D"), (2.17, "E"), (3.50, "F"), (float("inf"), "F+"),
)

# crowd_state → the risk band it implies (so pressure can raise density-risk).
_STATE_RISK = {"normal": "safe", "dense": "caution", "risky": "high", "critical": "critical"}


def level_of_service(density: float) -> str:
    for ceil, los in _LOS_BANDS:
        if density < ceil:
            return los
    return "F+"


def crowd_pressure(density: float, velocities) -> float:
    """Helbing crowd pressure P = ρ · Var(v). 0 when density is 0 or motion is
    uniform (no variance) — danger comes from the COMBINATION of packing and
    turbulent (high-variance) motion, not either alone."""
    vs = [float(v) for v in (velocities or []) if v is not None]
    if density <= 0 or len(vs) < 2:
        return 0.0
    return round(density * statistics.pvariance(vs), 4)


def assess(density: float, velocities, *,
           dense_density: float = 2.0,
           compression_density: float = 5.0,
           critical_density: float = 8.0,
           turbulence_cv: float = 0.75) -> dict:
    """Classify a zone from its density and per-person velocities.

    Returns the contract's pressure fields:
      pressure     – Helbing P = ρ·Var(v)
      pressure_cv  – velocity coefficient of variation (scale-free turbulence proxy)
      los          – Fruin Level-of-Service band
      crowd_state  – normal | dense | risky | critical
      turbulence   – stop-and-go / turbulent-flow precursor detected

    State machine (early-warning): turbulence at dense density escalates dense→risky,
    and at compression density escalates risky→critical — BEFORE the raw density
    alarm (critical_density) would trip.
    """
    vs = [float(v) for v in (velocities or []) if v is not None]
    if len(vs) >= 2:
        mean_v = statistics.fmean(vs)
        var_v  = statistics.pvariance(vs)
    else:
        mean_v = vs[0] if vs else 0.0
        var_v  = 0.0
    cv = (var_v ** 0.5 / mean_v) if mean_v > 1e-6 else 0.0
    pressure = round(density * var_v, 4) if density > 0 else 0.0

    turbulence = (density >= dense_density and cv >= turbulence_cv and len(vs) >= 3)

    if density >= critical_density or (turbulence and density >= compression_density):
        state = "critical"
    elif density >= compression_density or turbulence:
        state = "risky"
    elif density >= dense_density:
        state = "dense"
    else:
        state = "normal"

    return {
        "pressure":    pressure,
        "pressure_cv": round(cv, 3),
        "los":         level_of_service(density),
        "crowd_state": state,
        "turbulence":  bool(turbulence),
    }


def escalate_risk(density_risk: str, crowd_state: str) -> str:
    """Combine the density-based risk with the pressure-based crowd_state, taking
    the HIGHER of the two. Pressure can raise the alarm earlier; it never lowers it."""
    pr = _STATE_RISK.get(crowd_state, "safe")
    return density_risk if RISK_RANK.get(density_risk, 0) >= RISK_RANK.get(pr, 0) else pr


# Keep RISK_LEVELS importable from here too (single ordering for callers).
__all__ = ["level_of_service", "crowd_pressure", "assess", "escalate_risk", "RISK_LEVELS"]
