"""Regression tests for crowd/contract.py + crowd/pressure.py (P2 crowd-pressure
early-warning). Pure stdlib — runs under any python incl. WSL."""
from crowd import contract as C
from crowd import pressure as P

fails = []
def ck(label, cond):
    print(("  ok " if cond else "FAIL ") + label)
    if not cond: fails.append(label)

# ── contract ──────────────────────────────────────────────────────────────
good = {"zone_id": "z", "count": 5, "density": 1.2, "risk": "high"}
ck("contract.valid good meta", C.validate_meta(good) == [])
ck("contract.missing field", "missing required field: risk" in C.validate_meta(
    {"zone_id": "z", "count": 1, "density": 0.0}))
ck("contract.bad risk", any("risk" in p for p in C.validate_meta(
    {**good, "risk": "boom"})))
ck("contract.negative count", any("count" in p for p in C.validate_meta(
    {**good, "count": -1})))
ck("contract.bad crowd_state", any("crowd_state" in p for p in C.validate_meta(
    {**good, "crowd_state": "nope"})))
ck("contract.empty_meta valid", C.is_valid_meta(C.empty_meta(0, "z", "Z")))
ck("contract.not a dict", C.validate_meta(None) == ["meta is not a dict"])

# ── pressure value: P = density × Var(velocity) ─────────────────────────────
ck("pressure zero density", P.crowd_pressure(0.0, [1, 2, 3]) == 0.0)
ck("pressure no variance", P.crowd_pressure(2.0, [1, 1, 1, 1]) == 0.0)
ck("pressure empty velocities", P.crowd_pressure(3.0, []) == 0.0)
ck("pressure = rho*var", P.crowd_pressure(2.0, [0, 4]) == 8.0)   # var([0,4])=4 → 2*4

# ── Fruin level-of-service bands ────────────────────────────────────────────
ck("los A (free)", P.level_of_service(0.1) == "A")
ck("los escalates with density", C.RISK_RANK.get("x", 0) == 0 and
   P.level_of_service(9.0) == "F+")

# ── assess: the state machine that drives early-warning ─────────────────────
calm   = P.assess(1.0, [2, 2, 2])
dense  = P.assess(3.0, [2, 2, 2])
packed = P.assess(6.0, [2, 2, 2])
crush  = P.assess(9.0, [2, 2, 2])
ck("assess normal",   calm["crowd_state"] == "normal")
ck("assess dense",    dense["crowd_state"] == "dense")
ck("assess risky",    packed["crowd_state"] == "risky")
ck("assess critical", crush["crowd_state"] == "critical")

# turbulence (stop-and-go) escalates EARLY, below the critical density
turb_mid = P.assess(3.0, [0, 0, 10, 0, 10])   # dense density + turbulent flow
turb_hi  = P.assess(6.0, [0, 0, 10, 0, 10])   # compression density + turbulent
ck("turbulence flagged", turb_hi["turbulence"] is True)
ck("turbulence escalates dense->risky", turb_mid["crowd_state"] == "risky")
ck("turbulence escalates packed->critical (early warning)",
   turb_hi["crowd_state"] == "critical")
ck("assess returns contract fields", set(turb_hi) >=
   {"pressure", "pressure_cv", "los", "crowd_state", "turbulence"})

# ── escalate_risk: pressure can only RAISE the density risk, never lower it ──
ck("escalate raises", P.escalate_risk("safe", "risky") == "high")
ck("escalate keeps higher density risk", P.escalate_risk("critical", "normal") == "critical")
ck("escalate to critical", P.escalate_risk("caution", "critical") == "critical")
ck("escalate normal->safe", P.escalate_risk("safe", "normal") == "safe")

print("\n" + ("ALL P2 TESTS PASSED" if not fails else f"{len(fails)} FAILURES: {fails}"))
import sys; sys.exit(1 if fails else 0)
