"""Regression tests for crowd/sop.py — the SOP playbook engine, infrastructure
state model, closed-egress conflict detection, and ack/escalation tracking.
Pure stdlib — runs under any python incl. WSL."""
from crowd import sop as S

fails = []
def ck(label, cond):
    print(("  ok " if cond else "FAIL ") + label)
    if not cond: fails.append(label)

# ── playbook selection (by risk OR crowd_state, highest severity wins) ──────
ck("safe/normal → no playbook",   S.select_playbook("safe", "normal") is None)
ck("caution → SOP-1",             S.select_playbook("caution", "normal").id == "SOP-1")
ck("crowd_state dense → SOP-1",   S.select_playbook("safe", "dense").id == "SOP-1")
ck("high → SOP-2",                S.select_playbook("high", "normal").id == "SOP-2")
ck("critical → SOP-3",            S.select_playbook("critical", "normal").id == "SOP-3")
ck("crowd_state critical overrides low risk → SOP-3",
   S.select_playbook("caution", "critical").id == "SOP-3")

# ── prescribe returns structured, prioritized actions ───────────────────────
p = S.prescribe("zone_b", "Pontoon Bridge", "critical", "critical")
ck("prescribe sop_id", p["sop_id"] == "SOP-3")
ck("prescribe severity", p["severity"] == "critical")
ck("prescribe has actions", len(p["actions"]) >= 3)
ck("action shape", all({"type", "instruction", "status"} <= set(a) for a in p["actions"]))
ck("prescribe none when safe", S.prescribe("z", "Z", "safe", "normal") is None)

# ── infrastructure state model ──────────────────────────────────────────────
infra = S.Infrastructure([
    S.InfraElement("gate_b1", "North Gate", "gate", "zone_b", "open"),
    S.InfraElement("bridge_b1", "Pontoon Bridge", "bridge", "zone_b", "open"),
    S.InfraElement("exit_b1", "East Exit", "exit", "zone_b", "open"),
])
ck("for_zone filters by kind", [e.id for e in infra.for_zone("zone_b", "exit")] == ["exit_b1"])
ck("set_state ok", infra.set_state("exit_b1", "closed") is True)
ck("set_state bad value rejected", infra.set_state("exit_b1", "ajar") is False)
ck("closed() lists closed", [e.id for e in infra.closed()] == ["exit_b1"])

# ── the SANGAM failure: a CLOSED exit during a critical SOP is a CONFLICT ────
pc = S.prescribe("zone_b", "Pontoon Bridge", "critical", "critical", infra=infra)
ck("closed egress flagged as conflict", len(pc["conflicts"]) == 1)
ck("conflict names the element", pc["conflicts"][0]["id"] == "exit_b1")
# re-open it → conflict clears
infra.set_state("exit_b1", "open")
ck("re-opened egress clears conflict",
   S.prescribe("zone_b", "Pontoon Bridge", "critical", "critical", infra=infra)["conflicts"] == [])

# ── from_config reads zones.json-style infrastructure ───────────────────────
cfg = {"zones": [{"id": "zone_b", "infrastructure": [
    {"id": "g1", "name": "Gate 1", "kind": "gate", "state": "open"}]}]}
ic = S.Infrastructure.from_config(cfg)
ck("from_config builds elements", ic.get("g1") is not None and ic.get("g1").zone_id == "zone_b")

# ── ack / escalation tracker (clock injected, deterministic) ────────────────
t = S.SopTracker(escalate_after_s=120)
t.activate("a1", "zone_b", "SOP-3", "critical", now=1000.0)
ck("unacked not yet due", t.due_for_escalation(now=1000.0 + 60) == [])
due = t.due_for_escalation(now=1000.0 + 121)
ck("unacked critical escalates past timer", [a.id for a in due] == ["a1"])
ck("does not re-escalate", t.due_for_escalation(now=1000.0 + 200) == [])
t.activate("a2", "zone_c", "SOP-3", "critical", now=2000.0)
ck("ack stops escalation", t.acknowledge("a2") and
   t.due_for_escalation(now=2000.0 + 300) == [])
t.clear_zone("zone_b")
ck("clear_zone removes activations", all(a.zone_id != "zone_b" for a in t.active()))

print("\n" + ("ALL SOP TESTS PASSED" if not fails else f"{len(fails)} FAILURES: {fails}"))
import sys; sys.exit(1 if fails else 0)
