"""
crowd/sop.py
────────────
SOP (Standard Operating Procedure) playbook engine for the CIC.

Turns a crowd alert into PRESCRIBED, ordered ACTIONS instead of a hard-coded
"ACTIVATE SOP-3" string, references real infrastructure (gates / bridges / exits)
and their open/closed state, flags the lethal case — **a closed egress during a
critical SOP** (the documented 29-Jan-2025 Sangam failure: pontoon bridges closed
without explanation) — and tracks operator acknowledgement + escalation.

Deliberately STDLIB-ONLY (no cv2/config) so it's import-clean on an edge/command
node and runs unchanged at any scale — and is unit-testable under WSL python.
"""
from dataclasses import dataclass, field

from crowd.contract import RISK_RANK

SEVERITY_RANK = {"info": 0, "warning": 1, "high": 2, "critical": 3}
INFRA_STATES  = ("open", "restricted", "closed")
INFRA_KINDS   = ("gate", "bridge", "exit")
EGRESS_KINDS  = ("bridge", "exit")   # routes people leave by — must be OPEN under load


# ── Infrastructure state model ──────────────────────────────────────────────
@dataclass
class InfraElement:
    id:      str
    name:    str
    kind:    str          # gate | bridge | exit
    zone_id: str
    state:   str = "open"  # open | restricted | closed

    def as_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "kind": self.kind,
                "zone_id": self.zone_id, "state": self.state}


class Infrastructure:
    """Registry of controllable elements (gates/bridges/exits) with live state."""

    def __init__(self, elements=None):
        self._el = {e.id: e for e in (elements or [])}

    @classmethod
    def from_config(cls, zones_raw: dict) -> "Infrastructure":
        """Read infrastructure from zones.json: a per-zone `infrastructure` list
        and/or a top-level `infrastructure` list."""
        els = []
        for z in (zones_raw or {}).get("zones", []):
            zid = z.get("id", "")
            for it in z.get("infrastructure", []):
                els.append(InfraElement(it["id"], it.get("name", it["id"]),
                                        it.get("kind", "gate"), zid,
                                        it.get("state", "open")))
        for it in (zones_raw or {}).get("infrastructure", []):
            els.append(InfraElement(it["id"], it.get("name", it["id"]),
                                    it.get("kind", "gate"), it.get("zone_id", ""),
                                    it.get("state", "open")))
        return cls(els)

    def get(self, eid: str):
        return self._el.get(eid)

    def set_state(self, eid: str, state: str) -> bool:
        e = self._el.get(eid)
        if e is None or state not in INFRA_STATES:
            return False
        e.state = state
        return True

    def for_zone(self, zone_id: str, kind: str = None) -> list:
        return [e for e in self._el.values()
                if e.zone_id == zone_id and (kind is None or e.kind == kind)]

    def closed(self) -> list:
        return [e for e in self._el.values() if e.state == "closed"]

    def snapshot(self) -> list:
        return [e.as_dict() for e in self._el.values()]


# ── Playbooks ───────────────────────────────────────────────────────────────
@dataclass
class Action:
    type:        str
    instruction: str
    target_kind: str = ""   # infra kind this action operates on (gate/exit/...)


@dataclass
class Playbook:
    id:           str
    name:         str
    severity:     str            # info | warning | high | critical
    min_risk:     str            # triggers when zone risk rank >= this
    crowd_states: tuple          # also triggers when crowd_state in this set
    actions:      tuple


# The SOP ladder, aligned with the operator-assistant guidance (llm_ops):
#   caution/dense  → reinforce stewards
#   high/risky     → flow control (close a gate, open an alternate exit, alert cmd)
#   critical       → SOP-3 crowd-pressure protocol (halt inflow, open ALL exits, PA, medical)
DEFAULT_PLAYBOOKS = (
    Playbook("SOP-1", "Steward Reinforcement", "warning", "caution", ("dense",),
             (Action("deploy", "Deploy additional stewards to the zone entrance."),
              Action("monitor", "Increase monitoring cadence on this zone."))),
    Playbook("SOP-2", "Flow Control", "high", "high", ("risky",),
             (Action("restrict_inflow", "Close one entry gate to slow inflow.", "gate"),
              Action("open_exit", "Open an alternate exit to relieve pressure.", "exit"),
              Action("notify", "Alert the sector commander."))),
    Playbook("SOP-3", "Crowd-Pressure Protocol", "critical", "critical", ("critical",),
             (Action("halt_inflow", "HALT all new entries at every gate.", "gate"),
              Action("open_exit", "Open ALL exits immediately.", "exit"),
              Action("broadcast", "PA announcement: move calmly to the nearest exit."),
              Action("medical", "Alert the medical / rescue team."),
              Action("divert", "Divert incoming crowds to an alternate route/ghat."))),
)


def _triggers(pb: Playbook, risk: str, crowd_state: str) -> bool:
    by_risk  = RISK_RANK.get(risk, 0) >= RISK_RANK.get(pb.min_risk, 99)
    by_state = crowd_state in pb.crowd_states
    return by_risk or by_state


def select_playbook(risk: str, crowd_state: str = "normal",
                    playbooks=DEFAULT_PLAYBOOKS):
    """The highest-severity playbook triggered by either the density risk or the
    crowd-pressure state. None when nothing is triggered (safe/normal)."""
    matched = [p for p in playbooks if _triggers(p, risk, crowd_state)]
    if not matched:
        return None
    return max(matched, key=lambda p: SEVERITY_RANK.get(p.severity, 0))


def prescribe(zone_id: str, zone_name: str, risk: str, crowd_state: str = "normal",
              infra: Infrastructure = None, playbooks=DEFAULT_PLAYBOOKS):
    """Resolve the triggered playbook into concrete actions for this zone, with
    infrastructure targets attached and any closed-egress CONFLICTS surfaced.
    Returns None when no SOP is triggered."""
    pb = select_playbook(risk, crowd_state, playbooks)
    if pb is None:
        return None

    actions = []
    for a in pb.actions:
        targets = ([e.as_dict() for e in infra.for_zone(zone_id, a.target_kind)]
                   if (a.target_kind and infra is not None) else [])
        actions.append({"type": a.type, "instruction": a.instruction,
                        "target_kind": a.target_kind, "targets": targets,
                        "status": "pending"})

    # The lethal case: under a high/critical SOP, any egress (exit/bridge) that is
    # CLOSED blocks people from leaving — flag it loudly. This is the 2025 Sangam
    # failure modeled as an alertable state.
    conflicts = []
    if infra is not None and SEVERITY_RANK.get(pb.severity, 0) >= SEVERITY_RANK["high"]:
        for e in infra.for_zone(zone_id):
            if e.kind in EGRESS_KINDS and e.state == "closed":
                conflicts.append({
                    "id": e.id, "element": e.name, "kind": e.kind,
                    "issue": f"{e.kind} '{e.name}' is CLOSED — blocks egress during {pb.id}",
                })

    return {
        "sop_id":    pb.id,
        "name":      pb.name,
        "severity":  pb.severity,
        "headline":  f"{pb.id} — {pb.name}",
        "actions":   actions,
        "conflicts": conflicts,
    }


# ── Acknowledgement / escalation tracking ───────────────────────────────────
@dataclass
class Activation:
    id:         str
    zone_id:    str
    sop_id:     str
    severity:   str
    created_ts: float
    acked:      bool = False
    escalated:  bool = False

    def as_dict(self) -> dict:
        return {"id": self.id, "zone_id": self.zone_id, "sop_id": self.sop_id,
                "severity": self.severity, "created_ts": self.created_ts,
                "acked": self.acked, "escalated": self.escalated}


class SopTracker:
    """Tracks live SOP activations so an UNACKNOWLEDGED high/critical SOP
    escalates after a timer (chain-of-command), and clears when a zone recovers.
    The clock is injected (`now`) so it's deterministic and resume-safe."""

    def __init__(self, escalate_after_s: float = 120.0):
        self.escalate_after_s = escalate_after_s
        self._active: dict = {}

    def activate(self, activation_id, zone_id, sop_id, severity, now) -> Activation:
        act = Activation(activation_id, zone_id, sop_id, severity, now)
        self._active[activation_id] = act
        return act

    def acknowledge(self, activation_id) -> bool:
        a = self._active.get(activation_id)
        if a is None:
            return False
        a.acked = True
        return True

    def due_for_escalation(self, now) -> list:
        """Unacked high/critical activations past the timer, not yet escalated.
        Marks them escalated so they fire exactly once."""
        out = []
        for a in self._active.values():
            if (not a.acked and not a.escalated
                    and SEVERITY_RANK.get(a.severity, 0) >= SEVERITY_RANK["high"]
                    and (now - a.created_ts) >= self.escalate_after_s):
                a.escalated = True
                out.append(a)
        return out

    def clear_zone(self, zone_id) -> None:
        for aid in [aid for aid, a in self._active.items() if a.zone_id == zone_id]:
            del self._active[aid]

    def active(self) -> list:
        return list(self._active.values())
