"""Regression tests for crowd/counting.py — the pluggable crowd-counting head.
Pure stdlib (operates on boxes, not pixels) — runs under any python incl. WSL."""
from crowd import counting as C


def box(x1, y1, x2, y2): return (x1, y1, x2, y2)


fails = []
def ck(label, cond):
    print(("  ok " if cond else "FAIL ") + label)
    if not cond: fails.append(label)

W = H = 100  # 100×100 frame → area 10000

# ── detector backend (default) = raw count, no correction ───────────────────
boxes = [box(0, 0, 10, 10), box(20, 20, 30, 30), box(40, 40, 50, 50)]
d = C.estimate(boxes, W, H, backend="detector")
ck("detector count == raw", d["count"] == 3 and d["count_raw"] == 3)
ck("detector method", d["method"] == "detector" and d["factor"] == 1.0)

# ── occlusion backend: sparse scene → ~no correction ────────────────────────
# 3 small boxes (each 100px = 1% of frame) → coverage 0.03 → factor ~1.045
sp = C.estimate(boxes, W, H, backend="occlusion")
ck("occlusion sparse barely corrects", sp["count"] == 3 and sp["factor"] < 1.1)
ck("occlusion reports coverage", 0.0 < sp["coverage"] < 0.05 and sp["method"] == "occlusion")

# ── occlusion backend: dense scene → inflate (undercount correction) ────────
# 10 boxes each 40×40 = 1600px → sum 16000 > frame → coverage clamps to 1.0
dense = [box(i, i, i + 40, i + 40) for i in range(0, 100, 10)]
dn = C.estimate(dense, W, H, backend="occlusion", occlusion_gain=1.5, max_factor=2.5)
ck("dense coverage clamped to 1", dn["coverage"] == 1.0)
ck("dense factor hits cap", dn["factor"] == 2.5)
ck("dense count inflated above raw", dn["count"] == round(len(dense) * 2.5) > len(dense))

# ── guards ──────────────────────────────────────────────────────────────────
ck("raw < 2 → no correction", C.estimate([box(0, 0, 9, 9)], W, H, backend="occlusion")["count"] == 1)
ck("empty → 0", C.estimate([], W, H, backend="occlusion")["count"] == 0)
ck("zero frame area safe", C.estimate(dense, 0, 0, backend="occlusion")["method"] == "detector")
ck("max_factor respected", C.estimate(dense, W, H, backend="occlusion",
                                      occlusion_gain=10.0, max_factor=1.8)["factor"] == 1.8)
ck("unknown backend falls back to detector",
   C.estimate(dense, W, H, backend="bogus")["method"] == "detector")

print("\n" + ("ALL COUNTING TESTS PASSED" if not fails else f"{len(fails)} FAILURES: {fails}"))
import sys; sys.exit(1 if fails else 0)
