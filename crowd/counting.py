"""
crowd/counting.py
─────────────────
Pluggable crowd-counting head (capability C3, docs/CIC_MAHAKUMBH_ARCHITECTURE.md).

Object detectors (YOLO) systematically UNDERCOUNT dense crowds: above ~5–10 ppl/m²
heads occlude each other and boxes merge/drop. This module is the SEAM where a
learned point/density model (P2PNet / EBC-ZIP / PET) drops in later behind a stable
interface; today it ships a CPU-only, no-deps **occlusion correction** so dense
scenes stop reading low.

`estimate()` operates on detection BOXES, not pixels, so it's stdlib-only and
unit-testable under WSL. Default backend "detector" = today's raw count (zero
behaviour change); backend "occlusion" applies the correction.

Selection is config-gated (CIC_COUNTING). It is an ALTERNATIVE to tiling
(CIC_TILING) — use one dense-count strategy, not both.
"""

BACKENDS = ("detector", "occlusion")


def estimate(boxes, frame_w, frame_h, *, backend="detector",
             occlusion_gain: float = 1.5, max_factor: float = 2.5) -> dict:
    """Estimate the true person count from detection boxes.

    boxes: list of (x1, y1, x2, y2). Returns:
      count       – estimated count (corrected for the chosen backend)
      count_raw   – the raw detector count
      factor      – multiplier applied (1.0 for detector)
      coverage    – fraction of frame covered by person boxes (occlusion proxy)
      method      – which backend actually ran

    Occlusion model: person-box coverage of the frame is a density proxy. Sparse
    scenes (low coverage) get ~no correction — raw count is accurate. Dense scenes
    (boxes packing/overlapping the frame, coverage → 1) get inflated by
    1 + occlusion_gain·coverage, capped at max_factor, to compensate for hidden
    occluded people. Crude but monotonic and bounded — a labelled stopgap until a
    learned density head replaces it behind this same interface.
    """
    raw = len(boxes)
    frame_area = float(frame_w) * float(frame_h)
    if backend != "occlusion" or raw < 2 or frame_area <= 0:
        return {"count": raw, "count_raw": raw, "factor": 1.0,
                "coverage": 0.0, "method": "detector"}

    box_area = 0.0
    for (x1, y1, x2, y2) in boxes:
        box_area += max(0.0, x2 - x1) * max(0.0, y2 - y1)
    coverage = min(1.0, box_area / frame_area)
    # Floor at 1.0: this head only ever INFLATES (corrects undercount). A
    # misconfigured max_factor<1 or negative gain must never deflate the count
    # below the raw detections and silently lower density→risk→pressure.
    factor = max(1.0, min(max_factor, 1.0 + occlusion_gain * coverage))
    count = int(round(raw * factor))
    return {"count": count, "count_raw": raw, "factor": round(factor, 3),
            "coverage": round(coverage, 3), "method": "occlusion"}
