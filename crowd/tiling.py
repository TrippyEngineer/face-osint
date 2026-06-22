"""crowd/tiling.py — pure tile generation + box merge (NMS) for dense crowds."""


def generate_tiles(w: int, h: int, grid: str = "2x2", overlap: float = 0.2):
    """Return a list of (x1, y1, x2, y2) tile rects covering a w×h frame.
    `overlap` widens each tile by that fraction so people on seams are caught."""
    try:
        gx, gy = (int(p) for p in grid.lower().split("x"))
    except Exception:
        gx, gy = 2, 2
    gx, gy = max(1, gx), max(1, gy)
    tw, th = w / gx, h / gy
    ox, oy = tw * overlap, th * overlap
    tiles = []
    for j in range(gy):
        for i in range(gx):
            x1 = max(0, int(i * tw - ox)); y1 = max(0, int(j * th - oy))
            x2 = min(w, int((i + 1) * tw + ox)); y2 = min(h, int((j + 1) * th + oy))
            tiles.append((x1, y1, x2, y2))
    return tiles


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a[:4]
    bx1, by1, bx2, by2 = b[:4]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / float(area_a + area_b - inter)


def merge_boxes(boxes: list, iou_thresh: float = 0.5) -> list:
    """Greedy NMS: drop lower-confidence boxes that overlap a kept box ≥ iou_thresh."""
    kept = []
    for box in sorted(boxes, key=lambda b: b[4], reverse=True):
        if all(_iou(box, k) < iou_thresh for k in kept):
            kept.append(box)
    return kept
