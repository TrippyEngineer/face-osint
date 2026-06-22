"""crowd/clip.py — pure incident-clip path + encoder."""
import logging
import time
from pathlib import Path

import cv2

logger = logging.getLogger(__name__)


def incident_clip_path(out_dir, zone_id: str, ts=None) -> Path:
    ts = ts or time.time()
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(ts))
    return Path(out_dir) / f"{zone_id}_{stamp}.mp4"


def write_clip(frames: list, path, fps: int) -> bool:
    if not frames:
        return False
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (w, h))
    if not writer.isOpened():
        logger.warning(f"VideoWriter failed to open for {path}")
        return False
    try:
        for f in frames:
            if f.shape[:2] != (h, w):
                f = cv2.resize(f, (w, h))
            writer.write(f)
    finally:
        writer.release()
    return True
