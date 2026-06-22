import numpy as np
import cv2

from crowd.clip import incident_clip_path, write_clip


def test_incident_clip_path_shape(tmp_path):
    p = incident_clip_path(tmp_path, "zone_0", ts=1750000000)
    assert p.parent == tmp_path
    assert p.name.startswith("zone_0_") and p.suffix == ".mp4"


def test_write_clip_creates_readable_mp4(tmp_path):
    frames = [np.full((120, 160, 3), i, np.uint8) for i in range(0, 30)]
    out = tmp_path / "clip.mp4"
    ok = write_clip(frames, out, fps=5)
    assert ok is True
    assert out.exists() and out.stat().st_size > 0
    cap = cv2.VideoCapture(str(out))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert n >= 1


def test_write_clip_empty_returns_false(tmp_path):
    assert write_clip([], tmp_path / "x.mp4", fps=5) is False
