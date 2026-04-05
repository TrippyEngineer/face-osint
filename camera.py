"""
camera.py
──────────
WiFi (IP Webcam) and local webcam support.

Directly ports the proven FrameReader + probe_ip_camera architecture
from attendance.py — the same code that handles Iris Xe + IP Webcam
on Windows natively.

Key design decisions:
  • FrameReader runs in a daemon thread — main thread never blocks on cap.read()
  • cv2.error from cap.read() is caught inside the thread — never kills the reader
  • Resolution injected as URL query param (?640x480) — IP Webcam ignores
    CAP_PROP_FRAME_WIDTH/HEIGHT completely
  • Auto-reconnect after RECONNECT_AFTER consecutive empty reads
  • FPS measured continuously in the reader thread
"""

import cv2
import re
import time
import threading
import logging
import urllib.request
import urllib.error
import urllib.parse
from typing import Optional, Union

import config

logger = logging.getLogger(__name__)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  FRAME READER — background thread                                    ║
# ╚══════════════════════════════════════════════════════════════════════╝
class FrameReader:
    """
    Background daemon thread that drains the camera buffer continuously.
    Main thread always gets the freshest frame without ever blocking.

    Usage:
        reader = FrameReader(0)           # local webcam
        reader = FrameReader("http://...") # WiFi camera
        ret, frame = reader.read()
        reader.release()
    """

    def __init__(self, source: Union[int, str]):
        self.source  = source
        self.is_wifi = isinstance(source, str)

        self._lock   = threading.Lock()
        self._stop   = threading.Event()
        self._frame: Optional[object] = None
        self._ret    = False
        self._fps    = 0.0
        self._cap    = self._open(source)

        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()
        logger.info(f"FrameReader started | source={source} | wifi={self.is_wifi}")

    def _open(self, source: Union[int, str]):
        if isinstance(source, int):
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(source)

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        try:
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 4000)
        except Exception:
            pass

        if isinstance(source, str):
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FPS, 30)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logger.info(f"WiFi stream negotiated: {w}x{h}")

        return cap

    def _reader_loop(self):
        failed     = 0
        fps_frames = 0
        fps_t0     = time.time()

        while not self._stop.is_set():
            try:
                ret, frame = self._cap.read()
            except Exception as e:
                # cv2.error must NOT kill this thread
                logger.warning(f"FrameReader cap.read() raised {type(e).__name__}: {e}")
                ret, frame = False, None

            if not ret or frame is None:
                failed += 1
                time.sleep(0.05)
                if failed >= config.RECONNECT_AFTER:
                    logger.warning("FrameReader: too many empty reads — reconnecting")
                    try:
                        self._cap.release()
                    except Exception:
                        pass
                    time.sleep(1.5)
                    self._cap = self._open(self.source)
                    failed = 0
                continue

            failed     = 0
            fps_frames += 1
            elapsed    = time.time() - fps_t0
            if elapsed >= 1.0:
                with self._lock:
                    self._fps = round(fps_frames / elapsed, 1)
                fps_frames = 0
                fps_t0     = time.time()

            with self._lock:
                self._ret   = ret
                self._frame = frame

    def read(self):
        """Return (ret, frame_copy). Thread-safe."""
        with self._lock:
            if self._frame is None:
                return False, None
            return self._ret, self._frame.copy()

    def get_fps(self) -> float:
        with self._lock:
            return self._fps

    def is_opened(self) -> bool:
        return self._cap.isOpened()

    def release(self):
        self._stop.set()
        try:
            self._cap.release()
        except Exception:
            pass
        logger.info("FrameReader released")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  IP WEBCAM PROBE                                                     ║
# ╚══════════════════════════════════════════════════════════════════════╝
def _host_reachable(url: str, timeout: int = 3):
    try:
        with urllib.request.urlopen(
            urllib.request.Request(url, method="GET"), timeout=timeout
        ) as r:
            return True, r.status
    except urllib.error.HTTPError as e:
        return True, e.code
    except urllib.error.URLError as e:
        return False, str(e.reason)
    except Exception as e:
        return False, str(e)


def _inject_resolution(url: str, w: int, h: int) -> str:
    """
    Append ?WxH to the stream URL.
    IP Webcam (Android) silently ignores CAP_PROP_FRAME_WIDTH/HEIGHT —
    the query param is the only reliable way to control stream resolution.
    """
    parsed  = urllib.parse.urlparse(url)
    res_key = f"{w}x{h}"
    others  = [
        p for p in (parsed.query or "").split("&")
        if p and not re.match(r"^\d+x\d+$", p)
    ]
    new_q = "&".join([res_key] + others)
    return urllib.parse.urlunparse(parsed._replace(query=new_q))


def probe_ip_camera(base_url: str) -> Optional[str]:
    """
    Probe a WiFi camera URL. Returns first working low-res stream URL,
    or None if unreachable.

    Tries each endpoint in config.IP_WEBCAM_ENDPOINTS with multiple
    cap.read() attempts (MJPEG needs a warm-up burst).
    """
    base = base_url.rstrip("/")

    print(f"\n  Checking {base} ...", end=" ", flush=True)
    ok, reason = _host_reachable(base)
    if not ok:
        print("UNREACHABLE")
        print(f"    Reason : {reason}")
        print("    Fix    : phone and PC must be on the same WiFi")
        print("             Open IP Webcam → tap 'Start server'")
        logger.warning(f"WiFi camera unreachable: {base} — {reason}")
        return None
    print(f"OK (HTTP {reason})")

    last_seg     = base.split("?")[0].split("/")[-1]
    has_endpoint = any(ep.strip("/") == last_seg for ep in config.IP_WEBCAM_ENDPOINTS)
    candidates   = [base] if has_endpoint else [base + ep for ep in config.IP_WEBCAM_ENDPOINTS]

    ATTEMPTS = 6
    DELAY    = 0.4

    print(f"  Probing {len(candidates)} endpoint(s), {ATTEMPTS} reads each...")
    for url in candidates:
        print(f"  Testing {url} ... ", end="", flush=True)
        try:
            cap = cv2.VideoCapture(url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            try:
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 4000)
            except Exception:
                pass

            got = False
            for _ in range(ATTEMPTS):
                ret, frm = cap.read()
                if ret and frm is not None:
                    got = True
                    break
                time.sleep(DELAY)
            cap.release()

            if got:
                low_res = _inject_resolution(url, config.WIFI_WIDTH, config.WIFI_HEIGHT)
                print(f"WORKS → {config.WIFI_WIDTH}x{config.WIFI_HEIGHT}")
                logger.info(f"WiFi camera working: {low_res}")
                return low_res
            else:
                print(f"no frames after {ATTEMPTS} attempts")
        except Exception as e:
            print(f"error: {e}")
            logger.debug(f"Probe failed {url}: {e}")

    return None


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CAMERA SELECTION UI                                                 ║
# ╚══════════════════════════════════════════════════════════════════════╝
def select_camera() -> Union[int, str]:
    """
    Interactive camera selection.
    Returns camera index (int) for local webcam
    or stream URL (str) for WiFi camera.
    """
    print("\n" + "═" * 60)
    print("  CAMERA SELECTION")
    print("═" * 60)

    # Enumerate local cameras
    try:
        from cv2_enumerate_cameras import enumerate_cameras
        for cam in enumerate_cameras(cv2.CAP_MSMF):
            print(f"  [{cam.index}] {cam.name}")
    except Exception:
        for i in range(5):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.read()[0]:
                print(f"  [{i}] Camera {i}")
            cap.release()

    print("  [IP] WiFi phone camera (IP Webcam app)")
    print("─" * 60)
    print("  Tip: type IP directly  e.g.  192.168.1.7:8080")
    print("═" * 60)

    def _looks_like_ip(s: str) -> bool:
        s = s.lower().strip()
        return (
            s == "ip"
            or s.startswith("http")
            or bool(re.match(r"^\d{1,3}\.\d{1,3}", s))
        )

    def _normalise(raw: str) -> str:
        raw = raw.strip().strip("[]")
        if raw and not raw.startswith("http"):
            raw = "http://" + raw
        return raw

    def _try_ip(raw_input: str) -> Optional[str]:
        raw = _normalise(raw_input)
        if not raw.startswith("http"):
            print("  Cannot parse — try: 192.168.1.7:8080")
            return None
        return probe_ip_camera(raw)

    while True:
        choice = input("\nEnter camera number OR IP address: ").strip()

        if _looks_like_ip(choice):
            raw_input = (
                input("  IP address (e.g. 192.168.1.7:8080): ").strip()
                if choice.lower() == "ip"
                else choice
            )
            working = _try_ip(raw_input)
            if working:
                return working

            print("\n  No stream found.")
            print("  [R] Retry  [P] Different port  [0] Use laptop webcam")
            fb = input("  Choice: ").strip().upper()
            if fb == "0":
                return 0
            elif fb == "P":
                parsed = urllib.parse.urlparse(_normalise(raw_input))
                port   = input("  Port (e.g. 8080, 4747): ").strip()
                w2     = _try_ip(f"http://{parsed.hostname}:{port}")
                if w2:
                    return w2
        else:
            try:
                idx = int(choice)
                cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                ret, _ = cap.read()
                cap.release()
                if ret:
                    logger.info(f"Local camera selected: index {idx}")
                    return idx
                print(f"  Camera {idx} gave no frame.")
            except ValueError:
                print("  Enter a number (0) or IP (192.168.x.x:8080)")
