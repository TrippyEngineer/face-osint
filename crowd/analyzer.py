"""
crowd/analyzer.py
──────────────────
Per-camera AI analysis pipeline with real-deployment features.

  - YOLOv8n person detection at 5 FPS (CPU-viable, ~6MB model)
  - ByteTrack tracking (persist=True) with per-person history
  - Optical flow every 5th frame (Farneback)
  - Zone density: count / area_m2 → risk scoring
  - Behavioral intelligence: loitering, running, counter-flow detection
  - Attribute toggles: bbox, track ID, suspicious flag, child detection
  - Annotated frame overlay with color-coded risk border
"""

import base64
import logging
import threading
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)

RISK_COLORS = {
    "safe":     (34,  197,  94),
    "caution":  (245, 158,  11),
    "high":     (249, 115,  22),
    "critical": (239,  68,  68),
}
RISK_LABELS = {
    "safe": "SAFE", "caution": "CAUTION", "high": "HIGH RISK", "critical": "CRITICAL"
}

INFERENCE_FPS        = 5     # target inference FPS
FLOW_EVERY_N         = 5     # optical flow frame stride
LOITER_FRAMES        = getattr(config, "CIC_LOITER_FRAMES", 25)   # frames stationary before "loitering"
LOITER_PIXEL_THRESH  = 8     # px movement budget per frame to be "stationary"
RUNNING_SPEED_THRESH = getattr(config, "CIC_RUNNING_SPEED", 12.0) # px/frame velocity to flag as "running"


class CameraAnalyzer:
    def __init__(self, slot_id: int, zone_cfg: dict):
        self.slot_id  = slot_id
        self.zone_cfg = zone_cfg
        self._model   = None
        self._cap     = None
        self._source  = None
        self._active  = False
        self._thread: Optional[threading.Thread] = None
        self._lock    = threading.Lock()
        self._gen     = 0          # capture generation — bumped on every start/stop
                                   # so a stale _run thread can detect it's superseded

        # Per-person tracking history: {track_id: {pos, stationary_count, velocity}}
        self._person_hist:     dict = {}
        self._last_meta:       dict = {}
        self._last_frame:      Optional[np.ndarray] = None
        self._clip_fps   = INFERENCE_FPS
        self._clip_buf   = deque(maxlen=getattr(config, "CIC_CLIP_PRE_S", 10) * INFERENCE_FPS)
        self._recording  = False
        # Operator-triggered manual recording (arbitrary length, start/stop)
        self._manual_recording  = False
        self._manual_frames: Optional[list] = None
        self._manual_max_frames  = getattr(config, "CIC_CLIP_MANUAL_MAX_S", 300) * INFERENCE_FPS
        self._prev_gray:       Optional[np.ndarray] = None
        self._flow_counter     = 0
        self._frame_count      = 0
        # Throttle face-crop extraction: track_id → last capture timestamp
        self._face_capture_ts: dict = {}
        self._face_capture_q:  list = []   # [(crop_bgr, track_id)] pending
        self._face_thread: Optional[threading.Thread] = None

        # Feature toggles — can be updated from Flask route
        self.toggles = {
            "show_bbox":       True,
            "show_track_id":   True,
            "show_suspicious": True,
            "show_children":   True,
            "show_count":      True,
            "show_flow":       True,
        }

    # ── Public API ────────────────────────────────────────────────────────

    def start(self, source) -> bool:
        # Signal any existing capture thread to stop and wait briefly for it, so a
        # stale _run loop can't keep reading a capture we're about to replace.
        old_thread = None
        with self._lock:
            if self._active:
                self._active = False
                self._gen   += 1            # invalidate the running thread
                old_thread   = self._thread
        if old_thread is not None and old_thread.is_alive():
            old_thread.join(timeout=2.0)    # best-effort: a dead-IP-cam read() may outlast this

        cap = cv2.VideoCapture(source if isinstance(source, int) else str(source))
        if not cap.isOpened():
            logger.warning(f"Slot {self.slot_id}: cannot open source '{source}'")
            return False

        # Keep only the freshest frame buffered (best-effort; some FFMPEG/RTSP
        # backends ignore it, so _run also drains to the latest frame). Without
        # this a 25-30 FPS live source read at 5 FPS drifts seconds behind real
        # time and the heatmap/overlay visibly lag live state.
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        with self._lock:
            self._gen   += 1
            my_gen       = self._gen
            self._cap    = cap
            self._source = source
            self._active = True
            # fresh per-run analysis state
            self._last_frame  = None
            self._last_meta   = {}
            self._prev_gray   = None
            self._person_hist = {}
            self._frame_count = 0
            self._manual_recording = False
            self._manual_frames    = None

        # Pre-warm YOLO in a separate daemon thread so frames flow immediately
        threading.Thread(target=self._load_model, daemon=True,
                         name=f"CIC-yolo{self.slot_id}").start()

        # Each _run owns its capture (passed in) + generation and releases the
        # capture itself on exit — stop()/replacement never release a capture out
        # from under a thread that may be parked in cap.read().
        self._thread = threading.Thread(
            target=self._run, args=(cap, my_gen), daemon=True,
            name=f"CIC-slot{self.slot_id}"
        )
        self._thread.start()
        logger.info(f"Slot {self.slot_id}: started — source={source}")
        return True

    def stop(self):
        with self._lock:
            old_thread = self._thread
            self._stop_locked()
        if old_thread is not None and old_thread.is_alive():
            old_thread.join(timeout=2.0)    # best-effort; the thread releases its own capture on exit
        logger.info(f"Slot {self.slot_id}: stopped")

    def is_active(self) -> bool:
        with self._lock:
            return self._active

    def get_overlay_frame_b64(self) -> Optional[str]:
        with self._lock:
            frame = self._last_frame
        if frame is None:
            return None
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
        if not ok:
            return None
        return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()

    def get_meta(self) -> dict:
        with self._lock:
            return dict(self._last_meta)

    def record_incident(self, on_done) -> bool:
        """Snapshot the pre-buffer, keep appending for CIC_CLIP_POST_S, encode an
        mp4 in a background thread, then call on_done(path:str). Returns False if a
        recording is already in flight or clips disabled."""
        if not getattr(config, "CIC_CLIPS_ENABLED", True):
            return False
        with self._lock:
            if self._recording:
                return False
            self._recording = True
            pre = list(self._clip_buf)

        def _worker():
            from crowd.clip import incident_clip_path, write_clip
            post_n = getattr(config, "CIC_CLIP_POST_S", 10) * self._clip_fps
            collected = []
            for _ in range(post_n):
                with self._lock:
                    if not self._active:
                        break
                    lf = self._last_frame
                if lf is not None:
                    collected.append(lf.copy())
                time.sleep(1.0 / self._clip_fps)
            frames = pre + collected
            path = incident_clip_path(getattr(config, "CIC_INCIDENT_DIR"),
                                      self.zone_cfg.get("id", f"zone_{self.slot_id}"))
            ok = write_clip(frames, path, self._clip_fps)
            with self._lock:
                self._recording = False
            if ok:
                try:
                    on_done(str(path))
                except Exception:
                    pass

        threading.Thread(target=_worker, daemon=True,
                         name=f"CIC-clip{self.slot_id}").start()
        return True

    def snapshot(self):
        """Return a copy of the latest annotated frame (BGR), or None if idle."""
        with self._lock:
            f = self._last_frame
        return f.copy() if f is not None else None

    def is_manual_recording(self) -> bool:
        with self._lock:
            return self._manual_recording

    def start_manual_recording(self) -> bool:
        """Begin an operator-triggered recording, seeded with the pre-buffer so
        the moment just before the click is included. Returns False if the slot
        is inactive or a recording is already running."""
        with self._lock:
            if not self._active or self._manual_recording:
                return False
            self._manual_frames    = list(self._clip_buf)
            self._manual_recording = True
        return True

    def stop_manual_recording(self, on_done) -> bool:
        """Stop the manual recording, encode the captured frames to an mp4 in a
        background thread, then call on_done(path:str). Returns False if not
        recording or nothing was captured."""
        with self._lock:
            if not self._manual_recording:
                return False
            self._manual_recording = False
            frames = self._manual_frames or []
            self._manual_frames = None
        if not frames:
            return False

        def _worker():
            from crowd.clip import incident_clip_path, write_clip
            path = incident_clip_path(getattr(config, "CIC_INCIDENT_DIR"),
                                      self.zone_cfg.get("id", f"zone_{self.slot_id}"))
            ok = write_clip(list(frames), path, self._clip_fps)
            if ok:
                try:
                    on_done(str(path))
                except Exception:
                    pass

        threading.Thread(target=_worker, daemon=True,
                         name=f"CIC-mclip{self.slot_id}").start()
        return True

    # ── Background thread ─────────────────────────────────────────────────

    def _stop_locked(self):
        # Do NOT release self._cap here: the owning _run thread releases its own
        # capture on exit (in its finally). Releasing it from another thread while
        # _run may be parked inside cap.read() is undefined behavior. Bumping the
        # generation makes the running thread observe the stop and exit.
        self._active = False
        self._gen   += 1
        self._cap    = None          # drop our reference; the thread owns the object
        self._last_frame     = None
        self._last_meta      = {}
        self._prev_gray      = None
        self._person_hist    = {}
        self._frame_count    = 0
        self._manual_recording = False
        self._manual_frames    = None

    def _run(self, cap, my_gen):
        interval = 1.0 / INFERENCE_FPS
        is_file  = isinstance(self._source, str) and not self._source.startswith("http")

        try:
            while True:
                with self._lock:
                    if not self._active or self._gen != my_gen:
                        break

                t0  = time.time()
                ret, frame = cap.read()

                # Live sources buffer frames FIFO; consuming at 5 FPS while the
                # camera pushes 25-30 FPS makes cap.read() return progressively
                # staler frames, so the heatmap/overlay drift seconds behind real
                # time. Drain the already-buffered backlog to the newest frame.
                # Buffered grabs return near-instantly; the loop stops once a
                # grab has to wait on a fresh frame (budget exceeded), so it
                # costs at most ~one extra frame-time per cycle. Files are left
                # to play linearly.
                if ret and not is_file:
                    _dt = time.time()
                    while time.time() - _dt < 0.015:
                        if not cap.grab():
                            break
                        ok2, f2 = cap.retrieve()
                        if ok2 and f2 is not None:
                            frame = f2
                        else:
                            break

                if not ret or frame is None:
                    if is_file:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        time.sleep(0.1)
                        continue

                # Downscale for inference, preserving aspect ratio (no upscaling).
                # Forcing 640×480 squished non-4:3 footage and threw away the
                # resolution YOLO needs to spot small/distant people.
                _h0, _w0 = frame.shape[:2]
                _target  = getattr(config, "CIC_INFERENCE_SIZE", 960)
                _scale   = _target / max(_h0, _w0)
                if _scale < 1.0:
                    frame = cv2.resize(frame, (int(_w0 * _scale), int(_h0 * _scale)))
                self._frame_count += 1

                # Buffer raw (pre-overlay) frames for incident clips
                try:
                    self._clip_buf.append(frame.copy())
                except Exception:
                    pass

                # Manual recording: keep appending the raw frame until the
                # operator stops, or until the length cap is hit (then stop
                # growing but keep what's captured so Stop still saves it).
                if self._manual_recording:
                    with self._lock:
                        if (self._manual_recording and self._manual_frames is not None
                                and len(self._manual_frames) < self._manual_max_frames):
                            try:
                                self._manual_frames.append(frame.copy())
                            except Exception:
                                pass

                try:
                    meta, annotated = self._analyze(frame)
                except Exception as e:
                    logger.debug(f"Slot {self.slot_id} analyze error: {e}")
                    meta, annotated = {}, frame

                with self._lock:
                    if self._gen != my_gen:
                        break                    # superseded mid-cycle; don't clobber new state
                    self._last_meta  = meta
                    self._last_frame = annotated

                elapsed = time.time() - t0
                time.sleep(max(0.0, interval - elapsed))
        finally:
            try:
                cap.release()
            except Exception:
                pass
            logger.debug(f"Slot {self.slot_id}: thread exited (gen {my_gen})")

    # ── Core analysis ─────────────────────────────────────────────────────

    def _analyze(self, frame: np.ndarray):
        # Non-blocking: use model only if already loaded; background thread handles loading
        _m = self._model
        model = _m if (_m and _m is not False) else None
        detections = []
        h, w = frame.shape[:2]

        if model is not None:
            try:
                results = model.track(
                    frame, classes=[0], persist=True, verbose=False,
                    stream=False, iou=0.45,
                    conf=getattr(config, "CIC_YOLO_CONF", 0.25),
                    imgsz=getattr(config, "CIC_INFERENCE_SIZE", 960),
                    max_det=getattr(config, "CIC_MAX_DET", 1000),
                )
                if results and results[0].boxes is not None:
                    boxes = results[0].boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        trk_id = int(box.id[0]) if box.id is not None else -1
                        cx = (x1 + x2) / 2
                        cy = (y1 + y2) / 2
                        bh = y2 - y1

                        # Behavioral analysis using track history
                        behavior = self._update_person(trk_id, cx, cy, w, h)

                        detections.append({
                            "xyxy":     (int(x1), int(y1), int(x2), int(y2)),
                            "conf":     float(box.conf[0]),
                            "id":       trk_id,
                            "center":   (cx, cy),
                            "height_px": bh,
                            "is_child":  bh < h * getattr(config, "CIC_CHILD_HEIGHT_RATIO", 0.22),
                            **behavior,
                        })
            except Exception as e:
                logger.debug(f"Slot {self.slot_id} YOLO error: {e}")

        # Queue face crops for Khoya-Paya background extraction
        if detections:
            self._queue_face_crops(frame, detections)

        # Prune stale track history
        active_ids = {d["id"] for d in detections}
        stale = [k for k in self._person_hist if k not in active_ids and k != -1]
        for k in stale:
            del self._person_hist[k]

        # Optical flow
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow_data = self._compute_flow(gray)

        # Dense-crowd tiling: recount via overlapping tiles (config-gated).
        # Full-frame track() above still drives behavior flags + track IDs;
        # tiling only produces a more accurate count/density/heatmap.
        tiled_centers = None
        if model is not None and getattr(config, "CIC_TILING", False):
            try:
                from crowd.tiling import generate_tiles, merge_boxes
                tiles = generate_tiles(w, h,
                                       getattr(config, "CIC_TILE_GRID", "2x2"),
                                       getattr(config, "CIC_TILE_OVERLAP", 0.2))
                tboxes = []
                for (tx1, ty1, tx2, ty2) in tiles:
                    sub = frame[ty1:ty2, tx1:tx2]
                    if sub.size == 0:
                        continue
                    res = model(sub, classes=[0], verbose=False,
                                conf=getattr(config, "CIC_YOLO_CONF", 0.25),
                                max_det=getattr(config, "CIC_MAX_DET", 1000))
                    if res and res[0].boxes is not None:
                        for b in res[0].boxes:
                            x1, y1, x2, y2 = b.xyxy[0].tolist()
                            tboxes.append((x1 + tx1, y1 + ty1, x2 + tx1, y2 + ty1,
                                           float(b.conf[0])))
                merged = merge_boxes(tboxes, getattr(config, "CIC_TILE_NMS_IOU", 0.5))
                tiled_centers = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in merged]
            except Exception as e:
                logger.debug(f"Slot {self.slot_id} tiling error: {e}")

        # Zone density + risk. Tiling (if on) already produces a dense count;
        # otherwise the counting head can occlusion-correct the raw detector count.
        if tiled_centers is not None:
            count = len(tiled_centers)
            count_method = "tiling"
        else:
            from crowd import counting
            _ce = counting.estimate(
                [d["xyxy"] for d in detections if d.get("xyxy")], w, h,
                backend=getattr(config, "CIC_COUNTING", "detector"),
                occlusion_gain=getattr(config, "CIC_OCCLUSION_GAIN", 1.5),
                max_factor=getattr(config, "CIC_OCCLUSION_MAX_FACTOR", 2.5),
            )
            count        = _ce["count"]
            count_method = _ce["method"]
        # Density must use the camera's visible area, not the whole zone area,
        # otherwise count/area is microscopic and risk never escalates.
        fov_area = self.zone_cfg.get("fov_area_m2") or getattr(config, "CIC_FOV_AREA_M2", 100.0)
        density  = count / fov_area
        thresh  = self.zone_cfg.get("thresholds", {"caution": 1.5, "high": 3.0, "critical": 6.0})
        risk    = self._risk_level(density, thresh)

        # Crowd-PRESSURE early-warning: a density threshold alone is not stampede
        # prediction. Helbing pressure (density × velocity-variance) + turbulence
        # escalate the risk earlier. Velocities are already tracked per person.
        pa = {}
        if getattr(config, "CIC_PRESSURE_ENABLED", True):
            from crowd import pressure as _pressure
            velocities = [d.get("velocity", 0.0) for d in detections if d.get("id", -1) >= 0]
            pa = _pressure.assess(
                density, velocities,
                dense_density=getattr(config, "CIC_DENSE_DENSITY", 2.0),
                compression_density=getattr(config, "CIC_COMPRESSION_DENSITY", 5.0),
                critical_density=getattr(config, "CIC_CRITICAL_DENSITY", 8.0),
                turbulence_cv=getattr(config, "CIC_TURBULENCE_CV", 0.75),
            )
            risk = _pressure.escalate_risk(risk, pa["crowd_state"])

        # Suspicious persons count
        n_suspicious = sum(1 for d in detections if d.get("suspicious"))
        n_running    = sum(1 for d in detections if d.get("running"))
        n_loitering  = sum(1 for d in detections if d.get("loitering"))
        n_children   = sum(1 for d in detections if d.get("is_child"))

        meta = {
            "slot":          self.slot_id,
            "zone_id":       self.zone_cfg.get("id", f"zone_{self.slot_id}"),
            "zone_name":     self.zone_cfg.get("name", f"Zone {self.slot_id}"),
            "count":         count,
            "count_method":  count_method,
            "density":       round(density, 4),
            "risk":          risk,
            "flow":          flow_data,
            "timestamp":     time.time(),
            "detections":    detections,
            "n_suspicious":  n_suspicious,
            "n_running":     n_running,
            "n_loitering":   n_loitering,
            "n_children":    n_children,
            # Crowd-pressure early-warning fields (crowd/pressure.py)
            "pressure":      pa.get("pressure", 0.0),
            "pressure_cv":   pa.get("pressure_cv", 0.0),
            "los":           pa.get("los", "A"),
            "crowd_state":   pa.get("crowd_state", "normal"),
            "turbulence":    pa.get("turbulence", False),
            # Person coordinates for heat map (normalized 0-1)
            "heatmap_pts": [
                [round(cx / w, 3), round(cy / h, 3)]
                for (cx, cy) in (tiled_centers if tiled_centers is not None
                                 else [d["center"] for d in detections])
            ],
        }

        annotated = self._draw_overlay(frame.copy(), meta)
        return meta, annotated

    def _update_person(self, tid: int, cx: float, cy: float, fw: int, fh: int) -> dict:
        """Track per-person position history → derive behavioral flags."""
        if tid < 0:
            return {"suspicious": False, "loitering": False, "running": False, "velocity": 0.0}

        hist = self._person_hist.get(tid)
        if hist is None:
            self._person_hist[tid] = {"pos": (cx, cy), "stationary": 0, "velocity": 0.0}
            return {"suspicious": False, "loitering": False, "running": False, "velocity": 0.0}

        px, py  = hist["pos"]
        dist    = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
        velocity = dist  # pixels per frame at 5 FPS

        stationary = hist["stationary"]
        if dist < LOITER_PIXEL_THRESH:
            stationary += 1
        else:
            stationary = max(0, stationary - 2)

        loitering = stationary >= LOITER_FRAMES
        running   = velocity >= RUNNING_SPEED_THRESH
        suspicious = loitering or running

        self._person_hist[tid] = {
            "pos":        (cx, cy),
            "stationary": stationary,
            "velocity":   velocity,
        }
        return {
            "suspicious": suspicious,
            "loitering":  loitering,
            "running":    running,
            "velocity":   round(velocity, 1),
        }

    def _compute_flow(self, gray: np.ndarray) -> dict:
        self._flow_counter += 1
        if self._flow_counter % FLOW_EVERY_N != 0 or self._prev_gray is None:
            self._prev_gray = gray
            return {"dx": 0.0, "dy": 0.0, "speed": 0.0}
        try:
            flow = cv2.calcOpticalFlowFarneback(
                self._prev_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
            )
            dx = float(np.mean(flow[..., 0]))
            dy = float(np.mean(flow[..., 1]))
            self._prev_gray = gray
            return {"dx": round(dx, 3), "dy": round(dy, 3),
                    "speed": round((dx**2 + dy**2) ** 0.5, 3)}
        except Exception:
            self._prev_gray = gray
            return {"dx": 0.0, "dy": 0.0, "speed": 0.0}

    @staticmethod
    def _risk_level(density: float, thresh: dict) -> str:
        if density >= thresh.get("critical", 6.0):
            return "critical"
        if density >= thresh.get("high", 3.0):
            return "high"
        if density >= thresh.get("caution", 1.5):
            return "caution"
        return "safe"

    # ── Overlay rendering ─────────────────────────────────────────────────

    def _draw_overlay(self, frame: np.ndarray, meta: dict) -> np.ndarray:
        risk   = meta["risk"]
        count  = meta["count"]
        color  = RISK_COLORS[risk]
        h, w   = frame.shape[:2]
        tg     = self.toggles

        # Per-person bounding boxes
        if tg.get("show_bbox", True):
            for d in meta["detections"]:
                x1, y1, x2, y2 = d["xyxy"]
                is_susp  = d.get("suspicious") and tg.get("show_suspicious", True)
                is_child = d.get("is_child")    and tg.get("show_children",  True)

                if is_susp:
                    box_col = (0, 85, 255)     # orange-red (BGR)
                elif is_child:
                    box_col = (255, 200, 0)    # cyan-ish
                else:
                    box_col = color

                thick = 2 if is_susp else 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_col, thick, cv2.LINE_AA)

                # Track ID label
                if tg.get("show_track_id", True):
                    tid = d["id"]
                    if tid >= 0:
                        label  = f"#{tid}"
                        if is_child:
                            label += " C"
                        elif d.get("running"):
                            label += " RUN"
                        elif d.get("loitering"):
                            label += " LOITER"
                        cv2.putText(frame, label, (x1, max(y1 - 3, 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, box_col, 1, cv2.LINE_AA)

        # Suspicious person count badge (top-right warning)
        n_susp = meta.get("n_suspicious", 0)
        if n_susp > 0 and tg.get("show_suspicious", True):
            warn = f"! {n_susp} SUSP"
            tw, _ = cv2.getTextSize(warn, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0], 0
            cv2.putText(frame, warn, (w - 120, 44),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 85, 255), 2, cv2.LINE_AA)

        # Risk border
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 4, cv2.LINE_AA)

        # Count + density badge (top-left)
        if tg.get("show_count", True):
            cv2.rectangle(frame, (4, 4), (200, 58), (0, 0, 0), -1)
            cv2.putText(frame, f"Count: {count}", (9, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"{meta['density']:.3f} p/m2", (9, 44),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

        # Risk label (top-right)
        label = RISK_LABELS[risk]
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.putText(frame, label, (w - tw - 8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

        # Optical flow arrow (center)
        flow = meta.get("flow", {})
        if tg.get("show_flow", True) and flow.get("speed", 0) > 0.5:
            cx_, cy_ = w // 2, h // 2
            spd = flow["speed"]
            scale = min(60.0, spd * 10)
            ex = int(cx_ + flow["dx"] / max(0.001, spd) * scale)
            ey = int(cy_ + flow["dy"] / max(0.001, spd) * scale)
            cv2.arrowedLine(frame, (cx_, cy_), (ex, ey),
                            (255, 220, 0), 2, cv2.LINE_AA, tipLength=0.3)

        # Zone name (bottom-left)
        cv2.putText(frame, meta.get("zone_name", ""),
                    (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (180, 180, 180), 1, cv2.LINE_AA)

        # Children count (bottom-right)
        n_child = meta.get("n_children", 0)
        if n_child > 0 and tg.get("show_children", True):
            cv2.putText(frame, f"Children: {n_child}",
                        (w - 105, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        (255, 200, 0), 1, cv2.LINE_AA)

        return frame

    # ── Model loading ──────────────────────────────────────────────────────

    def _load_model(self):
        if self._model is not None:
            return self._model if self._model is not False else None
        try:
            from ultralytics import YOLO
            m = YOLO(getattr(config, "CIC_YOLO_MODEL", "yolov8n.pt"))
            m.overrides["verbose"] = False
            self._model = m
            logger.info(f"Slot {self.slot_id}: YOLOv8n loaded")
        except Exception as e:
            logger.warning(f"Slot {self.slot_id}: YOLOv8n unavailable ({e})")
            self._model = False
        return self._model if self._model else None

    # ── Face crop extraction for Khoya-Paya ───────────────────────────────

    FACE_CAPTURE_INTERVAL = 60   # seconds between captures per track_id

    def _queue_face_crops(self, frame: np.ndarray, detections: list):
        """Queue face crops for background DeepFace embedding extraction."""
        now = time.time()
        h, w = frame.shape[:2]
        for d in detections:
            tid = d.get("id", -1)
            if tid < 0:
                continue
            last = self._face_capture_ts.get(tid, 0)
            if now - last < self.FACE_CAPTURE_INTERVAL:
                continue
            self._face_capture_ts[tid] = now
            x1, y1, x2, y2 = d["xyxy"]
            # Crop the upper 50% of bbox (head region)
            head_h = max(30, int((y2 - y1) * 0.5))
            crop = frame[y1:y1 + head_h, x1:x2].copy()
            if crop.size == 0:
                continue
            if min(crop.shape[:2]) < getattr(config, "CIC_FACE_MIN_CROP_PX", 70):
                continue   # too small to yield a usable face embedding
            self._face_capture_q.append((crop, tid))
            if len(self._face_capture_q) > 20:   # safety cap
                self._face_capture_q.pop(0)

        # Start worker thread if idle
        if self._face_capture_q and (
            self._face_thread is None or not self._face_thread.is_alive()
        ):
            self._face_thread = threading.Thread(
                target=self._face_worker, daemon=True,
                name=f"CIC-face{self.slot_id}"
            )
            self._face_thread.start()

    def _face_worker(self):
        """Background thread: extract embeddings from queued crops and store."""
        from storage.database import Database
        try:
            import embedding as emb_mod
        except ImportError:
            return
        db = Database()
        zone_id   = self.zone_cfg.get("id",   f"zone_{self.slot_id}")
        zone_name = self.zone_cfg.get("name", f"Zone {self.slot_id}")
        # CIC-path-only capture quality controls (config CIC_FACE_*). enforce=True
        # drops crops with no detectable face so the Khoya index stays clean (a
        # sharp selfie can then actually match). These do NOT affect the OSINT
        # face pipeline, which calls extract() with no overrides.
        _enforce  = True if getattr(config, "CIC_FACE_ENFORCE", True) else None
        _detector = getattr(config, "CIC_FACE_DETECTOR", "") or None
        _diag     = getattr(config, "CIC_FACE_DIAG", False)

        while self._face_capture_q:
            try:
                crop, tid = self._face_capture_q.pop(0)
            except IndexError:
                break
            try:
                # Crowd cameras produce small person boxes; upscale so DeepFace
                # opencv detector can reliably find the face (needs ≥160px).
                h_c, w_c = crop.shape[:2]
                min_dim = min(h_c, w_c)
                if min_dim < 160:
                    scale = 160 / min_dim
                    crop = cv2.resize(
                        crop,
                        (max(160, int(w_c * scale)), max(160, int(h_c * scale))),
                        interpolation=cv2.INTER_LANCZOS4,
                    )
                # TEMP diagnostic (CIC_FACE_DIAG): does a STRICT detector find a
                # real face in this crop? Reveals how many stored embeddings are
                # genuine faces vs whole-crop fallbacks. Logged only; remove the
                # flag once the quality question is settled.
                if _diag:
                    try:
                        from deepface import DeepFace
                        DeepFace.represent(
                            img_path=crop, model_name=config.DEEPFACE_MODEL,
                            detector_backend=config.DEEPFACE_DETECTOR,
                            enforce_detection=True, align=True)
                        _strict = "FACE"
                    except Exception:
                        _strict = "NOFACE"

                result = emb_mod.extract(crop, enforce=_enforce, detector=_detector)
                _stored = bool(result and result.get("embedding") is not None)
                if _diag:
                    logger.info(
                        f"[FACEDIAG] slot{self.slot_id} track#{tid} "
                        f"crop={tuple(crop.shape[:2])} strict={_strict} "
                        f"stored={'Y' if _stored else 'N'} "
                        f"conf={(result or {}).get('confidence')}")
                if _stored:
                    import numpy as np
                    vec = np.array(result["embedding"], dtype=np.float32)
                    db.store_cic_capture(tid, self.slot_id, zone_id, zone_name, vec)
                    logger.debug(
                        f"Slot {self.slot_id}: stored face for track#{tid} in zone {zone_name}"
                    )
            except Exception as e:
                logger.debug(f"Slot {self.slot_id} face extract fail for #{tid}: {e}")
