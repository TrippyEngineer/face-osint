"""
Face OSINT System  —  main.py
══════════════════════════════
Windows-native. No Docker. No Redis. No external services required.
Runs on limited hardware — single process, controlled threading.

Architecture:
  Main thread     → OpenCV display loop (must own the GUI)
  FrameReader     → daemon thread, drains camera buffer
  SearchWorker    → daemon thread pool, runs scrapers + aggregation
  SQLite          → single file, zero-config persistence

Controls (click the camera window first):
  SPACE   — capture current frame, enter name, start OSINT search
  Q       — quit
  F       — flip / mirror feed
  D       — diagnostic: show face distances for person in frame
  H       — toggle HUD (help / status overlay)

Search pipeline (background thread):
  1. embedding.extract()              → 512D face vector
  2. database.find_similar_faces()    → check if seen before
  3. folder_writer.create_folder()    → write placeholder immediately
  4. scrapers (ThreadPoolExecutor)    → parallel scraping
  5. face_matcher.score_all_results() → compare scraped photos vs query
  6. scorer.score_all()               → compute combined scores
  7. resolver.resolve()               → entity resolution
  8. folder_writer.write_results()    → final info.txt + JSON
  9. database.insert_match() ×N       → persist to SQLite
"""

import os
import sys
import base64
import time
import uuid
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, wait as fut_wait, FIRST_COMPLETED
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ── Project imports ────────────────────────────────────────────────────────
import config
from config import setup_logging
import camera
import embedding
from storage.database     import Database
from storage.folder_writer import FolderWriter
from aggregator            import face_matcher, scorer, resolver

logger = logging.getLogger(__name__)

# ── Colours (BGR) ──────────────────────────────────────────────────────────
C_GREEN  = (0,   230, 100)
C_CYAN   = (200, 200,   0)
C_YELLOW = (0,   200, 220)
C_RED    = (0,    40, 220)
C_WHITE  = (255, 255, 255)
C_DARK   = (16,   16,  16)
C_GREY   = (120, 120, 120)

HUD_H        = 100   # pixels reserved at top for status overlay
PANEL_W      = 280   # sidebar width
WIN_NAME     = "Face OSINT  |  SPACE=search  F=flip  D=diag  Q=quit"


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  APPLICATION STATE  (thread-safe)                                    ║
# ╚══════════════════════════════════════════════════════════════════════╝
class AppState:
    def __init__(self):
        self._lock      = threading.Lock()
        self._status    = "Ready — press SPACE to capture"
        self._searching = False
        self._flip      = False
        self._show_hud  = True
        self._history   = []    # list of completed search summaries

    # ── Status ──────────────────────────────────────────────────────────
    @property
    def status(self):
        with self._lock: return self._status

    @status.setter
    def status(self, v):
        with self._lock:
            self._status = str(v)
            logger.info(f"Status: {v}")

    # ── Search slot ─────────────────────────────────────────────────────
    def try_start_search(self) -> bool:
        with self._lock:
            if self._searching:
                return False
            self._searching = True
            return True

    def done_search(self, summary: Optional[dict] = None):
        with self._lock:
            self._searching = False
            if summary:
                self._history.insert(0, summary)
                self._history = self._history[:20]  # keep last 20

    @property
    def is_searching(self):
        with self._lock: return self._searching

    # ── UI toggles ──────────────────────────────────────────────────────
    def toggle_flip(self):
        with self._lock:
            self._flip = not self._flip
            return self._flip

    @property
    def flip(self):
        with self._lock: return self._flip

    def toggle_hud(self):
        with self._lock:
            self._show_hud = not self._show_hud

    @property
    def show_hud(self):
        with self._lock: return self._show_hud

    def get_history(self):
        with self._lock: return list(self._history)


state = AppState()


# ── Helper functions (defined before run_search so order is explicit) ──────
def db_update_folder(db: Database, search_id: str, folder: str):
    """Update the output_folder field after it's been created."""
    import sqlite3
    with sqlite3.connect(str(db.path)) as conn:
        conn.execute(
            "UPDATE searches SET output_folder=? WHERE id=?",
            (folder, search_id)
        )


def _run_scraper(module_path: str, fn_name: str, context: dict) -> dict:
    """Import and call one scraper function. Isolated — exceptions caught upstream."""
    import importlib
    mod = importlib.import_module(module_path)
    fn  = getattr(mod, fn_name)
    return fn(context)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  SEARCH WORKER                                                        ║
# ║  Runs entirely in a background daemon thread.                        ║
# ║  Never touches the OpenCV window — all output goes to disk + DB.    ║
# ╚══════════════════════════════════════════════════════════════════════╝
def run_search(
    frame:     np.ndarray,
    name:      str,
    db:        Database,
    writer:    FolderWriter,
):
    """
    Full OSINT pipeline for one captured face.
    Called from a background thread — never blocks the display loop.
    """
    search_id = str(uuid.uuid4())
    ts_start  = time.time()

    try:
        state.status = f"Extracting face embedding..."
        logger.info(f"Search {search_id[:8]} started for '{name}'")

        # ── Step 1: Extract embedding ─────────────────────────────────
        result = embedding.extract(frame)
        if result is None:
            state.status = "No face detected — try again closer to camera"
            logger.warning(f"Search {search_id[:8]}: no face detected")
            state.done_search()
            return

        emb       = result["embedding"]
        face_crop = result["face_crop"]
        conf      = result["confidence"]
        logger.info(
            f"Search {search_id[:8]}: embedding extracted "
            f"conf={conf:.3f} bbox={result['bbox']}"
        )

        # ── Step 2: Check for previously seen faces ───────────────────
        state.status = "Checking face database..."
        similar = db.find_similar_faces(emb, top_k=3)
        if similar:
            top = similar[0]
            logger.info(
                f"Search {search_id[:8]}: similar face found "
                f"'{top['name']}' score={top['score']:.3f}"
            )
            state.status = f"Similar to previous search: {top['name']} ({top['score']:.2f})"
            time.sleep(1.5)   # let user see the message

        # ── Step 3: Create output folder immediately ──────────────────
        state.status = "Creating output folder..."
        db.create_search(search_id, name, output_folder="")
        folder = writer.create_folder(
            name      = name,
            search_id = search_id,
            frame     = frame,
            face_crop = face_crop,
        )
        db_update_folder(db, search_id, str(folder))
        logger.info(f"Search {search_id[:8]}: folder → {folder}")

        # ── Step 4: Store face vector ─────────────────────────────────
        db.store_vector(search_id, name, emb)

        # ── Step 5: Build scraper context ─────────────────────────────
        face_crop_path = folder / "face_crop.jpg"
        image_b64 = ""
        if face_crop_path.exists():
            image_b64 = (
                "data:image/jpeg;base64,"
                + base64.b64encode(face_crop_path.read_bytes()).decode()
            )

        context = {
            "name":           name,
            "company":        "",
            "location":       "",
            "embedding":      emb,
            "face_crop_path": str(face_crop_path),
            "image_b64":      image_b64,
        }

        # ── Step 6: Run all scrapers (parallel, resource-limited) ─────
        # Max 4 workers to stay within limited RAM budget.
        # Each scraper is isolated — one crash never kills others.
        # Per-scraper deadlines match app.py: reverse_face=90s, username=50s, etc.
        SCRAPERS = [
            # (label, module, function, context, per-scraper timeout seconds)
            ("reverse_face",   "scrapers.reverse_face",   "scrape",        context, 90),
            ("search_engines", "scrapers.search_engines", "scrape",        context, 25),
            ("academic",       "scrapers.academic",       "scrape",        context, 35),
            ("github",         "scrapers.platforms",      "scrape_github", context, 20),
            ("reddit",         "scrapers.platforms",      "scrape_reddit", context, 15),
            ("passive",        "scrapers.passive",        "scrape",        context, 20),
            ("username",       "scrapers.username",       "scrape",        context, 50),
        ]

        all_results = {}
        state.status = f"Scraping {len(SCRAPERS)} sources..."

        _t0 = time.time()
        _fut_dl: dict = {}   # future → its individual deadline

        with ThreadPoolExecutor(max_workers=4) as pool:
            futs: dict = {}
            for lbl, mod, fn, ctx, timeout_s in SCRAPERS:
                f = pool.submit(_run_scraper, mod, fn, ctx)
                futs[f]    = lbl
                _fut_dl[f] = _t0 + timeout_s
            pending = set(futs.keys())

            while pending:
                now = time.time()
                # Cancel any future that has exceeded its individual deadline
                for f in list(pending):
                    if now >= _fut_dl[f]:
                        lbl = futs[f]
                        f.cancel()
                        elapsed = int(now - _t0)
                        all_results[lbl] = {"matches": [], "error": "Timed out"}
                        logger.warning(
                            f"Search {search_id[:8]}: {lbl} timed out after {elapsed}s"
                        )
                        state.status = f"Timed out: {lbl}"
                        pending.discard(f)
                if not pending:
                    break

                # Poll until the nearest scraper deadline or next completion
                next_dl  = min(_fut_dl[f] for f in pending) - time.time()
                wait_t   = min(1.5, max(0.05, next_dl))
                done_futs, pending = fut_wait(
                    pending, timeout=wait_t, return_when=FIRST_COMPLETED
                )
                for f in done_futs:
                    lbl = futs[f]
                    try:
                        all_results[lbl] = f.result()
                        n = len(all_results[lbl].get("matches", []))
                        state.status = f"Done: {lbl} ({n} results)"
                        logger.info(f"Search {search_id[:8]}: {lbl} → {n} matches")
                    except Exception as e:
                        all_results[lbl] = {"matches": [], "error": str(e)}
                        logger.warning(f"Search {search_id[:8]}: {lbl} failed — {e}")

        # ── Step 7: Face-match scraped photos vs query embedding ──────
        state.status = "Comparing faces in scraped photos..."
        all_results  = face_matcher.score_all_results(all_results, emb)

        # ── Step 8: Score every match ─────────────────────────────────
        state.status = "Scoring matches..."
        flat_matches = [
            m for _s, d in all_results.items() if isinstance(d, dict)
            for m in d.get("matches", [])
        ]
        scored = scorer.score_all(flat_matches, query_name=name)

        # ── Step 9: Entity resolution ──────────────────────────────────
        state.status = "Resolving identity..."
        identity      = resolver.resolve(name, scored)

        # ── Step 10: Write info.txt + JSON ────────────────────────────
        state.status = "Writing report..."
        writer.write_results(
            folder      = folder,
            name        = name,
            search_id   = search_id,
            all_results = all_results,
            identity    = identity,
        )

        # ── Step 11: Persist matches to SQLite ────────────────────────
        db.complete_search(
            search_id,
            verdict       = identity.get("verdict", "unknown"),
            combined_score = identity.get("combined_score", 0.0),
        )
        for match in scored[:50]:   # store top 50
            db.insert_match(search_id, match.get("source", "unknown"), match)

        # ── Done ──────────────────────────────────────────────────────
        elapsed = time.time() - ts_start
        verdict = identity.get("verdict", "unknown").upper()
        score   = identity.get("combined_score", 0.0)
        n_total = sum(
            len(v.get("matches", []))
            for k, v in all_results.items()
            if isinstance(v, dict)
        )

        summary = {
            "name":    name,
            "verdict": verdict,
            "score":   score,
            "matches": n_total,
            "folder":  str(folder),
            "time":    datetime.now().strftime("%H:%M:%S"),
        }
        state.done_search(summary)
        state.status = (
            f"Done: {name} | {verdict} | score={score:.2f} | "
            f"{n_total} results | {elapsed:.0f}s"
        )
        logger.info(
            f"Search {search_id[:8]} COMPLETE — "
            f"verdict={verdict} score={score:.3f} "
            f"matches={n_total} elapsed={elapsed:.1f}s "
            f"folder={folder}"
        )

    except Exception as e:
        state.done_search()
        state.status = f"Search error — see logs"
        logger.exception(f"Search {search_id[:8]} unhandled exception: {e}")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  NAME INPUT — OpenCV overlay, no tkinter dependency                  ║
# ╚══════════════════════════════════════════════════════════════════════╝
def get_name_input(frozen_frame: np.ndarray) -> Optional[str]:
    """
    Show a name input overlay on top of the frozen camera frame.
    Returns the entered name, or None if cancelled.
    """
    h, w  = frozen_frame.shape[:2]
    name  = ""
    panel = frozen_frame.copy()

    # Dark overlay bar
    overlay = panel.copy()
    cv2.rectangle(overlay, (0, h // 2 - 65), (w, h // 2 + 65), C_DARK, -1)
    cv2.addWeighted(overlay, 0.82, panel, 0.18, 0, panel)

    while True:
        display = panel.copy()
        cv2.putText(display, "Enter name and press ENTER",
                    (20, h // 2 - 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, C_CYAN, 1, cv2.LINE_AA)
        cv2.putText(display, f"> {name}_",
                    (20, h // 2 + 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.82, C_WHITE, 2, cv2.LINE_AA)
        cv2.putText(display, "ESC to cancel",
                    (20, h // 2 + 54), cv2.FONT_HERSHEY_SIMPLEX,
                    0.40, C_GREY, 1, cv2.LINE_AA)
        cv2.imshow(WIN_NAME, display)

        key = cv2.waitKey(50) & 0xFF
        if key == 13:    # ENTER
            n = name.strip()
            return n if n else None
        elif key == 27:  # ESC
            return None
        elif key == 8 or key == 127:  # BACKSPACE
            name = name[:-1]
        elif 32 <= key <= 126:
            name += chr(key)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  DIAGNOSTIC — press D to see face distances for current frame        ║
# ╚══════════════════════════════════════════════════════════════════════╝
def run_diagnostic(frame: np.ndarray, db: Database):
    """Print face similarity against all stored vectors to console + log."""
    print("\n" + "═" * 58)
    print("  DIAGNOSTIC — face distances")
    print("═" * 58)

    result = embedding.extract(frame)
    if result is None:
        print("  No face detected in current frame.")
        print("═" * 58 + "\n")
        return

    emb   = result["embedding"]
    conf  = result["confidence"]
    print(f"  Detection confidence: {conf:.3f}")

    vectors = db.get_all_vectors()
    if not vectors:
        print("  No stored faces yet — run a search first.")
        print("═" * 58 + "\n")
        return

    from embedding import cosine_similarity, verdict as emb_verdict
    rows = []
    for entry in vectors:
        score = cosine_similarity(emb, entry["vector"])
        rows.append((entry["name"], score, emb_verdict(score)))

    rows.sort(key=lambda x: x[1], reverse=True)
    print(f"  {'Name':<25} {'Score':>6}  Verdict")
    print(f"  {'-'*25} {'-'*6}  {'-'*10}")
    for name, score, verd in rows[:15]:
        print(f"  {name:<25} {score:>6.3f}  {verd}")

    print("═" * 58 + "\n")
    logger.info("Diagnostic completed")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  HUD OVERLAY                                                          ║
# ╚══════════════════════════════════════════════════════════════════════╝
def draw_hud(frame: np.ndarray, fps: float) -> np.ndarray:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, HUD_H), C_DARK, -1)
    cv2.line(frame, (0, HUD_H), (w, HUD_H), (55, 55, 55), 1)

    st   = state.status
    st_c = C_GREEN if ("Done" in st or "Ready" in st) else C_CYAN
    cv2.putText(frame, st[:72], (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, st_c, 1, cv2.LINE_AA)

    if state.is_searching:
        cv2.putText(frame, "● SEARCHING", (10, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, C_YELLOW, 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "● READY",     (10, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, C_GREEN, 1, cv2.LINE_AA)

    fps_c = C_GREEN if fps >= 20 else C_YELLOW if fps >= 10 else C_RED
    cv2.putText(frame, f"FPS {fps:.0f}", (w - 96, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, fps_c, 1, cv2.LINE_AA)

    hint = "SPACE=search  F=flip  D=diag  H=hide  Q=quit"
    cv2.putText(frame, hint, (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, C_GREY, 1, cv2.LINE_AA)
    return frame


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  SIDEBAR — recent searches                                           ║
# ╚══════════════════════════════════════════════════════════════════════╝
def build_sidebar(panel_h: int) -> np.ndarray:
    panel     = np.full((panel_h, PANEL_W, 3), (22, 22, 22), dtype=np.uint8)
    history   = state.get_history()

    # Header
    cv2.rectangle(panel, (0, 0), (PANEL_W, 36), (47, 117, 181), -1)
    cv2.putText(panel, "Recent Searches",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.50, C_WHITE, 1, cv2.LINE_AA)

    if not history:
        cv2.putText(panel, "No searches yet",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.40, C_GREY, 1, cv2.LINE_AA)
        cv2.putText(panel, "Press SPACE to capture",
                    (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_GREY, 1, cv2.LINE_AA)
        return panel

    ROW_H = 48
    y0    = 44
    for i, h in enumerate(history[:6]):
        y = y0 + i * ROW_H
        if y + ROW_H > panel_h:
            break

        verdict = h.get("verdict", "?")
        dot_c   = C_GREEN if verdict == "CONFIRMED" else C_YELLOW if verdict == "POSSIBLE" else C_GREY
        cv2.circle(panel, (14, y + 14), 6, dot_c, -1, cv2.LINE_AA)

        name = h.get("name", "?")[:18]
        cv2.putText(panel, name, (26, y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, C_WHITE, 1, cv2.LINE_AA)

        score   = h.get("score", 0)
        matches = h.get("matches", 0)
        t       = h.get("time", "")
        cv2.putText(panel,
                    f"{verdict[:4]}  {score:.2f}  {matches}r  {t}",
                    (26, y + 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, C_GREY, 1, cv2.LINE_AA)

        if i < len(history) - 1:
            cv2.line(panel, (8, y + ROW_H - 2), (PANEL_W - 8, y + ROW_H - 2),
                     (50, 50, 50), 1)

    return panel


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  MAIN ENTRY POINT                                                     ║
# ╚══════════════════════════════════════════════════════════════════════╝
def main():
    # ── Logging ───────────────────────────────────────────────────────────
    setup_logging("osint")
    logger.info("=" * 58)
    logger.info("  Face OSINT System starting")
    logger.info(f"  Output  : {config.OUTPUT_DIR}")
    logger.info(f"  Database: {config.DB_PATH}")
    logger.info(f"  Models  : {config.MODELS_DIR}")
    logger.info("=" * 58)

    print("\n" + "═" * 58)
    print("  Face OSINT System")
    print("  Windows-native | No Docker | Limited-resource mode")
    print("═" * 58)

    # ── Storage ────────────────────────────────────────────────────────
    db     = Database()
    writer = FolderWriter()

    # ── Camera selection ───────────────────────────────────────────────
    cam_source = camera.select_camera()
    reader     = camera.FrameReader(cam_source)

    # Wait for first frame
    print("\n  Waiting for camera...", end="", flush=True)
    for _ in range(60):
        time.sleep(0.1)
        ret, _ = reader.read()
        if ret:
            break
    else:
        logger.error("Camera failed to produce a frame")
        print(" FAILED\n  Check camera connection and try again.")
        reader.release()
        sys.exit(1)
    print(" OK")

    logger.info(f"Camera ready: source={cam_source}")
    state.status = "Ready — press SPACE to capture a face"
    print(f"\n  Running. Click the window then:")
    print(f"    SPACE = capture & search")
    print(f"    F     = flip mirror")
    print(f"    D     = face diagnostic")
    print(f"    Q     = quit\n")

    # ── Display loop ───────────────────────────────────────────────────
    while True:
        ret, frame = reader.read()
        if not ret or frame is None:
            time.sleep(0.02)
            continue

        if state.flip:
            frame = cv2.flip(frame, 1)

        display = frame.copy()

        if state.show_hud:
            draw_hud(display, reader.get_fps())

        # Sidebar
        sidebar   = build_sidebar(display.shape[0])
        composite = np.hstack([display, sidebar])
        cv2.imshow(WIN_NAME, composite)

        # ── Key handling ───────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("f"):
            v = state.toggle_flip()
            state.status = f"Mirror {'ON' if v else 'OFF'}"
            logger.info(f"Mirror toggled: {v}")

        elif key == ord("h"):
            state.toggle_hud()

        elif key == ord("d"):
            threading.Thread(
                target=run_diagnostic,
                args=(frame.copy(), db),
                daemon=True,
            ).start()

        elif key == ord(" "):
            if state.is_searching:
                state.status = "Search in progress — please wait"
                continue

            # Freeze frame, get name
            name = get_name_input(frame.copy())
            if not name:
                state.status = "Cancelled"
                continue

            if not state.try_start_search():
                state.status = "Already searching — please wait"
                continue

            state.status = f"Starting search: {name}"
            threading.Thread(
                target=run_search,
                args=(frame.copy(), name, db, writer),
                daemon=True,
            ).start()

    # ── Shutdown ───────────────────────────────────────────────────────
    reader.release()
    cv2.destroyAllWindows()
    logger.info("Face OSINT System shutdown clean")
    print("\n  Session ended. Results saved to:", config.OUTPUT_DIR)


if __name__ == "__main__":
    main()
