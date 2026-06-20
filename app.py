"""
app.py — Face OSINT v4.2
─────────────────────────────────────────────────────────────────────
NEW IN v4.2:
  ✓ WiFi camera (IP Webcam) — backend FrameReader proxy, live browser preview
  ✓ Dark / Light mode toggle — full theme switch with CSS vars
  ✓ SSE fully wired — onmessage, onopen, onerror all explicit
  ✓ Camera stops immediately after Capture (stopCamera inside capture())
  ✓ Double-submit guard on Search button

PREVIOUS FIXES (v4.1):
  ✓ Multiple folders — image MD5 deduplication lock
  ✓ Session isolation — SSE cleaned immediately; repeat face modal
  ✓ Logging — mkdir guard + werkzeug silenced
  ✓ Wrong identity — scorer weights, resolver, location/company parsing
  ✓ host=127.0.0.1 — camera works on HTTP localhost; auto-opens browser

WiFi Camera setup:
    1. Install IP Webcam on Android (free, Pavel Khlebovich)
    2. Tap Start server — note the IP shown (e.g. 192.168.1.7:8080)
    3. In the app, click "📡 WiFi Cam" tab, enter IP, click Connect

RUN:
    python app.py
    Browser opens automatically at http://localhost:5000
"""

import os, sys, uuid, base64, json, queue, sqlite3, threading, time, logging, webbrowser, hashlib
import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, wait as fut_wait, FIRST_COMPLETED

import cv2
import numpy as np
import requests as _requests
from flask import Flask, Response, jsonify, request

# ── ISSUE 3 FIX: logging must be set up before ANY other import ──────────
# Ensure log dir exists BEFORE creating the handler
from pathlib import Path as _P
_P(__file__).parent.joinpath("logs").mkdir(parents=True, exist_ok=True)
_P(__file__).parent.joinpath("data", "output").mkdir(parents=True, exist_ok=True)
_P(__file__).parent.joinpath("data").mkdir(parents=True, exist_ok=True)

import config
from config import setup_logging
setup_logging("web")

# Suppress Flask/werkzeug noise from polluting the log
logging.getLogger("werkzeug").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
logger.info("=== app.py startup — logging confirmed ===")
logger.info(f"Log dir: {config.LOG_DIR}")
logger.info(f"Output dir: {config.OUTPUT_DIR}")

import embedding
from storage.database import Database
from storage.folder_writer import FolderWriter
from aggregator import face_matcher, scorer, resolver

app    = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024   # 10 MB max upload
db     = Database()
writer = FolderWriter()

_sse:    dict = {}   # sid → queue.Queue
_cancel: dict = {}   # sid → threading.Event
_sse_lock = threading.Lock()

# ── ISSUE 1 FIX: per-image deduplication — prevents SSE reconnect spawning N searches ──
_active_searches: dict = {}   # img_hash → sid
_sid_to_hash:     dict = {}   # sid → img_hash (for cleanup)
_active_lock = threading.Lock()

# ── WiFi camera — single shared FrameReader (one phone at a time) ────────────
_wifi_reader   = None   # camera.FrameReader instance
_wifi_url      = ""     # currently connected URL
_wifi_cam_lock = threading.Lock()


# ══════════════════════════════════════════════════════════════════════════
#  NAME PARSING — "John Doe, Mumbai" or "John Doe @ Google" or "John Doe"
# ══════════════════════════════════════════════════════════════════════════
def parse_name_input(raw: str) -> tuple[str, str, str]:
    """
    Parse the name field which may contain optional location/company hints.

    Formats accepted (all optional):
        "John Doe"
        "John Doe, Mumbai"
        "John Doe, Mumbai, India"
        "John Doe @ Google"
        "John Doe, Mumbai @ Google"
        "John Doe | TCS | Delhi"

    Returns (name, location, company)
    """
    raw      = raw.strip()
    company  = ""
    location = ""

    # Extract company after "@"
    if " @ " in raw:
        parts   = raw.split(" @ ", 1)
        raw     = parts[0].strip()
        company = parts[1].strip()

    # Extract location/extra after "|"
    if "|" in raw:
        parts    = [p.strip() for p in raw.split("|")]
        raw      = parts[0]
        location = parts[1] if len(parts) > 1 else ""
        if not company and len(parts) > 2:
            company = parts[2]

    # Extract location after first comma (only if no "@" was used)
    elif "," in raw:
        parts    = [p.strip() for p in raw.split(",", 1)]
        raw      = parts[0]
        location = parts[1] if len(parts) > 1 else ""

    name = raw.strip()
    logger.info(f"Parsed input → name='{name}' location='{location}' company='{company}'")
    return name, location, company


# ══════════════════════════════════════════════════════════════════════════
#  SSE HELPERS
# ══════════════════════════════════════════════════════════════════════════
def _push(sid: str, **kw):
    with _sse_lock:
        q = _sse.get(sid)
    if q:
        q.put(kw)

def _log(sid, level, source, msg, data=None):
    _push(sid, type="log",
          ts=datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3],
          level=level, source=source, msg=msg, data=data or {})

def _scrape(mod, fn, ctx):
    import importlib
    return getattr(importlib.import_module(mod), fn)(ctx)


# ══════════════════════════════════════════════════════════════════════════
#  SEARCH PIPELINE
# ══════════════════════════════════════════════════════════════════════════
def run_search(sid: str, frame: np.ndarray, name: str,
               location: str = "", company: str = ""):
    ev   = _cancel.get(sid, threading.Event())
    push = lambda **kw: _push(sid, **kw)
    log  = lambda lv, src, msg, d=None: _log(sid, lv, src, msg, d)

    def cancelled():
        if ev.is_set():
            push(step="error", msg="Cancelled by user.", done=True, ok=False)
            _cleanup_session(sid)
            return True
        return False

    try:
        # ── Step 1: Embedding ─────────────────────────────────────────
        push(step="embedding", msg="Extracting face embedding…")
        log("INFO", "embedding", "Running DeepFace Facenet512")
        res = embedding.extract(frame)
        if not res:
            log("ERROR", "embedding", "No face detected",
                {"tip": "Better lighting, face fills the frame, no glasses"})
            push(step="error",
                 msg="No face detected. Tips: better lighting · face fills frame · remove glasses.",
                 done=True, ok=False)
            _cleanup_session(sid)
            return

        emb_vec = res["embedding"]
        log("INFO", "embedding", "Face detected",
            {"confidence": round(float(res["confidence"]), 3),
             "bbox": list(res["bbox"]), "dim": int(emb_vec.shape[0])})
        push(step="embedding", msg=f"Detected · conf={res['confidence']:.3f}",
             done_step=True, confidence=round(float(res["confidence"]), 3))
        if cancelled(): return

        # ── Step 2: DB check ──────────────────────────────────────────
        push(step="db", msg="Searching face database…")
        similar = db.find_similar_faces(emb_vec, top_k=3)
        if similar:
            top = similar[0]
            log("INFO", "db", "Prior match found",
                {"name": top["name"], "score": top["score"]})
            push(step="db", msg=f"Prior record: {top['name']} · {top['score']:.3f}",
                 done_step=True, prior_name=top["name"], prior_score=top["score"])
        else:
            log("INFO", "db", "No prior match")
            push(step="db", msg="No prior record found", done_step=True)
        if cancelled(): return

        # ── Step 3: Folder ────────────────────────────────────────────
        push(step="folder", msg="Creating output folder…")
        db.create_search(sid, name, company=company, location=location)
        folder = writer.create_folder(name=name, search_id=sid,
                                      frame=frame, face_crop=res["face_crop"])
        with sqlite3.connect(str(db.path)) as conn:
            conn.execute("UPDATE searches SET output_folder=? WHERE id=?",
                         (str(folder), sid))
        db.store_vector(sid, name, emb_vec)
        log("INFO", "storage", "Folder created",
            {"path": str(folder),
             "note": "captured_photo.jpg = full frame. Use /api/cleanup to wipe."})
        push(step="folder", msg=folder.name, done_step=True)
        if cancelled(): return

        # ── Build context with location + company ─────────────────────
        ctx = {
            "name":           name,
            "company":        company,
            "location":       location,
            "embedding":      emb_vec,
            "face_crop_path": str(folder / "face_crop.jpg"),
            # image_b64: face crop as base64 JPEG — used by reverse_face.py
            # (reads from the saved face_crop.jpg so the context stays JSON-safe)
            "image_b64": (lambda p: (
                "data:image/jpeg;base64," + base64.b64encode(p.read_bytes()).decode()
                if p.exists() else ""
            ))(folder / "face_crop.jpg"),
        }

        # (label, module, function, context, per-scraper timeout seconds)
        # reverse_face needs ~60-80s (4 search engines + face-verify batch)
        # username needs ~35-50s (125 direct checks + Sherlock)
        # All others finish well within 30s.
        SCRAPERS = [
            # ── STAGE 1: Face is the query — runs first ────────────────────
            ("reverse_face",   "scrapers.reverse_face",  "scrape",        ctx, 90),
            # ── STAGE 2: Platform APIs ─────────────────────────────────────
            ("github",         "scrapers.platforms",      "scrape_github", ctx, 20),
            ("reddit",         "scrapers.platforms",      "scrape_reddit", ctx, 15),
            # ── STAGE 3: Broad text + academic sources ─────────────────────
            ("search_engines", "scrapers.search_engines", "scrape",        ctx, 25),
            ("academic",       "scrapers.academic",       "scrape",        ctx, 35),
            # ── STAGE 4: Passive + username intelligence ────────────────────
            ("passive",        "scrapers.passive",        "scrape",        ctx, 20),
            ("username",       "scrapers.username",       "scrape",        ctx, 50),
            # Note: gitlab (403 without token) and npm removed — add back
            # if you set GITLAB_TOKEN in .env
        ]

        # ── Step 4: Scraping ──────────────────────────────────────────
        push(step="scraping", msg=f"Launching {len(SCRAPERS)} sources…")
        log("INFO", "scraper", f"Starting {len(SCRAPERS)} scrapers",
            {"name": name, "location": location, "company": company,
             "workers": config.SCRAPER_MAX_WORKERS})

        all_results = {}
        _t0         = time.time()
        _fut_dl: dict = {}   # future → its individual deadline

        with ThreadPoolExecutor(max_workers=config.SCRAPER_MAX_WORKERS) as pool:
            futs: dict = {}
            for lbl, mod, fn, c, t in SCRAPERS:
                f = pool.submit(_scrape, mod, fn, c)
                futs[f]    = lbl
                _fut_dl[f] = _t0 + t
            pending = set(futs.keys())

            while pending:
                if cancelled():
                    for f in pending: f.cancel()
                    break

                now = time.time()
                # Cancel any future that has exceeded its individual deadline
                for f in list(pending):
                    if now >= _fut_dl[f]:
                        lbl = futs[f]; f.cancel()
                        elapsed = int(now - _t0)
                        all_results[lbl] = {"matches": [], "error": "Timed out"}
                        log("WARN", lbl, f"Timed out after {elapsed}s")
                        push(step="scraping", scraper=lbl, count=0,
                             msg=f"{lbl} timed out", err="Timed out")
                        pending.discard(f)
                if not pending:
                    break

                # Sleep until the nearest deadline or next completion
                next_dl = min(_fut_dl[f] for f in pending) - time.time()
                wait_t  = min(1.5, max(0.05, next_dl))
                done_futs, pending = fut_wait(pending, timeout=wait_t,
                                              return_when=FIRST_COMPLETED)
                for f in done_futs:
                    lbl = futs[f]
                    try:
                        result = f.result()
                        all_results[lbl] = result
                        n    = len(result.get("matches", []))
                        urls = [
                            m.get("url") or m.get("profile_url", "")
                            for m in result.get("matches", [])
                            if m.get("url") or m.get("profile_url")
                        ][:5]
                        # Tiny preview for the live-feed in the UI progress tab
                        preview = [
                            {"name":   m.get("name") or m.get("title", ""),
                             "url":    m.get("url") or m.get("profile_url", ""),
                             "source": lbl}
                            for m in result.get("matches", [])
                            if m.get("url") or m.get("profile_url")
                        ][:3]
                        log("INFO", lbl, f"{n} matches",
                            {"n": n, "urls": urls, "error": result.get("error")})
                        push(step="scraping", msg=f"{lbl} → {n} hits",
                             scraper=lbl, count=n, urls=urls, preview=preview)
                    except Exception as e:
                        all_results[lbl] = {"matches": [], "error": str(e)}
                        log("ERROR", lbl, f"Exception: {e}")
                        push(step="scraping", msg=f"{lbl} error",
                             scraper=lbl, count=0, err=str(e))

        # ── Bing HTML fallback if search_engines returned nothing ─────
        se = all_results.get("search_engines", {})
        if not se.get("matches"):
            log("INFO", "bing_html", "No search_engine results — running Bing HTML fallback")
            push(step="scraping", msg="Bing HTML fallback…", scraper="search_engines")
            from scrapers.search_engines import _bing_html
            bing_all = []
            # ISSUE 4 Layer 4: location-aware queries
            queries = [f'"{name}"']
            if location: queries.append(f'"{name}" "{location}"')
            if company:  queries.append(f'"{name}" "{company}"')
            queries += [
                f'"{name}" site:linkedin.com/in',
                f'"{name}" site:github.com OR site:twitter.com',
            ]
            for q in queries[:5]:
                bing_all.extend(_bing_html(q))
                time.sleep(0.35)
            seen_u, deduped_b = set(), []
            for m in bing_all:
                if m["url"] not in seen_u:
                    seen_u.add(m["url"]); deduped_b.append(m)
            all_results["search_engines"] = {"matches": deduped_b}
            log("INFO", "bing_html", f"{len(deduped_b)} results",
                {"urls": [m["url"] for m in deduped_b[:5]]})
            push(step="scraping", msg=f"Bing fallback → {len(deduped_b)} hits",
                 scraper="search_engines", count=len(deduped_b))

        # ── LinkedIn enrichment from face-confirmed reverse_face hits ──────
        # reverse_face sometimes returns LinkedIn URLs directly (face-confirmed).
        # If search_engines found nothing, promote those URLs into search_engines
        # so they go through the scorer and resolver as LinkedIn profiles.
        rf_matches = all_results.get("reverse_face", {}).get("matches", [])
        rf_linkedin = [
            m for m in rf_matches
            if "linkedin.com/in/" in (m.get("url") or "")
        ]
        if rf_linkedin:
            se = all_results.setdefault("search_engines", {"matches": []})
            existing_urls = {m.get("url") for m in se.get("matches", [])}
            added = 0
            for m in rf_linkedin:
                if m["url"] not in existing_urls:
                    se["matches"].append({
                        "url":      m["url"],
                        "title":    m.get("name", ""),
                        "snippet":  m.get("snippet", ""),
                        "source":   "reverse_face_linkedin",
                        "platform": "linkedin",
                        "is_linkedin": True,
                    })
                    existing_urls.add(m["url"])
                    added += 1
            if added:
                log("INFO", "enrich", f"Promoted {added} LinkedIn URL(s) from reverse_face")
                push(step="scraping", msg=f"LinkedIn from face search: {added}",
                     scraper="search_engines", count=len(se["matches"]))

        push(step="scraping", msg="All sources done", done_step=True)
        if cancelled(): return

        # ── Step 5: Face matching ─────────────────────────────────────
        push(step="matching", msg="Comparing scraped photos vs captured face…")
        log("INFO", "face_matcher", "Scoring profile photos",
            {"confirmed_threshold": config.FACE_CONFIRMED,
             "possible_threshold":  config.FACE_POSSIBLE})
        all_results  = face_matcher.score_all_results(all_results, emb_vec)
        face_scored  = [
            {"src": s, "user": m.get("username", m.get("name", "?")),
             "score": round(m["face_score"], 3)}
            for s, d in all_results.items() if isinstance(d, dict)
            for m in d.get("matches", [])
            if m.get("face_score") is not None
        ]
        log("INFO", "face_matcher", f"{len(face_scored)} photos scored",
            {"results": face_scored[:8]})
        push(step="matching", msg=f"{len(face_scored)} photos scored", done_step=True)
        if cancelled(): return

        # ── Step 6: Scoring ───────────────────────────────────────────
        push(step="scoring", msg="Computing confidence scores…")
        # Flatten all_results → flat list then score with face-first scorer
        flat_matches = [
            m for s, d in all_results.items() if isinstance(d, dict)
            for m in d.get("matches", [])
        ]
        scored_m = scorer.score_all(flat_matches, query_name=name,
                                    query_location=location,
                                    query_company=company)
        # Apply minimum score threshold
        scored_m = [m for m in scored_m
                    if m.get("combined_score", 0) >= getattr(config, "MIN_SCORE_KEEP", 0.25)]
        log("INFO", "scorer", f"{len(scored_m)} above threshold",
            {"threshold": config.MIN_SCORE_KEEP,
             "top5": [{"n": m.get("name", m.get("username", "?")),
                       "s": m.get("combined_score"), "src": m.get("source"),
                       "has_face": m.get("face_score") is not None}
                      for m in scored_m[:5]]})
        push(step="scoring", msg=f"{len(scored_m)} matches scored", done_step=True)
        if cancelled(): return

        # ── Step 7: Entity resolution ─────────────────────────────────
        push(step="resolving", msg="Building entity graph…")
        identity = resolver.resolve(name, scored_m)
        log("INFO", "resolver", "Resolution complete",
            {"verdict": identity.get("verdict"),
             "score":   round(identity.get("combined_score", 0), 3),
             "sources": identity.get("sources", []),
             "email":   identity.get("email"),
             "company": identity.get("company"),
             "profiles": len(identity.get("profile_urls", []))})
        push(step="resolving",
             msg=f"{identity.get('verdict','?').upper()} · {identity.get('combined_score',0):.3f}",
             done_step=True)
        if cancelled(): return

        # ── Step 8: Write report ──────────────────────────────────────
        push(step="writing", msg="Writing report…")
        writer.write_results(folder=folder, name=name, search_id=sid,
                             all_results=all_results, identity=identity)
        db.complete_search(sid,
                           verdict=identity.get("verdict", "unknown"),
                           combined_score=identity.get("combined_score", 0.0))
        for m in scored_m[:50]:
            db.insert_match(sid, m.get("source", "?"), m)

        stored_files = [
            {"path": str(p), "name": p.name,
             "rel":  str(p.relative_to(folder)),
             "size_kb": round(p.stat().st_size / 1024, 1)}
            for p in sorted(folder.rglob("*")) if p.is_file()
        ]
        log("INFO", "storage", "Report written",
            {"folder": str(folder), "files": stored_files,
             "WARNING": "captured_photo.jpg has your camera frame. "
                        "Use 🗑 Delete Images to remove sensitive files."})
        push(step="writing", msg=f"Saved {len(stored_files)} files", done_step=True)

        # ── Done ──────────────────────────────────────────────────────
        n_total = sum(
            len(v.get("matches", [])) for v in all_results.values()
            if isinstance(v, dict)
        )
        src_summary = {
            lbl: {"count": len(d.get("matches", [])), "error": d.get("error")}
            for lbl, d in all_results.items() if isinstance(d, dict)
        }
        clean_id = {k: v for k, v in identity.items() if k != "all_profiles"}
        push(step="done", msg="Complete!", done=True, ok=True,
             identity=clean_id, total=n_total, folder=str(folder),
             files=stored_files, source_summary=src_summary,
             all_matches=scored_m[:60])

        logger.info(
            f"Search {sid[:8]} DONE — "
            f"verdict={identity.get('verdict')} "
            f"score={identity.get('combined_score', 0):.3f} "
            f"matches={n_total}"
        )

    except Exception as e:
        logger.exception(f"Search {sid[:8]} crashed: {e}")
        push(step="error", msg=f"Crash: {e}", done=True, ok=False)
    finally:
        # ISSUE 2 FIX: Clean up SSE immediately after done — no 180s wait
        _cleanup_session(sid)


def _cleanup_session(sid: str):
    """ISSUE 2: Remove SSE queue immediately once search is done/cancelled."""
    # Small delay so the final 'done' event can be read by the browser
    def _do():
        time.sleep(3)
        with _sse_lock:
            _sse.pop(sid, None)
        _cancel.pop(sid, None)
        # ISSUE 1 FIX: also release the image hash lock so retries are allowed later
        with _active_lock:
            img_hash = _sid_to_hash.pop(sid, None)
            if img_hash:
                _active_searches.pop(img_hash, None)
        logger.debug(f"Session {sid[:8]} cleaned up")
    threading.Thread(target=_do, daemon=True).start()


# ══════════════════════════════════════════════════════════════════════════
#  FLASK ROUTES
# ══════════════════════════════════════════════════════════════════════════
@app.route("/")
def index():
    return _HTML, 200, {"Content-Type": "text/html; charset=utf-8"}

@app.route("/api/check_face", methods=["POST"])
def api_check_face():
    """
    ISSUE 2: Check if the submitted face was searched before.
    Returns prior match info if score >= FACE_CONFIRMED.
    Frontend shows a confirmation dialog before starting a full search.
    """
    d   = request.get_json() or {}
    img = d.get("image", "")
    if not img:
        return jsonify(prior=None)
    try:
        raw   = base64.b64decode(img.split(",", 1)[-1])
        frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify(prior=None)
        res = embedding.extract(frame)
        if not res:
            return jsonify(prior=None)
        similar = db.find_similar_faces(res["embedding"], top_k=1,
                                        threshold=config.FACE_CONFIRMED)
        if similar:
            t   = similar[0]
            row = db.get_search(t["search_id"])
            return jsonify(prior={
                "name":      t["name"],
                "score":     round(t["score"], 3),
                "search_id": t["search_id"],
                "verdict":   (row or {}).get("verdict", "unknown"),
                "date":      (row or {}).get("created_at", ""),
            })
        return jsonify(prior=None)
    except Exception as e:
        logger.debug(f"check_face error: {e}")
        return jsonify(prior=None)

@app.route("/api/search", methods=["POST"])
def api_search():
    d        = request.get_json() or {}
    raw_name = (d.get("name") or "").strip()
    img      = d.get("image", "")
    if not raw_name: return jsonify(error="Name required"), 400
    if len(raw_name) > 200: return jsonify(error="Name too long (max 200 characters)"), 400
    if not img:      return jsonify(error="Image required"), 400

    # Parse optional location/company from name field
    name, location, company = parse_name_input(raw_name)

    try:
        raw   = base64.b64decode(img.split(",", 1)[-1])
        frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if frame is None: raise ValueError("Image decode failed")
    except Exception as e:
        return jsonify(error=str(e)), 400

    # ISSUE 1 FIX: deduplicate by image hash — SSE reconnects & double-clicks
    # spawn the same base64 payload → return existing search_id instead of forking
    img_hash = hashlib.md5(raw).hexdigest()
    with _active_lock:
        if img_hash in _active_searches:
            existing_sid = _active_searches[img_hash]
            logger.info(f"Duplicate search for hash {img_hash[:8]} — reusing {existing_sid[:8]}")
            return jsonify(search_id=existing_sid, reused=True,
                           parsed={"name": name, "location": location, "company": company})
        sid = str(uuid.uuid4())
        _active_searches[img_hash] = sid
        _sid_to_hash[sid] = img_hash

    with _sse_lock:
        _sse[sid] = queue.Queue()
    _cancel[sid] = threading.Event()

    threading.Thread(
        target=run_search,
        args=(sid, frame, name, location, company),
        daemon=True,
    ).start()

    logger.info(f"Search {sid[:8]} started — name='{name}' loc='{location}' co='{company}'")
    return jsonify(search_id=sid, parsed={"name": name, "location": location, "company": company})

@app.route("/api/cancel/<sid>", methods=["POST"])
def api_cancel(sid):
    ev = _cancel.get(sid)
    if not ev: return jsonify(error="Not found"), 404
    ev.set()
    logger.info(f"Search {sid[:8]} killed by user")
    return jsonify(ok=True)

@app.route("/api/stream/<sid>")
def api_stream(sid):
    with _sse_lock:
        q = _sse.get(sid)
    if not q:
        return jsonify(error="Search not found or already completed"), 404
    def gen():
        while True:
            try:
                msg = q.get(timeout=25)
                yield f"data:{json.dumps(msg)}\n\n"
                if msg.get("done"): break
            except queue.Empty:
                yield 'data:{"hb":1}\n\n'
    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@app.route("/api/history")
def api_history():
    return jsonify(db.list_searches(20))


@app.route("/api/history/clear", methods=["POST"])
def api_history_clear():
    d = request.get_json() or {}
    if not d.get("confirm"):
        return jsonify(error="Pass {\"confirm\": true} to confirm bulk deletion"), 400
    with sqlite3.connect(str(db.path)) as conn:
        conn.execute("DELETE FROM searches")
        conn.execute("DELETE FROM matches")
        conn.execute("DELETE FROM face_vectors")
    logger.warning("History cleared — all search records deleted")
    return jsonify(ok=True)

@app.route("/api/result/<sid>")
def api_result(sid):
    """Return full structured result for a search — used by viewRep."""
    row = db.get_search(sid)
    if not row: return jsonify(error="Not found"), 404
    folder = Path(row.get("output_folder", ""))
    result = dict(row)
    # Try to load matches_summary.json for rich data
    ms = folder / "matches_summary.json"
    if ms.exists():
        try:
            result["matches"] = json.loads(ms.read_text("utf-8"))
        except Exception:
            pass
    # Load info.txt as text summary
    info = folder / "info.txt"
    if info.exists():
        result["info_text"] = info.read_text("utf-8")
    # Face crop as base64 for display
    fc = folder / "face_crop.jpg"
    if fc.exists():
        result["face_crop_b64"] = "data:image/jpeg;base64," + base64.b64encode(fc.read_bytes()).decode()
    return jsonify(result)
@app.route("/api/report/<sid>")
def api_report(sid):
    row = db.get_search(sid)
    if not row: return jsonify(error="Not found"), 404
    txt = Path(row.get("output_folder", "")) / "info.txt"
    if txt.exists():
        return txt.read_text("utf-8"), 200, {"Content-Type": "text/plain;charset=utf-8"}
    return jsonify(error="Report not ready"), 404

@app.route("/api/search/<sid>/report")
def api_search_report(sid):
    """Feature B: Download info.txt for a search as an attachment."""
    row = db.get_search(sid)
    if not row: return jsonify(error="Not found"), 404
    txt = Path(row.get("output_folder", "")) / "info.txt"
    if not txt.exists():
        return jsonify(error="Report not ready"), 404
    return txt.read_text("utf-8"), 200, {
        "Content-Type": "text/plain;charset=utf-8",
        "Content-Disposition": f'attachment; filename="report_{sid[:8]}.txt"',
    }

@app.route("/api/search/<sid>/face_crop")
def api_search_face_crop(sid):
    """Feature C: Return face_crop.jpg for a search."""
    row = db.get_search(sid)
    if not row: return jsonify(error="Not found"), 404
    fc = Path(row.get("output_folder", "")) / "face_crop.jpg"
    if not fc.exists():
        return jsonify(error="Face crop not found"), 404
    from flask import send_file
    return send_file(str(fc), mimetype="image/jpeg")

@app.route("/api/files/<sid>")
def api_files(sid):
    row = db.get_search(sid)
    if not row: return jsonify(error="Not found"), 404
    folder = Path(row.get("output_folder", ""))
    if not folder.exists():
        return jsonify(error="Folder not found", path=str(folder)), 404
    files = [
        {"name": p.name, "path": str(p),
         "rel":  str(p.relative_to(folder)),
         "size_kb": round(p.stat().st_size / 1024, 1),
         "modified": datetime.datetime.fromtimestamp(
             p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")}
        for p in sorted(folder.rglob("*")) if p.is_file()
    ]
    return jsonify(folder=str(folder), files=files)

@app.route("/api/cleanup/<sid>", methods=["POST"])
def api_cleanup(sid):
    d    = request.get_json() or {}
    row  = db.get_search(sid)
    if not row: return jsonify(error="Not found"), 404
    folder  = Path(row.get("output_folder", ""))
    deleted = []

    if d.get("images") or d.get("full"):
        for fname in ["captured_photo.jpg", "face_crop.jpg"]:
            fp = folder / fname
            if fp.exists():
                fp.unlink(); deleted.append(fname)
        sp = folder / "scraped_photos"
        if sp.exists():
            for f in sp.iterdir():
                f.unlink(); deleted.append(f"scraped_photos/{f.name}")
            try: sp.rmdir()
            except Exception: pass

    if d.get("embedding") or d.get("full"):
        with sqlite3.connect(str(db.path)) as conn:
            conn.execute("DELETE FROM face_vectors WHERE search_id=?", (sid,))
        deleted.append("face_vector (SQLite)")

    if d.get("full"):
        for fp in [folder / "info.txt", folder / "matches_summary.json"]:
            if fp.exists():
                fp.unlink(); deleted.append(fp.name)
        try: folder.rmdir(); deleted.append("output_folder/")
        except Exception: pass

    logger.info(f"Cleanup {sid[:8]}: deleted {deleted}")
    return jsonify(ok=True, deleted=deleted)


# ══════════════════════════════════════════════════════════════════════════
#  WIFI CAMERA ROUTES  (phone camera via IP Webcam app)
# ══════════════════════════════════════════════════════════════════════════
@app.route("/api/wifi/connect", methods=["POST"])
def api_wifi_connect():
    """
    Connect to a WiFi camera (IP Webcam app on Android).
    Body: {"url": "192.168.1.7:8080"}
    Probes the URL, starts a FrameReader, returns {ok, url, resolution}.
    """
    global _wifi_reader, _wifi_url
    d   = request.get_json() or {}
    raw = (d.get("url") or "").strip()
    if not raw:
        return jsonify(ok=False, error="URL required"), 400

    # Normalise — add http:// if missing
    if not raw.startswith("http"):
        raw = "http://" + raw

    with _wifi_cam_lock:
        # Release any previous reader
        if _wifi_reader is not None:
            try: _wifi_reader.release()
            except Exception: pass
            _wifi_reader = None

        from camera import probe_ip_camera, FrameReader
        working_url = probe_ip_camera(raw)
        if not working_url:
            return jsonify(ok=False, error=f"Cannot reach camera at {raw}. "
                           "Check phone and PC are on the same WiFi, and IP Webcam is running.")

        _wifi_reader = FrameReader(working_url)
        _wifi_url    = working_url

        # Wait up to 3s for the first frame
        for _ in range(30):
            ret, frame = _wifi_reader.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                logger.info(f"WiFi camera connected: {working_url} ({w}×{h})")
                return jsonify(ok=True, url=working_url, width=w, height=h)
            time.sleep(0.1)

        return jsonify(ok=False, error="Connected but no frames received — try again")


@app.route("/api/wifi/frame")
def api_wifi_frame():
    """
    Return the current WiFi camera frame as base64 JPEG.
    Called by the browser every ~80ms to display a live preview.
    """
    with _wifi_cam_lock:
        reader = _wifi_reader
    if reader is None:
        return jsonify(ok=False, error="No WiFi camera connected"), 404

    ret, frame = reader.read()
    if not ret or frame is None:
        return jsonify(ok=False, error="No frame available"), 503

    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        return jsonify(ok=False, error="Frame encode failed"), 500

    b64 = base64.b64encode(buf.tobytes()).decode()
    fps = round(reader.get_fps(), 1)
    return jsonify(ok=True, frame=f"data:image/jpeg;base64,{b64}", fps=fps)


@app.route("/api/wifi/disconnect", methods=["POST"])
def api_wifi_disconnect():
    """Release the WiFi camera FrameReader."""
    global _wifi_reader, _wifi_url
    with _wifi_cam_lock:
        if _wifi_reader is not None:
            try: _wifi_reader.release()
            except Exception: pass
            _wifi_reader = None
            _wifi_url    = ""
    logger.info("WiFi camera disconnected")
    return jsonify(ok=True)


# ══════════════════════════════════════════════════════════════════════════
#  CROWD INTELLIGENCE CENTER — ROUTES
# ══════════════════════════════════════════════════════════════════════════
from crowd.platform import get_platform

@app.route("/crowd/api/zones")
def cic_zones():
    return jsonify(get_platform().get_zones_raw())

@app.route("/crowd/api/status")
def cic_status():
    return jsonify(get_platform().get_state())

@app.route("/crowd/api/stream")
def cic_stream():
    plat = get_platform()
    q    = plat.subscribe()
    def gen():
        try:
            while True:
                try:
                    msg = q.get(timeout=25)
                    yield f"data:{json.dumps(msg, default=str)}\n\n"
                except queue.Empty:
                    yield 'data:{"hb":1}\n\n'
        finally:
            plat.unsubscribe(q)
    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@app.route("/crowd/api/slot/<int:slot>/start", methods=["POST"])
def cic_start_slot(slot):
    if slot < 0 or slot > 3:
        return jsonify(error="Slot must be 0–3"), 400
    d      = request.get_json() or {}
    source = d.get("source", "")
    if source == "webcam" or source == "" or source == "0":
        source = 0
    elif source.lstrip("-").isdigit():
        source = int(source)
    ok = get_platform().start_slot(slot, source)
    if not ok:
        return jsonify(error=f"Cannot open source '{source}'"), 400
    return jsonify(ok=True, slot=slot, source=str(source))

@app.route("/crowd/api/slot/<int:slot>/stop", methods=["POST"])
def cic_stop_slot(slot):
    get_platform().stop_slot(slot)
    return jsonify(ok=True)

@app.route("/crowd/api/frame/<int:slot>")
def cic_frame(slot):
    b64 = get_platform().get_slot_frame_b64(slot)
    if b64 is None:
        return jsonify(ok=False, error="No frame"), 404
    return jsonify(ok=True, frame=b64)

@app.route("/crowd/api/heatmap")
def cic_heatmap():
    return jsonify(points=get_platform().get_heatmap())

@app.route("/crowd/api/slot/<int:slot>/toggle", methods=["POST"])
def cic_slot_toggle(slot):
    d     = request.get_json() or {}
    name  = d.get("name", "")
    value = bool(d.get("value", True))
    get_platform().set_toggle(slot, name, value)
    return jsonify(ok=True)

@app.route("/crowd/api/ask", methods=["POST"])
def cic_ask():
    from crowd.llm_ops import ask as llm_ask
    d        = request.get_json() or {}
    question = (d.get("question") or "").strip()
    if not question:
        return jsonify(error="question required"), 400
    api_key  = config.ANTHROPIC_API_KEY
    state    = get_platform().get_state()
    def gen():
        for chunk in llm_ask(question, state, api_key):
            yield f"data:{json.dumps({'text': chunk})}\n\n"
        yield 'data:{"done":true}\n\n'
    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@app.route("/crowd/api/khoya", methods=["POST"])
def cic_khoya():
    """Khoya-Paya: search face against OSINT db + CIC crowd captures."""
    import base64, io, numpy as np
    d = request.get_json() or {}
    b64 = d.get("image", "")
    if not b64:
        return jsonify(error="image required"), 400
    try:
        # Decode image
        raw = base64.b64decode(b64.split(",", 1)[-1])
        arr = np.frombuffer(raw, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify(error="invalid image"), 400

        import embedding as emb_mod
        result = emb_mod.extract(frame)
        if not result or result.get("embedding") is None:
            return jsonify(found=False, reason="no_face")

        vec = np.array(result["embedding"], dtype=np.float32)

        # 1. Search existing OSINT face vectors
        osint_hits = db.find_similar_faces(vec, top_k=3, threshold=0.45)

        # 2. Search CIC crowd captures
        cic_hits = db.find_cic_captures(vec, top_k=3, threshold=0.45)
        cic_count = db.get_cic_capture_count()

        # Merge and rank
        matches = []
        for h in osint_hits:
            matches.append({
                "source":     "osint",
                "name":       h["name"],
                "score":      h["score"],
                "search_id":  h["search_id"],
            })
        for h in cic_hits:
            matches.append({
                "source":     "cic",
                "name":       f"Unknown (Track #{h['track_id']})",
                "score":      h["score"],
                "zone":       h["zone_name"],
                "slot":       h["slot_id"],
                "last_seen":  h["captured_at"],
            })
        matches.sort(key=lambda x: x["score"], reverse=True)

        return jsonify(
            found=bool(matches),
            matches=matches[:5],
            cic_faces_indexed=cic_count,
        )
    except Exception as e:
        logger.warning(f"Khoya-Paya error: {e}")
        return jsonify(error=str(e)), 500


# ══════════════════════════════════════════════════════════════════════════
#  EMBEDDED FRONTEND
# ══════════════════════════════════════════════════════════════════════════
_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FaceOSINT</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
/* ── Design tokens ───────────────────────────────────── */
:root{
  --bg-base:#0d0f17;
  --bg-card:#13161f;
  --bg-elevated:#1c1f2e;
  --border:#252836;
  --border-bright:#3a3f5c;
  --accent:#6366f1;
  --accent-hover:#818cf8;
  --accent-glow:rgba(99,102,241,0.15);
  --green:#22c55e;
  --green-dim:rgba(34,197,94,0.12);
  --yellow:#f59e0b;
  --yellow-dim:rgba(245,158,11,0.12);
  --red:#ef4444;
  --red-dim:rgba(239,68,68,0.12);
  --text-primary:#e2e8f0;
  --text-secondary:#94a3b8;
  --text-muted:#64748b;
  /* legacy aliases used by JS-generated HTML */
  --bg:#0d0f17;--s1:#13161f;--s2:#1c1f2e;--s3:#252836;
  --bd:#252836;--acc:#6366f1;--acc2:#818cf8;--grn:#22c55e;
  --yel:#f59e0b;--red2:#ef4444;--txt:#e2e8f0;--mute:#64748b;
  --sh:rgba(0,0,0,.55);
}
/* ── Global reset & base ─────────────────────────────── */
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%;overflow:hidden}
body{
  background:var(--bg-base);
  color:var(--text-primary);
  font-family:'Inter',sans-serif;
  display:flex;flex-direction:column;
  -webkit-font-smoothing:antialiased;
}

/* ── Scrollbars ──────────────────────────────────────── */
::-webkit-scrollbar{width:4px;height:4px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--border-bright);border-radius:4px}
::-webkit-scrollbar-thumb:hover{background:var(--text-muted)}

/* ── Animations ──────────────────────────────────────── */
@keyframes scn{from{top:0}to{top:100%}}
@keyframes spin{to{transform:rotate(360deg)}}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.35}}
@keyframes slideUp{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
@keyframes fadeIn{from{opacity:0}to{opacity:1}}
@keyframes pulse{0%,100%{box-shadow:0 0 0 0 var(--accent-glow)}50%{box-shadow:0 0 0 6px transparent}}

/* ── App shell ───────────────────────────────────────── */
.app-shell{flex:1;overflow:hidden;display:flex;flex-direction:column}

/* ── Top bar ─────────────────────────────────────────── */
.topbar{
  flex-shrink:0;
  display:flex;align-items:center;gap:12px;
  padding:0 16px;height:52px;
  background:var(--bg-card);
  border-bottom:1px solid var(--border);
  position:relative;z-index:30;
}
.logo{display:flex;align-items:center;gap:10px;flex-shrink:0;text-decoration:none}
.logo-icon{
  width:32px;height:32px;border-radius:8px;
  background:linear-gradient(135deg,var(--accent),var(--accent-hover));
  display:flex;align-items:center;justify-content:center;
  font-size:15px;color:#fff;flex-shrink:0;
  box-shadow:0 0 12px var(--accent-glow);
}
.logo-text{display:flex;flex-direction:column;line-height:1.1}
.logo-name{font-size:14px;font-weight:700;color:var(--text-primary);letter-spacing:-.3px}
.logo-sub{font-size:10px;color:var(--text-muted);font-weight:400}
.topbar-status{
  flex:1;max-width:440px;
  display:flex;align-items:center;gap:8px;
  background:var(--bg-elevated);border:1px solid var(--border);
  border-radius:20px;padding:6px 14px;
}
.status-dot{
  width:7px;height:7px;border-radius:50%;
  background:var(--green);flex-shrink:0;
  transition:background .3s;
}
.status-dot.busy{background:var(--accent);animation:pulse 2s ease infinite}
.status-dot.err{background:var(--red);animation:none}
#hdr-msg{
  font-size:12px;color:var(--text-muted);
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
  transition:color .2s;flex:1;
}
.topbar-actions{display:flex;gap:6px;margin-left:auto;flex-shrink:0;align-items:center}
.hb{
  display:flex;align-items:center;gap:5px;
  padding:5px 10px;border-radius:8px;
  border:1px solid var(--border);background:transparent;
  color:var(--text-muted);font-size:11px;font-weight:500;
  font-family:'Inter',sans-serif;
  cursor:pointer;transition:all .2s ease;white-space:nowrap;
}
.hb:hover{border-color:var(--accent);color:var(--accent-hover);background:var(--accent-glow)}
.hb.on{background:var(--accent-glow);border-color:var(--accent);color:var(--accent-hover)}
.hb.kill{border-color:rgba(239,68,68,.3);color:rgba(239,68,68,.5)}
.hb.kill:not(:disabled):hover{background:var(--red-dim);border-color:var(--red);color:var(--red)}
.hb:disabled{opacity:.25;cursor:not-allowed}
.bell-btn{
  width:34px;height:34px;border-radius:8px;
  border:1px solid var(--border);background:transparent;
  color:var(--text-muted);font-size:14px;cursor:pointer;
  display:flex;align-items:center;justify-content:center;
  transition:all .2s ease;
}
.bell-btn:hover{border-color:var(--accent);color:var(--accent-hover);background:var(--accent-glow)}
.bell-btn.on{background:var(--accent-glow);border-color:var(--accent);color:var(--accent-hover)}

/* ── Main layout ─────────────────────────────────────── */
.main-layout{
  flex:1;overflow:hidden;
  display:grid;grid-template-columns:320px 1fr;
  gap:0;
}

/* ── Left panel ──────────────────────────────────────── */
.left-panel{
  display:flex;flex-direction:column;
  border-right:1px solid var(--border);
  overflow:hidden;background:var(--bg-card);
}
.left-logo-area{
  padding:16px;
  border-bottom:1px solid var(--border);
  flex-shrink:0;
}
.left-logo-title{font-size:18px;font-weight:700;color:var(--text-primary);letter-spacing:-.4px;margin-bottom:2px}
.left-logo-title span{color:var(--accent)}
.left-logo-sub{font-size:11px;color:var(--text-muted)}

/* ── Input tabs (pill style) ─────────────────────────── */
.input-tabs-wrap{padding:12px 12px 0;flex-shrink:0}
.input-tabs{
  display:flex;gap:2px;
  background:var(--bg-elevated);border:1px solid var(--border);
  border-radius:10px;padding:3px;
}
.input-tab{
  flex:1;padding:6px 10px;border-radius:7px;border:none;
  background:transparent;color:var(--text-muted);
  font-size:11px;font-weight:500;font-family:'Inter',sans-serif;
  cursor:pointer;transition:all .2s ease;text-align:center;
}
.input-tab.act{
  background:var(--accent);color:#fff;
  box-shadow:0 2px 8px var(--accent-glow);
}
.input-tab:hover:not(.act){color:var(--text-primary);background:var(--border)}

/* ── Camera / WiFi / Upload panels ──────────────────── */
.input-content{flex-shrink:0}
.cam-w{
  position:relative;aspect-ratio:4/3;
  background:#000;overflow:hidden;
  margin:10px 12px;border-radius:10px;
}
#video,#cvs{width:100%;height:100%;object-fit:cover;display:block}
#cvs{display:none}
#video.mir{transform:scaleX(-1)}
.cam-ov{position:absolute;inset:0;pointer-events:none}
.cn{position:absolute;width:20px;height:20px;border-style:solid;border-color:var(--accent);opacity:.6}
.c-tl{top:8px;left:8px;border-width:2px 0 0 2px;border-radius:2px 0 0 0}
.c-tr{top:8px;right:8px;border-width:2px 2px 0 0;border-radius:0 2px 0 0}
.c-bl{bottom:8px;left:8px;border-width:0 0 2px 2px;border-radius:0 0 0 2px}
.c-br{bottom:8px;right:8px;border-width:0 2px 2px 0;border-radius:0 0 2px 0}
.scanl{
  position:absolute;left:0;right:0;height:1px;
  background:linear-gradient(90deg,transparent,var(--accent),transparent);
  opacity:.4;animation:scn 2.5s linear infinite;
}
#cam-off{
  display:none;position:absolute;inset:0;
  background:rgba(13,15,23,.8);
  align-items:center;justify-content:center;flex-direction:column;gap:8px;
  border-radius:10px;
}
#cam-off.show{display:flex}
#cam-off-ic{font-size:24px}
#cam-off-txt{font-size:11px;color:var(--green);font-weight:600;letter-spacing:.5px}

.drop-z{
  aspect-ratio:4/3;margin:10px 12px;border-radius:10px;
  border:2px dashed var(--border-bright);
  cursor:pointer;display:flex;flex-direction:column;
  align-items:center;justify-content:center;
  gap:8px;background:var(--bg-elevated);
  transition:all .2s ease;overflow:hidden;position:relative;
}
.drop-z:hover,.drop-z.ov{
  border-color:var(--accent);
  background:var(--accent-glow);
}
#upl-img{position:absolute;inset:0;width:100%;height:100%;object-fit:contain;display:none;border-radius:8px}
#file-in{display:none}
.dz-icon{font-size:28px;opacity:.5}
.dz-text{font-size:12px;color:var(--text-muted);font-weight:500}
.dz-hint{font-size:10px;color:var(--text-muted);opacity:.5}

/* ── WiFi panel ──────────────────────────────────────── */
.wifi-panel{padding:10px 12px;display:flex;flex-direction:column;gap:8px}
.wifi-row{display:flex;gap:6px;align-items:center}
.wifi-ip{
  flex:1;background:var(--bg-elevated);border:1px solid var(--border);
  border-radius:8px;padding:8px 12px;color:var(--text-primary);
  font-size:12px;font-family:'Inter',sans-serif;outline:none;transition:all .2s ease;
}
.wifi-ip:focus{border-color:var(--accent);box-shadow:0 0 0 3px var(--accent-glow)}
.wifi-ip::placeholder{color:var(--text-muted)}
.wifi-status{
  font-size:11px;display:flex;align-items:center;gap:6px;
  padding:6px 10px;border-radius:8px;border:1px solid var(--border);
  background:var(--bg-elevated);
}
.ws-dot{width:7px;height:7px;border-radius:50%;background:var(--text-muted);flex-shrink:0;transition:background .3s}
.ws-dot.on{background:var(--green);box-shadow:0 0 6px rgba(34,197,94,.4)}
.ws-dot.busy{background:var(--accent);animation:blink 1s ease infinite}
.ws-dot.err{background:var(--red)}
.wifi-prev{
  position:relative;aspect-ratio:4/3;background:#000;
  overflow:hidden;flex-shrink:0;border-radius:8px;margin:0 12px 10px;
}
.wifi-prev img{width:100%;height:100%;object-fit:cover;display:block}
.wifi-fps{
  position:absolute;bottom:6px;right:8px;font-size:9px;color:var(--green);
  background:rgba(0,0,0,.6);padding:2px 6px;border-radius:4px;pointer-events:none;
}
.wifi-prev-ph{
  display:flex;align-items:center;justify-content:center;
  width:100%;height:100%;color:var(--text-muted);
  font-size:12px;text-align:center;flex-direction:column;gap:6px;
}

/* ── Controls / inputs ───────────────────────────────── */
.ctrl{padding:10px 12px;display:flex;flex-direction:column;gap:8px;flex-shrink:0}
.field-label{
  font-size:10px;font-weight:600;letter-spacing:.05em;
  text-transform:uppercase;color:var(--text-muted);margin-bottom:4px;
}
.fi{
  width:100%;background:var(--bg-elevated);border:1px solid var(--border);
  border-radius:8px;padding:9px 12px;color:var(--text-primary);
  font-size:13px;font-family:'Inter',sans-serif;
  outline:none;transition:all .2s ease;
}
.fi:focus{border-color:var(--accent);box-shadow:0 0 0 3px var(--accent-glow)}
.fi::placeholder{color:var(--text-muted)}
.fi:disabled{opacity:.35;cursor:not-allowed}
.fi-hint{font-size:10px;color:var(--text-muted);margin-top:4px;line-height:1.5}
.fi-hint code{color:var(--accent-hover);font-family:'Inter',sans-serif}
.parsed-tags{display:flex;gap:4px;flex-wrap:wrap;margin-top:4px}
.ptag{
  font-size:10px;padding:2px 8px;border-radius:999px;
  font-weight:500;display:flex;align-items:center;gap:3px;
}
.ptag-loc{background:var(--accent-glow);color:var(--accent-hover);border:1px solid rgba(99,102,241,.3)}
.ptag-co{background:rgba(129,140,248,.1);color:#a78bfa;border:1px solid rgba(129,140,248,.25)}

/* ── Buttons ─────────────────────────────────────────── */
.btn{
  padding:8px 14px;border-radius:8px;border:none;
  font-size:12px;font-weight:600;font-family:'Inter',sans-serif;
  display:flex;align-items:center;justify-content:center;gap:6px;
  cursor:pointer;transition:all .2s ease;white-space:nowrap;
}
.btn-primary{
  background:var(--accent);color:#fff;
  box-shadow:0 2px 12px var(--accent-glow);
}
.btn-primary:hover:not(:disabled){
  background:var(--accent-hover);
  box-shadow:0 4px 20px rgba(99,102,241,.35);
  transform:translateY(-1px);
}
.btn-secondary{
  background:transparent;color:var(--text-secondary);
  border:1px solid var(--border);
}
.btn-secondary:hover:not(:disabled){
  border-color:var(--accent);color:var(--accent-hover);
  background:var(--accent-glow);
}
.btn-ghost{
  background:transparent;color:var(--text-muted);
  border:1px solid var(--border-bright);
}
.btn-ghost:hover:not(:disabled){border-color:var(--accent);color:var(--accent-hover)}
.btn-danger{background:var(--red-dim);color:var(--red);border:1px solid rgba(239,68,68,.3)}
.btn-danger:hover:not(:disabled){background:rgba(239,68,68,.2)}
.btn:disabled{opacity:.25;cursor:not-allowed;transform:none!important;box-shadow:none!important}
.btn-full{width:100%;height:44px;font-size:14px}
.btn-row{display:flex;gap:6px}

/* ── Files panel ─────────────────────────────────────── */
.card-section{
  background:var(--bg-card);border-top:1px solid var(--border);
  flex-shrink:0;
}
.section-header{
  display:flex;align-items:center;justify-content:space-between;
  padding:8px 14px;border-bottom:1px solid var(--border);
}
.section-title{
  font-size:10px;font-weight:600;letter-spacing:.08em;
  text-transform:uppercase;color:var(--text-muted);
}
.file-panel{max-height:120px;overflow-y:auto;padding:6px 12px}
.fp-item{
  display:flex;align-items:center;gap:8px;
  padding:5px 8px;background:var(--bg-elevated);
  border:1px solid var(--border);border-radius:6px;
  margin-bottom:4px;font-size:11px;transition:all .2s ease;
}
.fp-item:hover{border-color:var(--border-bright)}
.fp-nm{flex:1;word-break:break-all;color:var(--text-secondary)}
.fp-sz{color:var(--text-muted);flex-shrink:0;font-size:10px}
.del-btns{display:flex;gap:6px;padding:8px 12px}
.dbtn{
  flex:1;padding:6px 10px;border-radius:8px;border:none;
  font-size:11px;font-weight:600;font-family:'Inter',sans-serif;
  cursor:pointer;transition:all .2s ease;
}
.dbtn-warn{background:var(--yellow-dim);color:var(--yellow);border:1px solid rgba(245,158,11,.3)}
.dbtn-warn:not(:disabled):hover{background:rgba(245,158,11,.2)}
.dbtn-red{background:var(--red-dim);color:var(--red);border:1px solid rgba(239,68,68,.3)}
.dbtn-red:not(:disabled):hover{background:rgba(239,68,68,.2)}
.dbtn:disabled{opacity:.25;cursor:not-allowed}

/* ── History ─────────────────────────────────────────── */
.hist-section{flex:1;overflow:hidden;display:flex;flex-direction:column;min-height:0;border-top:1px solid var(--border)}
.hist{flex:1;overflow-y:auto;padding:8px 12px}
.hi{
  display:flex;align-items:center;gap:8px;
  padding:8px 10px;background:var(--bg-elevated);
  border:1px solid var(--border);border-radius:8px;
  cursor:pointer;transition:all .2s ease;margin-bottom:4px;
}
.hi:hover{border-color:var(--accent);transform:translateX(2px)}
.hi-dot{width:6px;height:6px;border-radius:50%;flex-shrink:0}
.hi-body{flex:1;min-width:0}
.hi-name{font-size:12px;font-weight:600;color:var(--text-primary);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.hi-meta{font-size:10px;color:var(--text-muted);margin-top:1px}
.hi-badge{flex-shrink:0}

/* ── Right panel ─────────────────────────────────────── */
.right-panel{
  display:flex;flex-direction:column;overflow:hidden;
  background:var(--bg-base);
}

/* ── Right tabs ──────────────────────────────────────── */
.right-tabs{
  display:flex;align-items:center;
  background:var(--bg-card);border-bottom:1px solid var(--border);
  padding:0 16px;flex-shrink:0;gap:2px;height:44px;
}
.rtab{
  padding:6px 14px;border-radius:6px;border:none;
  background:transparent;color:var(--text-muted);
  font-size:12px;font-weight:500;font-family:'Inter',sans-serif;
  cursor:pointer;transition:all .2s ease;
  display:flex;align-items:center;gap:6px;
  position:relative;
}
.rtab.act{color:var(--accent-hover);background:var(--accent-glow)}
.rtab:hover:not(.act){color:var(--text-primary)}
.rtab-badge{
  font-size:9px;font-weight:700;padding:1px 5px;
  border-radius:999px;background:var(--accent-glow);
  color:var(--accent-hover);min-width:16px;text-align:center;
}

/* ── Source matrix ───────────────────────────────────── */
.matrix{
  display:grid;grid-template-columns:repeat(4,1fr);
  gap:6px;padding:10px 14px;flex-shrink:0;
  border-bottom:1px solid var(--border);
}
.node{
  background:var(--bg-card);border:1px solid var(--border);
  border-radius:8px;padding:8px 6px;text-align:center;
  transition:all .2s ease;cursor:default;
}
.node.scanning{border-color:rgba(99,102,241,.4);background:var(--accent-glow)}
.node.found{border-color:rgba(34,197,94,.35);background:var(--green-dim)}
.node.empty{border-color:rgba(245,158,11,.25);background:var(--yellow-dim)}
.node.failed{border-color:rgba(239,68,68,.25);background:var(--red-dim)}
.node-ic{font-size:14px;margin-bottom:3px}
.node-nm{font-size:9px;font-weight:600;color:var(--text-muted);margin-bottom:1px;letter-spacing:.02em}
.node-st{font-size:8px;color:var(--text-muted);height:10px}
.node-cnt{font-size:10px;font-weight:700;color:var(--text-muted);margin-top:2px}
.node.scanning .node-nm{color:var(--accent-hover)}
.node.scanning .node-st{color:var(--accent)}
.node.found .node-nm{color:var(--green)}
.node.found .node-cnt{color:var(--green)}
.node.empty .node-nm{color:var(--yellow)}
.node.failed .node-nm{color:var(--red)}

/* ── Progress steps ──────────────────────────────────── */
.steps{flex:1;overflow-y:auto;padding:10px 14px;display:flex;flex-direction:column;gap:6px}
.step{
  display:flex;gap:10px;padding:8px 12px;border-radius:8px;
  border:1px solid transparent;background:transparent;
  opacity:.3;transition:all .2s ease;align-items:flex-start;
}
.step.act{
  opacity:1;border-color:rgba(99,102,241,.25);
  background:var(--accent-glow);
}
.step.ok{
  opacity:1;border-color:rgba(34,197,94,.2);
  background:var(--green-dim);
}
.step.err{
  opacity:1;border-color:rgba(239,68,68,.2);
  background:var(--red-dim);
}
.sic{
  width:22px;height:22px;border-radius:50%;
  background:var(--bg-elevated);border:1px solid var(--border);
  display:flex;align-items:center;justify-content:center;
  font-size:10px;flex-shrink:0;margin-top:1px;
}
.step.act .sic{animation:spin .7s linear infinite;border-color:var(--accent)}
.step.ok  .sic{border-color:var(--green)}
.step.err .sic{border-color:var(--red)}
.snm{font-size:11px;font-weight:600;color:var(--text-muted)}
.step.act .snm{color:var(--accent-hover)}
.step.ok  .snm{color:var(--green)}
.step.err .snm{color:var(--red)}
.sdt{font-size:11px;color:var(--text-muted);margin-top:2px;line-height:1.5;word-break:break-all}

/* ── Live feed ───────────────────────────────────────── */
.live-feed-wrap{
  flex-shrink:0;border-top:1px solid var(--border);
  background:var(--bg-card);
}
.live-feed-hdr{
  display:flex;align-items:center;justify-content:space-between;
  padding:6px 14px;
}
.live-feed-title{font-size:10px;font-weight:600;letter-spacing:.05em;color:var(--text-muted);text-transform:uppercase}
#live-feed{max-height:120px;overflow-y:auto;padding:0 14px 8px}
.lf-item{
  display:flex;align-items:center;gap:6px;
  padding:4px 0;border-bottom:1px solid var(--border);
  font-size:11px;animation:slideUp .2s ease;
}
.lf-item:last-child{border-bottom:none}
.lf-ic{flex-shrink:0;width:16px;text-align:center}
.lf-url{
  flex:1;color:var(--accent-hover);
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
  text-decoration:none;transition:color .2s;
}
.lf-url:hover{color:var(--accent);text-decoration:underline}
.lf-src{
  flex-shrink:0;font-size:9px;color:var(--text-muted);
  background:var(--bg-elevated);padding:1px 6px;
  border-radius:999px;border:1px solid var(--border);
}

/* ── Log ─────────────────────────────────────────────── */
.log-panel{flex:1;overflow-y:auto;padding:6px}
.le{
  display:flex;align-items:flex-start;gap:6px;
  padding:4px 8px;border-radius:6px;border-left:2px solid transparent;
  margin-bottom:2px;
}
.le.INFO{border-color:rgba(99,102,241,.4)}
.le.WARN{border-color:var(--yellow);background:var(--yellow-dim)}
.le.ERROR{border-color:var(--red);background:var(--red-dim)}
.lts{color:var(--text-muted);flex-shrink:0;font-size:9px;padding-top:1px;font-family:monospace}
.llv{
  font-size:8px;font-weight:700;padding:1px 5px;
  border-radius:999px;flex-shrink:0;letter-spacing:.04em;
}
.INFO .llv{background:var(--accent-glow);color:var(--accent-hover)}
.WARN .llv{background:var(--yellow-dim);color:var(--yellow)}
.ERROR .llv{background:var(--red-dim);color:var(--red)}
.lsrc{
  font-size:9px;font-weight:600;color:var(--accent-hover);flex-shrink:0;
  background:var(--accent-glow);padding:1px 6px;border-radius:999px;white-space:nowrap;
}
.lmsg{flex:1;font-size:11px;line-height:1.4;word-break:break-all;color:var(--text-secondary)}
.ldata{
  margin-top:3px;padding:4px 8px;border-radius:6px;
  background:var(--bg-elevated);border:1px solid var(--border);
  font-size:9px;color:var(--text-muted);white-space:pre-wrap;
  cursor:pointer;max-height:60px;overflow:hidden;transition:max-height .2s;
  font-family:monospace;
}
.ldata:hover{color:var(--text-primary)}
.ldata.open{max-height:240px}

/* ── Results ─────────────────────────────────────────── */
.res-body{flex:1;overflow-y:auto;padding:16px}

/* Verdict banner */
.verdict-banner{
  border-radius:10px;padding:14px 16px;margin-bottom:14px;
  display:flex;align-items:center;gap:12px;
  border:1px solid transparent;
}
.verdict-banner.vc{background:var(--green-dim);border-color:rgba(34,197,94,.3)}
.verdict-banner.vp{background:var(--yellow-dim);border-color:rgba(245,158,11,.3)}
.verdict-banner.vu{background:var(--red-dim);border-color:rgba(239,68,68,.3)}
.verdict-banner.vx{background:var(--bg-elevated);border-color:var(--border)}
.verdict-icon{
  width:44px;height:44px;border-radius:10px;
  display:flex;align-items:center;justify-content:center;
  font-size:20px;flex-shrink:0;font-weight:700;
}
.vc .verdict-icon{background:rgba(34,197,94,.2);color:var(--green)}
.vp .verdict-icon{background:rgba(245,158,11,.2);color:var(--yellow)}
.vu .verdict-icon{background:rgba(239,68,68,.2);color:var(--red)}
.vx .verdict-icon{background:var(--bg-elevated);color:var(--text-muted)}
.verdict-body{flex:1}
.verdict-label{font-size:11px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:var(--text-muted);margin-bottom:2px}
.verdict-text{font-size:20px;font-weight:700}
.vc .verdict-text{color:var(--green)}
.vp .verdict-text{color:var(--yellow)}
.vu .verdict-text{color:var(--red)}
.vx .verdict-text{color:var(--text-muted)}
.verdict-score-bar{margin-top:8px;height:4px;background:var(--bg-elevated);border-radius:999px;overflow:hidden}
.verdict-score-fill{height:100%;border-radius:999px;width:0;transition:width 1.2s cubic-bezier(.4,0,.2,1)}
.vc .verdict-score-fill{background:var(--green)}
.vp .verdict-score-fill{background:var(--yellow)}
.vu .verdict-score-fill{background:var(--red)}
.vx .verdict-score-fill{background:var(--text-muted)}
.verdict-score-pct{font-size:12px;font-weight:700;flex-shrink:0;margin-left:4px}
.vc .verdict-score-pct{color:var(--green)}
.vp .verdict-score-pct{color:var(--yellow)}
.vu .verdict-score-pct{color:var(--red)}

/* Identity summary card */
.id-card{
  background:var(--bg-card);border:1px solid var(--border);
  border-radius:10px;padding:14px;margin-bottom:14px;
  transition:all .2s ease;
}
.id-card:hover{border-color:var(--border-bright);transform:translateY(-1px)}
.id-card-head{display:flex;align-items:flex-start;gap:12px;margin-bottom:10px}
.av{
  width:52px;height:52px;border-radius:50%;flex-shrink:0;
  overflow:hidden;display:flex;align-items:center;justify-content:center;
  background:linear-gradient(135deg,var(--accent),var(--accent-hover));
  border:2px solid var(--border-bright);font-size:20px;
}
.av img{width:100%;height:100%;object-fit:cover}
.id-info{flex:1;min-width:0}
.id-name{font-size:17px;font-weight:700;color:var(--text-primary);margin-bottom:4px;letter-spacing:-.3px}
.id-badges{display:flex;flex-wrap:wrap;gap:4px;margin-bottom:6px}
.vb{
  font-size:9px;font-weight:700;letter-spacing:.08em;
  text-transform:uppercase;padding:2px 10px;border-radius:999px;
}
.vc{background:var(--green-dim);color:var(--green);border:1px solid rgba(34,197,94,.35)}
.vp{background:var(--yellow-dim);color:var(--yellow);border:1px solid rgba(245,158,11,.35)}
.vu{background:var(--red-dim);color:var(--red);border:1px solid rgba(239,68,68,.35)}
.id-meta-row{display:flex;align-items:center;gap:6px;font-size:11px;color:var(--text-muted);margin-top:2px}
.id-fields{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-top:8px}
.id-field{background:var(--bg-elevated);border:1px solid var(--border);border-radius:7px;padding:8px 10px}
.id-field-label{font-size:9px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:var(--text-muted);margin-bottom:3px}
.id-field-val{font-size:11px;color:var(--text-primary);word-break:break-all;line-height:1.4}
.id-field-val a{color:var(--accent-hover);text-decoration:none}
.id-field-val a:hover{text-decoration:underline}
.sbar-row{margin:6px 0}
.sbar-h{display:flex;justify-content:space-between;font-size:10px;font-weight:600;color:var(--text-muted);margin-bottom:4px}
.sbar{height:4px;background:var(--bg-elevated);border-radius:999px;overflow:hidden}
.sfill{height:100%;border-radius:999px;width:0;transition:width 1.2s cubic-bezier(.4,0,.2,1)}
.sf1{background:linear-gradient(90deg,var(--accent),var(--accent-hover))}
.sf2{background:linear-gradient(90deg,#a855f7,var(--accent-hover))}
.plinks{display:flex;flex-direction:column;gap:4px;margin-top:8px}
.plink{
  display:flex;align-items:center;gap:6px;padding:6px 10px;
  background:var(--bg-elevated);border:1px solid var(--border);border-radius:8px;
  color:var(--accent-hover);text-decoration:none;font-size:11px;
  transition:all .2s ease;word-break:break-all;
}
.plink:hover{border-color:var(--accent);background:var(--accent-glow);transform:translateY(-1px)}
.plink.visited{color:var(--text-muted);border-color:var(--border)}
.mc-url.visited{opacity:.5}
.result-link{color:var(--accent-hover);text-decoration:none;transition:color .2s}
.result-link:hover{text-decoration:underline;color:var(--accent)}

/* Source breakdown */
.ss-grid{display:flex;flex-direction:column;gap:4px;margin-top:6px}
.ss-row{
  display:flex;align-items:center;gap:8px;padding:6px 10px;
  background:var(--bg-elevated);border:1px solid var(--border);
  border-radius:8px;font-size:11px;
}
.ss-lbl{width:90px;flex-shrink:0;font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.ss-bar{flex:1;height:3px;background:var(--bg-card);border-radius:999px;overflow:hidden}
.ss-fill{height:100%;border-radius:999px;transition:width .8s ease}
.ss-n{flex-shrink:0;width:28px;text-align:right;font-weight:700;font-size:11px}

/* Match cards */
.mc-grid{display:flex;flex-direction:column;gap:8px;margin-top:6px}
.mc{
  background:var(--bg-card);border:1px solid var(--border);
  border-radius:10px;padding:12px 14px;
  transition:all .2s ease;
}
.mc:hover{border-color:var(--border-bright);transform:translateY(-1px);box-shadow:0 4px 16px rgba(0,0,0,.2)}
.mc-head{display:flex;gap:10px;align-items:flex-start}
.mc-av{
  width:40px;height:40px;border-radius:50%;
  object-fit:cover;flex-shrink:0;border:1px solid var(--border);
}
.mc-av-ph{
  width:40px;height:40px;border-radius:50%;
  background:var(--bg-elevated);display:flex;
  align-items:center;justify-content:center;font-size:18px;flex-shrink:0;
  border:1px solid var(--border);
}
.mc-info{flex:1;min-width:0}
.mc-top{display:flex;align-items:center;gap:5px;flex-wrap:wrap;margin-bottom:4px}
.mc-plat{font-size:11px;font-weight:600;color:var(--accent-hover)}
.mc-fv{
  font-size:9px;font-weight:700;letter-spacing:.05em;
  background:var(--green-dim);color:var(--green);
  padding:1px 6px;border-radius:999px;border:1px solid rgba(34,197,94,.3);
}
.mc-src{font-size:9px;color:var(--text-muted);margin-left:auto}
.mc-score-wrap{display:flex;align-items:center;gap:6px;margin-top:4px}
.mc-score-bar-bg{flex:1;height:3px;background:var(--bg-elevated);border-radius:999px;overflow:hidden}
.mc-score-bar-fill{height:100%;border-radius:999px}
.mc-score-val{font-size:10px;font-weight:600;flex-shrink:0;width:32px;text-align:right}
.mc-name{font-size:12px;font-weight:700;color:var(--text-primary);margin-bottom:2px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.mc-meta{font-size:10px;color:var(--text-muted);line-height:1.5}
.mc-snip{
  font-size:10px;color:var(--text-muted);margin-top:6px;
  line-height:1.5;border-left:2px solid var(--border);
  padding-left:8px;
}
.mc-url{
  display:block;font-size:10px;color:var(--accent-hover);
  margin-top:6px;text-decoration:none;
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
  transition:color .2s;
}
.mc-url:hover{text-decoration:underline;color:var(--accent)}
.mc-url.visited{color:#8b5cf6;opacity:.7}
.mc-badge-row{display:flex;align-items:center;gap:4px;margin-top:4px;flex-wrap:wrap}

/* Verdict badges reuse .vb .vc .vp .vu above */

/* Face quality warning */
.face-warn{
  display:none;padding:8px 14px;
  background:var(--yellow-dim);border-left:3px solid var(--yellow);
  color:var(--yellow);font-size:11px;font-weight:500;
  align-items:center;gap:8px;line-height:1.4;flex-shrink:0;
}
.face-warn.show{display:flex}
.face-warn-close{
  margin-left:auto;cursor:pointer;opacity:.7;font-size:14px;
  padding:0 4px;background:none;border:none;color:var(--yellow);
}
.face-warn-close:hover{opacity:1}

/* Score filter toggle */
.toggle-row{display:flex;align-items:center;gap:8px;padding:8px 0 4px;cursor:pointer;user-select:none}
.toggle-switch{
  position:relative;width:34px;height:18px;flex-shrink:0;
}
.toggle-switch input{opacity:0;width:0;height:0;position:absolute}
.toggle-slider{
  position:absolute;inset:0;border-radius:999px;
  background:var(--border-bright);cursor:pointer;
  transition:background .2s;
}
.toggle-slider::before{
  content:'';position:absolute;
  width:12px;height:12px;border-radius:50%;
  left:3px;top:3px;background:#fff;
  transition:transform .2s;box-shadow:0 1px 3px rgba(0,0,0,.3);
}
.toggle-switch input:checked + .toggle-slider{background:var(--accent)}
.toggle-switch input:checked + .toggle-slider::before{transform:translateX(16px)}
.toggle-label{font-size:11px;color:var(--text-muted)}
.toggle-row:hover .toggle-label{color:var(--text-primary)}

/* Query face thumbnail */
.qface-wrap{
  display:flex;align-items:center;gap:10px;padding:8px 10px;
  background:var(--bg-elevated);border:1px solid var(--border);
  border-radius:8px;
}
.qface-img{width:40px;height:40px;border-radius:8px;object-fit:cover;flex-shrink:0;border:1px solid var(--border)}
.qface-lbl{font-size:10px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:var(--text-muted)}

/* Toolbar */
.res-toolbar{
  display:none;flex-shrink:0;padding:8px 16px;
  border-bottom:1px solid var(--border);
  align-items:center;gap:10px;flex-wrap:wrap;
  background:var(--bg-card);
}

/* Section headings within results */
.res-section-title{
  font-size:10px;font-weight:700;letter-spacing:.08em;
  text-transform:uppercase;color:var(--text-muted);
  margin:14px 0 6px;display:flex;align-items:center;gap:6px;
}
.res-section-title::after{
  content:'';flex:1;height:1px;background:var(--border);
}

/* Wait / empty state */
.wait{
  flex:1;display:flex;flex-direction:column;
  align-items:center;justify-content:center;
  padding:40px 20px;text-align:center;gap:12px;
}
.wait-ic{font-size:36px;opacity:.6}
.wait-t{font-size:15px;font-weight:600;color:var(--text-primary)}
.wait-s{font-size:12px;color:var(--text-muted);max-width:280px;line-height:1.7}

/* Toasts */
#toasts{position:fixed;bottom:16px;right:16px;display:flex;flex-direction:column;gap:6px;z-index:998}
.toast{
  background:var(--bg-card);border:1px solid var(--border);
  border-radius:10px;padding:10px 14px;font-size:12px;
  min-width:220px;display:flex;align-items:center;gap:8px;
  animation:slideUp .2s ease;box-shadow:0 8px 24px rgba(0,0,0,.4);
  font-family:'Inter',sans-serif;
}
.toast.ok{border-color:rgba(34,197,94,.35)}
.toast.er{border-color:rgba(239,68,68,.35)}
.toast-icon{font-size:14px;flex-shrink:0}

/* Repeat-face modal */
.modal-bg{
  display:none;position:fixed;inset:0;
  background:rgba(0,0,0,.7);backdrop-filter:blur(4px);
  z-index:999;align-items:center;justify-content:center;
}
.modal-bg.show{display:flex}
.modal{
  background:var(--bg-card);border:1px solid var(--border);
  border-radius:14px;padding:28px;max-width:400px;width:90%;
  box-shadow:0 20px 60px rgba(0,0,0,.5);
  animation:slideUp .25s ease;
}
.modal h2{font-size:16px;font-weight:700;margin-bottom:10px;color:var(--yellow)}
.modal p{font-size:12px;color:var(--text-muted);line-height:1.7;margin-bottom:18px}
.modal-btns{display:flex;gap:8px}

/* Shortcuts modal */
.shortcuts-modal{
  background:var(--bg-card);border:1px solid var(--border);
  border-radius:14px;padding:24px;max-width:340px;width:90%;
  box-shadow:0 20px 60px rgba(0,0,0,.5);animation:slideUp .25s ease;
}
.shortcuts-modal h3{font-size:14px;font-weight:700;margin-bottom:14px;color:var(--text-primary)}
.shortcut-row{display:flex;align-items:center;justify-content:space-between;padding:5px 0;font-size:12px}
.shortcut-desc{color:var(--text-muted)}
.kbd{
  background:var(--bg-elevated);border:1px solid var(--border-bright);
  border-radius:5px;padding:2px 7px;font-size:10px;font-weight:600;
  color:var(--text-primary);font-family:monospace;letter-spacing:.03em;
}

/* Misc utility */
.it{background:var(--bg-elevated);border:1px solid var(--border);border-radius:8px;padding:8px 12px}
.lbl{font-size:10px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:var(--text-muted);margin-bottom:3px}
.val{font-size:11px;word-break:break-all;color:var(--text-primary)}

/* ════════ CROWD INTELLIGENCE CENTER (CIC) ════════════════════════════ */
#cic-overlay{display:none;position:fixed;inset:0;z-index:200;background:var(--bg-base);flex-direction:column}
#cic-overlay.open{display:flex}
.cic-topbar{display:flex;align-items:center;gap:10px;padding:8px 14px;background:var(--bg-card);border-bottom:1px solid var(--border);flex-shrink:0;flex-wrap:wrap}
.cic-title{font-weight:700;font-size:14px;color:var(--accent);white-space:nowrap}
.cic-tabs{display:flex;gap:5px;flex:1;flex-wrap:wrap}
.cic-tab{background:transparent;border:1px solid var(--border);color:var(--text-secondary);border-radius:6px;padding:4px 11px;font-size:11px;cursor:pointer;transition:background .15s,color .15s}
.cic-tab:hover{background:var(--bg-elevated);color:var(--text-primary)}
.cic-tab.active{background:var(--accent);border-color:var(--accent);color:#fff}
.cic-close{background:var(--red-dim);border:1px solid rgba(239,68,68,.4);color:var(--red);border-radius:6px;padding:4px 12px;font-size:11px;cursor:pointer;white-space:nowrap;transition:background .15s}
.cic-close:hover{background:var(--red);color:#fff}
#cic-total-badge{font-size:11px;color:var(--text-secondary);white-space:nowrap}
#cic-total-badge b{color:var(--accent)}
.hb.cic-btn{border-color:rgba(99,102,241,.4);color:rgba(99,102,241,.85)}
.hb.cic-btn:hover,.hb.cic-btn.on{background:var(--accent-glow);border-color:var(--accent);color:var(--accent)}
.cic-panel{display:none;flex:1;overflow:hidden;flex-direction:column;min-height:0}
.cic-panel.active{display:flex}
.cic-status-dot{width:7px;height:7px;border-radius:50%;display:inline-block;flex-shrink:0}
.cic-status-dot.live{background:var(--green);box-shadow:0 0 5px var(--green)}
.cic-status-dot.offline{background:var(--text-muted)}
.cic-cam-grid{display:grid;grid-template-columns:1fr 1fr;grid-template-rows:1fr 1fr;gap:8px;padding:8px;flex:1;min-height:0}
.cic-cam-tile{background:var(--bg-card);border:2px solid var(--border);border-radius:8px;display:flex;flex-direction:column;overflow:hidden;transition:border-color .3s}
.cic-cam-tile.offline{opacity:.5}
.cic-cam-header{display:flex;align-items:center;gap:5px;padding:5px 8px;background:var(--bg-elevated);border-bottom:1px solid var(--border);flex-shrink:0}
.cic-cam-name{font-weight:600;color:var(--text-primary);flex:1;font-size:10px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.cic-cam-badge{background:var(--accent);color:#fff;border-radius:10px;padding:1px 7px;font-size:9px;font-weight:700;white-space:nowrap}
.cic-cam-img{width:100%;flex:1;object-fit:cover;display:block;background:#0a0c14;min-height:0}
.cic-cam-footer{display:flex;align-items:center;gap:8px;padding:4px 8px;font-size:10px;flex-shrink:0;border-top:1px solid var(--border)}
.cic-slot-ctrl{display:flex;gap:3px;margin-left:auto}
.cic-btn-sm{background:var(--bg-base);border:1px solid var(--border);color:var(--text-secondary);border-radius:4px;padding:2px 6px;font-size:9px;cursor:pointer;transition:.15s}
.cic-btn-sm:hover{background:var(--accent);color:#fff;border-color:var(--accent)}
.cic-risk-safe{color:var(--green)}
.cic-risk-caution{color:var(--yellow)}
.cic-risk-high{color:#f97316}
.cic-risk-critical{color:var(--red);font-weight:700;animation:cic-blink 1s ease-in-out infinite}
@keyframes cic-blink{0%,100%{opacity:1}50%{opacity:.35}}
#cic-map{flex:1;min-height:0}
.cic-alerts-layout{flex:1;display:flex;flex-direction:column;min-height:0;overflow:hidden}
.cic-alert-log{flex:1;overflow-y:auto;padding:8px;min-height:0}
.cic-alert-item{display:flex;align-items:flex-start;gap:8px;padding:8px 10px;border-radius:6px;margin-bottom:5px;border-left:3px solid;font-size:11px}
.cic-alert-item.warning{border-color:var(--yellow);background:var(--yellow-dim)}
.cic-alert-item.high{border-color:#f97316;background:rgba(249,115,22,.1)}
.cic-alert-item.critical{border-color:var(--red);background:var(--red-dim)}
.cic-alert-zone{font-weight:700;color:var(--text-primary);white-space:nowrap}
.cic-alert-msg{color:var(--text-secondary);flex:1}
.cic-alert-ts{font-size:9px;color:var(--text-muted);white-space:nowrap}
.cic-llm-panel{border-top:1px solid var(--border);padding:10px;display:flex;flex-direction:column;gap:6px;flex-shrink:0;background:var(--bg-card)}
.cic-llm-title{font-size:11px;font-weight:700;color:var(--accent)}
.cic-llm-response{background:var(--bg-base);border:1px solid var(--border);border-radius:6px;padding:8px;font-size:11px;color:var(--text-primary);min-height:55px;max-height:130px;overflow-y:auto;white-space:pre-wrap;line-height:1.5}
.cic-llm-input-row{display:flex;gap:6px}
#cic-llm-q{flex:1;background:var(--bg-elevated);border:1px solid var(--border);border-radius:6px;padding:6px 10px;font-size:11px;color:var(--text-primary);outline:none}
#cic-llm-q:focus{border-color:var(--accent)}
.cic-analytics-layout{flex:1;display:flex;gap:10px;padding:10px;min-height:0}
.cic-chart-card{flex:1;background:var(--bg-card);border:1px solid var(--border);border-radius:8px;padding:12px;display:flex;flex-direction:column;min-width:0}
.cic-chart-title{font-size:11px;font-weight:600;color:var(--text-secondary);margin-bottom:8px;flex-shrink:0}
.cic-chart-wrap{flex:1;position:relative;min-height:0}
.cic-chart-wrap canvas{position:absolute;inset:0;width:100%!important;height:100%!important}
.cic-khoya-layout{padding:20px;display:flex;flex-direction:column;gap:14px;max-width:560px;margin:0 auto;flex:1;overflow-y:auto}
.cic-khoya-title{font-size:16px;font-weight:700;color:var(--text-primary)}
.cic-khoya-sub{font-size:12px;color:var(--text-secondary);line-height:1.5}
.cic-upload-zone{border:2px dashed var(--border-bright);border-radius:8px;padding:24px;text-align:center;color:var(--text-muted);font-size:12px;cursor:pointer;transition:border-color .15s;display:flex;flex-direction:column;align-items:center;gap:8px}
.cic-upload-zone:hover{border-color:var(--accent);color:var(--text-secondary)}
.cic-khoya-results{background:var(--bg-card);border:1px solid var(--border);border-radius:8px;padding:12px;font-size:12px;line-height:1.7}
.cic-toggle-bar{display:flex;align-items:center;gap:5px;padding:5px 10px;background:var(--bg-elevated);border-bottom:1px solid var(--border);flex-shrink:0;flex-wrap:wrap}
.cic-toggle-sep{flex:1}
.cic-tog{background:var(--bg-base);border:1px solid var(--border);color:var(--text-muted);border-radius:4px;padding:2px 7px;font-size:9px;cursor:pointer;transition:.15s}
.cic-tog.on{background:var(--accent-glow);border-color:var(--accent);color:var(--accent)}
</style>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" crossorigin=""/>
</head>
<body>

<!-- Repeat-face modal -->
<div class="modal-bg" id="modal-bg">
  <div class="modal">
    <h2>&#9888; Face Recognised</h2>
    <p id="modal-txt">This face was previously searched.</p>
    <div class="modal-btns">
      <button class="btn btn-secondary" style="flex:1" onclick="modalCancel()">Cancel</button>
      <button class="btn btn-primary" style="flex:2" onclick="modalProceed()">Search Again</button>
    </div>
  </div>
</div>

<!-- Shortcuts modal -->
<div class="modal-bg" id="shortcuts-bg" onclick="if(event.target===this)this.classList.remove('show')">
  <div class="shortcuts-modal">
    <h3>Keyboard Shortcuts</h3>
    <div class="shortcut-row"><span class="shortcut-desc">Run search</span><span class="kbd">Ctrl + Enter</span></div>
    <div class="shortcut-row"><span class="shortcut-desc">Cancel / clear</span><span class="kbd">Esc</span></div>
    <div class="shortcut-row"><span class="shortcut-desc">Flip camera</span><span class="kbd">F</span></div>
    <div class="shortcut-row"><span class="shortcut-desc">Show shortcuts</span><span class="kbd">?</span></div>
    <div style="margin-top:16px;display:flex;justify-content:flex-end">
      <button class="btn btn-secondary" onclick="document.getElementById('shortcuts-bg').classList.remove('show')">Close</button>
    </div>
  </div>
</div>

<div class="app-shell">

<!-- Top bar -->
<div class="topbar">
  <div class="logo">
    <div class="logo-icon">&#9672;</div>
    <div class="logo-text">
      <span class="logo-name">FaceOSINT</span>
      <span class="logo-sub">Open-source face intelligence</span>
    </div>
  </div>

  <div class="topbar-status">
    <div class="status-dot" id="sd"></div>
    <span id="hdr-msg">System ready</span>
  </div>

  <div class="topbar-actions">
    <button class="bell-btn" id="bell-btn" onclick="toggleNotifications()" title="Browser notifications">&#128276;</button>
    <button class="hb" id="hb-mir" onclick="toggleMirror()">Mirror</button>
    <button class="hb" id="hb-dbg" onclick="toggleDebug()">Debug</button>
    <button class="hb" id="hb-theme" onclick="toggleTheme()" title="Toggle theme">&#9788; Light</button>
    <button class="hb" onclick="document.getElementById('shortcuts-bg').classList.add('show')" title="Keyboard shortcuts">?</button>
    <button class="hb kill" id="hb-kill" disabled onclick="killSearch()">&#9760; Kill</button>
    <button class="hb cic-btn" id="hb-cic" onclick="toggleCIC()" title="Crowd Intelligence Center">&#9889; CIC</button>
  </div>
</div>

<!-- Main layout -->
<div class="main-layout">

<!-- LEFT PANEL -->
<div class="left-panel">

  <div class="left-logo-area">
    <div class="left-logo-title"><span>&#9672;</span> FaceOSINT</div>
    <div class="left-logo-sub">Open-source face intelligence</div>
  </div>

  <!-- Input tabs -->
  <div class="input-tabs-wrap">
    <div class="input-tabs">
      <button class="input-tab act" id="t-cam"  onclick="switchInput('cam')">Camera</button>
      <button class="input-tab"     id="t-upl"  onclick="switchInput('upl')">Upload</button>
      <button class="input-tab"     id="t-wifi" onclick="switchInput('wifi')">WiFi</button>
    </div>
  </div>

  <div class="input-content">

    <!-- Camera panel -->
    <div id="cam-panel">
      <div class="cam-w">
        <video id="video" autoplay muted playsinline></video>
        <canvas id="cvs"></canvas>
        <div id="cam-off">
          <div id="cam-off-ic">&#128274;</div>
          <div id="cam-off-txt">CAMERA OFF</div>
        </div>
        <div class="cam-ov">
          <div class="cn c-tl"></div><div class="cn c-tr"></div>
          <div class="cn c-bl"></div><div class="cn c-br"></div>
          <div class="scanl"></div>
        </div>
      </div>
    </div>

    <!-- Upload panel -->
    <div id="upl-panel" style="display:none">
      <div class="drop-z" id="drop-z"
           onclick="document.getElementById('file-in').click()"
           ondragover="event.preventDefault();this.classList.add('ov')"
           ondragleave="this.classList.remove('ov')"
           ondrop="onDrop(event)">
        <img id="upl-img" alt="">
        <div id="dz-hint">
          <div class="dz-icon">&#128444;</div>
          <div class="dz-text">Click or drag image here</div>
          <div class="dz-hint">JPG &middot; PNG &middot; WEBP &middot; BMP</div>
        </div>
      </div>
      <input type="file" id="file-in" accept="image/*" onchange="onFileSelect(event)">
    </div>

    <!-- WiFi camera panel -->
    <div id="wifi-panel" style="display:none">
      <div class="wifi-panel">
        <div class="field-label">Phone IP (IP Webcam app)</div>
        <div class="wifi-row">
          <input class="wifi-ip" id="wifi-ip" type="text"
                 placeholder="192.168.1.7:8080" autocomplete="off"
                 onkeydown="if(event.key==='Enter')wifiConnect()">
          <button class="btn btn-secondary" id="wifi-btn-con" onclick="wifiConnect()" style="flex:none;padding:6px 12px;font-size:11px">Connect</button>
          <button class="btn btn-secondary" id="wifi-btn-dis" onclick="wifiDisconnect()" style="flex:none;padding:6px 12px;font-size:11px;display:none">Disconnect</button>
        </div>
        <div class="wifi-status">
          <div class="ws-dot" id="ws-dot"></div>
          <span id="ws-txt" style="color:var(--text-muted)">Not connected</span>
        </div>
      </div>
      <div class="wifi-prev" id="wifi-prev">
        <div class="wifi-prev-ph" id="wifi-prev-ph">
          <div style="font-size:24px">&#128225;</div>
          <div>Connect to see preview</div>
        </div>
        <img id="wifi-img" style="display:none" alt="WiFi preview">
        <div class="wifi-fps" id="wifi-fps" style="display:none">--fps</div>
        <canvas id="wifi-cvs" style="display:none"></canvas>
      </div>
    </div>

  </div><!-- /input-content -->

  <!-- Controls -->
  <div class="ctrl">
    <div>
      <div class="field-label">Name &amp; optional hints</div>
      <input class="fi" id="name-in" type="text"
             placeholder="Name | Location @ Company (optional)"
             autocomplete="off"
             oninput="onNameInput(this.value)">
      <div class="fi-hint">
        <code>John Doe, Mumbai</code> &middot; <code>John Doe @ Google</code> &middot; <code>John Doe | TCS | Delhi</code>
      </div>
      <div class="parsed-tags" id="parsed-tags"></div>
    </div>
    <div class="btn-row">
      <button class="btn btn-secondary" id="btn-cap" onclick="capture()">&#128247; Capture</button>
      <button class="btn btn-secondary" id="btn-ret" onclick="retake()" style="display:none">&#8617; Retake</button>
    </div>
    <button class="btn btn-primary btn-full" id="btn-srch" disabled onclick="startSearch()">&#128269; Search OSINT</button>
  </div>

  <!-- Files -->
  <div class="card-section" id="card-files" style="display:none">
    <div class="section-header">
      <span class="section-title">Stored Files</span>
    </div>
    <div class="file-panel" id="file-panel"></div>
    <div class="del-btns">
      <button class="dbtn dbtn-warn" id="dbtn-img" onclick="cleanup('images')" disabled>&#128465; Del Images</button>
      <button class="dbtn dbtn-red"  id="dbtn-all" onclick="cleanup('full')"   disabled>&#9760; Wipe All</button>
    </div>
  </div>

  <!-- History -->
  <div class="hist-section">
    <div class="section-header">
      <span class="section-title">History</span>
      <div style="display:flex;gap:6px">
        <button class="hb" style="padding:3px 8px;font-size:10px" onclick="loadHistory()">&#8635;</button>
        <button class="hb" style="padding:3px 8px;font-size:10px;color:var(--red)" onclick="clearHistory()">Clear</button>
      </div>
    </div>
    <div class="hist" id="hist-list">
      <div class="wait" style="padding:16px"><div class="wait-s">No searches yet</div></div>
    </div>
  </div>

</div><!-- /left-panel -->

<!-- RIGHT PANEL -->
<div class="right-panel">

  <!-- Right tabs -->
  <div class="right-tabs">
    <button class="rtab act" id="tr-prog" onclick="switchTab('prog')">&#128225; Progress</button>
    <button class="rtab"     id="tr-log"  onclick="switchTab('log')">Debug <span class="rtab-badge" id="lcnt">0</span></button>
    <button class="rtab"     id="tr-res"  onclick="switchTab('res')" style="display:none">&#128203; Results</button>
  </div>

  <!-- Source matrix -->
  <div class="matrix" id="matrix"></div>

  <!-- Progress tab -->
  <div id="tp-prog" style="display:flex;flex-direction:column;flex:1;overflow:hidden;min-height:0">
    <div class="steps" id="steps-list">
      <div class="wait">
        <div class="wait-ic">&#128752;</div>
        <div class="wait-t">Ready to Investigate</div>
        <div class="wait-s">
          Capture or upload a face, enter the target's name
          (optionally add city or employer), then click Search OSINT.<br><br>
          Each search hits 10+ intelligence sources simultaneously.
        </div>
      </div>
    </div>
    <div id="live-feed-wrap" class="live-feed-wrap" style="display:none">
      <div class="live-feed-hdr">
        <span class="live-feed-title" id="live-feed-hd">Live matches (0)</span>
      </div>
      <div id="live-feed"></div>
    </div>
  </div>

  <!-- Log tab -->
  <div id="tp-log" style="display:none;flex-direction:column;flex:1;overflow:hidden;min-height:0">
    <div style="display:flex;align-items:center;justify-content:space-between;padding:8px 14px;border-bottom:1px solid var(--border);flex-shrink:0">
      <span style="font-size:10px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:var(--text-muted)">Activity Log</span>
      <button class="hb" style="padding:3px 8px;font-size:10px" onclick="clearLog()">Clear</button>
    </div>
    <div class="log-panel" id="log-panel">
      <div class="wait" style="padding:20px"><div class="wait-s">Log appears here during search</div></div>
    </div>
  </div>

  <!-- Results tab -->
  <div id="tp-res" style="display:none;flex:1;overflow:hidden;min-height:0;flex-direction:column">
    <div class="face-warn" id="face-warn-banner">
      <span>&#9888;</span>
      <span>Low quality face detected &mdash; accuracy may be reduced</span>
      <button class="face-warn-close" onclick="document.getElementById('face-warn-banner').classList.remove('show')" title="Dismiss">&#10005;</button>
    </div>
    <div id="res-toolbar" class="res-toolbar">
      <div id="qface-wrap" class="qface-wrap" style="display:none;flex:none">
        <img id="qface-img" class="qface-img" alt="Query Face" src="">
        <div class="qface-lbl">Query Face</div>
      </div>
      <a id="btn-dl-report" class="btn btn-ghost" style="display:none;flex:none;font-size:11px;padding:6px 12px;text-decoration:none" download>&#8681; Report</a>
      <label class="toggle-row" id="score-filter-wrap" style="display:none">
        <span class="toggle-switch">
          <input type="checkbox" id="chk-show-unlikely" onchange="applyScoreFilter()">
          <span class="toggle-slider"></span>
        </span>
        <span class="toggle-label">Show unlikely (score &lt; 0.55)</span>
      </label>
    </div>
    <div class="res-body" id="res-panel">
      <div class="wait"><div class="wait-ic">&#128203;</div><div class="wait-t">No results yet</div></div>
    </div>
  </div>

</div><!-- /right-panel -->

</div><!-- /main-layout -->
</div><!-- /app-shell -->

<!-- ╔══════════════════════════════════════════════════════════════════╗
     ║  CROWD INTELLIGENCE CENTER OVERLAY                               ║
     ╚══════════════════════════════════════════════════════════════════╝ -->
<div id="cic-overlay">
  <!-- CIC top bar -->
  <div class="cic-topbar">
    <span class="cic-title">&#9889; Crowd Intelligence Center</span>
    <div class="cic-tabs">
      <button class="cic-tab active" id="cic-tabn-cameras" onclick="cicTab('cameras')">&#128249; Cameras</button>
      <button class="cic-tab" id="cic-tabn-map"       onclick="cicTab('map')">&#128205; Zone Map</button>
      <button class="cic-tab" id="cic-tabn-alerts"    onclick="cicTab('alerts')">&#128680; Alerts &amp; SOP</button>
      <button class="cic-tab" id="cic-tabn-analytics" onclick="cicTab('analytics')">&#128202; Analytics</button>
      <button class="cic-tab" id="cic-tabn-khoya"     onclick="cicTab('khoya')">&#128269; Lost Person</button>
    </div>
    <div id="cic-total-badge">Total: <b>0</b> persons</div>
    <button class="cic-close" onclick="toggleCIC()">&#10005; Exit CIC</button>
  </div>

  <!-- Tab 1: Live Cameras 2x2 grid -->
  <div class="cic-panel active" id="cic-panel-cameras">
    <!-- Overlay toggles toolbar -->
    <div class="cic-toggle-bar" id="cic-toggle-bar">
      <span style="font-size:10px;color:var(--text-muted);white-space:nowrap">OVERLAY:</span>
      <button class="cic-tog on" id="tog-show_bbox"       onclick="cicToggle('show_bbox')">Boxes</button>
      <button class="cic-tog on" id="tog-show_track_id"   onclick="cicToggle('show_track_id')">Track&nbsp;ID</button>
      <button class="cic-tog on" id="tog-show_suspicious" onclick="cicToggle('show_suspicious')">Suspicious</button>
      <button class="cic-tog on" id="tog-show_children"   onclick="cicToggle('show_children')">Children</button>
      <button class="cic-tog on" id="tog-show_flow"       onclick="cicToggle('show_flow')">Flow&nbsp;Arrow</button>
      <button class="cic-tog on" id="tog-show_count"      onclick="cicToggle('show_count')">Count</button>
      <span class="cic-toggle-sep"></span>
      <span style="font-size:10px;color:var(--text-muted)">ALARM:</span>
      <button class="cic-tog" id="tog-audio" onclick="cicToggleAudio()">&#128266; Audio</button>
    </div>
    <!-- 2x2 camera grid -->
    <div class="cic-cam-grid">
      <div class="cic-cam-tile offline" id="cic-tile-0">
        <div class="cic-cam-header">
          <span class="cic-status-dot offline" id="cic-dot-0"></span>
          <span class="cic-cam-name">Slot 0 &mdash; Sangam Ghat</span>
          <span class="cic-cam-badge" id="cic-badge-0">--</span>
          <div class="cic-slot-ctrl">
            <button class="cic-btn-sm" onclick="cicStartSlot(0)">&#9654; Start</button>
            <button class="cic-btn-sm" onclick="cicStopSlot(0)">&#9632; Stop</button>
          </div>
        </div>
        <img class="cic-cam-img" id="cic-frame-0" alt=""/>
        <div class="cic-cam-footer">
          <span class="cic-risk-safe" id="cic-risk-0">OFFLINE</span>
          <span style="color:var(--text-muted);font-size:10px" id="cic-dens-0">0.000 p/m&#178;</span>
          <span style="color:var(--text-muted);font-size:9px;margin-left:auto" id="cic-bhv-0"></span>
        </div>
      </div>
      <div class="cic-cam-tile offline" id="cic-tile-1">
        <div class="cic-cam-header">
          <span class="cic-status-dot offline" id="cic-dot-1"></span>
          <span class="cic-cam-name">Slot 1 &mdash; Pontoon Bridge</span>
          <span class="cic-cam-badge" id="cic-badge-1">--</span>
          <div class="cic-slot-ctrl">
            <button class="cic-btn-sm" onclick="cicStartSlot(1)">&#9654; Start</button>
            <button class="cic-btn-sm" onclick="cicStopSlot(1)">&#9632; Stop</button>
          </div>
        </div>
        <img class="cic-cam-img" id="cic-frame-1" alt=""/>
        <div class="cic-cam-footer">
          <span class="cic-risk-safe" id="cic-risk-1">OFFLINE</span>
          <span style="color:var(--text-muted);font-size:10px" id="cic-dens-1">0.000 p/m&#178;</span>
          <span style="color:var(--text-muted);font-size:9px;margin-left:auto" id="cic-bhv-1"></span>
        </div>
      </div>
      <div class="cic-cam-tile offline" id="cic-tile-2">
        <div class="cic-cam-header">
          <span class="cic-status-dot offline" id="cic-dot-2"></span>
          <span class="cic-cam-name">Slot 2 &mdash; Sector 4 Entry</span>
          <span class="cic-cam-badge" id="cic-badge-2">--</span>
          <div class="cic-slot-ctrl">
            <button class="cic-btn-sm" onclick="cicStartSlot(2)">&#9654; Start</button>
            <button class="cic-btn-sm" onclick="cicStopSlot(2)">&#9632; Stop</button>
          </div>
        </div>
        <img class="cic-cam-img" id="cic-frame-2" alt=""/>
        <div class="cic-cam-footer">
          <span class="cic-risk-safe" id="cic-risk-2">OFFLINE</span>
          <span style="color:var(--text-muted);font-size:10px" id="cic-dens-2">0.000 p/m&#178;</span>
          <span style="color:var(--text-muted);font-size:9px;margin-left:auto" id="cic-bhv-2"></span>
        </div>
      </div>
      <div class="cic-cam-tile offline" id="cic-tile-3">
        <div class="cic-cam-header">
          <span class="cic-status-dot offline" id="cic-dot-3"></span>
          <span class="cic-cam-name">Slot 3 &mdash; Approach Road</span>
          <span class="cic-cam-badge" id="cic-badge-3">--</span>
          <div class="cic-slot-ctrl">
            <button class="cic-btn-sm" onclick="cicStartSlot(3)">&#9654; Start</button>
            <button class="cic-btn-sm" onclick="cicStopSlot(3)">&#9632; Stop</button>
          </div>
        </div>
        <img class="cic-cam-img" id="cic-frame-3" alt=""/>
        <div class="cic-cam-footer">
          <span class="cic-risk-safe" id="cic-risk-3">OFFLINE</span>
          <span style="color:var(--text-muted);font-size:10px" id="cic-dens-3">0.000 p/m&#178;</span>
          <span style="color:var(--text-muted);font-size:9px;margin-left:auto" id="cic-bhv-3"></span>
        </div>
      </div>
    </div>
  </div>

  <!-- Tab 2: Zone Map (Leaflet.js) -->
  <div class="cic-panel" id="cic-panel-map">
    <div style="display:flex;align-items:center;gap:8px;padding:6px 10px;background:var(--bg-card);border-bottom:1px solid var(--border);flex-shrink:0;font-size:11px">
      <span style="color:var(--text-muted)">Map layers:</span>
      <button class="cic-btn-sm" id="map-toggle-heat" onclick="cicToggleHeatmap()">&#127777; Heat Map OFF</button>
      <button class="cic-btn-sm" onclick="cicRefreshMap()">&#8635; Refresh Zones</button>
      <span style="margin-left:auto;color:var(--text-muted)">Prayagraj Kumbh Mela — Triveni Sangam</span>
    </div>
    <div id="cic-map" style="flex:1;min-height:0"></div>
  </div>

  <!-- Tab 3: Alerts + LLM SOP assistant -->
  <div class="cic-panel" id="cic-panel-alerts">
    <div class="cic-alerts-layout">
      <div class="cic-alert-log" id="cic-alert-log">
        <div style="color:var(--text-muted);font-size:12px;padding:12px">No alerts &mdash; monitoring active zones&hellip;</div>
      </div>
      <div class="cic-llm-panel">
        <div class="cic-llm-title">&#9889; AI Operator Assistant (Claude)</div>
        <div class="cic-llm-response" id="cic-llm-response" style="color:var(--text-muted)">Ask about current crowd conditions, which zones need attention, or SOP guidance&hellip;</div>
        <div class="cic-llm-input-row">
          <input id="cic-llm-q" type="text" placeholder="e.g. Which zones need immediate attention?" onkeydown="if(event.key==='Enter')cicAsk()"/>
          <button class="cic-btn-sm" onclick="cicAsk()" style="padding:5px 14px;font-size:11px">Ask</button>
        </div>
      </div>
    </div>
  </div>

  <!-- Tab 4: Analytics -->
  <div class="cic-panel" id="cic-panel-analytics">
    <div class="cic-analytics-layout">
      <div class="cic-chart-card">
        <div class="cic-chart-title">Zone Occupancy (persons detected)</div>
        <div class="cic-chart-wrap"><canvas id="cic-bar-canvas"></canvas></div>
      </div>
      <div class="cic-chart-card">
        <div class="cic-chart-title">Zone Density (persons / m&#178;)</div>
        <div class="cic-chart-wrap"><canvas id="cic-density-canvas"></canvas></div>
      </div>
    </div>
  </div>

  <!-- Tab 5: Khoya-Paya (Lost Person) -->
  <div class="cic-panel" id="cic-panel-khoya">
    <div class="cic-khoya-layout">
      <div class="cic-khoya-title">&#128269; Khoya-Paya &mdash; Lost Person Search</div>
      <div class="cic-khoya-sub">Upload a photo of the missing person. The system will search face embeddings collected from all active camera feeds and historical searches.</div>
      <div class="cic-upload-zone" onclick="document.getElementById('cic-khoya-input').click()">
        <img id="cic-khoya-preview" style="max-height:130px;max-width:200px;border-radius:6px;display:none" alt=""/>
        <div>&#128247; Click to upload photo</div>
        <div style="font-size:10px;color:var(--text-muted)">JPG, PNG, WEBP supported</div>
      </div>
      <input type="file" id="cic-khoya-input" accept="image/*" style="display:none" onchange="cicKhoyaFile(this)"/>
      <div class="cic-khoya-results" id="cic-khoya-results" style="display:none"></div>
    </div>
  </div>
</div>
<!-- /CIC overlay -->

<div id="toasts"></div>

<script>
const STEP_DEFS = [
  {k:'embedding',ic:'🧠',nm:'Face Embedding'},
  {k:'db',       ic:'🗃',nm:'DB Check'},
  {k:'folder',   ic:'📁',nm:'Output Folder'},
  {k:'scraping', ic:'🌐',nm:'Intel Scraping'},
  {k:'matching', ic:'👤',nm:'Face Matching'},
  {k:'scoring',  ic:'📊',nm:'Scoring'},
  {k:'resolving',ic:'🔗',nm:'Entity Resolve'},
  {k:'writing',  ic:'📝',nm:'Report Write'},
];
const SRC_NODES = [
  {k:'reverse_face',  ic:'🔍',nm:'Face Search'},
  {k:'search_engines',ic:'🌐',nm:'Web Search'},
  {k:'github',        ic:'🐙',nm:'GitHub'},
  {k:'reddit',        ic:'🟠',nm:'Reddit'},
  {k:'academic',      ic:'📚',nm:'Academic'},
  {k:'passive',       ic:'🕵',nm:'Passive'},
  {k:'username',      ic:'🔑',nm:'Usernames'},
];
const SLBL={search_engines:'Web',academic:'Academic',github:'GitHub',
  reddit:'Reddit',passive:'Passive',reverse_face:'FaceSearch',
  username:'Usernames'};

// State
let captured=null, searchId=null, es=null, curSid=null;
let mirrored=false, debugOn=false, activeTab='prog', logCount=0;
let camStream=null, pendingSearch=null;
// WiFi camera state
let wifiPollTimer=null, wifiConnected=false, inputMode='cam';

// Build source matrix
(function(){
  const m=document.getElementById('matrix');
  SRC_NODES.forEach(({k,ic,nm})=>{
    const d=document.createElement('div');
    d.className='node'; d.id='node-'+k;
    d.innerHTML=`<div class="node-ic">${ic}</div>`+
      `<div class="node-nm">${nm}</div>`+
      `<div class="node-st" id="ns-${k}">IDLE</div>`+
      `<div class="node-cnt" id="nc-${k}">—</div>`;
    m.appendChild(d);
  });
})();

// Camera init
const vid=document.getElementById('video');
const cvs=document.getElementById('cvs');
(async()=>{
  try{
    camStream=await navigator.mediaDevices.getUserMedia({
      video:{width:{ideal:1280},height:{ideal:720},facingMode:'user'}
    });
    vid.srcObject=camStream;
  }catch(e){ toast('Camera: '+e.message,'er'); }
})();

// ISSUE 1 FIX + v4.2: capture dispatches to WiFi mode if active
function capture(){
  if(inputMode==='wifi'){wifiCapture();return;}
  if(!vid.videoWidth){toast('Camera not ready','er');return;}
  cvs.width=vid.videoWidth; cvs.height=vid.videoHeight;
  const ctx2=cvs.getContext('2d');
  if(mirrored){ ctx2.translate(cvs.width,0); ctx2.scale(-1,1); }
  ctx2.drawImage(vid,0,0);
  if(mirrored) ctx2.setTransform(1,0,0,1,0,0);
  captured=cvs.toDataURL('image/jpeg',.93);

  // STOP camera immediately — turns LED off
  stopCamera();

  vid.style.display='none';
  cvs.style.display='block';
  document.getElementById('cam-off').classList.add('show');
  document.getElementById('btn-cap').style.display='none';
  document.getElementById('btn-ret').style.display='';
  document.getElementById('btn-srch').disabled=false;
  toast('Photo captured — camera off ✓','ok');
}

function stopCamera(){
  if(camStream){
    camStream.getTracks().forEach(t=>t.stop());
    camStream=null;
    vid.srcObject=null;
  }
}

function retake(){
  if(inputMode==='wifi'){wifiRetake();return;}
  captured=null;
  document.getElementById('cam-off').classList.remove('show');
  vid.style.display='block'; cvs.style.display='none';
  document.getElementById('btn-cap').style.display='';
  document.getElementById('btn-ret').style.display='none';
  document.getElementById('btn-srch').disabled=true;
  // Restart local camera
  (async()=>{
    try{
      camStream=await navigator.mediaDevices.getUserMedia({
        video:{width:{ideal:1280},height:{ideal:720},facingMode:'user'}
      });
      vid.srcObject=camStream;
      vid.classList.toggle('mir',mirrored);
    }catch(e){toast('Camera error: '+e.message,'er');}
  })();
}

function toggleMirror(){
  mirrored=!mirrored;
  vid.classList.toggle('mir',mirrored);
  document.getElementById('hb-mir').classList.toggle('on',mirrored);
}
function toggleDebug(){
  debugOn=!debugOn;
  document.getElementById('hb-dbg').classList.toggle('on',debugOn);
  if(debugOn) switchTab('log');
  toast(debugOn?'Debug ON':'Debug OFF','ok');
}

// ── Input mode switcher (cam / wifi / upl) ───────────────────────────────
function switchInput(m){
  inputMode=m;
  document.getElementById('cam-panel').style.display  =m==='cam' ?'':'none';
  document.getElementById('wifi-panel').style.display =m==='wifi'?'':'none';
  document.getElementById('upl-panel').style.display  =m==='upl' ?'':'none';
  document.getElementById('btn-cap').style.display =m==='upl' ?'none':'';
  // input-tab uses "act" class (pill style)
  ['cam','upl','wifi'].forEach(k=>{
    const el=document.getElementById('t-'+k);
    if(el) el.classList.toggle('act',k===m);
  });
  if(m!=='cam') stopCamera();
  if(m!=='wifi'){
    wifiStopPoll();
  } else {
    if(wifiConnected) wifiStartPoll();
  }
  if(m==='upl'&&captured) document.getElementById('btn-srch').disabled=false;
}

// ── WiFi camera ───────────────────────────────────────────────────────────
function wifiSetStatus(state,msg){
  const dot=document.getElementById('ws-dot');
  const txt=document.getElementById('ws-txt');
  dot.className='ws-dot'+(state==='on'?' on':state==='busy'?' busy':state==='err'?' err':'');
  txt.textContent=msg;
  txt.style.color=state==='on'?'var(--green)':state==='err'?'var(--red)':state==='busy'?'var(--accent)':'var(--text-muted)';
}

async function wifiConnect(){
  const ip=(document.getElementById('wifi-ip').value||'').trim();
  if(!ip){toast('Enter IP address first','er');return;}
  document.getElementById('wifi-btn-con').disabled=true;
  wifiSetStatus('busy','Probing '+ip+' …');
  toast('Connecting to '+ip+' …','ok');
  try{
    const r=await fetch('/api/wifi/connect',{
      method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({url:ip})
    });
    const d=await r.json();
    if(!d.ok){
      wifiSetStatus('err',d.error||'Connection failed');
      toast(d.error||'WiFi connect failed','er');
      document.getElementById('wifi-btn-con').disabled=false;
      return;
    }
    wifiConnected=true;
    wifiSetStatus('on','Connected · '+d.width+'×'+d.height+' · '+ip);
    toast('WiFi camera connected ✓','ok');
    document.getElementById('wifi-btn-con').style.display='none';
    document.getElementById('wifi-btn-dis').style.display='';
    document.getElementById('wifi-btn-con').disabled=false;
    wifiStartPoll();
  }catch(e){
    wifiSetStatus('err','Error: '+e.message);
    toast('WiFi connect error: '+e.message,'er');
    document.getElementById('wifi-btn-con').disabled=false;
  }
}

async function wifiDisconnect(){
  wifiStopPoll();
  wifiConnected=false;
  await fetch('/api/wifi/disconnect',{method:'POST'}).catch(()=>{});
  wifiSetStatus('','Not connected — enter IP above');
  document.getElementById('wifi-btn-con').style.display='';
  document.getElementById('wifi-btn-dis').style.display='none';
  document.getElementById('wifi-prev-ph').style.display='flex';
  document.getElementById('wifi-img').style.display='none';
  document.getElementById('wifi-fps').style.display='none';
  captured=null;
  document.getElementById('btn-srch').disabled=true;
  toast('WiFi camera disconnected','ok');
}

function wifiStartPoll(){
  wifiStopPoll();
  wifiPollTimer=setInterval(wifiPollFrame,80);
}
function wifiStopPoll(){
  if(wifiPollTimer){clearInterval(wifiPollTimer);wifiPollTimer=null;}
}

let _wifiPollActive=false;
async function wifiPollFrame(){
  if(_wifiPollActive) return;
  _wifiPollActive=true;
  try{
    const r=await fetch('/api/wifi/frame');
    if(!r.ok){_wifiPollActive=false;return;}
    const d=await r.json();
    if(!d.ok){_wifiPollActive=false;return;}
    const img=document.getElementById('wifi-img');
    const ph=document.getElementById('wifi-prev-ph');
    const fps=document.getElementById('wifi-fps');
    if(img.style.display==='none'){
      img.style.display='block';
      ph.style.display='none';
      fps.style.display='';
    }
    img.src=d.frame;
    fps.textContent=(d.fps||'?')+'fps';
  }catch(e){/* ignore poll errors */}
  _wifiPollActive=false;
}

function wifiCapture(){
  const img=document.getElementById('wifi-img');
  if(!img||img.style.display==='none'){toast('No WiFi frame available — is camera connected?','er');return;}
  const cvs=document.getElementById('wifi-cvs');
  cvs.width=img.naturalWidth||img.width||640;
  cvs.height=img.naturalHeight||img.height||480;
  const ctx=cvs.getContext('2d');
  ctx.drawImage(img,0,0);
  captured=cvs.toDataURL('image/jpeg',.93);
  // Show overlay on preview
  const ph=document.getElementById('wifi-prev-ph');
  ph.innerHTML='<div style="font-size:20px">📸</div><div>Captured ✓</div>';
  ph.style.display='flex';
  ph.style.background='rgba(5,5,15,.65)';
  img.style.opacity='.35';
  wifiStopPoll();
  document.getElementById('btn-srch').disabled=false;
  document.getElementById('btn-ret').style.display='';
  toast('WiFi photo captured ✓','ok');
}

function wifiRetake(){
  // Reset capture overlay and resume poll
  captured=null;
  const img=document.getElementById('wifi-img');
  const ph=document.getElementById('wifi-prev-ph');
  img.style.opacity='1';
  ph.style.display='none';
  ph.style.background='';
  document.getElementById('btn-srch').disabled=true;
  document.getElementById('btn-ret').style.display='none';
  if(wifiConnected) wifiStartPoll();
}

// ── Theme toggle ──────────────────────────────────────────────────────────
(function(){
  const saved=localStorage.getItem('osint_theme')||'dark';
  applyTheme(saved);
})();

function toggleTheme(){
  const cur=document.documentElement.getAttribute('data-theme')||'dark';
  applyTheme(cur==='dark'?'light':'dark');
}
function applyTheme(t){
  document.documentElement.setAttribute('data-theme',t);
  // Update CSS vars for light mode
  if(t==='light'){
    document.documentElement.style.setProperty('--bg-base','#f0f4f8');
    document.documentElement.style.setProperty('--bg-card','#ffffff');
    document.documentElement.style.setProperty('--bg-elevated','#f0f4f8');
    document.documentElement.style.setProperty('--border','#d1d9e0');
    document.documentElement.style.setProperty('--border-bright','#9ba8b5');
    document.documentElement.style.setProperty('--text-primary','#1e293b');
    document.documentElement.style.setProperty('--text-secondary','#475569');
    document.documentElement.style.setProperty('--text-muted','#94a3b8');
    // legacy vars
    document.documentElement.style.setProperty('--bg','#f0f4f8');
    document.documentElement.style.setProperty('--s1','#ffffff');
    document.documentElement.style.setProperty('--s2','#f0f4f8');
    document.documentElement.style.setProperty('--s3','#d1d9e0');
    document.documentElement.style.setProperty('--bd','#d1d9e0');
    document.documentElement.style.setProperty('--txt','#1e293b');
    document.documentElement.style.setProperty('--mute','#94a3b8');
  } else {
    document.documentElement.style.setProperty('--bg-base','#0d0f17');
    document.documentElement.style.setProperty('--bg-card','#13161f');
    document.documentElement.style.setProperty('--bg-elevated','#1c1f2e');
    document.documentElement.style.setProperty('--border','#252836');
    document.documentElement.style.setProperty('--border-bright','#3a3f5c');
    document.documentElement.style.setProperty('--text-primary','#e2e8f0');
    document.documentElement.style.setProperty('--text-secondary','#94a3b8');
    document.documentElement.style.setProperty('--text-muted','#64748b');
    document.documentElement.style.setProperty('--bg','#0d0f17');
    document.documentElement.style.setProperty('--s1','#13161f');
    document.documentElement.style.setProperty('--s2','#1c1f2e');
    document.documentElement.style.setProperty('--s3','#252836');
    document.documentElement.style.setProperty('--bd','#252836');
    document.documentElement.style.setProperty('--txt','#e2e8f0');
    document.documentElement.style.setProperty('--mute','#64748b');
  }
  const btn=document.getElementById('hb-theme');
  if(btn){
    btn.innerHTML=t==='dark'?'&#9788; Light':'&#9790; Dark';
    btn.classList.toggle('on',t==='light');
  }
  try{localStorage.setItem('osint_theme',t);}catch(e){}
}

// ── Notifications ─────────────────────────────────────────────────────────
let notifEnabled=false;
function toggleNotifications(){
  if(notifEnabled){
    notifEnabled=false;
    document.getElementById('bell-btn').classList.remove('on');
    toast('Notifications off','ok');
  } else {
    if(typeof Notification==='undefined'){toast('Notifications not supported','er');return;}
    if(Notification.permission==='granted'){
      notifEnabled=true;
      document.getElementById('bell-btn').classList.add('on');
      toast('Notifications on','ok');
    } else if(Notification.permission!=='denied'){
      Notification.requestPermission().then(p=>{
        if(p==='granted'){
          notifEnabled=true;
          document.getElementById('bell-btn').classList.add('on');
          toast('Notifications enabled','ok');
        } else {
          toast('Notification permission denied','er');
        }
      });
    } else {
      toast('Notifications blocked in browser settings','er');
    }
  }
}
function _sendNotification(title,body){
  if(!notifEnabled||typeof Notification==='undefined') return;
  if(Notification.permission==='granted'){
    try{ new Notification(title,{body}); }catch(e){}
  }
}

// ── Keyboard shortcuts ────────────────────────────────────────────────────
function flipCamera(){ toggleMirror(); }
document.addEventListener('keydown',e=>{
  if(e.target&&e.target.matches('input,textarea,select')) return;
  if((e.ctrlKey||e.metaKey)&&e.key==='Enter'){ e.preventDefault(); startSearch(); return; }
  if(e.key==='Escape'){
    // Close any open modal first
    const modals=['modal-bg','shortcuts-bg'];
    for(const id of modals){
      const el=document.getElementById(id);
      if(el&&el.classList.contains('show')){ el.classList.remove('show'); return; }
    }
    // Cancel active search
    if(searchId){ killSearch(); return; }
    // Clear captured image
    if(captured){ retake(); }
    return;
  }
  if(e.key==='F'||e.key==='f'){ flipCamera(); return; }
  if(e.key==='?'){ document.getElementById('shortcuts-bg').classList.toggle('show'); return; }
});
function onFileSelect(e){const f=e.target.files[0];if(f)readImg(f);}
function onDrop(e){
  e.preventDefault(); document.getElementById('drop-z').classList.remove('ov');
  const f=e.dataTransfer.files[0];
  if(f&&f.type.startsWith('image/')) readImg(f);
}
function readImg(file){
  const fr=new FileReader();
  fr.onload=ev=>{
    captured=ev.target.result;
    const p=document.getElementById('upl-img');
    p.src=captured; p.style.display='block';
    document.getElementById('dz-hint').style.display='none';
    document.getElementById('btn-srch').disabled=false;
    toast('Image loaded ✓','ok');
  };
  fr.readAsDataURL(file);
}

// Name parsing preview (client side)
function onNameInput(v){
  const tags=document.getElementById('parsed-tags');
  if(!v.trim()){tags.innerHTML='';return;}
  let loc='', co='';
  let tmp=v;
  if(tmp.includes(' @ ')){const p=tmp.split(' @ ',2);tmp=p[0];co=p[1];}
  if(tmp.includes('|')){const p=tmp.split('|');tmp=p[0];loc=p[1]||'';if(!co&&p[2])co=p[2];}
  else if(tmp.includes(',')){const p=tmp.split(',',2);tmp=p[0];loc=p[1]||'';}
  let html='';
  if(loc.trim()) html+=`<span class="ptag ptag-loc">📍 ${esc(loc.trim())}</span>`;
  if(co.trim())  html+=`<span class="ptag ptag-co">🏢 ${esc(co.trim())}</span>`;
  tags.innerHTML=html;
}

// Tab switch
function switchTab(tab){
  activeTab=tab;
  const defs={prog:{id:'tp-prog',d:'flex'},log:{id:'tp-log',d:'flex'},res:{id:'tp-res',d:'flex'}};
  Object.entries(defs).forEach(([t,{id,d}])=>{
    const el=document.getElementById(id);
    const btn=document.getElementById('tr-'+t);
    const show=t===tab;
    el.style.display=show?d:'none';
    if(show) el.style.flexDirection='column';
    if(btn) btn.classList.toggle('act',show);
  });
}

// ISSUE 2: check face before search, show modal if repeat
async function startSearch(){
  const nameRaw=document.getElementById('name-in').value.trim();
  if(!nameRaw){toast('Enter a name first','er');document.getElementById('name-in').focus();return;}
  if(!captured){toast('Capture or upload a photo first','er');return;}

  // Check for repeat face
  setStatus('busy','Checking face database…');
  try{
    const cr=await fetch('/api/check_face',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({image:captured}),
    });
    const cd=await cr.json();
    if(cd.prior){
      pendingSearch={nameRaw};
      const p=cd.prior;
      document.getElementById('modal-txt').innerHTML=
        `Face matched <strong style="color:var(--yellow)">${esc(p.name)}</strong> `+
        `from <strong>${esc(p.date.split(' ')[0])}</strong>`+
        ` &nbsp;·&nbsp; score <strong>${(p.score*100).toFixed(1)}%</strong>`+
        ` &nbsp;·&nbsp; verdict <strong>${esc(p.verdict).toUpperCase()}</strong>`+
        `<br><br>Do you want to run a fresh search anyway?`;
      document.getElementById('modal-bg').classList.add('show');
      setStatus('ok','Ready');
      return;
    }
  }catch(e){
    logger.debug&&console.debug('check_face err',e);
  }
  doSearch(nameRaw);
}

function modalCancel(){
  document.getElementById('modal-bg').classList.remove('show');
  pendingSearch=null;
  setStatus('ok','Ready — search cancelled');
}
function modalProceed(){
  document.getElementById('modal-bg').classList.remove('show');
  if(pendingSearch) doSearch(pendingSearch.nameRaw);
  pendingSearch=null;
}

async function doSearch(nameRaw){
  // Guard: disable search button immediately (before async fetch) to block double-submit
  const srchBtn = document.getElementById('btn-srch');
  if(srchBtn) srchBtn.disabled=true;
  setUI(false);
  initSteps();
  logCount=0;
  document.getElementById('lcnt').textContent='0';
  document.getElementById('log-panel').innerHTML='';
  document.getElementById('tr-res').style.display='none';
  document.getElementById('card-files').style.display='none';
  // Reset face quality warning and toolbar for new search
  const _fwb=document.getElementById('face-warn-banner');if(_fwb)_fwb.classList.remove('show');
  const _rtb=document.getElementById('res-toolbar');if(_rtb)_rtb.style.display='none';
  const _qw=document.getElementById('qface-wrap');if(_qw)_qw.style.display='none';
  const _dlb=document.getElementById('btn-dl-report');if(_dlb)_dlb.style.display='none';
  const _sfw=document.getElementById('score-filter-wrap');if(_sfw)_sfw.style.display='none';
  // Reset live feed
  _liveFeed=[];
  const _lfw=document.getElementById('live-feed-wrap');
  const _lf=document.getElementById('live-feed');
  const _lfh=document.getElementById('live-feed-hd');
  if(_lfw) _lfw.style.display='none';
  if(_lf)  _lf.innerHTML='';
  if(_lfh) _lfh.textContent='Live matches (0)';
  SRC_NODES.forEach(({k})=>{
    const n=document.getElementById('node-'+k); if(n) n.className='node';
    const s=document.getElementById('ns-'+k); if(s) s.textContent='IDLE';
    const c=document.getElementById('nc-'+k); if(c) c.textContent='—';
  });
  setStatus('busy','Starting search: '+nameRaw);
  document.getElementById('hb-kill').disabled=false;

  try{
    const r=await fetch('/api/search',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({name:nameRaw,image:captured}),
    });
    const d=await r.json();
    if(!r.ok||d.error) throw new Error(d.error||'Request failed');
    searchId=curSid=d.search_id;
    // Show parsed location/company in header
    if(d.parsed){
      const {location,company}=d.parsed;
      let extra='';
      if(location) extra+=` · 📍${location}`;
      if(company)  extra+=` · 🏢${company}`;
      if(extra) setStatus('busy',`Searching: ${d.parsed.name}${extra}`);
    }
    openSSE(d.search_id,d.parsed?.name||nameRaw);
  }catch(e){
    toast('Error: '+e.message,'er');
    setUI(true); setStatus('err','Search failed');
    document.getElementById('hb-kill').disabled=true;
  }
}

function openSSE(sid,name){
  if(es){es.close();es=null;}
  es=new EventSource('/api/stream/'+sid);
  es.onopen=()=>{ /* connected */ };
  es.onmessage=ev=>{
    const m=JSON.parse(ev.data);
    if(m.hb) return;
    handleEvent(m,name);
    if(m.done){es.close();es=null;}
  };
  es.onerror=()=>{
    // Close immediately — do NOT let EventSource auto-reconnect
    // (auto-reconnect would re-open the stream after the search finishes,
    //  or keep spamming a 404 if the queue is already cleaned up)
    if(es){es.close();es=null;}
    setUI(true);
  };
}

// ── Live feed ─────────────────────────────────────────────────────────────
let _liveFeed=[];
function addLiveFeedItems(items){
  if(!items||!items.length) return;
  const wrap=document.getElementById('live-feed-wrap');
  const feed=document.getElementById('live-feed');
  if(!feed) return;
  if(wrap&&wrap.style.display==='none') wrap.style.display='';
  items.forEach(m=>{
    if(!m.url&&!m.name) return;
    _liveFeed.push(m);
    const d=document.createElement('div');
    d.className='lf-item';
    const icon=_platIcon(m.url||'');
    const nm=esc(m.name||'');
    const u=m.url||'';
    d.innerHTML=`<span class="lf-ic">${icon}</span>`+
      (u?`<a href="${esc(u)}" target="_blank" rel="noopener" class="lf-url">${nm||esc(u.length>55?u.slice(0,52)+'…':u)}</a>`
        :`<span class="lf-url">${nm}</span>`)+
      `<span class="lf-src">${esc(m.source||'')}</span>`;
    feed.appendChild(d);
    feed.scrollTop=feed.scrollHeight;
  });
  const hd=document.getElementById('live-feed-hd');
  if(hd) hd.textContent=`Live matches (${_liveFeed.length})`;
}

function handleEvent(m,name){
  if(m.type==='log'){addLogEntry(m);return;}
  if(m.msg) setStatus('busy',m.msg);
  if(m.step&&m.step!=='done'&&m.step!=='error'){
    if(m.done_step) markStep(m.step,'ok',m.msg,'✓');
    else            markStep(m.step,'act',m.msg);
  }
  // Feature D: face quality warning — show banner if confidence is in 0.50–0.65 range
  if(m.step==='embedding' && m.done_step && m.confidence!=null){
    setFaceQualityWarning(m.confidence);
  }
  if(m.scraper){
    const st=m.scraper;
    if(!m.done_step&&!m.err) setNode(st,'scanning',undefined);
    else setNode(st,'done',m.count,m.err);
    if(m.preview&&m.preview.length) addLiveFeedItems(m.preview);
  }
  if(m.done){
    document.getElementById('hb-kill').disabled=true;
    setUI(true);
    // Hide live feed — Results tab has full scored data now
    const _lfw2=document.getElementById('live-feed-wrap');
    if(_lfw2) _lfw2.style.display='none';
    if(m.ok){
      const verdict=(m.identity||{}).verdict||'unknown';
      const score=(m.identity||{}).combined_score||0;
      setStatus('ok','Complete \u2014 '+name);
      toast('Search complete \u2014 '+verdict.toUpperCase(),'ok');
      _sendNotification('FaceOSINT \u2014 Search Complete',`${name}: ${verdict} (${score.toFixed(2)})`);
      buildResults(m,name);
      document.getElementById('tr-res').style.display='';
      switchTab('res');
      if(m.files) showFiles(m.files);
      loadHistory();
    }else{
      setStatus('err',m.msg||'Failed');
      toast(m.msg||'Search failed','er');
    }
  }
}

function initSteps(){
  const sl=document.getElementById('steps-list');
  sl.innerHTML='';
  STEP_DEFS.forEach(({k,ic,nm})=>{
    const d=document.createElement('div');
    d.className='step'; d.id='s-'+k;
    d.innerHTML=`<div class="sic" id="si-${k}">${ic}</div>`+
      `<div style="flex:1"><div class="snm">${nm}</div>`+
      `<div class="sdt" id="sd-${k}">Waiting…</div></div>`;
    sl.appendChild(d);
  });
  switchTab('prog');
}
function markStep(k,cls,dt,ic){
  const e=document.getElementById('s-'+k);
  const d=document.getElementById('sd-'+k);
  const i=document.getElementById('si-'+k);
  if(e) e.className='step '+cls;
  if(d&&dt) d.textContent=dt;
  if(i&&ic) i.textContent=ic;
}
function setNode(k,state,count,err){
  const n=document.getElementById('node-'+k);
  const s=document.getElementById('ns-'+k);
  const c=document.getElementById('nc-'+k);
  if(!n) return;
  n.className='node '+(err?'failed':count>0?'found':state==='scanning'?'scanning':'empty');
  if(s) s.textContent=err?'ERROR':state==='scanning'?'SCANNING':count>0?'FOUND':'DONE';
  if(c&&count!==undefined) c.textContent=count||'—';
}

function addLogEntry(m){
  logCount++;
  document.getElementById('lcnt').textContent=logCount;
  const p=document.getElementById('log-panel');
  if(logCount===1) p.innerHTML='';
  const e=document.createElement('div');
  e.className='le '+(m.level||'INFO');
  const hasD=m.data&&Object.keys(m.data).length;
  e.innerHTML=`<span class="lts">${esc(m.ts||'')}</span>`+
    `<span class="llv">${m.level||'INFO'}</span>`+
    `<span class="lsrc">${esc(m.source||'')}</span>`+
    `<div style="flex:1"><div class="lmsg">${esc(m.msg||'')}</div>`+
    (hasD?`<div class="ldata" onclick="this.classList.toggle('open')">${esc(JSON.stringify(m.data,null,2))}</div>`:'')+
    `</div>`;
  p.appendChild(e);
  p.scrollTop=p.scrollHeight;
  if(debugOn&&activeTab==='prog') switchTab('log');
}
function clearLog(){
  document.getElementById('log-panel').innerHTML='<div class="wait" style="padding:14px;flex:none"><div class="wait-s">Log cleared</div></div>';
  logCount=0; document.getElementById('lcnt').textContent='0';
}

function _platIcon(url){
  if(!url) return '🌐';
  const u=url.toLowerCase();
  if(u.includes('linkedin')) return '💼';
  if(u.includes('github')) return '🐙';
  if(u.includes('twitter')||u.includes('x.com')) return '🐦';
  if(u.includes('instagram')) return '📸';
  if(u.includes('facebook')) return '👤';
  if(u.includes('reddit')) return '🟠';
  if(u.includes('researchgate')) return '🎓';
  if(u.includes('scholar.google')) return '🎓';
  if(u.includes('orcid')) return '🔬';
  if(u.includes('medium')) return '✍';
  if(u.includes('youtube')) return '▶';
  if(u.includes('stackoverflow')) return '📚';
  return '🔗';
}
function _platName(url){
  if(!url) return 'Web';
  const u=url.toLowerCase();
  if(u.includes('linkedin.com/in/')) return 'LinkedIn';
  if(u.includes('linkedin')) return 'LinkedIn';
  if(u.includes('github.com')) return 'GitHub';
  if(u.includes('twitter.com')||u.includes('x.com')) return 'Twitter/X';
  if(u.includes('instagram')) return 'Instagram';
  if(u.includes('facebook')) return 'Facebook';
  if(u.includes('reddit')) return 'Reddit';
  if(u.includes('researchgate')) return 'ResearchGate';
  if(u.includes('scholar.google')) return 'Google Scholar';
  if(u.includes('semanticscholar')) return 'Semantic Scholar';
  if(u.includes('openalex')) return 'OpenAlex';
  if(u.includes('orcid')) return 'ORCID';
  if(u.includes('medium')) return 'Medium';
  if(u.includes('youtube')) return 'YouTube';
  if(u.includes('stackoverflow')) return 'Stack Overflow';
  try{ return new URL(url).hostname.replace('www.',''); }catch(e){ return 'Web'; }
}
function _matchCard(m, idx){
  const url = m.url||m.profile_url||'';
  const name = m.name||m.title||'';
  const snippet = m.snippet||m.bio||'';
  const photo = m.photo_url||m.avatar_url||'';
  const platform = _platName(url);
  const icon = _platIcon(url);
  const fv = m.face_verified;
  const fs = m.face_score||m.face_similarity||null;
  const src = m.source||'';
  const company = m.company||m.affiliation||'';
  const location = m.location||'';
  const username = m.username||'';
  const papers = m.paper_count||m.papers||null;
  const hindex = m.h_index||null;
  const combinedScore = m.combined_score!=null ? m.combined_score : (fs||0);
  const scorePct = Math.round((fs!=null?fs:combinedScore)*100);
  const scoreCol = fv ? 'var(--green)' : (fs!=null&&fs>0.35) ? 'var(--yellow)' : 'var(--text-muted)';
  let scoreBar = '';
  if(fs!=null){
    scoreBar = `<div class="mc-score-wrap">
      <div class="mc-score-bar-bg"><div class="mc-score-bar-fill" style="width:${scorePct}%;background:${scoreCol}"></div></div>
      <span class="mc-score-val" style="color:${scoreCol}">${fv?'&#10003; ':''}${scorePct}%</span>
    </div>`;
  }
  const meta = [
    company && `<span title="Company">&#127962; ${esc(company)}</span>`,
    location && `<span title="Location">&#128205; ${esc(location)}</span>`,
    username && `<span title="Username">@${esc(username)}</span>`,
    papers!=null && `<span title="Papers">&#128196; ${papers} papers</span>`,
    hindex!=null && `<span title="h-index">h=${hindex}</span>`,
  ].filter(Boolean).join(' &middot; ');
  // verdict badge for individual match
  const mv = m.verdict||'';
  const mvc = mv==='confirmed'?'vc':mv==='possible'?'vp':mv?'vu':'';
  return `<div class="mc" data-url="${esc(url)}" data-score="${combinedScore.toFixed(4)}">
    <div class="mc-head">
      ${photo ? `<img class="mc-av" src="${esc(photo)}" onerror="this.style.display='none'" loading="lazy" alt="">` : `<div class="mc-av-ph">${icon}</div>`}
      <div class="mc-info">
        <div class="mc-top">
          <span class="mc-plat">${icon} ${esc(platform)}</span>
          ${fv ? '<span class="mc-fv">&#10003; Face</span>' : ''}
          ${mvc ? `<span class="vb ${mvc}" style="font-size:8px">${esc(mv.toUpperCase())}</span>` : ''}
          <span class="mc-src">${esc(src.replace(/_/g,' '))}</span>
        </div>
        ${name ? `<div class="mc-name">${esc(name)}</div>` : ''}
        ${meta ? `<div class="mc-meta">${meta}</div>` : ''}
        ${scoreBar}
      </div>
    </div>
    ${snippet ? `<div class="mc-snip">${esc(snippet.slice(0,180))}</div>` : ''}
    ${url ? `<div style="margin-top:6px;display:flex;gap:6px;align-items:center;flex-wrap:wrap">
      ${company ? `<span class="mc-meta">&#127962; ${esc(company)}</span>` : ''}
      ${location ? `<span class="mc-meta">&#128205; ${esc(location)}</span>` : ''}
    </div>` : ''}
    ${url ? `<a class="mc-url result-link" href="${esc(url)}" target="_blank" rel="noopener noreferrer" onclick="_markVisited(this)">${esc(url.length>72?url.slice(0,69)+'\u2026':url)}</a>` : ''}
  </div>`;
}
function _markVisited(el){
  el.classList.add('visited');
  const url = el.href;
  let v = JSON.parse(localStorage.getItem('visited_urls')||'[]');
  if(!v.includes(url)){ v.push(url); localStorage.setItem('visited_urls', JSON.stringify(v.slice(-500))); }
}
function _applyVisited(){
  const v = new Set(JSON.parse(localStorage.getItem('visited_urls')||'[]'));
  document.querySelectorAll('.mc-url').forEach(a=>{ if(v.has(a.href)) a.classList.add('visited'); });
}
function buildResults(m,name){
  const id=m.identity||{};
  const v=id.verdict||'unknown';
  const vc=v==='confirmed'?'vc':v==='possible'?'vp':v==='no_results'?'vx':'vu';
  const vIcon=v==='confirmed'?'&#10003;':v==='possible'?'?':v==='no_results'?'&mdash;':'&#10007;';
  const vLabel=v==='confirmed'?'Confirmed':v==='possible'?'Possible':v==='no_results'?'No Results':'Unlikely';
  const sc=id.combined_score||0;
  const fs=id.face_score;
  const ss=m.source_summary||{};
  const urls=(id.profile_urls||[]).filter(Boolean);
  const maxN=Math.max(1,...Object.values(ss).map(s=>s.count||0));
  const allMatches = (m.all_matches||[]).slice().sort((a,b)=>(b.combined_score||0)-(a.combined_score||0));

  const srcHtml=Object.entries(ss).map(([lbl,s])=>{
    const n=s.count||0,ok=!s.error;
    const pct=Math.round(n/maxN*100);
    return `<div class="ss-row">`+
      `<div class="ss-lbl" style="color:${ok?'var(--text-primary)':'var(--text-muted)'}">${esc(SLBL[lbl]||lbl)}</div>`+
      `<div class="ss-bar"><div class="ss-fill" style="width:${pct}%;background:${ok?'var(--accent)':'var(--red)'}"></div></div>`+
      `<div class="ss-n" style="color:${ok?'var(--green)':'var(--red)'}">${n}${s.error?' &#10007;':''}</div></div>`;
  }).join('');

  const photo = id.photo_url||'';
  const faceCrop = m.face_crop_b64||'';

  document.getElementById('res-panel').innerHTML=
    // ── Verdict banner ────────────────────────────────────────
    `<div class="verdict-banner ${vc}">` +
      `<div class="verdict-icon">${vIcon}</div>` +
      `<div class="verdict-body">` +
        `<div class="verdict-label">Verdict</div>` +
        `<div class="verdict-text">${esc(vLabel)}</div>` +
        `<div style="display:flex;align-items:center;gap:8px;margin-top:6px">` +
          `<div class="verdict-score-bar" style="flex:1"><div class="verdict-score-fill" id="vsf" style="width:0"></div></div>` +
          `<span class="verdict-score-pct" id="vsf-pct">${(sc*100).toFixed(1)}%</span>` +
        `</div>` +
      `</div>` +
    `</div>` +
    // ── Identity summary card ─────────────────────────────────
    `<div class="id-card">` +
      `<div class="id-card-head">` +
        (photo ? `<div class="av"><img src="${esc(photo)}" onerror="this.parentElement.innerHTML='&#128100;'" loading="lazy" alt=""></div>` :
                 faceCrop ? `<div class="av"><img src="${esc(faceCrop)}" loading="lazy" alt=""></div>` :
                 `<div class="av">&#128100;</div>`) +
        `<div class="id-info">` +
          `<div class="id-name">${esc(id.resolved_name||name)}</div>` +
          `<div class="id-badges">` +
            `<span class="vb ${vc}">${esc(vLabel.toUpperCase())}</span>` +
            (m.total ? `<span style="font-size:10px;color:var(--text-muted)">${m.total} results</span>` : '') +
            ((id.sources||[]).length ? `<span style="font-size:10px;color:var(--text-muted)">${id.sources.length} sources</span>` : '') +
          `</div>` +
        `</div>` +
      `</div>` +
      `<div class="id-fields">` +
        (id.email ? `<div class="id-field"><div class="id-field-label">Email</div><div class="id-field-val"><a href="mailto:${esc(id.email)}" class="result-link">${esc(id.email)}</a></div></div>` : '') +
        (id.company ? `<div class="id-field"><div class="id-field-label">Company</div><div class="id-field-val">${esc(id.company)}</div></div>` : '') +
        (id.location ? `<div class="id-field"><div class="id-field-label">Location</div><div class="id-field-val">${esc(id.location)}</div></div>` : '') +
        (id.bio ? `<div class="id-field" style="grid-column:1/-1"><div class="id-field-label">Bio</div><div class="id-field-val">${esc(id.bio.slice(0,160))}</div></div>` : '') +
      `</div>` +
    `</div>` +
    // ── Confidence bars ───────────────────────────────────────
    `<div class="sbar-row"><div class="sbar-h"><span>Combined Confidence</span>` +
    `<span style="color:var(--accent-hover)">${(sc*100).toFixed(1)}%</span></div>` +
    `<div class="sbar"><div class="sfill sf1" id="sf1"></div></div></div>` +
    (fs!=null ? `<div class="sbar-row"><div class="sbar-h"><span>Face Match</span>` +
    `<span style="color:#a78bfa">${(fs*100).toFixed(1)}%</span></div>` +
    `<div class="sbar"><div class="sfill sf2" id="sf2"></div></div></div>` : '') +
    // ── Profile links ─────────────────────────────────────────
    (urls.length ? `<div class="res-section-title">Profile Links</div>` +
      `<div class="plinks">${urls.slice(0,10).map(u =>
        `<a class="plink mc-url result-link" href="${esc(u)}" target="_blank" rel="noopener noreferrer" onclick="_markVisited(this)">${_platIcon(u)} ${esc(u.length>58?u.slice(0,55)+'\u2026':u)}</a>`
      ).join('')}</div>` : '') +
    // ── Source breakdown ──────────────────────────────────────
    (srcHtml ? `<div class="res-section-title">Source Breakdown</div>` +
    `<div class="ss-grid">${srcHtml}</div>` : '') +
    // ── Match cards ───────────────────────────────────────────
    (allMatches.length ? `<div class="res-section-title">All Matches (${allMatches.length})</div>` +
      `<div class="mc-grid">${allMatches.slice(0,50).map((mm,i)=>_matchCard(mm,i)).join('')}</div>` : '') +
    // ── Report path ───────────────────────────────────────────
    (m.folder ? `<div class="it" style="margin-top:12px"><div class="lbl">Report saved to</div>` +
    `<div class="val" style="font-size:10px;color:var(--text-muted)">${esc(m.folder)}</div></div>` : '');

  // Score filter
  const filterWrap = document.getElementById('score-filter-wrap');
  if(filterWrap) filterWrap.style.display = allMatches.length ? '' : 'none';
  const chk = document.getElementById('chk-show-unlikely');
  if(chk) chk.checked = false;

  // Download report link
  const dlBtn = document.getElementById('btn-dl-report');
  if(dlBtn && curSid) {
    dlBtn.href = '/api/search/'+curSid+'/report';
    dlBtn.style.display = '';
  }

  // Query face thumbnail
  const qwrap = document.getElementById('qface-wrap');
  const qimg  = document.getElementById('qface-img');
  if(qwrap && qimg && curSid) {
    qimg.src = '/api/search/'+curSid+'/face_crop';
    qimg.onerror = function(){ qwrap.style.display='none'; };
    qwrap.style.display = '';
  }

  // Show toolbar
  const toolbar = document.getElementById('res-toolbar');
  if(toolbar) toolbar.style.display = 'flex';

  setTimeout(()=>{
    const vsf=document.getElementById('vsf'); if(vsf) vsf.style.width=(sc*100)+'%';
    const a=document.getElementById('sf1'); if(a) a.style.width=(sc*100)+'%';
    const b=document.getElementById('sf2'); if(b&&fs!=null) b.style.width=(fs*100)+'%';
    _applyVisited();
    applyScoreFilter();
  },80);
}

function showFiles(files){
  const card=document.getElementById('card-files');
  const panel=document.getElementById('file-panel');
  card.style.display='';
  document.getElementById('dbtn-img').disabled=false;
  document.getElementById('dbtn-all').disabled=false;
  const icons={'.jpg':'🖼','.jpeg':'🖼','.png':'🖼','.txt':'📄','.json':'📋','.db':'🗄'};
  panel.innerHTML=files.map(f=>{
    const ext=f.name.slice(f.name.lastIndexOf('.')).toLowerCase();
    const sens=['captured_photo.jpg','face_crop.jpg'].includes(f.name);
    return `<div class="fp-item" style="${sens?'border-color:rgba(255,59,92,.25)':''}">`+
      `<span>${icons[ext]||'📁'}</span>`+
      `<div class="fp-nm" title="${esc(f.path)}">${esc(f.rel||f.name)}${sens?' 🔴':''}</div>`+
      `<span class="fp-sz">${f.size_kb}kb</span></div>`;
  }).join('');
}

async function cleanup(mode){
  if(!curSid){toast('No active search','er');return;}
  const full=mode==='full';
  if(!confirm(full?'Wipe ALL files for this search (including report)?':
    'Delete captured_photo.jpg, face_crop.jpg and scraped photos?')) return;
  try{
    const r=await fetch('/api/cleanup/'+curSid,{
      method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({images:mode==='images'||full,embedding:full,full})
    });
    const d=await r.json();
    toast('Deleted: '+(d.deleted||[]).join(', '),'ok');
    document.getElementById('file-panel').innerHTML=
      '<div class="wait" style="padding:10px;flex:none"><div class="wait-s">Files wiped ✓</div></div>';
    document.getElementById('dbtn-img').disabled=true;
    document.getElementById('dbtn-all').disabled=true;
    loadHistory();
  }catch(e){toast('Cleanup failed: '+e.message,'er');}
}

async function killSearch(){
  if(!searchId) return;
  document.getElementById('hb-kill').disabled=true;
  try{await fetch('/api/cancel/'+searchId,{method:'POST'});toast('Kill signal sent','ok');}
  catch(e){toast('Kill failed','er');document.getElementById('hb-kill').disabled=false;}
}

async function loadHistory(){
  try{
    const data=await(await fetch('/api/history')).json();
    const el=document.getElementById('hist-list');
    if(!data.length){
      el.innerHTML='<div class="wait" style="padding:16px"><div class="wait-s">No searches yet</div></div>';
      return;
    }
    el.innerHTML=data.map(s=>{
      const v=s.verdict||'unknown';
      const dotCol=v==='confirmed'?'var(--green)':v==='possible'?'var(--yellow)':'var(--text-muted)';
      const badgeCls=v==='confirmed'?'vc':v==='possible'?'vp':'vu';
      const sc=s.combined_score?((s.combined_score*100).toFixed(0)+'%'):'—';
      const dt=(s.created_at||'').split(' ')[0]||'';
      return `<div class="hi" onclick="viewRep('${s.id}','${esc(s.name||'')}')">` +
        `<div class="hi-dot" style="background:${dotCol}"></div>`+
        `<div class="hi-body">`+
          `<div class="hi-name">${esc(s.name||'?')}</div>`+
          `<div class="hi-meta">${sc} &middot; ${dt}</div>`+
        `</div>`+
        `<span class="vb ${badgeCls}" style="font-size:8px">${v.toUpperCase()}</span>`+
        `</div>`;
    }).join('');
  }catch(e){console.error(e);}
}

async function clearHistory(){
  if(!confirm('Clear all search history? This cannot be undone.')) return;
  try{
    await fetch('/api/history/clear',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({confirm:true})
    });
    loadHistory();
    toast('History cleared','ok');
  }catch(e){ toast('Failed to clear: '+e.message,'er'); }
}

async function viewRep(sid,name){
  curSid=sid;
  document.getElementById('tr-res').style.display='';
  switchTab('res');
  document.getElementById('res-panel').innerHTML='<div class="wait"><div class="wait-ic">&#8987;</div><div class="wait-t">Loading\u2026</div></div>';
  try{
    // Fetch structured result data
    const r=await fetch('/api/result/'+sid);
    if(!r.ok) throw new Error('Not found');
    const data=await r.json();

    // Build a synthetic "done" message compatible with buildResults
    const matches_raw = data.matches||[];
    const identity = {
      resolved_name: data.name||name,
      verdict: data.verdict||'unknown',
      combined_score: data.combined_score||0,
      photo_url: data.photo_url||'',
      email: data.email||'',
      company: data.company||'',
      location: data.location||'',
      bio: data.bio||'',
      profile_urls: matches_raw.filter(m=>m.url&&(m.platform||'').match(/LinkedIn|GitHub|Twitter|Instagram|ResearchGate/i)).map(m=>m.url).filter(Boolean).slice(0,10),
      sources: [...new Set(matches_raw.map(m=>m.source).filter(Boolean))],
    };
    // Best photo: face-verified match photo or face crop
    const fvMatch = matches_raw.find(m=>m.face_verified&&m.photo_url);
    if(fvMatch&&!identity.photo_url) identity.photo_url=fvMatch.photo_url;

    const source_summary = {};
    matches_raw.forEach(m=>{
      const s=m.source||'unknown';
      if(!source_summary[s]) source_summary[s]={count:0};
      source_summary[s].count++;
    });

    buildResults({
      identity,
      total: matches_raw.length,
      source_summary,
      folder: data.output_folder||'',
      face_crop_b64: data.face_crop_b64||'',
      all_matches: matches_raw,
    }, data.name||name);

    // Also show files
    const fr=await(await fetch('/api/files/'+sid)).json();
    if(fr.files) showFiles(fr.files);
  }catch(e){ toast('Cannot load report: '+e.message,'er'); }
}

// Feature A: score filter — toggle visibility of match cards below threshold
function applyScoreFilter(){
  const chk = document.getElementById('chk-show-unlikely');
  const showUnlikely = chk && chk.checked;
  document.querySelectorAll('.mc[data-score]').forEach(card=>{
    const score = parseFloat(card.dataset.score);
    card.style.display = (score < 0.55 && !showUnlikely) ? 'none' : '';
  });
}

// Feature D: show/hide low-confidence face warning
function setFaceQualityWarning(confidence){
  const banner = document.getElementById('face-warn-banner');
  if(!banner) return;
  if(confidence >= 0.50 && confidence < 0.65){
    banner.classList.add('show');
  } else {
    banner.classList.remove('show');
  }
}

function setStatus(mode,txt){
  const d=document.getElementById('sd');
  const m=document.getElementById('hdr-msg');
  d.className='status-dot'+(mode==='busy'?' busy':mode==='err'?' err':'');
  m.textContent=txt;
  m.style.color=mode==='ok'?'var(--green)':mode==='err'?'var(--red)':'var(--text-muted)';
}
function setUI(on){
  ['btn-srch','btn-cap','name-in'].forEach(id=>{
    const el=document.getElementById(id);
    if(el) el.disabled=!on||(id==='btn-srch'&&!captured);
  });
  const r=document.getElementById('btn-ret'); if(r) r.disabled=!on;
}
function toast(msg,type){
  const c=document.getElementById('toasts');
  const d=document.createElement('div');
  d.className='toast '+(type||'ok');
  d.innerHTML=`<span class="toast-icon">${type==='er'?'&#10007;':'&#10003;'}</span>${esc(msg)}`;
  c.appendChild(d);
  setTimeout(()=>d.remove(),4500);
}
function esc(s){
  return s==null?'':String(s)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

loadHistory();

/* ════════════════════════════════════════════════════════════════════════
   CROWD INTELLIGENCE CENTER — JavaScript
   ════════════════════════════════════════════════════════════════════════ */

// ── State ─────────────────────────────────────────────────────────────────
var _cicOpen       = false;
var _cicPollTimer  = null;
var _cicSSE        = null;
var _cicMapObj     = null;
var _cicZonePolys  = {};
var _cicHeatLayer  = null;
var _cicHeatOn     = false;
var _cicHeatTimer  = null;
var _cicBarChart   = null;
var _cicDenChart   = null;
var _cicActiveSlots = new Set();
var _cicZoneSlotMap = {};   // zoneId → slot int
var _cicZoneList    = [];   // raw zones from API
var _cicToggleState = {     // mirrors server toggle state
  show_bbox: true, show_track_id: true, show_suspicious: true,
  show_children: true, show_flow: true, show_count: true
};
var _cicAudioOn    = false;
var _cicLastCritical = 0;   // timestamp of last audio alarm

var _RISK_FILL = {safe:'#22c55e', caution:'#f59e0b', high:'#f97316', critical:'#ef4444'};
var _RISK_BORDER = {safe:'var(--green)', caution:'var(--yellow)', high:'#f97316', critical:'var(--red)'};

// ── Toggle overlay ─────────────────────────────────────────────────────────
function toggleCIC() {
  _cicOpen = !_cicOpen;
  var el  = document.getElementById('cic-overlay');
  var btn = document.getElementById('hb-cic');
  el.classList.toggle('open', _cicOpen);
  if (btn) btn.classList.toggle('on', _cicOpen);
  if (_cicOpen) {
    _cicLoadZones();
    _cicSyncSlots();      // restore server-side active slots
    _cicConnectSSE();
    _cicStartPolling();
  } else {
    _cicStopPolling();
    if (_cicSSE) { _cicSSE.close(); _cicSSE = null; }
    if (_cicHeatTimer) { clearInterval(_cicHeatTimer); _cicHeatTimer = null; }
  }
}

// ── Sync server slot state into browser ───────────────────────────────────
function _cicSyncSlots() {
  fetch('/crowd/api/status').then(function(r) { return r.json(); }).then(function(d) {
    (d.slots || []).forEach(function(s) {
      if (!s.active) return;
      var slot = s.slot;
      _cicActiveSlots.add(slot);
      var tile = document.getElementById('cic-tile-' + slot);
      var dot  = document.getElementById('cic-dot-' + slot);
      if (tile) tile.classList.remove('offline');
      if (dot)  dot.className = 'cic-status-dot live';
    });
  }).catch(function() {});
}

// ── Tab switching ──────────────────────────────────────────────────────────
function cicTab(name) {
  document.querySelectorAll('.cic-panel').forEach(function(p) { p.classList.remove('active'); });
  document.querySelectorAll('.cic-tab').forEach(function(t) { t.classList.remove('active'); });
  var panel = document.getElementById('cic-panel-' + name);
  var tab   = document.getElementById('cic-tabn-' + name);
  if (panel) panel.classList.add('active');
  if (tab)   tab.classList.add('active');
  if (name === 'map') {
    if (!_cicMapObj) _cicInitMap();
    else setTimeout(function() { if (_cicMapObj) _cicMapObj.invalidateSize(); }, 50);
  }
  if (name === 'analytics' && !_cicBarChart) {
    setTimeout(_cicInitCharts, 50);
  }
}

// ── Zone config ────────────────────────────────────────────────────────────
function _cicLoadZones() {
  fetch('/crowd/api/zones').then(function(r) { return r.json(); }).then(function(data) {
    _cicZoneList = data.zones || [];
    _cicZoneList.forEach(function(z) {
      _cicZoneSlotMap[z.id] = z.camera_slot;
    });
  }).catch(function() {});
}

// ── Frame polling ──────────────────────────────────────────────────────────
function _cicStartPolling() {
  if (_cicPollTimer) return;
  _cicPollTimer = setInterval(_cicFetchFrames, 300);
}
function _cicStopPolling() {
  if (_cicPollTimer) { clearInterval(_cicPollTimer); _cicPollTimer = null; }
}
function _cicFetchFrames() {
  _cicActiveSlots.forEach(function(slot) {
    fetch('/crowd/api/frame/' + slot)
      .then(function(r) { return r.json(); })
      .then(function(d) {
        if (d.ok) {
          var img = document.getElementById('cic-frame-' + slot);
          if (img) img.src = d.frame;
        }
      }).catch(function() {});
  });
}

// ── Platform SSE ───────────────────────────────────────────────────────────
function _cicConnectSSE() {
  if (_cicSSE) { _cicSSE.close(); _cicSSE = null; }
  _cicSSE = new EventSource('/crowd/api/stream');
  _cicSSE.onmessage = function(e) {
    try {
      var d = JSON.parse(e.data);
      if (d.hb || !d.type) return;
      if (d.type === 'update') _cicHandleUpdate(d);
    } catch (ex) {}
  };
  _cicSSE.onerror = function() {
    if (_cicSSE) { _cicSSE.close(); _cicSSE = null; }
    if (_cicOpen) setTimeout(_cicConnectSSE, 4000);
  };
}

function _cicHandleUpdate(d) {
  var zones = d.zones || {};
  var total = d.total_count || 0;

  // Total badge
  var tb = document.getElementById('cic-total-badge');
  if (tb) tb.innerHTML = 'Total: <b>' + total + '</b> persons';

  // Per-slot badges + tile borders
  Object.keys(zones).forEach(function(zid) {
    var z    = zones[zid];
    var slot = _cicZoneSlotMap[zid];
    if (slot === undefined) return;

    var badge = document.getElementById('cic-badge-' + slot);
    if (badge) badge.textContent = z.count;

    var riskEl = document.getElementById('cic-risk-' + slot);
    if (riskEl) {
      riskEl.textContent  = z.risk.toUpperCase();
      riskEl.className    = 'cic-risk-' + z.risk;
    }
    var densEl = document.getElementById('cic-dens-' + slot);
    if (densEl) densEl.textContent = (z.density || 0).toFixed(3) + ' p/m²';

    var tile = document.getElementById('cic-tile-' + slot);
    if (tile) tile.style.borderColor = _RISK_BORDER[z.risk] || 'var(--border)';

    // Behavioral footer badge
    var bhvEl = document.getElementById('cic-bhv-' + slot);
    if (bhvEl) {
      var parts = [];
      if (z.n_suspicious > 0) parts.push('! ' + z.n_suspicious + ' susp');
      if (z.n_running    > 0) parts.push(z.n_running + ' run');
      if (z.n_children   > 0) parts.push(z.n_children + ' child');
      bhvEl.textContent = parts.join(' | ');
      bhvEl.style.color = z.n_suspicious > 0 ? '#f97316' : 'var(--text-muted)';
    }

    // Leaflet zone polygon color
    _cicUpdateZonePoly(zid, z.risk);
  });

  // New alerts + audio alarm
  if (d.alerts && d.alerts.length) {
    d.alerts.forEach(_cicAddAlert);
    var hasCritical = d.alerts.some(function(a) { return a.severity === 'critical'; });
    if (hasCritical) _cicPlayAlarm();
  }

  // Charts live update
  if (_cicBarChart) _cicLiveUpdateCharts(zones);
}

// ── Overlay toggles ─────────────────────────────────────────────────────────
function cicToggle(name) {
  _cicToggleState[name] = !_cicToggleState[name];
  var btn = document.getElementById('tog-' + name);
  if (btn) btn.classList.toggle('on', _cicToggleState[name]);
  _cicActiveSlots.forEach(function(slot) {
    fetch('/crowd/api/slot/' + slot + '/toggle', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({name: name, value: _cicToggleState[name]})
    }).catch(function() {});
  });
}

function cicToggleAudio() {
  _cicAudioOn = !_cicAudioOn;
  var btn = document.getElementById('tog-audio');
  if (btn) btn.classList.toggle('on', _cicAudioOn);
}

function _cicPlayAlarm() {
  if (!_cicAudioOn) return;
  var now = Date.now();
  if (now - _cicLastCritical < 30000) return;  // 30s cooldown
  _cicLastCritical = now;
  try {
    var ctx = new (window.AudioContext || window.webkitAudioContext)();
    [880, 1100, 880].forEach(function(freq, i) {
      var osc  = ctx.createOscillator();
      var gain = ctx.createGain();
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.frequency.value = freq;
      osc.type = 'square';
      gain.gain.setValueAtTime(0.3, ctx.currentTime + i * 0.25);
      gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + i * 0.25 + 0.2);
      osc.start(ctx.currentTime + i * 0.25);
      osc.stop(ctx.currentTime + i * 0.25 + 0.25);
    });
  } catch(e) {}
}

// ── Slot management ────────────────────────────────────────────────────────
function cicStartSlot(slot) {
  var dataDir = 'Place your video in data\\ folder (e.g. data\\crowd.mp4)';
  var src = prompt(
    'Camera Slot ' + slot + ' — enter video source:\n\n' +
    '  0                       = local webcam\n' +
    '  data\\crowd.mp4          = video file in project data\\ folder\n' +
    '  D:\\full\\path\\video.mp4  = full Windows path\n' +
    '  192.168.x.x:8080        = IP camera (auto-probes endpoints)\n' +
    '  rtsp://...               = RTSP stream URL\n\n' +
    'Tip: ' + dataDir,
    slot === 0 ? '0' : 'data\\'
  );
  if (src === null || src === undefined) return;
  src = src.trim();
  fetch('/crowd/api/slot/' + slot + '/start', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({source: src || '0'})
  }).then(function(r) { return r.json(); }).then(function(d) {
    if (d.ok) {
      _cicActiveSlots.add(slot);
      var tile = document.getElementById('cic-tile-' + slot);
      var dot  = document.getElementById('cic-dot-' + slot);
      if (tile) tile.classList.remove('offline');
      if (dot)  { dot.className = 'cic-status-dot live'; }
      if (typeof toast === 'function') toast('Slot ' + slot + ' started', 'ok');
    } else {
      if (typeof toast === 'function') toast('Cannot open: ' + (d.error || 'unknown error'), 'er');
      else alert('Cannot open: ' + (d.error || 'unknown error'));
    }
  }).catch(function(e) {
    if (typeof toast === 'function') toast('Network error', 'er');
  });
}

function cicStopSlot(slot) {
  fetch('/crowd/api/slot/' + slot + '/stop', {method: 'POST'}).catch(function() {});
  _cicActiveSlots.delete(slot);
  var img  = document.getElementById('cic-frame-' + slot);
  var tile = document.getElementById('cic-tile-' + slot);
  var dot  = document.getElementById('cic-dot-' + slot);
  var risk = document.getElementById('cic-risk-' + slot);
  var badge= document.getElementById('cic-badge-' + slot);
  if (img)   { img.src = ''; }
  if (tile)  { tile.classList.add('offline'); tile.style.borderColor = ''; }
  if (dot)   { dot.className = 'cic-status-dot offline'; }
  if (risk)  { risk.textContent = 'OFFLINE'; risk.className = 'cic-risk-safe'; }
  if (badge) { badge.textContent = '--'; }
}

// ── Alert log ──────────────────────────────────────────────────────────────
function _cicAddAlert(a) {
  var log = document.getElementById('cic-alert-log');
  if (!log) return;
  // Remove placeholder
  var placeholder = log.querySelector('div[style]');
  if (placeholder && placeholder.textContent.includes('No alerts')) placeholder.remove();

  var div = document.createElement('div');
  div.className = 'cic-alert-item ' + (a.severity || 'warning');
  div.innerHTML =
    '<span class="cic-alert-ts">' + (a.timestamp || '') + '</span>' +
    '<span class="cic-alert-zone">' + (a.zone || '') + '</span>' +
    '<span class="cic-alert-msg">' + (a.message || '') + '</span>';
  log.insertBefore(div, log.firstChild);
  while (log.children.length > 60) log.removeChild(log.lastChild);

  // Flash alerts tab if not on it
  var alertsTab = document.getElementById('cic-tabn-alerts');
  if (alertsTab && !alertsTab.classList.contains('active')) {
    alertsTab.style.animation = 'cic-blink 0.4s ease-in-out 3';
    setTimeout(function() { alertsTab.style.animation = ''; }, 1300);
  }
}

// ── Leaflet GIS Map ────────────────────────────────────────────────────────
function _cicInitMap() {
  if (typeof L === 'undefined') {
    var mapEl = document.getElementById('cic-map');
    if (mapEl) mapEl.innerHTML = '<div style="padding:20px;color:var(--text-muted);font-size:12px">Leaflet.js not available. Check internet connection.</div>';
    return;
  }
  fetch('/crowd/api/zones').then(function(r) { return r.json(); }).then(function(data) {
    var venue  = data.venue || {};
    var center = venue.center || [25.4232, 81.8845];
    var zoom   = venue.zoom   || 14;

    _cicMapObj = L.map('cic-map').setView(center, zoom);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; OpenStreetMap contributors',
      maxZoom: 19
    }).addTo(_cicMapObj);

    (data.zones || []).forEach(function(z) {
      var latlngs = (z.lat_lon || []).map(function(p) { return [p[0], p[1]]; });
      var poly = L.polygon(latlngs, {
        color: z.color || '#6366f1',
        fillColor: z.color || '#6366f1',
        fillOpacity: 0.3,
        weight: 2
      }).addTo(_cicMapObj);
      poly.bindPopup(
        '<b>' + z.name + '</b><br>' + (z.description || '') +
        '<br>Area: ' + z.area_m2 + ' m² &bull; Capacity: ' + z.capacity
      );
      _cicZonePolys[z.id] = poly;
    });
  }).catch(function(e) {
    console.warn('CIC map init error:', e);
  });
}

function _cicUpdateZonePoly(zid, risk) {
  var poly = _cicZonePolys[zid];
  if (!poly) return;
  var col = _RISK_FILL[risk] || '#6366f1';
  poly.setStyle({fillColor: col, color: col, fillOpacity: risk === 'safe' ? 0.2 : 0.45});
}

function cicToggleHeatmap() {
  _cicHeatOn = !_cicHeatOn;
  var btn = document.getElementById('map-toggle-heat');
  if (btn) btn.textContent = (_cicHeatOn ? 'Heat Map ON' : 'Heat Map OFF');
  if (_cicHeatOn) {
    _cicUpdateHeatmap();
    _cicHeatTimer = setInterval(_cicUpdateHeatmap, 2000);
  } else {
    if (_cicHeatTimer) { clearInterval(_cicHeatTimer); _cicHeatTimer = null; }
    if (_cicHeatLayer && _cicMapObj) { _cicMapObj.removeLayer(_cicHeatLayer); _cicHeatLayer = null; }
  }
}

function _cicUpdateHeatmap() {
  if (!_cicMapObj || typeof L === 'undefined') return;
  fetch('/crowd/api/heatmap').then(function(r) { return r.json(); }).then(function(d) {
    var pts = d.points || [];
    if (_cicHeatLayer) _cicMapObj.removeLayer(_cicHeatLayer);
    if (typeof L.heatLayer === 'undefined') {
      // Load Leaflet.heat plugin dynamically
      var s = document.createElement('script');
      s.src = 'https://cdn.jsdelivr.net/npm/leaflet.heat@0.2.0/dist/leaflet-heat.js';
      s.onload = function() {
        _cicHeatLayer = L.heatLayer(pts, {radius:25, blur:20, maxZoom:17, max:1.0,
          gradient:{0.0:'blue',0.3:'lime',0.6:'yellow',0.8:'orange',1.0:'red'}}).addTo(_cicMapObj);
      };
      document.head.appendChild(s);
    } else {
      _cicHeatLayer = L.heatLayer(pts, {radius:25, blur:20, maxZoom:17,
        gradient:{0.0:'blue',0.3:'lime',0.6:'yellow',0.8:'orange',1.0:'red'}}).addTo(_cicMapObj);
    }
  }).catch(function() {});
}

function cicRefreshMap() {
  if (_cicMapObj) { _cicMapObj.remove(); _cicMapObj = null; _cicZonePolys = {}; }
  _cicInitMap();
}

// ── Chart.js Analytics ─────────────────────────────────────────────────────
function _cicInitCharts() {
  if (typeof Chart === 'undefined') return;
  fetch('/crowd/api/status').then(function(r) { return r.json(); }).then(function(d) {
    var zones   = d.zones || {};
    var names   = Object.values(zones).map(function(z) { return z.name; });
    var counts  = Object.values(zones).map(function(z) { return z.count || 0; });
    var densities = Object.values(zones).map(function(z) { return +(z.density || 0).toFixed(4); });
    var colors  = Object.values(zones).map(function(z) { return z.color || '#6366f1'; });

    var barCtx = document.getElementById('cic-bar-canvas').getContext('2d');
    _cicBarChart = new Chart(barCtx, {
      type: 'bar',
      data: {
        labels: names,
        datasets: [{
          label: 'Persons',
          data: counts,
          backgroundColor: colors.map(function(c) { return c + 'aa'; }),
          borderColor: colors,
          borderWidth: 1.5,
          borderRadius: 4
        }]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: {legend: {display: false}},
        scales: {
          x: {ticks: {color:'#94a3b8', font:{size:10}}, grid: {color:'#252836'}},
          y: {ticks: {color:'#94a3b8', font:{size:10}}, grid: {color:'#252836'}, beginAtZero: true}
        }
      }
    });

    var denCtx = document.getElementById('cic-density-canvas').getContext('2d');
    _cicDenChart = new Chart(denCtx, {
      type: 'bar',
      data: {
        labels: names,
        datasets: [{
          label: 'p/m²',
          data: densities,
          backgroundColor: densities.map(function(v) {
            return v >= 6 ? '#ef4444aa' : v >= 3 ? '#f97316aa' : v >= 1.5 ? '#f59e0baa' : '#22c55eaa';
          }),
          borderColor: densities.map(function(v) {
            return v >= 6 ? '#ef4444' : v >= 3 ? '#f97316' : v >= 1.5 ? '#f59e0b' : '#22c55e';
          }),
          borderWidth: 1.5,
          borderRadius: 4
        }]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: {legend: {display: false}},
        scales: {
          x: {ticks: {color:'#94a3b8', font:{size:10}}, grid: {color:'#252836'}},
          y: {ticks: {color:'#94a3b8', font:{size:10}}, grid: {color:'#252836'}, beginAtZero: true}
        }
      }
    });
  }).catch(function() {});
}

function _cicLiveUpdateCharts(zones) {
  if (!_cicBarChart || !_cicDenChart) return;
  var names = _cicBarChart.data.labels;
  var counts = [], dens = [];
  names.forEach(function(name) {
    var z = Object.values(zones).find(function(zz) { return zz.name === name; });
    counts.push(z ? (z.count || 0) : 0);
    dens.push(z   ? +(z.density || 0).toFixed(4) : 0);
  });
  _cicBarChart.data.datasets[0].data = counts;
  _cicBarChart.update('none');

  _cicDenChart.data.datasets[0].data = dens;
  _cicDenChart.data.datasets[0].backgroundColor = dens.map(function(v) {
    return v >= 6 ? '#ef4444aa' : v >= 3 ? '#f97316aa' : v >= 1.5 ? '#f59e0baa' : '#22c55eaa';
  });
  _cicDenChart.data.datasets[0].borderColor = dens.map(function(v) {
    return v >= 6 ? '#ef4444' : v >= 3 ? '#f97316' : v >= 1.5 ? '#f59e0b' : '#22c55e';
  });
  _cicDenChart.update('none');
}

// ── LLM Operator Assistant ─────────────────────────────────────────────────
function cicAsk() {
  var qEl  = document.getElementById('cic-llm-q');
  var resp = document.getElementById('cic-llm-response');
  if (!qEl || !resp) return;
  var q = qEl.value.trim();
  if (!q) return;

  resp.textContent = 'Thinking…';
  resp.style.color = 'var(--text-muted)';

  fetch('/crowd/api/ask', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({question: q})
  }).then(function(r) {
    var reader  = r.body.getReader();
    var decoder = new TextDecoder();
    var text    = '';
    resp.textContent = '';
    resp.style.color = 'var(--text-primary)';

    function pump() {
      reader.read().then(function(res) {
        if (res.done) return;
        var raw = decoder.decode(res.value, {stream: true});
        raw.split('\n').forEach(function(line) {
          if (line.startsWith('data:')) {
            try {
              var msg = JSON.parse(line.slice(5));
              if (msg.done) return;
              if (msg.text) { text += msg.text; resp.textContent = text; resp.scrollTop = resp.scrollHeight; }
            } catch (ex) {}
          }
        });
        pump();
      }).catch(function() {});
    }
    pump();
  }).catch(function(e) {
    resp.textContent = 'Error: ' + e.message;
    resp.style.color = 'var(--red)';
  });
}

// ── Khoya-Paya (Lost Person Search) ───────────────────────────────────────
function cicKhoyaFile(input) {
  var file = input.files[0];
  if (!file) return;
  var reader = new FileReader();
  reader.onload = function(e) {
    var b64 = e.target.result;
    var prev = document.getElementById('cic-khoya-preview');
    var res  = document.getElementById('cic-khoya-results');
    if (prev) { prev.src = b64; prev.style.display = 'block'; }
    if (res)  { res.style.display = 'block'; res.innerHTML = '<span style="color:var(--text-muted)">Searching face database…</span>'; }

    fetch('/crowd/api/khoya', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({image: b64})
    }).then(function(r) { return r.json(); }).then(function(d) {
      if (!res) return;
      if (d.error === 'no_face') {
        res.innerHTML =
          '<div style="color:var(--yellow);font-weight:600;margin-bottom:6px">&#9888; No face detected</div>' +
          'Upload a clear, well-lit frontal photo. Avoid sunglasses or head coverings.<br>' +
          '<small style="color:var(--text-muted)">Faces indexed from cameras: ' + (d.cic_faces_indexed||0) + '</small>';
        return;
      }
      if (d.error) {
        res.innerHTML = '<span style="color:var(--red)">Error: ' + d.error + '</span>';
        return;
      }
      if (d.found && d.matches && d.matches.length) {
        var html = '<div style="color:var(--green);font-weight:700;margin-bottom:10px">&#10003; ' + d.matches.length + ' match(es) found</div>';
        d.matches.forEach(function(m, i) {
          var pct = (m.score * 100).toFixed(1);
          var icon = m.source === 'cic' ? '&#128247;' : '&#128269;';
          html += '<div style="border:1px solid var(--border);border-radius:6px;padding:8px;margin-bottom:8px">';
          html += '<b>' + icon + ' #' + (i+1) + ' — ' + m.name + '</b><br>';
          html += 'Confidence: <b style="color:var(--green)">' + pct + '%</b><br>';
          if (m.source === 'cic') {
            html += 'Source: Live Camera (Slot ' + m.slot + ')<br>';
            html += 'Zone: <b>' + m.zone + '</b><br>';
            html += 'Last seen: ' + m.last_seen;
          } else {
            html += 'Source: OSINT database<br>';
            html += 'Search ID: ' + m.search_id;
          }
          html += '</div>';
        });
        html += '<small style="color:var(--text-muted)">Faces indexed from cameras: ' + (d.cic_faces_indexed||0) + '</small>';
        res.innerHTML = html;
      } else {
        var indexed = d.cic_faces_indexed || 0;
        res.innerHTML =
          '<div style="color:var(--yellow);font-weight:600;margin-bottom:6px">No match found</div>' +
          'This person was not found in the face database.<br><br>' +
          '<b>Faces indexed from cameras: ' + indexed + '</b><br>' +
          (indexed === 0
            ? '<small style="color:var(--text-muted)">Start a camera slot and wait ~60s for faces to be captured automatically.</small>'
            : '<small style="color:var(--text-muted)">Try a clearer frontal photo, or the person may not have been captured yet.</small>');
      }
    }).catch(function(e) {
      if (res) res.textContent = 'Error: ' + e.message;
    });
  };
  reader.readAsDataURL(file);
}

// Load Leaflet + Chart.js CDN lazily when CIC first opens
(function() {
  function loadScript(src, id, cb) {
    if (document.getElementById(id)) { if (cb) cb(); return; }
    var s = document.createElement('script');
    s.src = src; s.id = id; s.onload = cb || null;
    document.head.appendChild(s);
  }
  // Expose helper so toggleCIC can trigger load
  window._cicLoadLibs = function(cb) {
    loadScript('https://unpkg.com/leaflet@1.9.4/dist/leaflet.js', 'cic-leaflet-js', function() {
      loadScript('https://cdn.jsdelivr.net/npm/chart.js@4.4.2/dist/chart.umd.min.js', 'cic-chartjs', cb);
    });
  };
})();

// Patch toggleCIC to load libs first
var _toggleCICOrig = toggleCIC;
toggleCIC = function() {
  if (!_cicOpen) {
    window._cicLoadLibs(function() { _toggleCICOrig(); });
  } else {
    _toggleCICOrig();
  }
};

loadHistory();
</script>
</body>
</html>"""


# ══════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    PORT = 5000
    URL  = f"http://localhost:{PORT}"

    sep = "=" * 56
    print("\n" + sep)
    print("  Face OSINT v4.2  +  Crowd Intelligence Center")
    print(sep)
    print(f"\n  Opening: {URL}")
    print(f"  Output:  {config.OUTPUT_DIR}")
    print(f"  DB:      {config.DB_PATH}")
    print(f"  Logs:    {config.LOG_DIR}")
    print("\n  CIC: click [CIC] button in top bar")
    print(sep + "\n")

    # ISSUE 5 FIX: bind to 127.0.0.1 — camera works, no IP leak
    # Auto-open browser after 1.2s (gives Flask time to start)
    def _open_browser():
        time.sleep(1.2)
        webbrowser.open_new_tab(URL)

    threading.Thread(target=_open_browser, daemon=True).start()

    app.run(
        host="127.0.0.1",   # localhost only — camera works on HTTP
        port=PORT,
        debug=False,        # debug=True with 0.0.0.0 is a security risk
        threaded=True,      # required for SSE streams
    )