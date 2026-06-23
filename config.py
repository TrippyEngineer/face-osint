"""
config.py
──────────
Single source of truth for all configuration.
Every constant, threshold, and path lives here.
No magic numbers anywhere else in the codebase.

Loads .env automatically if present.
All values have safe defaults so the system runs without any API keys —
free-tier sources (academic, passive, reverse image) work out of the box.
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env from project root ───────────────────────────────────────────
ROOT = Path(__file__).parent.resolve()
load_dotenv(ROOT / ".env", override=False)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  PATHS                                                               ║
# ╚══════════════════════════════════════════════════════════════════════╝
OUTPUT_DIR   = ROOT / "data" / "output"
LOG_DIR      = ROOT / "logs"
DB_PATH      = ROOT / "data" / "face_osint.db"
MODELS_DIR   = ROOT / "data" / "models"        # DeepFace model cache

# Auto-create required directories on import
for _d in [OUTPUT_DIR, LOG_DIR, DB_PATH.parent, MODELS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  LOGGING                                                             ║
# ╚══════════════════════════════════════════════════════════════════════╝
LOG_LEVEL      = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_MAX_BYTES  = 10 * 1024 * 1024   # 10 MB per log file
LOG_BACKUP_COUNT = 5                 # keep 5 rotated files


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CAMERA                                                              ║
# ╚══════════════════════════════════════════════════════════════════════╝
WIFI_WIDTH       = 640
WIFI_HEIGHT      = 480
RECONNECT_AFTER  = 30     # consecutive empty reads before reconnect
PREVIEW_WIDTH    = 960
PREVIEW_HEIGHT   = 540

IP_WEBCAM_ENDPOINTS = [
    "/videofeed",
    "/video",
    "/video?submenu=mjpg",
]


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  FACE EMBEDDING (DeepFace)                                           ║
# ╚══════════════════════════════════════════════════════════════════════╝
DEEPFACE_MODEL    = "Facenet512"     # 512D embeddings, best accuracy
DEEPFACE_DETECTOR = "opencv"         # fast, reliable, no extra deps
DEEPFACE_ENFORCE  = False            # don't crash if no face detected

# Cosine similarity thresholds for face matching
# Calibrated for Facenet512 — same values used in attendance.py
FACE_CONFIRMED  = 0.68   # high confidence — same person
FACE_POSSIBLE   = 0.50   # possible match — review manually
FACE_REJECTED   = 0.35   # different person — discard
# Skip face-matching candidate images smaller than this on the long side —
# favicons/thumbnails (e.g. 32x32) embed to noise and yield spurious high scores.
FACE_MATCH_MIN_PX = 120


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  SCRAPER BEHAVIOUR                                                   ║
# ╚══════════════════════════════════════════════════════════════════════╝
SCRAPER_TIMEOUT_S   = 20     # max seconds per scraper before it is abandoned
# = number of scrapers (run_search dispatches 7) so none queue. Queueing was the
# bug: with 4 workers the last-submitted scrapers (username/passive) started ~20s
# late, so their per-scraper deadline (measured from scraping start) expired while
# they were still queued and their results were discarded. Mostly I/O-bound; the
# two heavy ones (reverse_face, username) each spawn a browser, so peak is ~3
# headless chromium — fine on any machine that already runs the DeepFace/YOLO stack.
SCRAPER_MAX_WORKERS = 7      # one worker per scraper — no queueing, deadlines accurate
SEARCH_DEDUP_TTL_S  = 300    # max age of an image-hash dedup entry before a TTL sweep drops it
HTTP_TIMEOUT_S      = 12     # per HTTP request timeout
HTTP_RETRIES        = 2      # retry transient failures this many times

# Browser headers — prevents trivial bot detection
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  AGGREGATION SCORING                                                 ║
# ╚══════════════════════════════════════════════════════════════════════╝
# Scorer weights — must sum to 1.0
# combined = W_FACE*face + W_NAME*name + W_SOCIAL*social + W_PHOTO*photo + W_SOURCES*sources
W_FACE    = 0.70   # face similarity is primary signal
W_NAME    = 0.10   # name match is a hint only
W_SOCIAL  = 0.08   # bonus for real social profile URL
W_PHOTO   = 0.05   # bonus for having a downloadable photo
W_SOURCES = 0.07   # bonus for appearing across multiple scrapers

# Score thresholds for final verdict
VERDICT_CONFIRMED_HIGH = 0.85
VERDICT_CONFIRMED_LOW  = 0.70
VERDICT_POSSIBLE       = 0.55
MIN_SCORE_KEEP         = 0.25   # drop results below this


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  API KEYS  (all optional — free sources work without any keys)       ║
# ╚══════════════════════════════════════════════════════════════════════╝
# GOOGLE_CSE_KEY + GOOGLE_CSE_ID:
#   1. Go to https://programmablesearchengine.google.com → New Search Engine
#   2. Sites: linkedin.com, twitter.com, instagram.com (or leave blank for web)
#   3. Get API key at https://console.cloud.google.com → Custom Search API
#   4. 100 free queries/day. Best source for LinkedIn results.
GOOGLE_CSE_KEY  = os.getenv("GOOGLE_CSE_KEY",  "")
GOOGLE_CSE_ID   = os.getenv("GOOGLE_CSE_ID",   "")
BING_API_KEY    = os.getenv("BING_API_KEY",    "")
# BING_SEARCH_KEY is the same credential used for Bing Visual Search API
BING_SEARCH_KEY = os.getenv("BING_SEARCH_KEY", os.getenv("BING_API_KEY", ""))
BRAVE_API_KEY   = os.getenv("BRAVE_API_KEY",   "")
GITHUB_TOKEN    = os.getenv("GITHUB_TOKEN",    "")
GITLAB_TOKEN    = os.getenv("GITLAB_TOKEN",    "")
SERPAPI_KEY     = os.getenv("SERPAPI_KEY",     "")
IMGBB_API_KEY   = os.getenv("IMGBB_API_KEY",   "")
REDDIT_CLIENT_ID     = os.getenv("REDDIT_CLIENT_ID",     "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT    = os.getenv("REDDIT_USER_AGENT",    "FaceOSINT/1.0")
HUNTER_API_KEY   = os.getenv("HUNTER_API_KEY",   "")
FACECHECK_API_KEY = os.getenv("FACECHECK_API_KEY", "")
# PimEyes reverse face search (paid, https://pimeyes.com)
PIMEYES_API_KEY  = os.getenv("PIMEYES_API_KEY",  "")
# Optional: set this to your email to get into OpenAlex's "polite pool" (higher rate limits)
OPENALEX_MAILTO  = os.getenv("OPENALEX_MAILTO",  "")
# Claude API key — used by the CIC LLM operator assistant
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  CROWD INTELLIGENCE CENTER (CIC)                                     ║
# ╚══════════════════════════════════════════════════════════════════════╝
# Default risk thresholds (persons / m²) — overridden per-zone in zones.json
CIC_DENSITY_CAUTION  = 1.5   # → yellow warning
CIC_DENSITY_HIGH     = 3.0   # → orange high risk
CIC_DENSITY_CRITICAL = 6.0   # → red critical / stampede alert

CIC_INFERENCE_FPS    = 5     # target FPS for YOLOv8n inference per slot
CIC_MAX_SLOTS        = 4     # simultaneous camera slots

# ── CIC detection / calibration ────────────────────────────────────────────
# Detection model + thresholds. Swap CIC_YOLO_MODEL to "yolov8s.pt"/"yolov8m.pt"
# for higher recall in dense crowds (auto-downloads; slower on CPU).
CIC_YOLO_MODEL        = "yolov8n.pt"
CIC_YOLO_CONF         = 0.25    # lower than the old 0.35 → catches more people
CIC_INFERENCE_SIZE    = 960     # long-side px fed to YOLO (was a forced 640×480)
CIC_MAX_DET           = 1000    # per-frame detection cap (was YOLO default 300)

# Density = count / camera field-of-view area. Per-zone fov_area_m2 lives in
# zones.json; this is the fallback when a zone omits it. NOTE: density must use
# the camera's visible area, NOT the whole zone area, or risk never escalates.
CIC_FOV_AREA_M2       = float(os.getenv("CIC_FOV_AREA_M2", "100.0"))

# Behavioural heuristics (crude — tune per deployment). Stricter than the old
# 25-frame / 12-px defaults to cut false positives in crowds.
CIC_CHILD_HEIGHT_RATIO = 0.22   # bbox height < ratio*frame height → "child"
CIC_LOITER_FRAMES      = 40     # frames stationary before "loitering" (~8s @5fps)
CIC_RUNNING_SPEED      = 18.0   # px/frame velocity to flag "running"
CIC_FACE_MIN_CROP_PX   = 70     # skip Khoya-Paya capture for crops smaller than this

# ── Crowd-PRESSURE early-warning (crowd/pressure.py) ────────────────────────
# Density bands (persons/m²): a density threshold alone is NOT stampede prediction
# (the 2025 Sangam crush fired its alarm too late). Helbing pressure = density ×
# velocity-variance + turbulence escalate the risk EARLIER. Calibrate per camera.
CIC_PRESSURE_ENABLED       = os.getenv("CIC_PRESSURE_ENABLED", "1") not in ("0", "false", "False")
CIC_DENSE_DENSITY          = float(os.getenv("CIC_DENSE_DENSITY", "2.0"))         # ppl/m² — comfortable-but-crowded
CIC_COMPRESSION_DENSITY    = float(os.getenv("CIC_COMPRESSION_DENSITY", "5.0"))   # ppl/m² — body compression begins (risky)
CIC_CRITICAL_DENSITY       = float(os.getenv("CIC_CRITICAL_DENSITY", "8.0"))      # ppl/m² — crush regime (critical)
CIC_TURBULENCE_CV          = float(os.getenv("CIC_TURBULENCE_CV", "0.75"))        # velocity coeff-of-variation → stop-and-go/turbulence

# ── SOP playbook engine (crowd/sop.py) ──────────────────────────────────────
CIC_SOP_ESCALATE_S         = int(os.getenv("CIC_SOP_ESCALATE_S", "120"))  # unacked high/critical SOP escalates after this


# ── CIC Phase 4: persistence / retention ────────────────────────────────────
CIC_READING_PERSIST_S = 10     # min seconds between persisted zone-reading snapshots
CIC_DATA_TTL_DAYS     = 30     # prune cic_alerts / cic_zone_readings older than this

# ── CIC Phase 4: outbound webhook notifications ─────────────────────────────
CIC_WEBHOOK_URL          = os.getenv("CIC_WEBHOOK_URL", "")               # empty → disabled
CIC_WEBHOOK_MIN_SEVERITY = os.getenv("CIC_WEBHOOK_MIN_SEVERITY", "high")  # warning|high|critical
CIC_WEBHOOK_HEADERS      = os.getenv("CIC_WEBHOOK_HEADERS", "")           # optional JSON string
CIC_WEBHOOK_TIMEOUT_S    = int(os.getenv("CIC_WEBHOOK_TIMEOUT_S", "6"))
CIC_DISCORD_WEBHOOK_URL  = os.getenv("CIC_DISCORD_WEBHOOK_URL", "")       # Discord channel webhook
CIC_NOTIFY_WORKERS       = int(os.getenv("CIC_NOTIFY_WORKERS", "3"))      # bounded notifier dispatch pool

# ── CIC Phase 4: incident clips ─────────────────────────────────────────────
CIC_CLIPS_ENABLED = os.getenv("CIC_CLIPS_ENABLED", "1") not in ("0", "false", "False")
CIC_CLIP_PRE_S    = int(os.getenv("CIC_CLIP_PRE_S", "10"))
CIC_CLIP_POST_S   = int(os.getenv("CIC_CLIP_POST_S", "10"))
CIC_INCIDENT_DIR  = OUTPUT_DIR / "cic_incidents"
CIC_INCIDENT_DIR.mkdir(parents=True, exist_ok=True)

# ── CIC Phase 4: dense-crowd tiling ─────────────────────────────────────────
CIC_TILING       = os.getenv("CIC_TILING", "0") in ("1", "true", "True")
CIC_TILE_GRID    = os.getenv("CIC_TILE_GRID", "2x2")
CIC_TILE_OVERLAP = float(os.getenv("CIC_TILE_OVERLAP", "0.2"))
CIC_TILE_NMS_IOU = float(os.getenv("CIC_TILE_NMS_IOU", "0.5"))


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  WEB / UPLOAD                                                        ║
# ╚══════════════════════════════════════════════════════════════════════╝
MAX_UPLOAD_MB         = 512     # max request body (CIC videos); was a hard 10 MB


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  LOGGING SETUP  (call once at startup)                               ║
# ╚══════════════════════════════════════════════════════════════════════╝
def setup_logging(session_name: str = "session") -> logging.Logger:
    """
    Configure root logger with:
      • RotatingFileHandler → logs/YYYYMMDD_HHMMSS.log
      • StreamHandler       → console (INFO+ only, concise format)

    Call once from main.py. All other modules use:
        logger = logging.getLogger(__name__)
    """
    import logging.handlers
    from datetime import datetime

    log_file = LOG_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{session_name}.log"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Suppress noisy third-party loggers
    for noisy in ["urllib3", "httpx", "httpcore", "tensorflow",
                  "absl", "h11", "asyncio"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # ── File handler — full DEBUG detail ──────────────────────────────
    fh = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes    = LOG_MAX_BYTES,
        backupCount = LOG_BACKUP_COUNT,
        encoding    = "utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(name)-28s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    # ── Console handler — INFO+ only, cleaner ─────────────────────────
    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    ch.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    ))

    root.addHandler(fh)
    root.addHandler(ch)

    root.info(f"Logging started → {log_file}")
    return root
