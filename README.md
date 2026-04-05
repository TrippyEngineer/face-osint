# Face OSINT

> **Capture a face. Get an identity.**
> Point your camera at someone, upload a photo, or stream from a phone — Face OSINT searches 10+ open-source intelligence sources simultaneously, face-verifies every result, and delivers a ranked, de-duplicated identity profile in under two minutes.

**No Docker. No Redis. No cloud accounts. No API keys required to start.**
Runs entirely on your local machine with a single `pip install`.

---

## What it does

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│   📷  Camera / 🖼 Photo / 📡 WiFi Cam                                  │
│                      │                                                  │
│                      ▼                                                  │
│          ┌───────────────────────┐                                      │
│          │  DeepFace Facenet512  │  512-D face vector in ~0.3s          │
│          └──────────┬────────────┘                                      │
│                     │                                                   │
│          ┌──────────▼────────────┐                                      │
│          │   7 scrapers fire     │  parallel, isolated, deadline-capped │
│          │   simultaneously      │                                      │
│          │                       │                                      │
│          │  🔍 Reverse face     │  Yandex · SerpApi Lens · Google CSE  │
│          │  🔎 Search engines   │  LinkedIn · Twitter · Instagram +8   │
│          │  🎓 Academic         │  Scholar · OpenAlex · ORCID          │
│          │  💻 GitHub / Reddit  │  public API, no auth needed          │
│          │  🕵️  Username        │  125 direct checks + Sherlock        │
│          │  🗂  Passive          │  Wayback · GDELT · crt.sh · PGP     │
│          └──────────┬────────────┘                                      │
│                     │                                                   │
│          ┌──────────▼────────────┐                                      │
│          │   Face verification   │  download + cosine-compare photos   │
│          │   of every result     │  against your query embedding        │
│          └──────────┬────────────┘                                      │
│                     │                                                   │
│          ┌──────────▼────────────┐                                      │
│          │   Weighted scoring    │  70% face · 10% name · 18% meta     │
│          │   Entity resolution   │  Union-Find identity graph           │
│          └──────────┬────────────┘                                      │
│                     │                                                   │
│          ┌──────────▼────────────┐                                      │
│          │   📄 info.txt         │  readable report                     │
│          │   📊 JSON summary     │  full scored match data              │
│          │   🖼  scraped photos  │  face-verified profile images        │
│          │   🗄  SQLite history  │  search history + face vectors       │
│          └───────────────────────┘                                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Quick start

```bash
# Python 3.10 or 3.11 required  (3.12 breaks TensorFlow 2.16)
pip install -r requirements.txt

cp .env.example .env          # all keys optional — works without any

python diagnose.py             # health check: packages, keys, network
python app.py                  # web UI — browser opens automatically
```

First run downloads Facenet512 model weights (~600 MB) into `data/models/` once.

---

## Architecture

```
╔══════════════════════════════════════════════════════════════════════════╗
║  ENTRY POINTS                                                            ║
║                                                                          ║
║   app.py  ──────  Flask + SSE web UI  ──────  browser camera / upload   ║
║   main.py  ─────  OpenCV display loop  ────  local webcam only          ║
╚══════════════════╤═══════════════════════════════════════════════════════╝
                   │  shared pipeline
                   ▼
╔══════════════════════════════════════════════════════════════════════════╗
║  LAYER 1 — EMBEDDING                        embedding.py                ║
║                                                                          ║
║  DeepFace.represent()  →  512-D Facenet512 vector  →  L2-normalised     ║
║  Detector: opencv  |  Confidence threshold: 0.50  |  Align: true        ║
║  Thread-safe: stateless, model cached internally by DeepFace            ║
╚══════════════════╤═══════════════════════════════════════════════════════╝
                   │
                   ▼
╔══════════════════════════════════════════════════════════════════════════╗
║  LAYER 2 — FACE DATABASE                    storage/database.py         ║
║                                                                          ║
║  SQLite WAL  ·  cosine search across stored vectors                      ║
║  Tables: searches · matches · face_vectors                               ║
║  Thread-safe: each method opens its own connection                       ║
╚══════════════════╤═══════════════════════════════════════════════════════╝
                   │
                   ▼
╔══════════════════════════════════════════════════════════════════════════╗
║  LAYER 3 — PARALLEL SCRAPING                scrapers/                   ║
║                                                                          ║
║  ThreadPoolExecutor(max_workers=4)  ·  per-scraper deadline enforcement  ║
║                                                                          ║
║  ┌────────────────┬────────┬──────────────────────────────────────────┐  ║
║  │ Scraper        │Timeout │ Sources                                  │  ║
║  ├────────────────┼────────┼──────────────────────────────────────────┤  ║
║  │ reverse_face   │  90s   │ Yandex · SerpApi Lens · Google CSE       │  ║
║  │                │        │ 4 engines parallel · face-verify top 25  │  ║
║  ├────────────────┼────────┼──────────────────────────────────────────┤  ║
║  │ search_engines │  25s   │ Google CSE → DDG → Bing (cascade)        │  ║
║  │                │        │ LinkedIn-first + 10 social platforms      │  ║
║  ├────────────────┼────────┼──────────────────────────────────────────┤  ║
║  │ academic       │  35s   │ Semantic Scholar · OpenAlex · ORCID      │  ║
║  │                │        │ Scholar dedup by user ID · 1hr cache      │  ║
║  ├────────────────┼────────┼──────────────────────────────────────────┤  ║
║  │ github         │  20s   │ GitHub API · public profile + repos       │  ║
║  ├────────────────┼────────┼──────────────────────────────────────────┤  ║
║  │ reddit         │  15s   │ PRAW or public JSON fallback              │  ║
║  ├────────────────┼────────┼──────────────────────────────────────────┤  ║
║  │ passive        │  20s   │ Wayback Machine · GDELT · crt.sh         │  ║
║  │                │        │ Gravatar · PGP keyservers                 │  ║
║  ├────────────────┼────────┼──────────────────────────────────────────┤  ║
║  │ username       │  50s   │ 125 direct platform checks               │  ║
║  │                │        │ + Sherlock 300+ platforms (15s cap)       │  ║
║  └────────────────┴────────┴──────────────────────────────────────────┘  ║
║                                                                          ║
║  LinkedIn enrichment: reverse_face confirmed hits re-fed into           ║
║  search_engines for deeper profile discovery                             ║
╚══════════════════╤═══════════════════════════════════════════════════════╝
                   │
                   ▼
╔══════════════════════════════════════════════════════════════════════════╗
║  LAYER 4 — FACE MATCHING                    aggregator/face_matcher.py  ║
║                                                                          ║
║  URL-deduplicated download cache  ·  parallel embedding extraction       ║
║  Sets face_score · face_similarity · face_verified on every match        ║
║  Photo URL priority: photo_url → avatar_url → preview_url               ║
╚══════════════════╤═══════════════════════════════════════════════════════╝
                   │
                   ▼
╔══════════════════════════════════════════════════════════════════════════╗
║  LAYER 5 — SCORING                          aggregator/scorer.py        ║
║                                                                          ║
║  combined = 0.70·face + 0.10·name + 0.08·social + 0.05·photo           ║
║           + 0.07·sources                                                 ║
║                                                                          ║
║  Cosine thresholds (Facenet512-calibrated):                              ║
║    ≥ 0.68  CONFIRMED  ·  0.50–0.68  POSSIBLE  ·  < 0.35  REJECTED      ║
║                                                                          ║
║  Text-only max reachable score: 0.28  <  POSSIBLE (0.50)                ║
║  → name-only hits can never reach a "possible" verdict                  ║
╚══════════════════╤═══════════════════════════════════════════════════════╝
                   │
                   ▼
╔══════════════════════════════════════════════════════════════════════════╗
║  LAYER 6 — ENTITY RESOLUTION                aggregator/resolver.py      ║
║                                                                          ║
║  Union-Find identity graph  ·  edge weights:                             ║
║    face-match 1.0  ·  email 0.90  ·  username cross-ref 0.85            ║
║    company 0.45  ·  name 0.40  ·  location 0.30                         ║
║                                                                          ║
║  Best cluster → resolved identity with canonical verdict + score         ║
╚══════════════════╤═══════════════════════════════════════════════════════╝
                   │
                   ▼
╔══════════════════════════════════════════════════════════════════════════╗
║  OUTPUT                                     storage/folder_writer.py    ║
║                                                                          ║
║  data/output/Name_YYYYMMDD_HHMM_hash/                                   ║
║    captured_photo.jpg  ·  face_crop.jpg  ·  info.txt                    ║
║    matches_summary.json  ·  scraped_photos/                              ║
║                                                                          ║
║  SQLite row inserted  ·  Live SSE progress feed in browser               ║
╚══════════════════════════════════════════════════════════════════════════╝
```

---

## Verdict system

| Combined Score | Verdict | What it means |
|---|---|---|
| ≥ 0.85 | **CONFIRMED** (high) | Face verified at high confidence + consistent metadata |
| ≥ 0.70 | **CONFIRMED** (low) | Face verified, limited corroborating metadata |
| ≥ 0.55 | **POSSIBLE** | Near-match face OR strong metadata without face proof |
| < 0.55 | **UNLIKELY** | Name match only — no photographic evidence found |

Face similarity alone determines whether evidence reaches "possible" — text metadata alone cannot.

---

## Input modes

**Browser camera** — click the Camera tab, allow browser access, click Capture.

**File upload** — drag and drop or select any JPEG / PNG / WebP.

**WiFi / IP camera** — install [IP Webcam](https://play.google.com/store/apps/details?id=com.pas.webcam) (Android, free), tap Start server, enter the IP shown (e.g. `192.168.1.5:8080`) in the WiFi Cam tab. Phone and PC must be on the same network.

**Name hints** — common names benefit from context:
```
John Smith | New York               ← adds location
John Smith @ Acme Corp             ← adds employer
John Smith | New York @ Acme Corp  ← both
```

---

## API keys

All keys are optional. The system runs on ~8 free sources with no configuration.

Copy `.env.example` → `.env` and fill in what you have.

| Key | Where to get | Free tier | Unlocks |
|---|---|---|---|
| `GOOGLE_CSE_KEY` + `GOOGLE_CSE_ID` | [programmablesearchengine.google.com](https://programmablesearchengine.google.com) | 100 req/day | Best LinkedIn + social search |
| `SERPAPI_KEY` | [serpapi.com](https://serpapi.com) | 100 req/month | Google Lens + Yandex face search |
| `IMGBB_API_KEY` | [imgbb.com → API](https://imgbb.com) | Free | Image hosting for SerpApi upload |
| `GITHUB_TOKEN` | GitHub → Settings → Developer tokens | Free | 5,000 req/hr (vs 60/hr unauthenticated) |
| `GITLAB_TOKEN` | GitLab → Profile → Access Tokens | Free | GitLab user search (required since 2024) |
| `BING_API_KEY` | Azure → Cognitive Services → Bing v7 | 1,000 req/month | Bing web + visual search |
| `BRAVE_API_KEY` | [api.search.brave.com](https://api.search.brave.com) | 2,000 req/month | Brave Search results |
| `HUNTER_API_KEY` | [hunter.io/api-keys](https://hunter.io/api-keys) | 25 req/month | Email intelligence |
| `REDDIT_CLIENT_ID` + `SECRET` | reddit.com/prefs/apps | Free | Authenticated Reddit API |
| `OPENALEX_MAILTO` | any email address | Free | OpenAlex polite pool (higher rate limits) |

Run `python diagnose.py` at any time to see which keys are active and test connectivity.

---

## Output structure

```
data/output/
└── John_Doe_20260404_1430_a1b2c3d4/
    ├── captured_photo.jpg      original camera frame
    ├── face_crop.jpg           160×160 aligned face crop (Facenet512 input)
    ├── info.txt                human-readable plaintext report
    ├── matches_summary.json    full structured data — all scores + metadata
    └── scraped_photos/
        ├── github_johndoe_a1b2.jpg
        ├── reverse_face_yandex_c3d4.jpg
        └── ...
```

Search history and face vectors are persisted in `data/face_osint.db` (SQLite WAL).

---

## CLI interface

```bash
python main.py
```

Runs the full pipeline in an OpenCV window. Controls (click the window first):

| Key | Action |
|---|---|
| `SPACE` | Freeze frame → enter name → start full OSINT search |
| `F` | Flip / mirror the camera feed |
| `D` | Diagnostic — print cosine distances against all stored face vectors |
| `H` | Toggle HUD overlay |
| `Q` | Quit |

---

## Tech stack

| Component | Library | Why |
|---|---|---|
| Face embedding | [DeepFace](https://github.com/serengil/deepface) + Facenet512 | 512-D L2-normalised vectors, ArcFace-level accuracy |
| Web UI | Flask + Server-Sent Events | Zero-install, live progress streaming |
| Camera | OpenCV (`cv2`) | Native Windows webcam + MJPEG WiFi streams |
| Persistence | SQLite (WAL mode) | Single file, zero config, thread-safe |
| HTML parsing | BeautifulSoup + lxml | Fast, lenient scraping of search results |
| Name matching | rapidfuzz | Fuzzy string scoring for entity resolution |
| Username OSINT | [Sherlock](https://github.com/sherlock-project/sherlock) + direct checks | 300+ platforms |

**Design constraints:** single process · no Celery/Redis/Docker/PostgreSQL · daemon threads + `ThreadPoolExecutor` · all state behind `threading.Lock`.

---

## Requirements

- Python **3.10** or **3.11** (3.12 breaks TensorFlow 2.16)
- ~600 MB disk for model weights (downloaded once on first run)
- Webcam or WiFi camera (or just use file upload)

```bash
pip install -r requirements.txt
```

Optional extras for better coverage:
```bash
pip install sherlock-project    # username scraper Layer 1 (300+ platforms)
pip install socid-extractor     # username scraper Layer 2 (social ID extraction)
pip install praw                # authenticated Reddit API
```

---

## Troubleshooting

**"No face detected"** — face must fill at least 20% of the frame; avoid backlighting; try moving closer.

**"no_results" verdict on a common name** — add context: `John Smith | London` or `John Smith @ Google`.

**Search takes ~90 seconds** — expected; `reverse_face` downloads and face-verifies up to 25 candidate photos against your query embedding.

**TensorFlow import errors** — you must use Python 3.10 or 3.11. TensorFlow 2.16 does not support Python 3.12.

**LinkedIn always returns 0 results** — set `GOOGLE_CSE_KEY` + `GOOGLE_CSE_ID` in `.env` (DDG and Bing are fallbacks but return fewer LinkedIn results).

**WiFi camera won't connect** — phone and PC must be on the same WiFi subnet. Test `http://IP:8080/video` in a browser first before using the app.

Run `python diagnose.py` for an end-to-end health check with targeted fix instructions.

---

## Data & privacy

Everything stays on your machine:

| Path | Contents |
|---|---|
| `data/face_osint.db` | Search history, match records, 512-D face vectors |
| `data/output/` | Per-search folders: photos, reports, JSON |
| `data/models/` | DeepFace model weights (~600 MB, downloaded once) |
| `logs/` | Rotating logs, 10 MB × 5 files |

To wipe all search data: delete `data/face_osint.db` and `data/output/`. The tool makes no outbound connections except to the scraping sources listed above — no telemetry, no cloud sync.

---

## Responsible use

This tool is built for **legitimate OSINT research** — verifying identities, locating missing persons, investigating fraud, academic research, and penetration testing with explicit authorisation.

Do not use it to stalk, harass, or build unauthorised profiles of private individuals. Comply with applicable laws in your jurisdiction. Respect platform terms of service.
