<div align="center">

# Face OSINT + Crowd Intelligence Center

**Capture a face. Identify an individual. Command a crowd.**

Two systems in one: a deep OSINT identity pipeline powered by face recognition,
and a real-time AI crowd management platform modeled on India's ICCC deployments at Kumbh Mela scale.

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-SSE%20Live%20UI-000000?style=flat-square&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![YOLOv8](https://img.shields.io/badge/YOLOv8n-Person%20Detection-00B4D8?style=flat-square)](https://docs.ultralytics.com)
[![DeepFace](https://img.shields.io/badge/DeepFace-Facenet512-FF6B6B?style=flat-square)](https://github.com/serengil/deepface)
[![Claude](https://img.shields.io/badge/Claude%20API-LLM%20Operator-8B5CF6?style=flat-square)](https://anthropic.com)
[![Leaflet](https://img.shields.io/badge/Leaflet.js-GIS%20Heat%20Map-199900?style=flat-square&logo=leaflet&logoColor=white)](https://leafletjs.com)
[![SQLite](https://img.shields.io/badge/SQLite-WAL%20Mode-003B57?style=flat-square&logo=sqlite&logoColor=white)](https://sqlite.org)
[![License](https://img.shields.io/badge/License-Research%20%2F%20Educational-yellow?style=flat-square)](#responsible-use)

**No Docker. No Redis. No cloud accounts. Runs on a laptop.**  
Single `pip install` → `python app.py` → browser opens.

> This project is for research and educational purposes only.

</div>

---

## Two Modes, One Application

```
┌──────────────────────────────────────────────────────────┐
│                    localhost:5000                         │
│                                                          │
│  [ Face OSINT ]  ──────────  [ ⚡ CIC ]                  │
│                                                          │
│  Identity search             Crowd management            │
│  Face recognition            Real-time surveillance      │
│  OSINT scraping              AI operator interface       │
│  Entity resolution           GIS density heat maps       │
└──────────────────────────────────────────────────────────┘
```

Click **[CIC]** in the top bar to switch into Crowd Intelligence Center mode.
Click **[Face OSINT]** to return to identity investigation mode.
Both run on the same Flask server, same camera infrastructure, same SQLite database.

---

## Crowd Intelligence Center (CIC)

A laptop-scale depiction of an Integrated Command and Control Centre (ICCC) as deployed at India's Maha Kumbh Mela and similar mega-events. Real detection algorithms. Real risk scoring. Real LLM-powered operator interface.

### Live Architecture

```mermaid
flowchart LR
    subgraph CAMERAS["Camera Inputs (4 slots)"]
        C0["Slot 0\nWebcam / IP cam"]
        C1["Slot 1\nVideo clip / RTSP"]
        C2["Slot 2\nVideo clip / RTSP"]
        C3["Slot 3\nVideo clip / RTSP"]
    end

    subgraph AI["Per-Camera AI Pipeline  ·  crowd/analyzer.py"]
        direction TB
        YOLO["YOLOv8n\nPerson detection\n5 FPS · CPU · 6MB"]
        TRACK["ByteTrack\nPersistent IDs\ntrajectory history"]
        FLOW["Farneback\nOptical flow\ndirection + speed"]
        BEH["Behavioral AI\nLoitering · Running\nChildren · Suspicious"]
        RISK["Zone Density\npersons / m²\nSAFE→CAUTION→CRITICAL"]
        YOLO --> TRACK --> BEH
        FLOW --> BEH
        BEH --> RISK
    end

    subgraph PLAT["Platform Aggregator  ·  crowd/platform.py"]
        AGG["Metric fusion\n1 Hz broadcast\nAlert generation"]
        HIST["5-min rolling\nhistory per zone\nChart.js trend"]
        SSE["Server-Sent Events\nLive push to all\nbrowser tabs"]
        AGG --> HIST --> SSE
    end

    subgraph UI["Dashboard UI  ·  5 tabs"]
        CAM["Live Cameras\n2×2 grid + overlay\ntoggle controls"]
        MAP["Zone Map\nLeaflet GIS\nDensity heat map"]
        ALERT["Alerts & SOPs\nLLM operator\nAcknowledgement"]
        CHART["Analytics\nBar + trend charts\nOccupancy history"]
        LOST["Khoya-Paya\nLost person search\nFace match from DB"]
    end

    CAMERAS --> AI --> PLAT --> UI
```

### Features — Real Deployment Grade

| Feature | What it does |
|---|---|
| **YOLOv8n Detection** | 5 FPS person detection on CPU; 6MB model auto-downloads; 8–15 persons/frame typical |
| **ByteTrack Tracking** | Persistent ID assignment across frames; trajectory history per person |
| **Optical Flow** | OpenCV Farneback every 5th frame; dominant direction vector shown as arrow overlay |
| **Behavioral AI** | Loitering (25 frames < 8px movement), running (velocity > 12px/frame), children (bbox height < 22% frame) |
| **Zone Density** | `count / area_m²` → SAFE / CAUTION / HIGH / CRITICAL risk with per-zone thresholds |
| **GIS Heat Map** | Leaflet.js map with person-position heat layer; updates every 2s |
| **Zone Polygons** | Lat/lon zone boundaries with color fill reflecting live density risk |
| **ICCC Dashboard** | 5-tab operator interface with live SSE push — no page refresh needed |
| **LLM Operator** | Claude API streaming chat: "Which zones are critical and what should I do?" → structured SOP response |
| **IP Camera** | Enter `192.168.x.x:8080` → auto-probes 4 common MJPEG endpoints |
| **RTSP / Video** | Any `rtsp://` URL or `.mp4` / `.avi` file (loops); Windows paths supported |
| **Overlay Toggles** | Per-slot toggle bar: Boxes · Track ID · Suspicious · Children · Flow Arrow · Count |
| **Audio Alarm** | Web Audio API synthetic alarm on CRITICAL density; 30s cooldown; on/off toggle |
| **Alert Dedup** | 60s cooldown per (zone, type) pair; alert log newest-first with severity badges |
| **Analytics Charts** | Chart.js bar charts (zone occupancy) and line trend charts (5-min history) |
| **Khoya-Paya** | Upload a photo → DeepFace cosine search across **two indexes**: (1) historical OSINT face vectors and (2) CIC crowd captures — returns name/track-ID, last-seen zone, camera slot, and timestamp |
| **CIC Face Indexing** | Every tracked person's face crop is extracted every 60 s, upscaled to ≥ 160 px, and embedded via Facenet512 → stored in `cic_face_captures` SQLite table for Khoya-Paya retrieval |

### Kumbh Mela Reference Zones

The default `zones.json` models Prayagraj's Kumbh Mela grounds:

| Zone | Camera Slot | Area | Capacity | CRITICAL at |
|---|---|---|---|---|
| Sangam Ghat | 0 | 8,000 m² | 50,000 | 48 persons/frame |
| Pontoon Bridge | 1 | 2,000 m² | 8,000 | 12 persons/frame |
| Sector 4 Entry Plaza | 2 | 12,000 m² | 80,000 | 72 persons/frame |
| Approach Road | 3 | 5,000 m² | 30,000 | 30 persons/frame |

### Starting the CIC

```bash
python app.py          # starts at localhost:5000
```

1. Click **[⚡ CIC]** in the top bar
2. Click a camera tile → enter a source:
   - `0` → webcam
   - `data\crowd.mp4` → video file in the project `data\` folder
   - `192.168.1.100:8080` → IP camera (auto-probes endpoints)
   - `rtsp://user:pass@192.168.1.100/stream` → RTSP
3. Frames appear immediately; YOLO detection kicks in after ~15–20s (model loads once and caches)
4. Switch to **Zone Map** tab → click **Heat Map ON** to see live density overlay
5. Switch to **Alerts & SOPs** tab → type a question in the chat to consult the AI operator

### LLM Operator Interface

The **Alerts & SOPs** tab has a live Claude-powered ICCC operator assistant:

```
You: Which zones are near critical and what actions should I take?

AI: ## Current Crowd Status

CAUTION: Sangam Ghat — 9 persons, 0.0011 p/m² (density rising)
SAFE: Pontoon Bridge — 0 persons

Recommended Actions:
• Deploy 2 stewards to Sangam Ghat to manage crowd flow
• Monitor Pontoon Bridge entry rate — current trajectory suggests
  CAUTION threshold in ~8 minutes at current rate
• Prepare SOP-2 (crowd diversion) for Sangam Ghat if density exceeds 2.0 p/m²
```

Put your `ANTHROPIC_API_KEY` in `.env` to enable this feature.

---

## Face OSINT Pipeline

```mermaid
flowchart TD
    INPUT(["`**Input**
    Camera · Photo · WiFi Cam`"])

    EMB["`**Layer 1 — Embedding**
    DeepFace · Facenet512
    512-D L2-normalised vector`"]

    DB[("`**SQLite WAL**
    Face vectors
    Search history`")]

    SCRAPE["`**Layer 2 — Parallel Scraping**
    7 scrapers · ThreadPoolExecutor`"]

    S1["`Reverse Face Search
    Yandex · SerpApi Lens · Google CSE`"]
    S2["`Search Engines
    Google CSE · DDG · Bing`"]
    S3["`Academic
    Scholar · OpenAlex · ORCID`"]
    S4["`Developer
    GitHub · GitLab`"]
    S5["`Social
    Reddit · PRAW`"]
    S6["`Passive OSINT
    Wayback · GDELT · crt.sh`"]
    S7["`Username
    125 direct + Sherlock 300+`"]

    MATCH["`**Layer 3 — Face Matching**
    Download · embed · cosine compare`"]

    SCORE["`**Layer 4 — Scoring**
    70% face · 10% name · 18% meta`"]

    RESOLVE["`**Layer 5 — Entity Resolution**
    Union-Find identity graph`"]

    OUT["`**Output**
    info.txt · JSON · Photos · SQLite`"]

    INPUT --> EMB
    EMB <--> DB
    EMB --> SCRAPE
    SCRAPE --> S1 & S2 & S3 & S4 & S5 & S6 & S7
    S1 & S2 & S3 & S4 & S5 & S6 & S7 --> MATCH
    MATCH --> SCORE
    SCORE --> RESOLVE
    RESOLVE --> OUT
```

---

## Quick Start

```bash
# Python 3.10 or 3.11 required (TensorFlow 2.16 breaks on 3.12)
pip install -r requirements.txt

cp .env.example .env    # all keys optional — works without any

python diagnose.py      # health check: packages, keys, network
python app.py           # opens http://localhost:5000 automatically
```

**First run downloads:**
- Facenet512 model weights (~600 MB) into `data/models/` — for Face OSINT
- YOLOv8n weights (~6 MB) into ultralytics cache — for CIC detection

Both download once and are cached indefinitely.

---

## Input Modes

| Mode | Where | How |
|---|---|---|
| **Browser webcam** | Face OSINT | Camera tab → Allow → Capture |
| **File upload** | Face OSINT | Drag-and-drop any JPEG/PNG/WebP |
| **WiFi/IP camera** | Face OSINT + CIC | Enter `IP:port` — auto-probes MJPEG endpoints |
| **Video file** | CIC only | Enter full path or `data\filename.mp4` |
| **RTSP stream** | CIC only | Enter `rtsp://...` URL |
| **Name hints** | Face OSINT | `John Smith \| New York @ Acme Corp` |

---

## API Keys

All optional. The system runs on free sources with no configuration.

| Key | Free tier | Unlocks |
|---|---|---|
| `ANTHROPIC_API_KEY` | Pay-as-you-go | CIC LLM operator interface (Claude) |
| `GOOGLE_CSE_KEY` + `GOOGLE_CSE_ID` | 100 req/day | LinkedIn + social search |
| `SERPAPI_KEY` | 100 req/month | Google Lens + Yandex face search |
| `IMGBB_API_KEY` | Free | Image hosting for SerpApi upload |
| `GITHUB_TOKEN` | Free | 5,000 req/hr (vs 60/hr unauthenticated) |
| `GITLAB_TOKEN` | Free | GitLab user search |
| `BING_API_KEY` | 1,000 req/month | Bing web + visual search |
| `BRAVE_API_KEY` | 2,000 req/month | Brave Search results |
| `HUNTER_API_KEY` | 25 req/month | Email intelligence |
| `REDDIT_CLIENT_ID` + `SECRET` | Free | Authenticated Reddit API |

```bash
python diagnose.py    # shows which keys are active and tests connectivity
```

---

## Output Structure

```
data/output/
└── John_Doe_20260619_1430_a1b2c3d4/
    ├── captured_photo.jpg          original input frame
    ├── face_crop.jpg               160×160 aligned face crop (Facenet512 input)
    ├── info.txt                    human-readable plaintext report
    ├── matches_summary.json        full structured data — all scores + metadata
    └── scraped_photos/
        ├── github_johndoe_a1b2.jpg
        ├── reverse_face_yandex_c3d4.jpg
        └── ...
```

CIC annotated frames can be saved via the `/crowd/api/frame/<slot>` endpoint (returns base64 JPEG).

---

## Tech Stack

| Layer | Library | Role |
|---|---|---|
| Face embedding | DeepFace + Facenet512 | 512-D L2-normalised vectors |
| Person detection | YOLOv8n (ultralytics) | 5 FPS CPU inference, 6MB model |
| Person tracking | ByteTrack (built-in ultralytics) | Persistent IDs, trajectory history |
| Optical flow | OpenCV Farneback | Crowd direction + speed |
| GIS map | Leaflet.js + Leaflet.heat | Zone polygons + person heat layer |
| LLM operator | Anthropic Claude API | Streaming SSE chat with zone context |
| Live charts | Chart.js | Occupancy bars + 5-min trend lines |
| Web UI | Flask + Server-Sent Events | Zero-install, live push |
| Camera | OpenCV (`cv2`) | Webcam + MJPEG + RTSP + video files |
| Persistence | SQLite WAL mode | Thread-safe, single-file, no config |
| HTML parsing | BeautifulSoup + lxml | Scraping search results |
| Name matching | rapidfuzz | Fuzzy string entity resolution |
| Username OSINT | Sherlock + direct checks | 300+ platforms |

---

## Verdict System (Face OSINT)

```mermaid
graph LR
    A[Combined Score] --> B{Threshold}
    B -->|≥ 0.85| C["CONFIRMED (high)\nFace verified · consistent metadata"]
    B -->|≥ 0.70| D["CONFIRMED (low)\nFace verified · limited metadata"]
    B -->|≥ 0.55| E["POSSIBLE\nNear-match face OR strong metadata"]
    B -->|< 0.55| F["UNLIKELY\nName match only · no photo evidence"]

    style C fill:#14532d,stroke:#4ade80,color:#f1f5f9
    style D fill:#1a3a1a,stroke:#86efac,color:#f1f5f9
    style E fill:#431407,stroke:#fb923c,color:#f1f5f9
    style F fill:#3b0a0a,stroke:#f87171,color:#f1f5f9
```

Score weights: `0.70 × face_similarity + 0.10 × name_match + 0.08 × social_signals + 0.05 × photo_count + 0.07 × source_diversity`

---

## Risk Levels (CIC)

| Level | Density | Border | Action |
|---|---|---|---|
| **SAFE** | < 1.5 p/m² | Green | Normal monitoring |
| **CAUTION** | 1.5–3.0 p/m² | Amber | Alert generated; deploy stewards |
| **HIGH RISK** | 3.0–6.0 p/m² | Orange | Activate crowd diversion |
| **CRITICAL** | > 6.0 p/m² | Red | SOP-3 crush prevention; audio alarm |

*Thresholds are per-zone configurable in `crowd/zones.json`.*

---

## Requirements

- Python **3.10** or **3.11** — TensorFlow 2.16 does not support 3.12
- ~610 MB disk (Facenet512 600MB + YOLOv8n 6MB — both downloaded once)
- Webcam, IP camera, video file, or RTSP stream for CIC

```bash
pip install -r requirements.txt
```

Optional extras:

```bash
pip install sherlock-project    # username scraper (300+ platforms)
pip install socid-extractor     # social ID extraction
pip install praw                # authenticated Reddit API
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| **CIC shows video but no bounding boxes** | YOLO loads in ~15–20s on first use per session; boxes appear automatically |
| **CIC total count shows but no video tile** | Open CIC overlay while slot is active — it auto-syncs server state |
| **LLM chat returns error** | Check `ANTHROPIC_API_KEY` is set in `.env` |
| **IP camera not connecting** | Ensure IP and PC are on the same subnet; try `http://IP:port/video` in browser first |
| **"No face detected" (Face OSINT)** | Face must fill ≥ 20% of frame · avoid backlighting · move closer |
| **Khoya-Paya finds 0 faces in CIC index** | Faces are captured every 60 s per tracked person; run a slot for at least 60–90 s and check the DB: `SELECT COUNT(*) FROM cic_face_captures` |
| **TensorFlow import errors** | Use Python 3.10 or 3.11 — TF 2.16 does not support 3.12; also install `tf-keras` |
| **LinkedIn always 0 results** | Set `GOOGLE_CSE_KEY` + `GOOGLE_CSE_ID` in `.env` |

```bash
python diagnose.py    # end-to-end health check with targeted fix instructions
```

---

## Data & Privacy

Everything stays on your machine — no telemetry, no cloud sync.

| Path | Contents |
|---|---|
| `data/face_osint.db` | SQLite WAL — tables: `searches`, `matches`, `face_vectors` (512-D BLOB), `cic_face_captures` |
| `data/output/` | Per-search folders: captured photo, face crop, info.txt, matches_summary.json, scraped photos |
| `data/models/` | DeepFace Facenet512 weights (~600 MB, downloaded once) |
| `crowd/zones.json` | CIC zone definitions — lat/lon polygons, area_m², density thresholds (editable) |
| `logs/` | Rotating logs, 10 MB × 5 files |

**`cic_face_captures`** stores each crowd-camera face embedding (track ID, slot, zone, timestamp, 512-D vector). Queried by Khoya-Paya lost-person search. Grows as cameras run; safe to truncate between sessions.

To wipe all search data: delete `data/face_osint.db` and `data/output/`.

---

## Responsible Use

Built for **legitimate OSINT research** — verifying identities, locating missing persons, investigating fraud, academic research, penetration testing with explicit authorization, and event safety management demonstration.

Do not use to stalk, harass, or build unauthorized profiles of private individuals.  
CIC camera analysis must only be performed on footage you have the legal right to process.  
Comply with applicable laws in your jurisdiction. Respect platform terms of service.
