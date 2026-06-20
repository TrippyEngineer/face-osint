<div align="center">

# Face OSINT + Crowd Intelligence Center

**Capture a face. Identify an individual. Command a crowd.**

Two systems in one: a deep OSINT identity pipeline powered by face recognition,
and a real-time AI crowd management platform modeled on India's ICCC deployments at Kumbh Mela scale.

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-SSE%20Live%20UI-000000?style=flat-square&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![YOLOv8](https://img.shields.io/badge/YOLOv8n-Person%20Detection-00B4D8?style=flat-square)](https://docs.ultralytics.com)
[![DeepFace](https://img.shields.io/badge/DeepFace-Facenet512-FF6B6B?style=flat-square)](https://github.com/serengil/deepface)
[![Playwright](https://img.shields.io/badge/Playwright-Browser%20Scraping-2EAD33?style=flat-square&logo=playwright&logoColor=white)](https://playwright.dev)
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

## Full System Architecture

```mermaid
flowchart TD
    subgraph INPUT["Input Layer"]
        WC["Browser Webcam"]
        FU["File Upload\nJPEG · PNG · WebP"]
        IP["WiFi / IP Camera\nMJPEG · RTSP"]
        VF["Video File\n.mp4 · .avi"]
    end

    subgraph FLASK["Flask Web Server — app.py (SSE)"]
        direction LR
        API["REST API\n/search\n/results\n/cancel"]
        SSE_SRV["SSE Stream\nper-session queue\nUUID isolation"]
        CIC_API["CIC API\n/crowd/api/*\nKhoya-Paya"]
    end

    subgraph FACE["Face OSINT Pipeline"]
        direction TB
        EMB["embedding.py\nFacenet512 · 512-D L2"]
        DB_VEC[("SQLite WAL\nface_vectors\nsearches · matches")]
        SCRAPE["Scraper Dispatcher\nThreadPoolExecutor · 7 scrapers"]
        AGG["Aggregator\nface_matcher → scorer → resolver"]
        OUT["folder_writer\ninfo.txt · JSON · photos"]
    end

    subgraph SCRAPERS["Scraper Layer — all parallel"]
        direction TB
        RF["reverse_face.py\n7 engines"]
        SE["search_engines.py\nGoogle CSE · DDG · Bing"]
        AC["academic.py\nScholar · OpenAlex · ORCID"]
        PL["platforms.py\nGitHub · GitLab · Reddit"]
        PA["passive.py\nWayback · GDELT · crt.sh"]
        UN["username.py\n4 layers · 300+ sites"]
    end

    subgraph RF_ENGINES["reverse_face.py — 7 engines in parallel"]
        direction LR
        E1["SerpApi\nGoogle Lens"]
        E2["SerpApi\nYandex"]
        E3["Yandex\nDirect CBIR"]
        E4["Google CSE\nImage + Social"]
        E5["PimEyes\npaid stub"]
        E6["🎭 Playwright\nGoogle Lens\nfree · no key"]
        E7["🎭 Playwright\nBing Visual\nfree · no key"]
    end

    subgraph UN_LAYERS["username.py — 4 layers"]
        direction LR
        L1["Layer 1\nSherlock\n300+ platforms"]
        L2["Layer 2\nsocid_extractor\nhidden IDs"]
        L3["Layer 3\n25 direct HEAD\nno keys"]
        L4["🎭 Layer 4\nPlaywright verify\nJS-gated profiles"]
    end

    subgraph CIC["Crowd Intelligence Center"]
        direction TB
        YOLO["YOLOv8n\nPerson detection\n5 FPS · CPU"]
        TRACK["ByteTrack\nPersistent IDs"]
        FLOW["Farneback\nOptical flow"]
        BEH["Behavioral AI\nLoitering · Running\nChildren · Suspicious"]
        RISK["Zone Density\np/m² → Risk level"]
        LLM["Claude API\nLLM Operator\nStreaming chat"]
        GIS["Leaflet.js\nGIS Heat Map"]
        KP["Khoya-Paya\nLost person search\nFace cosine match"]
    end

    INPUT --> FLASK
    FLASK --> FACE
    FACE --> SCRAPE
    SCRAPE --> SCRAPERS
    RF --> RF_ENGINES
    UN --> UN_LAYERS
    SCRAPERS --> AGG
    AGG --> OUT
    AGG --> DB_VEC
    EMB --> DB_VEC
    FLASK -.->|SSE live progress| INPUT
    FLASK --> CIC
```

---

## Face OSINT — Request Sequence Flow

```mermaid
sequenceDiagram
    actor U as User
    participant B as Browser
    participant F as Flask (app.py)
    participant E as embedding.py
    participant DB as SQLite WAL
    participant RF as reverse_face.py
    participant UN as username.py
    participant SC as other scrapers ×5
    participant AG as Aggregator
    participant OT as Output

    U->>B: Capture / upload face photo
    B->>F: POST /search {image_b64, name, location}
    F->>F: Dedup check (MD5 → _active_searches)
    F->>E: extract_embedding(face_crop)
    E-->>F: 512-D Facenet512 vector
    F->>DB: INSERT face_vectors, searches → sid
    DB-->>F: sid (UUID)
    F-->>B: {sid} — SSE stream opens

    Note over F,OT: All 7 scrapers run in parallel (ThreadPoolExecutor max_workers=4)

    par Scraper 1 — reverse_face (90s budget)
        F->>RF: scrape(context)
        Note over RF: Engine 1: SerpApi Google Lens (paid key)<br/>Engine 2: SerpApi Yandex (paid key)<br/>Engine 3: Yandex Direct CBIR (free)<br/>Engine 4: Google CSE (free tier)<br/>Engine 5: PimEyes (paid stub)<br/>Engine 6: 🎭 Playwright Google Lens (free)<br/>Engine 7: 🎭 Playwright Bing Visual (free)
        RF-->>F: {matches[], names_found[]}
        F-->>B: SSE: reverse_face complete
    and Scraper 2 — username (50s budget)
        F->>UN: scrape(context)
        Note over UN: Layer 1: Sherlock 300+ platforms<br/>Layer 2: socid_extractor hidden IDs<br/>Layer 3: 25 direct HEAD checks<br/>Layer 4: 🎭 Playwright JS-profile verify
        UN-->>F: {matches[], variants[]}
        F-->>B: SSE: username complete
    and Scrapers 3–7 (25–35s each)
        F->>SC: search_engines / academic / platforms / passive
        SC-->>F: {matches[]}
        F-->>B: SSE: each scraper complete
    end

    F->>AG: face_matcher.verify_all(candidates)
    Note over AG: Download candidate photos<br/>DeepFace cosine compare vs query<br/>≥ 0.50 similarity → face_verified=True

    AG->>AG: scorer.score(verified_matches)
    Note over AG: 0.70×face + 0.10×name<br/>+ 0.08×social + 0.05×photo<br/>+ 0.07×source_count

    AG->>AG: resolver.resolve(scored_matches)
    Note over AG: Union-Find identity graph<br/>Edge weights by signal type<br/>Clusters → best match wins

    AG->>DB: INSERT matches
    AG->>OT: folder_writer.write()
    Note over OT: captured_photo.jpg<br/>face_crop.jpg<br/>info.txt<br/>matches_summary.json<br/>scraped_photos/

    AG-->>F: Final results
    F-->>B: SSE: search complete
    B->>F: GET /results/{sid}
    F-->>U: Results page rendered
```

---

## Face OSINT — Scraper Architecture

```mermaid
flowchart TD
    IMG[/"Face Image (bytes)"/]

    subgraph RF["reverse_face.py — ENGINE POOL (max_workers=6)"]
        direction LR
        E1["Engine 1\nSerpApi Google Lens\n🔑 SERPAPI_KEY"]
        E2["Engine 2\nSerpApi Yandex\n🔑 SERPAPI_KEY"]
        E3["Engine 3\nYandex Direct CBIR\n✅ free · no key"]
        E4["Engine 4\nGoogle CSE\n🔑 GOOGLE_CSE_KEY"]
        E5["Engine 5\nPimEyes\n🔑 PIMEYES_API_KEY"]
        E6["Engine 6\n🎭 Playwright\nGoogle Lens\n✅ free · no key"]
        E7["Engine 7\n🎭 Playwright\nBing Visual\n✅ free · no key"]
    end

    subgraph UN["username.py — LAYER STACK"]
        direction TB
        L3["Layer 3 — Direct HEAD checks\n25 hand-picked platforms\n✅ free · instant"]
        L1["Layer 1 — Sherlock\n300+ platforms via subprocess\n✅ free · 15s cap"]
        L2["Layer 2 — socid_extractor\nhidden IDs · cross-platform links\n✅ free"]
        L4["Layer 4 — 🎭 Playwright verify\nInstagram · TikTok · YouTube\nPinterest · Spotify · Telegram · Twitch\ndrops false positives · extracts profile data"]
        L3 --> L1 --> L2 --> L4
    end

    DEDUP["Dedup + social-first sort"]
    VERIFY["Face Verification\nDeepFace cosine × candidates\nmax_workers=6"]
    ENRICH["Identity Enrichment\nog:title · GitHub API\nname extraction"]

    IMG --> RF
    E1 & E2 & E3 & E4 & E5 & E6 & E7 --> DEDUP
    IMG --> UN
    DEDUP --> VERIFY
    VERIFY --> ENRICH
    ENRICH --> OUT[/"Verified matches[]"/]
```

---

## Crowd Intelligence Center (CIC)

A laptop-scale depiction of an Integrated Command and Control Centre (ICCC) as deployed at India's Maha Kumbh Mela and similar mega-events. Real detection algorithms. Real risk scoring. Real LLM-powered operator interface.

### CIC Architecture

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
        FACE_IDX["Face Indexing\nper-track every 60s\nFacenet512 → cic_face_captures"]
        YOLO --> TRACK --> BEH
        FLOW --> BEH
        BEH --> RISK
        TRACK --> FACE_IDX
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

    subgraph KP_FLOW["Khoya-Paya Search Flow"]
        KP_IN["Upload photo"]
        KP_EMB["Facenet512 embed"]
        KP_DB1["Search OSINT\nface_vectors"]
        KP_DB2["Search CIC\ncic_face_captures"]
        KP_OUT["Results: name · track ID\nlast zone · timestamp"]
        KP_IN --> KP_EMB --> KP_DB1 & KP_DB2 --> KP_OUT
    end

    CAMERAS --> AI --> PLAT --> UI
    FACE_IDX --> KP_FLOW
```

### CIC Features — Real Deployment Grade

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

### Optional: Playwright Browser Scrapers

Adds two free engines (Google Lens + Bing Visual Search) and JS-profile verification — no API keys needed:

```bash
pip install playwright
playwright install chromium

# WSL2 only — install browser OS dependencies:
playwright install-deps chromium
```

Once installed, `ENGINE 6` (Playwright Google Lens) and `ENGINE 7` (Playwright Bing Visual) activate automatically alongside the existing scrapers. No configuration needed.

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

All optional. The system runs on free sources with zero configuration. Playwright engines unlock Google Lens and Bing Visual Search for free.

| Key | Free tier | Unlocks |
|---|---|---|
| *(none)* | — | Yandex Direct CBIR · 25 direct username checks · Playwright Google Lens · Playwright Bing Visual |
| `ANTHROPIC_API_KEY` | Pay-as-you-go | CIC LLM operator interface (Claude) |
| `GOOGLE_CSE_KEY` + `GOOGLE_CSE_ID` | 100 req/day | LinkedIn + social profile search |
| `SERPAPI_KEY` | 100 req/month | SerpApi Google Lens + SerpApi Yandex |
| `IMGBB_API_KEY` | Free | Image hosting for SerpApi upload |
| `GITHUB_TOKEN` | Free | 5,000 req/hr (vs 60/hr unauthenticated) |
| `GITLAB_TOKEN` | Free | GitLab user search |
| `BING_API_KEY` | 1,000 req/month | Bing web + text search (separate from Playwright visual) |
| `BRAVE_API_KEY` | 2,000 req/month | Brave Search results |
| `HUNTER_API_KEY` | 25 req/month | Email intelligence |
| `REDDIT_CLIENT_ID` + `SECRET` | Free | Authenticated Reddit API |
| `PIMEYES_API_KEY` | Paid | PimEyes face search |

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
        ├── reverse_face_playwright_lens_c3d4.jpg
        ├── reverse_face_yandex_e5f6.jpg
        └── ...
```

CIC annotated frames can be saved via the `/crowd/api/frame/<slot>` endpoint (returns base64 JPEG).

---

## Scoring & Verdicts

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

**Score formula:** `0.70 × face_similarity + 0.10 × name_match + 0.08 × social_signals + 0.05 × photo_count + 0.07 × source_diversity`

**Face cosine thresholds:** confirmed ≥ 0.68 · possible ≥ 0.50 · different < 0.35  
**Guardrails:** face-only match floors at 0.72 · name-only caps at 0.49 (cannot reach CONFIRMED without face evidence)

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

## Tech Stack

Every dependency and the exact job it does in this application, grouped by layer.

### Backend & web
| Technology | Role in this app |
|---|---|
| **Flask** | HTTP server and all routes (`/api/*` for Face OSINT, `/crowd/api/*` for CIC) |
| **Server-Sent Events (SSE)** | Live one-way push to the browser — scraper progress, the new preliminary-results event, and the CIC live status stream |
| **Flask-CORS** | Cross-origin headers so the single-page UI can call the API |
| **python-dotenv** | Loads `.env` (API keys, thresholds) into `config.py` at startup |
| **SQLite** (WAL mode) | Single-file store for searches, matches, and face vectors; WAL allows the scraper threads + request threads to read/write concurrently |
| **Filesystem** | One folder per search under `data/output/` holds the captured photo, face crop, `info.txt`, and `matches_summary.json` |

### Face recognition & computer vision
| Technology | Role in this app |
|---|---|
| **DeepFace** | High-level API that wraps detection + embedding; called in `embedding.py`. Pre-warmed in a background thread at startup so the first search skips the ~30-40s cold load |
| **TensorFlow + tf-keras** | Deep-learning runtime that executes the Facenet512 network |
| **OpenCV** (`cv2`) | Camera/video capture (webcam, MJPEG, RTSP, files), JPEG decode, face cropping, and Farneback optical flow for crowd direction/speed |
| **NumPy** | Embedding vectors, cosine similarity, frame buffers |
| **Pillow** | Decodes downloaded profile photos before re-embedding them for face matching |

### Crowd Intelligence Center (CIC)
| Technology | Role in this app |
|---|---|
| **Ultralytics YOLOv8n** | Per-frame person detection that drives the density/occupancy and stampede-risk logic |
| **ByteTrack** (bundled with ultralytics) | Persistent per-person track IDs and trajectory history |
| **Leaflet.js + Leaflet.heat** | Zone-map tab — zone polygons and the live person heat layer |
| **Chart.js** | Analytics tab — occupancy bars and density trend lines |

### OSINT scrapers
| Technology | Role in this app |
|---|---|
| **requests** | All plain HTTP scraping and API calls (GitHub, Reddit, academic, search engines, passive) |
| **BeautifulSoup4 + lxml** | Parse search-result and profile HTML |
| **Playwright** (Chromium) | Renders JS-gated pages — Google Lens + Bing Visual reverse-image engines, and username JS-profile verification |
| **Sherlock + socid_extractor** | Username OSINT across 300+ platforms + extraction of hidden cross-platform IDs |
| **rapidfuzz** | Fuzzy name matching during entity resolution (graceful pure-Python fallback if absent) |

### AI operator
| Technology | Role in this app |
|---|---|
| **Anthropic Claude API** | The CIC "AI Operator Assistant" (`crowd/llm_ops.py`) — answers operator questions over the live zone state and gives SOP guidance, streamed token-by-token over SSE |

---

## Models

Three models power the application. The two local ones download once and are cached under `data/models/`.

| Model | Used by | Role | Size | Runs where |
|---|---|---|---|---|
| **Facenet512** (via DeepFace) | Face OSINT + Khoya-Paya | Produces the 512-D L2-normalised face embedding that is the **primary identity signal** (70% of the match score). Compared by cosine similarity against scraped profile photos and the stored face DB | ~600 MB | Local, on TensorFlow (CPU or GPU); pre-warmed at startup |
| **OpenCV detector** (`opencv` backend) | DeepFace detection step | Locates and aligns the face in the frame before embedding (`DEEPFACE_DETECTOR="opencv"`) | bundled with OpenCV | Local, CPU |
| **YOLOv8n** (Ultralytics) | Crowd Intelligence Center | Detects people per frame for crowd counting, density (persons/m²), and stampede-risk thresholds; tracked across frames by ByteTrack | ~6 MB | Local, CPU (~5 FPS) |
| **Claude** (Anthropic, `claude-*` via API) | CIC AI Operator Assistant | Natural-language Q&A over the live crowd state and standard-operating-procedure guidance | Cloud API | Anthropic API (needs `ANTHROPIC_API_KEY`) |

Score thresholds for the face model (Facenet512, cosine similarity) live in `config.py`: `FACE_CONFIRMED = 0.68`, `FACE_POSSIBLE = 0.50`, `FACE_REJECTED = 0.35`.

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
pip install socid-extractor     # social ID extraction from profile pages
pip install praw                # authenticated Reddit API

# Browser-based scraping (Google Lens + Bing Visual, free):
pip install playwright
playwright install chromium
playwright install-deps chromium   # WSL2 / Linux only
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
| **Playwright: "Event loop already running"** | Ensure `playwright install chromium` was run after `pip install playwright` |
| **Playwright: no results from Google Lens** | Google may have changed the UI; check logs for `playwright_google_lens` warnings |

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
