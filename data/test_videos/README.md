# CIC test videos (real humans) — pressure.py end-to-end demo

Three real-human clips for testing the crowd-pressure early-warning
(`crowd/pressure.py`) end to end on the dashboard. **The video files themselves are
git-ignored** (binaries) — re-download with the commands at the bottom if missing.

| file | scene | res · fps | best for |
|------|-------|-----------|----------|
| `pedestrians_vtest.avi` | plaza, people crossing many directions | 768×576 · 10fps | **best** — directional spread → velocity variance → turbulence |
| `people-detection.mp4` | pedestrians on a path | 768×432 · 12fps | steady flow |
| `store-aisle-detection.mp4` | shoppers in an aisle | 720×404 · 30fps | sparse / slow |

## How to run it
1. Start the app on Windows python: `python app.py` (open http://localhost:5000).
2. CIC dashboard → **Cameras** tab → pick a slot → **Upload** → choose
   `data/test_videos/pedestrians_vtest.avi`. The slot starts and loops the file.
3. Watch the slot's risk pill + the footer line. With pressure on you'll see e.g.
   **`⚠ TURBULENT | RISKY · LoS E`**, and alerts fire in the Alerts tab.

## Calibration — IMPORTANT (else the alarm won't fire)
These clips are *sparse* (~5–15 people), so at the default `CIC_FOV_AREA_M2=100`
density is ~0.1 ppl/m² = "normal" and nothing escalates. The pressure thresholds
and FOV area are now `.env`-overridable. For a visible demo, add to `.env`:

```env
# Pretend each camera sees a small ~3 m² patch so ~12 people ≈ 4 ppl/m².
CIC_FOV_AREA_M2=3
# (optional) make turbulence trip a touch easier on low-fps test clips
CIC_TURBULENCE_CV=0.6
```

With `CIC_FOV_AREA_M2=3`: ~9 people → ~3 ppl/m² (**dense**), ~15 → ~5 (**risky**),
and directional/stop-go motion → **turbulence** escalates the band toward
**critical**. Tune `CIC_FOV_AREA_M2` down for a stronger alarm, up for calmer.
Restart `app.py` after editing `.env` (no auto-reload).

> Real deployments calibrate `CIC_FOV_AREA_M2` to each camera's actual ground
> footprint and keep the default density bands (compression 5 / critical 8 ppl/m²).

## Re-download (files are git-ignored)
```bash
mkdir -p data/test_videos && cd data/test_videos
curl -fsSL -o pedestrians_vtest.avi      https://github.com/opencv/opencv/raw/4.x/samples/data/vtest.avi
curl -fsSL -o people-detection.mp4       https://github.com/intel-iot-devkit/sample-videos/raw/master/people-detection.mp4
curl -fsSL -o store-aisle-detection.mp4  https://github.com/intel-iot-devkit/sample-videos/raw/master/store-aisle-detection.mp4
```
