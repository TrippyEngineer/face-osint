# CIC persons/m² Calibration

Density = persons ÷ **camera-visible area (m²)**, NOT the whole zone area — otherwise
risk never escalates. Set `fov_area_m2` per zone in `crowd/zones.json`.

## Method A — quick (rectangular ground patch)
1. Identify the ground area the camera actually sees (the visible footprint).
2. Measure its real width and depth in metres (site plan, pacing, or known landmarks
   like a 1.8 m doorway).
3. `fov_area_m2 = width_m × depth_m`. Enter it on the zone in `zones.json`.

## Method B — homography (accurate, perspective-correct)
1. Pick 4 ground points visible in frame with known real-world metres (corners of a
   marked rectangle, paving grid, court lines).
2. Compute the image→ground homography (`cv2.findHomography`).
3. Project the frame's ground-visible polygon to metres; its area is `fov_area_m2`.
   For finer density, project each person's foot point and bin into m² cells.

## Thresholds
Per-zone `thresholds` (`caution`/`high`/`critical`, persons/m²) override
`config.CIC_DENSITY_*`. Crowd-safety references: ~4 p/m² comfortable upper bound,
≥5–6 p/m² crush risk. Tune per venue.

## With tiling
`CIC_TILING=1` raises recall (more true heads) so counts rise; re-check thresholds
against the calibrated area after enabling it.
