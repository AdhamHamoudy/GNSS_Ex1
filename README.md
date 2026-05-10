# Drone Visual Navigation — GNSS-Denied Localization

> **Ex1 — Visual Navigation for Drones**
> Given a drone flight video and SRT telemetry **with** GNSS, build a visual reference database.
> Then, given a new flight video and telemetry **without** GNSS, compute the GPS coordinate
> of the ground point at the centre of the camera in real-time.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
  - [Step 1 — Preprocessing](#step-1--preprocessing-with-gnss)
  - [Step 2 — Localization](#step-2--gnss-denied-localization)
  - [Combining Multiple Flights](#combining-multiple-flights)
  - [Night Video](#night-video)
  - [Videos Without SRT](#videos-without-srt)
- [Sample Data](#sample-data)
- [Results](#results)
- [Algorithm](#algorithm)
- [Limitations](#limitations)

---

## Problem Statement

Standard drone navigation relies on GNSS (GPS) for positioning. When GNSS is unavailable
(jamming, indoor flight, denied airspace), the drone must navigate using only its onboard
sensors and camera feed — a problem called **GNSS-denied visual navigation**.

This project solves the following specific problem:

```
PREPROCESSING (offline, with GNSS):
  Video + SRT telemetry (lat, lon, altitude, camera angle)
       ↓
  Reference database of geo-tagged frames

NAVIGATION (real-time, without GNSS):
  New video stream + SRT telemetry (altitude, camera angle — NO lat/lon)
       ↓
  GPS coordinate of the ground point at the image centre, per frame
```

---

## How It Works

### Step 1 — Preprocessing (`src/prepare_dataset.py`)

1. Parses the DJI SRT telemetry file to extract per-frame GNSS position,
   barometric altitude, camera angle, and heading.
2. Samples video frames at a configurable interval (default: every 1 second).
3. For each frame, computes the GPS coordinate of the ground point visible
   at the image centre using camera geometry:
   ```
   ground_distance = altitude / tan(camera_pitch)
   center_point    = move(drone_position, bearing, ground_distance)
   ```
4. Saves a reference CSV with all metadata, extracted JPEG frames, and a
   KML path file viewable in Google Earth.

### Step 2 — GNSS-Denied Localization (`src/visual_localization.py`)

For each query frame from the new flight (no GNSS used):

1. **Feature extraction** — ORB keypoints and descriptors are computed.
2. **Database matching** — the query frame is matched against every reference
   frame using BFMatcher with Lowe's ratio test (threshold 0.75).
3. **Geometric verification** — RANSAC homography is estimated; frames with
   fewer than `--min-inliers` inliers are rejected.
4. **Centre-point mapping** — the query image centre is projected through the
   homography into the reference image's pixel space.
5. **GPS estimation** — the projected pixel offset from the reference centre
   is converted to a real-world (north, east) displacement using:
   - **Real-time barometric altitude** from the query SRT telemetry
   - **Real-time camera pitch** from the query SRT telemetry (if available)
   - Reference frame bearing and stored centre GPS coordinate
6. The result is a `(lat, lon)` coordinate output per frame — no GNSS used.

---

## Project Structure

```
GNSS_EX1/
├── src/
│   ├── prepare_dataset.py       # Step 1: offline preprocessing
│   ├── visual_localization.py   # Step 2: real-time GNSS-denied localization
│   └── utils.py                 # Shared: SRT parsing, geodesy, geometry
├── data/
│   ├── DJI_20260427152735_0019_D.SRT   # Flight 0019 telemetry (118 s)
│   └── DJI_20260427152226_0017_D.SRT   # Flight 0017 telemetry (252 s)
├── requirements.txt
├── .gitignore
└── README.md
```

### Key outputs (generated, not committed)

```
outputs_step1/
├── reference_dataset.csv      # geo-tagged frame database
├── frames/                    # extracted JPEG frames
└── estimated_center_path.kml  # viewable in Google Earth

outputs_step2/
├── visual_localization_results.csv   # predicted coords + error per frame
└── visual_localization_paths.kml     # predicted path (red) vs truth (green)
```

---

## Setup

**Requirements:** Python 3.10+, pip

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/GNSS_EX1.git
cd GNSS_EX1

# Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

`requirements.txt`:
```
opencv-python
numpy
yt-dlp
```

---

## Usage

### Step 1 — Preprocessing (with GNSS)

```bash
python src/prepare_dataset.py \
    --video        data/DJI_20260427152735_0019_D.MP4 \
    --srt          data/DJI_20260427152735_0019_D.SRT \
    --out          outputs_step1 \
    --sample-every 1.0 \
    --camera-pitch 45.0
```

| Argument | Default | Description |
|---|---|---|
| `--video` | required | Path to the drone MP4 recording |
| `--srt` | required | Path to the matching DJI SRT telemetry file |
| `--out` | `outputs_step1` | Output folder |
| `--sample-every` | `1.0` | Extract one frame every N seconds |
| `--camera-pitch` | `45.0` | Fallback camera tilt in degrees — used only when `drone_pitch` is absent from the SRT |

**Example output:**
```
Video      : DJI_20260427152735_0019_D.MP4
Resolution : 1920×1080
FPS        : 29.970
Duration   : 117.95 s  (video) / 117.92 s  (SRT)
SRT rows   : 3535
Bearing src: gps_difference

Done.
  Samples extracted : 118  (this run)
  Coverage diagonal : 459.3 m
```

---

### Step 2 — GNSS-Denied Localization

```bash
python src/visual_localization.py \
    --video         data/DJI_20260427152735_0019_D.MP4 \
    --srt           data/DJI_20260427152735_0019_D.SRT \
    --reference-csv outputs_step1/reference_dataset.csv \
    --out           outputs_step2 \
    --sample-every  2.0 \
    --camera-pitch  45.0 \
    --hfov          82.0 \
    --min-inliers   15
```

| Argument | Default | Description |
|---|---|---|
| `--video` | required | Query MP4 — GNSS lat/lon is never read during prediction |
| `--srt` | optional | SRT file — read during prediction for non-GNSS telemetry (altitude, camera pitch) only. Lat/lon are never used during prediction; they are read only after prediction for error evaluation. Omit with `--no-eval` when no SRT is available. |
| `--reference-csv` | required | CSV produced by `prepare_dataset.py` |
| `--out` | `outputs_step2` | Output folder |
| `--sample-every` | `2.0` | Process one frame every N seconds |
| `--camera-pitch` | `45.0` | Fallback camera tilt in degrees — used only when `drone_pitch` is absent from the SRT |
| `--hfov` | `82.0` | Horizontal field of view in degrees |
| `--min-inliers` | `15` | Minimum RANSAC inliers to accept a prediction |
| `--resize-width` | `960` | Resize frames before matching (0 = full resolution) |
| `--exclude-near-time` | `0.5` | Skip reference frames within N seconds of query time |
| `--no-eval` | off | Skip ground-truth evaluation when no SRT is available |
| `--night-mode` | off | Enable CLAHE contrast enhancement for low-light footage |

**Example output:**
```
Real-time telemetry fields : rel_alt

Processing query video: DJI_20260427152735_0019_D.MP4
  Duration : 117.95 s  (video) / 117.92 s (SRT)
  Min inliers threshold : 15

  t=  0.00s | ref=  1.00s | inliers=1423 | pred=(32.10293118, 35.20968559) | error=0.198 m
  t=  2.00s | ref=  3.00s | inliers= 371 | pred=(32.10243171, 35.20893504) | error=89.986 m
  ...

============================================================
Summary
============================================================
  Queries processed    : 59
  Accepted predictions : 59
  Coverage             : 100.0 %
  Mean error           : 8.564 m
  Median error         : 3.304 m
  RMSE                 : 19.115 m
  Max error            : 89.986 m
============================================================
```

---

### Combining Multiple Flights

Run `prepare_dataset.py` twice with the same `--out` folder.
The second run automatically appends to the existing CSV without overwriting:

```bash
python src/prepare_dataset.py \
    --video data/DJI_20260427152735_0019_D.MP4 \
    --srt   data/DJI_20260427152735_0019_D.SRT \
    --out   outputs_combined --camera-pitch 45.0

python src/prepare_dataset.py \
    --video data/DJI_20260427152226_0017_D.MP4 \
    --srt   data/DJI_20260427152226_0017_D.SRT \
    --out   outputs_combined --camera-pitch 45.0

# → 370 reference frames covering 1044 m diagonally
```

---

### Night Video

Use `--night-mode` to apply CLAHE contrast enhancement before feature extraction.
This is required for low-light footage where ORB cannot find features in dark frames:

```bash
python src/visual_localization.py \
    --video         night_flight.mp4 \
    --reference-csv outputs_combined/reference_dataset.csv \
    --out           outputs_night \
    --night-mode \
    --min-inliers   10 \
    --no-eval
```

---

### Videos Without SRT (YouTube downloads)

Download videos using yt-dlp:

```bash
yt-dlp -f "bestvideo[height<=1080][ext=mp4]+bestaudio/best" \
    --merge-output-format mp4 -o "video.mp4" "YOUTUBE_URL"
```

Then run localization without evaluation:

```bash
python src/visual_localization.py \
    --video         video.f137.mp4 \
    --reference-csv outputs_combined/reference_dataset.csv \
    --out           outputs_youtube \
    --no-eval \
    --exclude-near-time 0
```

> **Note:** When no SRT is provided, the system runs in prediction-only mode and uses
> fallback altitude and pitch assumptions from `--camera-pitch` and the matched reference
> frame. Real-time telemetry is unavailable, so no error statistics can be computed.

---

## Sample Data

Two DJI Air 3 SRT telemetry files are included under `data/` for testing:

| File | Flight | Duration | Area Covered |
|---|---|---|---|
| `DJI_20260427152735_0019_D.SRT` | Flight 0019 | 118 s | 459 m diagonal |
| `DJI_20260427152226_0017_D.SRT` | Flight 0017 | 252 s | 1044 m diagonal |

> The corresponding MP4 videos are not included due to file size limits.
> Download them from the assignment's [Google Drive folder](https://drive.google.com/drive/folders/1UnJIRpjtdLXDm6mmHdGWE9yallkxSAam).

---

## Results

Tested on DJI Air 3 footage at 45° camera pitch, 120 m altitude:

| Test | Reference DB | Query | Median Error | Coverage |
|---|---|---|---|---|
| Self-test 0019 | 0019 SRT — 118 frames | 0019 video | **3.3 m** | 100% |
| Self-test 0017 | 0017 SRT — 252 frames | 0017 video | **2.7 m** | 100% |
| Cross-test Air 3 v1 | Combined — 370 frames | YouTube video | no ground truth (no SRT) | 100% accepted visual matches |
| Cross-test Air 3 v2 | Combined — 370 frames | YouTube video | no ground truth (no SRT) | 100% accepted visual matches |
| Night video NV1 | Combined — 370 frames | Night video | no ground truth (no SRT) | 49.8% accepted visual matches |

The combined reference database (0017 + 0019) covering **1044 m diagonally**
achieved consistently high inlier counts (1600–1900) on daytime query videos,
demonstrating that a larger reference area significantly improves match confidence.

---

## Algorithm

```
PREPROCESSING
─────────────
SRT file ──► parse_srt()
               │
               ├─ latitude, longitude  (GNSS — preprocessing only)
               ├─ rel_alt              (barometric altitude)
               ├─ drone_yaw / speed vector / GPS-diff ──► bearing
               └─ drone_pitch          (camera tilt angle)
                       │
                       ▼
           estimate_center_ground_coordinate()
           ground_distance = altitude / tan(pitch)
           center = destination_point(drone_pos, bearing, distance)
                       │
                       ▼
           reference_dataset.csv + frames/


NAVIGATION (per query frame — no GNSS)
──────────────────────────────────────
Query frame
    │
    ▼
ORB feature extraction
    │
    ▼
BFMatcher + Lowe ratio test  ──► good matches
    │
    ▼
RANSAC homography  ──► inlier count
    │
    ├─ inliers < min_inliers ──► REJECTED
    │
    └─ inliers ≥ min_inliers
           │
           ▼
    Map query centre through H ──► (mapped_x, mapped_y)
           │
           ▼
    Real-time telemetry (no GNSS):
      rt_alt   = parse_srt_no_gnss() → rel_alt    (barometric)
      rt_pitch = parse_srt_no_gnss() → drone_pitch (gimbal angle)
           │
           ▼
    predict_center_gps()
      slant   = rt_alt / sin(rt_pitch)
      cross_m = (dx / w/2) × slant × tan(HFOV/2)
      along_m = rt_alt/tan(query_angle) − rt_alt/tan(rt_pitch)
      rotate (north_m, east_m) by reference bearing
           │
           ▼
    (pred_lat, pred_lon)  ◄── output, no GNSS used
```

---

## Limitations

- **Repetitive textures** — uniform roads or fields can produce geometrically
  consistent but semantically incorrect homography matches (false positives
  with high inlier counts). Possible mitigation: check that the mapped centre
  point falls within the reference image bounds before accepting a prediction.

- **Night vs. daytime domain gap** — a daytime reference database achieves
  ~50% coverage on night queries even with CLAHE. A proper solution requires
  a night-time reference database or a learned feature extractor such as
  SuperPoint.

- **Real-time performance** — brute-force O(N) matching over all reference
  frames is accurate but slow for large databases. Future improvement: replace
  BFMatcher with FLANN or a learned descriptor index (e.g. FAISS).

- **Bearing accuracy** — bearing is estimated from GPS trajectory differences
  when `drone_yaw` is not present in the SRT. This degrades when the drone
  hovers or moves sideways.
