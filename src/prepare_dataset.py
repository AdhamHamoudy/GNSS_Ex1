"""
prepare_dataset.py – Step 1: preprocessing stage.

Given a drone flight recording (MP4 + DJI SRT telemetry file that contains
GNSS data), this script:

  1. Parses the SRT file to extract per-frame telemetry
     (position, altitude, bearing, …).
  2. Samples video frames at a configurable interval.
  3. Estimates the GPS coordinate of the ground point visible at the image
     centre using camera geometry (altitude + camera pitch angle).
  4. Writes a reference CSV that visual_localization.py will use at query
     time, along with the extracted JPEG frames and a KML path file.

Usage
-----
python prepare_dataset.py \\
    --video  DJI_20260427152735_0019_D.MP4 \\
    --srt    DJI_20260427152735_0019_D.SRT \\
    --out    outputs_step1 \\
    --sample-every 1.0 \\
    --camera-pitch 45.0
"""

import argparse
import csv
from pathlib import Path

import cv2

from utils import (
    estimate_bearing_from_gnss,
    estimate_center_ground_coordinate,
    haversine_m,
    nearest_row_by_time,
    parse_srt,
    realtime_camera_pitch,
)


# ---------------------------------------------------------------------------
# KML export
# ---------------------------------------------------------------------------

def write_kml(csv_path: Path, kml_path: Path) -> None:
    """Write a KML LineString of the estimated centre-point path."""
    points: list[tuple[str, str]] = []

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lon = row.get("center_lon", "")
            lat = row.get("center_lat", "")
            if lon and lat:
                points.append((lon, lat))

    coords = "\n".join(f"{lon},{lat},0" for lon, lat in points)

    kml_text = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<kml xmlns="http://www.opengis.net/kml/2.2">\n'
        "  <Document>\n"
        "    <name>Estimated centre-point path</name>\n"
        "    <Placemark>\n"
        "      <name>Centre-point path</name>\n"
        "      <LineString>\n"
        "        <tessellate>1</tessellate>\n"
        "        <coordinates>\n"
        f"{coords}\n"
        "        </coordinates>\n"
        "      </LineString>\n"
        "    </Placemark>\n"
        "  </Document>\n"
        "</kml>\n"
    )

    kml_path.write_text(kml_text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Main dataset builder
# ---------------------------------------------------------------------------

def build_dataset(
    video_path: Path,
    srt_path: Path,
    out_dir: Path,
    sample_every_s: float,
    camera_pitch_deg: float,
) -> None:
    """
    Build the preprocessing reference dataset from a video + matching SRT.

    Parameters
    ----------
    video_path : Path
        Path to the drone MP4 recording.
    srt_path : Path
        Path to the matching DJI SRT telemetry file.
    out_dir : Path
        Destination folder.  Created if it does not exist.
    sample_every_s : float
        Extract one video frame every this many seconds.
    camera_pitch_deg : float
        Camera tilt below the horizon in degrees (must be in (0, 90)).
        For the DJI Air 3 50-metre clip at 45° use 45.0.
    """
    # ---- Validate inputs ----
    if sample_every_s <= 0:
        raise ValueError("sample_every_s must be greater than 0.")
    if not (0 < camera_pitch_deg < 90):
        raise ValueError("camera_pitch_deg must be strictly between 0 and 90.")
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # ---- Setup output dirs ----
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # ---- Parse SRT ----
    rows = parse_srt(srt_path)
    if not rows:
        raise RuntimeError("No telemetry rows were parsed from the SRT file.")

    # ---- Open video ----
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_s = frame_count / fps if fps else 0.0

        if fps <= 0 or frame_count <= 0 or duration_s <= 0:
            raise RuntimeError("Could not read valid FPS / frame count from the video.")

        bearing_source = _detect_bearing_source(rows)

        print(f"Video      : {video_path.name}")
        print(f"Resolution : {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}×"
              f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        print(f"FPS        : {fps:.3f}")
        print(f"Duration   : {duration_s:.2f} s  (video) / "
              f"{rows[-1]['start_s']:.2f} s  (SRT)")
        print(f"SRT rows   : {len(rows)}")
        print(f"Bearing src: {bearing_source}")
        print()

        max_time = min(duration_s, rows[-1]["start_s"])
        csv_path = out_dir / "reference_dataset.csv"
        fieldnames = [
            "sample_id",
            "time_s",
            "frame_file",
            "drone_lat",
            "drone_lon",
            "rel_alt",
            "bearing_deg",
            "bearing_source",
            "camera_pitch_deg",
            "center_lat",
            "center_lon",
        ]

        # If CSV already exists (combining multiple flights), append to it.
        # Read the highest existing sample_id so IDs stay unique.
        append_mode = csv_path.exists()
        sample_id = 0
        if append_mode:
            with csv_path.open("r", encoding="utf-8", newline="") as f:
                existing = list(csv.DictReader(f))
                if existing:
                    sample_id = max(int(r["sample_id"]) for r in existing) + 1
            print(f"  Appending to existing CSV (starting at sample_id={sample_id})")

        # Fix 3: separate counter so the summary prints only this run's samples
        samples_written_this_run = 0

        file_mode = "a" if append_mode else "w"
        with csv_path.open(file_mode, encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not append_mode:
                writer.writeheader()

            t = 0.0
            while t <= max_time:
                row = nearest_row_by_time(rows, t)

                # ---- Extract frame ----
                frame_index = int(round(t * fps))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = cap.read()
                if not ok:
                    print(f"  Warning: could not read frame at t={t:.2f} s")
                    t += sample_every_s
                    continue

                # ---- Estimate bearing and centre coordinate ----
                # Fix 1 & 2: use drone_pitch from SRT when available so the
                # actual gimbal angle is used rather than the CLI fallback.
                bearing = estimate_bearing_from_gnss(rows, t)
                actual_pitch = realtime_camera_pitch(row, fallback_deg=camera_pitch_deg)
                center_lat, center_lon = estimate_center_ground_coordinate(
                    row, bearing, actual_pitch
                )

                # ---- Save frame ----
                frame_name = f"frame_{sample_id:05d}_{t:.2f}s.jpg"
                frame_path = frames_dir / frame_name
                cv2.imwrite(str(frame_path), frame)

                writer.writerow(
                    {
                        "sample_id": sample_id,
                        "time_s": f"{t:.3f}",
                        "frame_file": frame_path.as_posix(),
                        "drone_lat": row["latitude"],
                        "drone_lon": row["longitude"],
                        "rel_alt": row["rel_alt"],
                        "bearing_deg": f"{bearing:.3f}",
                        "bearing_source": bearing_source,
                        # Fix 2: write the actual pitch used, not the CLI default
                        "camera_pitch_deg": f"{actual_pitch:.2f}",
                        "center_lat": f"{center_lat:.8f}",
                        "center_lon": f"{center_lon:.8f}",
                    }
                )

                sample_id += 1
                samples_written_this_run += 1
                t += sample_every_s

    finally:
        cap.release()

    # ---- KML export ----
    kml_path = out_dir / "estimated_center_path.kml"
    write_kml(csv_path, kml_path)

    # ---- Summary ----
    print("Done.")
    print(f"  Samples extracted : {samples_written_this_run}  (this run)")
    print(f"  CSV               : {csv_path}")
    print(f"  KML               : {kml_path}")
    print(f"  Frames            : {frames_dir}")

    _print_coverage_stats(csv_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_bearing_source(rows: list[dict]) -> str:
    """
    Inspect the first few rows to determine which bearing source is available.
    Returns a human-readable string logged in the CSV and on stdout.
    """
    sample = rows[:min(5, len(rows))]
    has_yaw = any(r.get("drone_yaw") is not None for r in sample)
    has_speed = any(
        r.get("speed_north") is not None and r.get("speed_east") is not None
        for r in sample
    )

    if has_yaw:
        return "drone_yaw"
    if has_speed:
        return "speed_vector"
    return "gps_difference"


def _print_coverage_stats(csv_path: Path) -> None:
    """Print a quick sanity check on the estimated centre-point spread."""
    lats, lons = [], []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                lats.append(float(row["center_lat"]))
                lons.append(float(row["center_lon"]))
            except (KeyError, ValueError):
                pass

    if len(lats) < 2:
        return

    diag_m = haversine_m(min(lats), min(lons), max(lats), max(lons))
    print(f"  Coverage diagonal : {diag_m:.1f} m  "
          f"(lat {min(lats):.6f}→{max(lats):.6f}, "
          f"lon {min(lons):.6f}→{max(lons):.6f})")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Step 1 – parse DJI SRT telemetry, extract video frames, and "
            "estimate the image-centre ground coordinate for each sample."
        )
    )

    parser.add_argument("--video", required=True, help="Path to the drone MP4 video.")
    parser.add_argument("--srt", required=True, help="Path to the matching DJI SRT file.")
    parser.add_argument("--out", default="outputs_step1", help="Output folder.")
    parser.add_argument(
        "--sample-every",
        type=float,
        default=1.0,
        metavar="SECONDS",
        help="Extract one frame every N seconds (default: 1.0).",
    )
    parser.add_argument(
        "--camera-pitch",
        type=float,
        default=45.0,
        metavar="DEGREES",
        help=(
            "Camera tilt below the horizon in degrees.  "
            "Must be strictly between 0 and 90.  "
            "For the DJI Air 3 clip at 45° use 45.0 (default)."
        ),
    )

    args = parser.parse_args()

    build_dataset(
        video_path=Path(args.video),
        srt_path=Path(args.srt),
        out_dir=Path(args.out),
        sample_every_s=args.sample_every,
        camera_pitch_deg=args.camera_pitch,
    )


if __name__ == "__main__":
    main()
