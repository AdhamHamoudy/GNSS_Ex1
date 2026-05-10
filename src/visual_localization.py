"""
visual_localization.py – Step 2: GNSS-denied visual localization.

Given:
  • A query drone video (no GNSS used during prediction).
  • The reference CSV + frames produced by prepare_dataset.py.
  • The matching SRT file (used ONLY after prediction, for error evaluation).

The script:
  1. Loads the reference database (CSV + pre-saved JPEG frames).
  2. Extracts ORB features from every reference image.
  3. For each query frame (sampled every N seconds):
       a. Extracts ORB features.
       b. Finds the best-matching reference frame via BFMatcher + Lowe ratio
          test + RANSAC homography (no GNSS involved).
       c. Maps the query image centre through the homography into reference
          image space.
       d. Converts that reference pixel coordinate to a GPS coordinate using
          reference-frame altitude, bearing, and camera FOV.
  4. After prediction, reads the SRT to compute ground-truth error.
  5. Writes a results CSV, a dual-path KML, and prints error statistics.

Usage
-----
python visual_localization.py \\
    --video        DJI_20260427152735_0019_D.MP4 \\
    --srt          DJI_20260427152735_0019_D.SRT \\
    --reference-csv outputs_step1/reference_dataset.csv \\
    --out          outputs_step2 \\
    --sample-every 2.0 \\
    --camera-pitch 45.0 \\
    --hfov         82.0 \\
    --min-inliers  15
"""

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np

from utils import (
    estimate_bearing_from_gnss,
    estimate_center_ground_coordinate,
    haversine_m,
    local_offset_to_gps,
    nearest_row_by_time,
    parse_srt,
    parse_srt_no_gnss,
    realtime_altitude,
    realtime_camera_pitch,
)


# ---------------------------------------------------------------------------
# Reference database
# ---------------------------------------------------------------------------

def load_reference_database(reference_csv: Path) -> list[dict]:
    """
    Load the reference CSV produced by prepare_dataset.py.

    Returns a list of dicts; image data and keypoints are populated later by
    ``build_reference_features``.
    """
    if not reference_csv.exists():
        raise FileNotFoundError(f"Reference CSV not found: {reference_csv}")

    refs: list[dict] = []

    with reference_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                refs.append(
                    {
                        "sample_id": int(row["sample_id"]),
                        "time_s": float(row["time_s"]),
                        "frame_file": row["frame_file"],
                        "drone_lat": float(row["drone_lat"]),
                        "drone_lon": float(row["drone_lon"]),
                        "rel_alt": float(row["rel_alt"]),
                        "bearing_deg": float(row["bearing_deg"]),
                        "center_lat": float(row["center_lat"]),
                        "center_lon": float(row["center_lon"]),
                    }
                )
            except (KeyError, ValueError):
                continue

    return refs


def build_reference_features(
    refs: list[dict], orb: cv2.ORB, resize_width: int, night_mode: bool = False
) -> list[dict]:
    """
    Read reference images from disk and compute ORB keypoints + descriptors.

    Entries whose image cannot be read or that yield fewer than 10 keypoints
    are silently dropped and a warning is printed.
    """
    valid: list[dict] = []

    for ref in refs:
        img = cv2.imread(ref["frame_file"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  Warning: could not read reference image: {ref['frame_file']}")
            continue

        img = _resize_keep_aspect(img, resize_width)
        if night_mode:
            img = apply_clahe(img)
        kp, desc = orb.detectAndCompute(img, None)

        if desc is None or len(kp) < 10:
            print(f"  Warning: too few features in reference image: {ref['frame_file']}")
            continue

        ref["image_shape"] = img.shape   # (height, width)
        ref["keypoints"] = kp
        ref["descriptors"] = desc
        valid.append(ref)

    return valid


# ---------------------------------------------------------------------------
# Feature matching
# ---------------------------------------------------------------------------

def _resize_keep_aspect(gray: np.ndarray, resize_width: int) -> np.ndarray:
    """Downscale *gray* so its width equals *resize_width*, keeping aspect ratio."""
    if resize_width <= 0:
        return gray
    h, w = gray.shape[:2]
    if w <= resize_width:
        return gray
    scale = resize_width / float(w)
    return cv2.resize(gray, (resize_width, int(round(h * scale))), interpolation=cv2.INTER_AREA)


def apply_clahe(gray: np.ndarray, clip_limit: float = 3.0, tile_size: int = 8) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to a
    grayscale image.  This dramatically improves ORB feature detection in
    low-light / night-time footage by normalising local contrast.

    Parameters
    ----------
    gray       : Grayscale input image.
    clip_limit : Threshold for contrast limiting (higher = more enhancement).
    tile_size  : Size of the grid tiles for local histogram equalisation.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return clahe.apply(gray)


def match_query_to_reference(
    query_kp: list,
    query_desc: np.ndarray,
    ref: dict,
    matcher: cv2.BFMatcher,
    ratio: float = 0.75,
) -> dict | None:
    """
    Match *query_desc* against one reference frame and estimate a homography.

    Returns a result dict with keys:
        good_matches – number of matches passing Lowe's ratio test
        inliers      – RANSAC homography inlier count  (0 if no homography)
        homography   – 3×3 numpy array, or None
        score        – combined ranking score (higher = better match)

    Returns ``None`` when either descriptor set is empty.
    """
    if query_desc is None or ref.get("descriptors") is None:
        return None

    raw = matcher.knnMatch(query_desc, ref["descriptors"], k=2)

    good = [m for pair in raw if len(pair) == 2
            for m, n in [pair] if m.distance < ratio * n.distance]

    H: np.ndarray | None = None
    inliers = 0

    if len(good) >= 8:
        src = np.float32([query_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([ref["keypoints"][m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if mask is not None:
            inliers = int(mask.ravel().sum())

    return {
        "good_matches": len(good),
        "inliers": inliers,
        "homography": H,
        # Inliers weighted heavily so geometric quality dominates the ranking.
        "score": inliers * 10 + len(good),
    }


def find_best_reference(
    query_kp: list,
    query_desc: np.ndarray,
    refs: list[dict],
    matcher: cv2.BFMatcher,
    exclude_near_time: float,
    query_time: float,
) -> tuple[dict | None, dict | None]:
    """
    Iterate over all reference frames and return the best-matching one.

    Reference frames within *exclude_near_time* seconds of *query_time* are
    skipped to prevent trivial same-clip matches during self-evaluation.

    Returns ``(best_ref, best_match)``; both are ``None`` if nothing matched.
    """
    best_ref: dict | None = None
    best_match: dict | None = None

    for ref in refs:
        if exclude_near_time > 0 and abs(ref["time_s"] - query_time) <= exclude_near_time:
            continue

        result = match_query_to_reference(query_kp, query_desc, ref, matcher)
        if result is None:
            continue

        if best_match is None or result["score"] > best_match["score"]:
            best_match = result
            best_ref = ref

    return best_ref, best_match


# ---------------------------------------------------------------------------
# GPS prediction from a matched reference frame
# ---------------------------------------------------------------------------

def predict_center_gps(
    ref: dict,
    mapped_point: tuple[float, float],
    camera_pitch_deg: float,
    hfov_deg: float,
    query_altitude_m: float | None = None,
    query_pitch_deg: float | None = None,
) -> tuple[float, float]:
    """
    Estimate the GPS coordinate of the query frame's image centre.

    The query image centre was mapped through the homography into reference
    image space, giving *mapped_point* = (x, y) in reference pixel coordinates.
    We then convert the pixel offset from the reference image centre into a
    real-world (north, east) offset and add it to the reference centre's GPS
    coordinate.

    Geometry
    --------
    Let:
        h, w        = reference image height and width
        altitude    = drone altitude at the reference frame (metres)
        pitch       = camera pitch below horizon (radians)
        hfov, vfov  = horizontal / vertical field of view (radians)

    Slant range (distance from drone to image centre along viewing ray):
        slant = altitude / sin(pitch)

    Horizontal (cross-track) ground offset:
        cross_m = (dx / (w/2)) * slant * tan(hfov/2)

    Vertical (along-track) ground offset – uses angular interpolation to avoid
    the small-angle approximation breaking down near nadir or near horizon:
        query_angle  = pitch + (dy / (h/2)) * (vfov/2)
        along_m      = altitude/tan(query_angle) - altitude/tan(pitch)

    The offsets are then rotated by the reference bearing and added to the
    reference centre GPS coordinate via a flat-Earth approximation (accurate
    to within centimetres at these scales).

    Parameters
    ----------
    ref : dict
        Reference row (must contain rel_alt, bearing_deg, center_lat, center_lon,
        image_shape).
    mapped_point : (x, y)
        Query image centre mapped into reference image pixel coordinates.
    camera_pitch_deg : float
        Camera tilt below the horizon in degrees (must be in (0, 90)).
    hfov_deg : float
        Horizontal field of view in degrees.

    Returns
    -------
    (pred_lat, pred_lon) in decimal degrees.
    """
    h, w = ref["image_shape"]
    x, y = mapped_point

    dx = x - w / 2.0   # positive → right of reference centre
    dy = y - h / 2.0   # positive → below reference centre

    # Use real-time altitude/pitch from query telemetry when available.
    # This is the key real-time improvement: the drone's barometric altitude
    # and gimbal angle are known even without GNSS, and using them gives a
    # more accurate ground-distance calculation than falling back to the
    # reference frame values.
    altitude = query_altitude_m if query_altitude_m is not None else ref["rel_alt"]
    pitch_used = query_pitch_deg if query_pitch_deg is not None else camera_pitch_deg
    bearing = ref["bearing_deg"]

    pitch_rad = math.radians(pitch_used)
    hfov_rad = math.radians(hfov_deg)
    # Vertical FOV derived from horizontal FOV and image aspect ratio.
    vfov_rad = 2.0 * math.atan(math.tan(hfov_rad / 2.0) * (h / w))

    # --- Cross-track (left-right) displacement ---
    slant_range = altitude / max(math.sin(pitch_rad), 1e-6)
    cross_m = (dx / (w / 2.0)) * slant_range * math.tan(hfov_rad / 2.0)

    # --- Along-track (forward-backward) displacement ---
    delta_v_angle = (dy / (h / 2.0)) * (vfov_rad / 2.0)
    query_angle = pitch_rad + delta_v_angle
    # Clamp to a valid range so tan() stays well-behaved.
    query_angle = min(max(query_angle, math.radians(5.0)), math.radians(85.0))

    ref_ground_dist = altitude / math.tan(pitch_rad)
    query_ground_dist = altitude / math.tan(query_angle)
    along_m = query_ground_dist - ref_ground_dist

    # --- Rotate into (north, east) frame ---
    b = math.radians(bearing)
    right_b = b + math.pi / 2.0

    north_m = along_m * math.cos(b) + cross_m * math.cos(right_b)
    east_m = along_m * math.sin(b) + cross_m * math.sin(right_b)

    return local_offset_to_gps(ref["center_lat"], ref["center_lon"], north_m, east_m)


# ---------------------------------------------------------------------------
# KML export
# ---------------------------------------------------------------------------

def write_paths_kml(csv_path: Path, kml_path: Path) -> None:
    """Write a KML file with two LineStrings: predicted path and SRT-derived path."""
    predicted: list[tuple[str, str]] = []
    real: list[tuple[str, str]] = []

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("pred_lat") and row.get("pred_lon"):
                predicted.append((row["pred_lon"], row["pred_lat"]))
            if row.get("eval_lat") and row.get("eval_lon"):
                real.append((row["eval_lon"], row["eval_lat"]))

    def _linestring(name: str, coords: list[tuple[str, str]], color_hex: str) -> str:
        coord_str = "\n".join(f"{lon},{lat},0" for lon, lat in coords)
        return (
            f"    <Placemark>\n"
            f"      <name>{name}</name>\n"
            f"      <Style><LineStyle>"
            f"<color>ff{color_hex}</color><width>3</width>"
            f"</LineStyle></Style>\n"
            f"      <LineString><tessellate>1</tessellate>"
            f"<coordinates>\n{coord_str}\n</coordinates></LineString>\n"
            f"    </Placemark>\n"
        )

    kml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<kml xmlns="http://www.opengis.net/kml/2.2">\n'
        "  <Document>\n"
        "    <name>Visual localisation result</name>\n"
        + _linestring("Predicted path", predicted, "0000ff")       # red
        + _linestring("SRT ground-truth path", real, "00ff00")     # green
        + "  </Document>\n"
        "</kml>\n"
    )

    kml_path.write_text(kml, encoding="utf-8")


# ---------------------------------------------------------------------------
# Main localization pipeline
# ---------------------------------------------------------------------------

def run_visual_localization(
    video_path: Path,
    srt_path: Path | None,
    reference_csv: Path,
    out_dir: Path,
    sample_every_s: float,
    camera_pitch_deg: float,
    hfov_deg: float,
    exclude_near_time: float,
    resize_width: int,
    min_inliers: int,
    no_eval: bool = False,
    night_mode: bool = False,
) -> None:
    """
    Run GNSS-denied visual localisation on a query video.

    GNSS is NOT read from the SRT during prediction.  The SRT is parsed only
    after all predictions are made so that per-frame error can be computed.
    """
    # ---- Validate inputs ----
    if sample_every_s <= 0:
        raise ValueError("sample_every_s must be greater than 0.")
    if not (0 < camera_pitch_deg < 90):
        raise ValueError("camera_pitch_deg must be strictly between 0 and 90.")
    if not (0 < hfov_deg < 180):
        raise ValueError("hfov_deg must be strictly between 0 and 180.")
    if exclude_near_time < 0:
        raise ValueError("exclude_near_time must be >= 0.")
    if resize_width < 0:
        raise ValueError("resize_width must be >= 0.")
    if min_inliers < 0:
        raise ValueError("min_inliers must be >= 0.")
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load + index reference database ----
    refs = load_reference_database(reference_csv)
    if not refs:
        raise RuntimeError("No reference rows loaded. Check the reference CSV path.")

    orb = cv2.ORB_create(nfeatures=2000)
    # BFMatcher with Hamming distance (required for ORB binary descriptors).
    # crossCheck=False is required for knnMatch(k=2) used in Lowe ratio test.
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    print("Loading reference images and computing ORB features …")
    refs = build_reference_features(refs, orb, resize_width, night_mode=night_mode)
    print(f"  Valid reference frames : {len(refs)}")
    if not refs:
        raise RuntimeError("No valid reference images found. Check frame paths in the CSV.")

    # ---- Parse SRT (evaluation only – not used during prediction) ----
    eval_rows = []
    if not no_eval:
        if srt_path is None or not srt_path.exists():
            print("  Warning: no SRT provided – running in prediction-only mode.")
            no_eval = True
        else:
            eval_rows = parse_srt(srt_path)
            if not eval_rows:
                print("  Warning: SRT parsed but empty – running in prediction-only mode.")
                no_eval = True

    # ---- Open query video ----
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    csv_path = out_dir / "visual_localization_results.csv"
    fieldnames = [
        "query_id",
        "time_s",
        "best_ref_id",
        "best_ref_time_s",
        "best_ref_frame",
        "good_matches",
        "inliers",
        "score",
        "status",
        "mapped_x",
        "mapped_y",
        "pred_lat",
        "pred_lon",
        "eval_lat",
        "eval_lon",
        "error_m",
    ]

    errors: list[float] = []
    query_id = 0
    rejected_count = 0

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_s = frame_count / fps if fps else 0.0

        if fps <= 0 or frame_count <= 0 or duration_s <= 0:
            raise RuntimeError("Could not read valid FPS / frame count from the video.")

        max_time = duration_s if no_eval else min(duration_s, eval_rows[-1]["start_s"])
        # ── Parse query telemetry (GNSS-denied) ──────────────────────────────
        # The SRT is available in real life even without GNSS; we extract only
        # altitude and camera pitch — never lat/lon — so this is genuinely
        # GNSS-free.  Falls back gracefully when the SRT has no altitude data.
        query_telemetry: list[dict] = []
        if srt_path is not None:
            try:
                query_telemetry = parse_srt_no_gnss(srt_path)
                if query_telemetry:
                    # Report what non-GNSS fields are available
                    sample = query_telemetry[0]
                    available = [k for k in ("rel_alt", "drone_pitch", "drone_yaw")
                                 if sample.get(k) is not None]
                    print(f"  Real-time telemetry fields : {', '.join(available) or 'none'}")
            except Exception as e:
                print(f"  Warning: could not parse query SRT for telemetry: {e}")

        print(f"\nProcessing query video: {video_path.name}")
        if no_eval:
            print(f"  Duration : {duration_s:.2f} s  (video) | prediction-only mode (no SRT)")
        else:
            print(f"  Duration : {duration_s:.2f} s  (video) / {eval_rows[-1]['start_s']:.2f} s (SRT)")
        print(f"  Min inliers threshold : {min_inliers}")
        print()

        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            t = 0.0
            while t <= max_time:
                # ---- Extract query frame ----
                frame_index = int(round(t * fps))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = cap.read()
                if not ok:
                    t += sample_every_s
                    continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = _resize_keep_aspect(gray, resize_width)
                if night_mode:
                    gray = apply_clahe(gray)
                query_kp, query_desc = orb.detectAndCompute(gray, None)

                if query_desc is None or len(query_kp) < 10:
                    print(f"  t={t:6.2f}s | skipped – too few query features")
                    t += sample_every_s
                    continue

                # ---- Match to reference database (no GNSS used here) ----
                best_ref, best_match = find_best_reference(
                    query_kp, query_desc, refs, matcher, exclude_near_time, t
                )

                if best_ref is None or best_match is None:
                    print(f"  t={t:6.2f}s | skipped – no reference match found")
                    t += sample_every_s
                    continue

                # ---- Map query image centre into reference pixel space ----
                h, w = gray.shape
                query_center = np.float32([[[w / 2.0, h / 2.0]]])

                if best_match["homography"] is not None and best_match["inliers"] >= 8:
                    mapped = cv2.perspectiveTransform(
                        query_center, best_match["homography"]
                    )[0][0]
                    mapped_x, mapped_y = float(mapped[0]), float(mapped[1])
                else:
                    # Fallback: treat the reference image centre as the mapped point.
                    ref_h, ref_w = best_ref["image_shape"]
                    mapped_x, mapped_y = ref_w / 2.0, ref_h / 2.0

                # ---- Accept / reject based on inlier count ----
                accepted = best_match["inliers"] >= min_inliers
                status = "accepted" if accepted else "rejected_low_inliers"

                pred_lat_str = pred_lon_str = error_str = ""
                real_lat_str = real_lon_str = ""

                if accepted:
                    # ── Real-time telemetry lookup (no GNSS) ────────────────
                    # At this point in a live system the drone would expose
                    # barometric altitude and gimbal angle via its telemetry
                    # bus.  We read those same fields from the SRT here.
                    if query_telemetry:
                        q_row = nearest_row_by_time(query_telemetry, t)
                        rt_alt   = realtime_altitude(q_row, fallback_m=best_ref["rel_alt"])
                        rt_pitch = realtime_camera_pitch(q_row, fallback_deg=camera_pitch_deg)
                    else:
                        rt_alt   = best_ref["rel_alt"]
                        rt_pitch = camera_pitch_deg

                    pred_lat_v, pred_lon_v = predict_center_gps(
                        best_ref,
                        (mapped_x, mapped_y),
                        camera_pitch_deg=camera_pitch_deg,
                        hfov_deg=hfov_deg,
                        query_altitude_m=rt_alt,
                        query_pitch_deg=rt_pitch,
                    )
                    pred_lat_str = f"{pred_lat_v:.8f}"
                    pred_lon_str = f"{pred_lon_v:.8f}"

                    if not no_eval:
                        eval_row = nearest_row_by_time(eval_rows, t)
                        eval_bearing = estimate_bearing_from_gnss(eval_rows, t)
                        # Fix: use actual SRT pitch for ground-truth so it matches
                        # reality rather than the CLI default fallback.
                        eval_pitch = realtime_camera_pitch(eval_row, fallback_deg=camera_pitch_deg)
                        real_lat, real_lon = estimate_center_ground_coordinate(
                            eval_row, eval_bearing, eval_pitch
                        )
                        real_lat_str = f"{real_lat:.8f}"
                        real_lon_str = f"{real_lon:.8f}"
                        error = haversine_m(pred_lat_v, pred_lon_v, real_lat, real_lon)
                        errors.append(error)
                        error_str = f"{error:.3f}"
                        print(
                            f"  t={t:6.2f}s | ref={best_ref['time_s']:6.2f}s | "
                            f"inliers={best_match['inliers']:4d} | "
                            f"pred=({pred_lat_str}, {pred_lon_str}) | "
                            f"error={error_str} m"
                        )
                    else:
                        print(
                            f"  t={t:6.2f}s | ref={best_ref['time_s']:6.2f}s | "
                            f"inliers={best_match['inliers']:4d} | "
                            f"pred=({pred_lat_str}, {pred_lon_str})"
                        )
                else:
                    rejected_count += 1
                    print(
                        f"  t={t:6.2f}s | ref={best_ref['time_s']:6.2f}s | "
                        f"inliers={best_match['inliers']:4d} | REJECTED (< {min_inliers})"
                    )

                writer.writerow(
                    {
                        "query_id": query_id,
                        "time_s": f"{t:.3f}",
                        "best_ref_id": best_ref["sample_id"],
                        "best_ref_time_s": f"{best_ref['time_s']:.3f}",
                        "best_ref_frame": best_ref["frame_file"],
                        "good_matches": best_match["good_matches"],
                        "inliers": best_match["inliers"],
                        "score": best_match["score"],
                        "status": status,
                        "mapped_x": f"{mapped_x:.2f}",
                        "mapped_y": f"{mapped_y:.2f}",
                        "pred_lat": pred_lat_str,
                        "pred_lon": pred_lon_str,
                        "eval_lat": real_lat_str,
                        "eval_lon": real_lon_str,
                        "error_m": error_str,
                    }
                )

                query_id += 1
                t += sample_every_s

    finally:
        cap.release()

    # ---- KML export ----
    kml_path = out_dir / "visual_localization_paths.kml"
    write_paths_kml(csv_path, kml_path)

    # ---- Summary ----
    accepted_count = query_id - rejected_count
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Results CSV          : {csv_path}")
    print(f"  KML path file        : {kml_path}")
    print(f"  Queries processed    : {query_id}")
    print(f"  Accepted predictions : {accepted_count}")
    print(f"  Rejected predictions : {rejected_count}")

    if query_id > 0:
        coverage = accepted_count / query_id * 100.0
        print(f"  Coverage             : {coverage:.1f} %")

    if errors:
        errors_np = np.array(errors, dtype=float)
        print(f"  Mean error           : {errors_np.mean():.3f} m")
        print(f"  Median error         : {float(np.median(errors_np)):.3f} m")
        print(f"  RMSE                 : {math.sqrt(float(np.mean(errors_np ** 2))):.3f} m")
        print(f"  Max error            : {errors_np.max():.3f} m")
    else:
        print("  No accepted predictions – no error statistics computed.")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Step 2 – GNSS-denied visual localisation using a preprocessed "
            "visual reference database."
        )
    )

    parser.add_argument(
        "--video", required=True,
        help="Query MP4 video. GNSS is NOT used from this video during prediction.",
    )
    parser.add_argument(
        "--srt", required=False, default=None,
        help="Matching SRT file. Used ONLY after prediction to compute evaluation error. Omit or use with --no-eval when no SRT is available.",
    )
    parser.add_argument(
        "--reference-csv", required=True,
        help="reference_dataset.csv produced by prepare_dataset.py.",
    )
    parser.add_argument("--out", default="outputs_step2", help="Output folder.")
    parser.add_argument(
        "--sample-every", type=float, default=2.0, metavar="SECONDS",
        help="Process one query frame every N seconds (default: 2.0).",
    )
    parser.add_argument(
        "--camera-pitch", type=float, default=45.0, metavar="DEGREES",
        help="Camera tilt below the horizon in degrees, strictly between 0 and 90 (default: 45.0).",
    )
    parser.add_argument(
        "--hfov", type=float, default=82.0, metavar="DEGREES",
        help="Horizontal field of view in degrees (default: 82.0, suitable for DJI Air 3).",
    )
    parser.add_argument(
        "--resize-width", type=int, default=960, metavar="PIXELS",
        help=(
            "Resize frames to this width before feature matching. "
            "Use 0 for full resolution (slower). Default: 960."
        ),
    )
    parser.add_argument(
        "--exclude-near-time", type=float, default=0.5, metavar="SECONDS",
        help=(
            "Skip reference frames within this many seconds of the query time. "
            "Prevents trivial same-clip matches during self-evaluation (default: 0.5)."
        ),
    )
    parser.add_argument(
        "--min-inliers", type=int, default=15, metavar="N",
        help=(
            "Reject predictions with fewer than N RANSAC inliers. "
            "Raise to 30–50 for higher confidence at the cost of coverage (default: 15)."
        ),
    )

    parser.add_argument(
        "--no-eval",
        action="store_true",
        default=False,
        help="Skip SRT-based ground truth evaluation. Use when no SRT is available.",
    )

    parser.add_argument(
        "--night-mode",
        action="store_true",
        default=False,
        help=(
            "Apply CLAHE contrast enhancement before feature extraction. "
            "Use for low-light or night-time footage where ORB struggles to "
            "find features due to poor contrast."
        ),
    )

    args = parser.parse_args()

    run_visual_localization(
        video_path=Path(args.video),
        srt_path=Path(args.srt) if args.srt else None,
        reference_csv=Path(args.reference_csv),
        out_dir=Path(args.out),
        sample_every_s=args.sample_every,
        camera_pitch_deg=args.camera_pitch,
        hfov_deg=args.hfov,
        exclude_near_time=args.exclude_near_time,
        resize_width=args.resize_width,
        min_inliers=args.min_inliers,
        no_eval=args.no_eval,
        night_mode=args.night_mode,
    )


if __name__ == "__main__":
    main()
