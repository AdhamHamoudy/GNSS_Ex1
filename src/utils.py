"""
utils.py – shared GPS, SRT, and geometry helpers for the drone visual-navigation pipeline.

Both prepare_dataset.py and visual_localization.py import from here so that
every formula and parser lives in exactly one place.
"""

import math
import re
from pathlib import Path


# ---------------------------------------------------------------------------
# SRT parsing
# ---------------------------------------------------------------------------

def srt_time_to_seconds(t: str) -> float:
    """Convert SRT timestamp ``HH:MM:SS,mmm`` to seconds."""
    h, m, rest = t.split(":")
    s, ms = rest.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def parse_srt(srt_path: Path) -> list[dict]:
    """
    Parse a DJI SRT telemetry file and return a list of per-frame dicts.

    Extracted fields
    ----------------
    srt_index, start_s, end_s, datetime, framecnt,
    latitude, longitude, rel_alt, abs_alt, focal_len,
    drone_yaw   – heading in degrees (0 = North, clockwise), or None
    speed_north – northward speed in m/s, or None
    speed_east  – eastward speed in m/s, or None
    speed_down  – downward speed in m/s, or None

    The drone_yaw field is tried first when estimating bearing because it is
    more accurate than deriving a heading from two consecutive GPS positions.
    speed_north/speed_east are used as a second fallback.

    DJI Air 3 SRT example line
    --------------------------
    [drone_speedX: -0.2 drone_speedY: 1.3 drone_speedZ: 0.1]
    [drone_yaw: 32.6 drone_pitch: -45.0 drone_roll: 0.1]

    speedX  = East component (positive → East)
    speedY  = North component (positive → North)
    speedZ  = Down component  (positive → Down)
    drone_yaw is the drone body yaw (North = 0, clockwise positive).
    """
    if not srt_path.exists():
        raise FileNotFoundError(f"SRT file not found: {srt_path}")

    text = srt_path.read_text(encoding="utf-8", errors="ignore").strip()
    blocks = re.split(r"\n\s*\n", text)

    rows: list[dict] = []

    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 3:
            continue

        try:
            srt_index = int(lines[0])
        except ValueError:
            continue

        time_match = re.search(
            r"(\d\d:\d\d:\d\d,\d\d\d)\s*-->\s*(\d\d:\d\d:\d\d,\d\d\d)",
            lines[1],
        )
        if not time_match:
            continue

        start_s = srt_time_to_seconds(time_match.group(1))
        end_s = srt_time_to_seconds(time_match.group(2))

        body = " ".join(lines[2:])
        body = re.sub(r"<[^>]+>", " ", body)

        frame_match = re.search(r"FrameCnt:\s*(\d+)", body)
        date_match = re.search(r"(\d{4}-\d\d-\d\d\s+\d\d:\d\d:\d\d\.\d+)", body)

        def extract_float(pattern: str):
            m = re.search(pattern, body)
            return float(m.group(1)) if m else None

        row = {
            "srt_index": srt_index,
            "start_s": start_s,
            "end_s": end_s,
            "datetime": date_match.group(1) if date_match else "",
            "framecnt": int(frame_match.group(1)) if frame_match else None,
            "latitude": extract_float(r"\[latitude:\s*([-\d.]+)\]"),
            "longitude": extract_float(r"\[longitude:\s*([-\d.]+)\]"),
            "rel_alt": extract_float(r"\[rel_alt:\s*([-\d.]+)\s+abs_alt:"),
            "abs_alt": extract_float(r"abs_alt:\s*([-\d.]+)\]"),
            "focal_len": extract_float(r"\[focal_len:\s*([-\d.]+)\]"),
            # ---- optional richer fields (present in many DJI models) ----
            # drone_yaw: body heading, 0 = North, clockwise
            "drone_yaw": extract_float(r"\[drone_yaw:\s*([-\d.]+)"),
            # DJI speedX = East, speedY = North, speedZ = Down
            "speed_east": extract_float(r"drone_speedX:\s*([-\d.]+)"),
            "speed_north": extract_float(r"drone_speedY:\s*([-\d.]+)"),
            "speed_down": extract_float(r"drone_speedZ:\s*([-\d.]+)"),
            # drone_pitch: gimbal tilt in degrees (negative = tilted down in DJI convention)
            # abs() converts to the camera_pitch_deg used in geometry functions
            "drone_pitch": extract_float(r"drone_pitch:\s*([-\d.]+)"),
        }

        if row["latitude"] is not None and row["longitude"] is not None:
            rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# GNSS-denied SRT parser (navigation stage)
# ---------------------------------------------------------------------------

def parse_srt_no_gnss(srt_path: Path) -> list[dict]:
    """
    Parse a DJI SRT file for real-time telemetry used during GNSS-denied navigation.

    Unlike ``parse_srt``, this function does NOT require latitude/longitude to
    be present.  It extracts only the fields that are available without GNSS:

        start_s      – frame timestamp in seconds
        rel_alt      – barometric altitude above takeoff point (metres)
        abs_alt      – absolute altitude (metres), fallback if rel_alt missing
        drone_pitch  – gimbal tilt in degrees (DJI: negative = tilted down)
        focal_len    – focal length in mm (proxy for zoom / FOV changes)
        drone_yaw    – heading in degrees (from compass, not GNSS)
        speed_north  – northward velocity from IMU (m/s)
        speed_east   – eastward velocity from IMU (m/s)

    This is the telemetry stream a drone would expose in a real GNSS-denied
    scenario — everything the flight controller knows except absolute position.
    """
    if not srt_path.exists():
        raise FileNotFoundError(f"SRT file not found: {srt_path}")

    text = srt_path.read_text(encoding="utf-8", errors="ignore").strip()
    blocks = re.split(r"\n\s*\n", text)
    rows: list[dict] = []

    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 3:
            continue
        try:
            int(lines[0])
        except ValueError:
            continue

        time_match = re.search(
            r"(\d\d:\d\d:\d\d,\d\d\d)\s*-->\s*(\d\d:\d\d:\d\d,\d\d\d)", lines[1]
        )
        if not time_match:
            continue

        start_s = srt_time_to_seconds(time_match.group(1))
        body = " ".join(lines[2:])
        body = re.sub(r"<[^>]+>", " ", body)

        def ef(pattern: str):
            m = re.search(pattern, body)
            return float(m.group(1)) if m else None

        row = {
            "start_s":     start_s,
            "rel_alt":     ef(r"\[rel_alt:\s*([-\d.]+)\s+abs_alt:"),
            "abs_alt":     ef(r"abs_alt:\s*([-\d.]+)\]"),
            "drone_pitch": ef(r"drone_pitch:\s*([-\d.]+)"),
            "focal_len":   ef(r"\[focal_len:\s*([-\d.]+)\]"),
            "drone_yaw":   ef(r"\[drone_yaw:\s*([-\d.]+)"),
            "speed_north": ef(r"drone_speedY:\s*([-\d.]+)"),
            "speed_east":  ef(r"drone_speedX:\s*([-\d.]+)"),
        }

        # Accept every row that has at least an altitude reading.
        if row["rel_alt"] is not None or row["abs_alt"] is not None:
            rows.append(row)

    return rows


def realtime_altitude(row: dict, fallback_m: float) -> float:
    """
    Extract the best available altitude from a GNSS-denied telemetry row.

    Prefers ``rel_alt`` (barometric, relative to takeoff) over ``abs_alt``.
    Falls back to *fallback_m* (e.g. the matched reference frame altitude)
    when neither field is present.
    """
    if row.get("rel_alt") is not None:
        return float(row["rel_alt"])
    if row.get("abs_alt") is not None:
        return float(row["abs_alt"])
    return fallback_m


def realtime_camera_pitch(row: dict, fallback_deg: float) -> float:
    """
    Extract the camera pitch (tilt below horizon) from a GNSS-denied telemetry row.

    DJI stores gimbal tilt as ``drone_pitch`` in degrees where negative means
    tilted downward (e.g. -45 = 45° below horizontal).  We convert to the
    positive convention used throughout the pipeline (0° = horizontal,
    90° = straight down).

    Falls back to *fallback_deg* when the field is absent.
    """
    dp = row.get("drone_pitch")
    if dp is not None:
        pitch = abs(float(dp))        # -45 → 45
        if 0 < pitch < 90:
            return pitch
    return fallback_deg


# ---------------------------------------------------------------------------
# Telemetry helpers
# ---------------------------------------------------------------------------

def nearest_row_by_time(rows: list[dict], t: float) -> dict:
    """Return the telemetry row whose ``start_s`` is closest to *t*."""
    return min(rows, key=lambda r: abs(r["start_s"] - t))


# ---------------------------------------------------------------------------
# Geodesy
# ---------------------------------------------------------------------------

_EARTH_RADIUS_M = 6_371_000.0


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres between two WGS-84 coordinates."""
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)

    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * _EARTH_RADIUS_M * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Forward azimuth (bearing) in degrees from point 1 to point 2.
    0° = North, 90° = East, clockwise positive.
    """
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dl = math.radians(lon2 - lon1)

    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)

    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def destination_point(
    lat: float, lon: float, bearing_d: float, distance_m: float
) -> tuple[float, float]:
    """
    Compute the WGS-84 coordinate reached by travelling *distance_m* metres
    from (*lat*, *lon*) along *bearing_d* degrees.
    """
    br = math.radians(bearing_d)
    p1 = math.radians(lat)
    l1 = math.radians(lon)
    d = distance_m / _EARTH_RADIUS_M

    p2 = math.asin(
        math.sin(p1) * math.cos(d) + math.cos(p1) * math.sin(d) * math.cos(br)
    )
    l2 = l1 + math.atan2(
        math.sin(br) * math.sin(d) * math.cos(p1),
        math.cos(d) - math.sin(p1) * math.sin(p2),
    )

    return math.degrees(p2), (math.degrees(l2) + 540.0) % 360.0 - 180.0


def local_offset_to_gps(
    lat: float, lon: float, north_m: float, east_m: float
) -> tuple[float, float]:
    """
    Flat-Earth approximation: convert a local (north_m, east_m) offset into
    a WGS-84 coordinate.  Accurate to well within a metre for offsets up to
    a few hundred metres.
    """
    new_lat = lat + math.degrees(north_m / _EARTH_RADIUS_M)
    new_lon = lon + math.degrees(
        east_m / (_EARTH_RADIUS_M * math.cos(math.radians(lat)))
    )
    return new_lat, new_lon


# ---------------------------------------------------------------------------
# Bearing estimation
# ---------------------------------------------------------------------------

_MIN_SPEED_MS = 0.3   # m/s – below this the speed vector is too noisy to trust
_MIN_DIST_M = 0.5     # m   – minimum position displacement to compute bearing


def estimate_bearing_from_row(row: dict) -> float | None:
    """
    Try to read the bearing directly from a single telemetry row.

    Priority order
    --------------
    1. ``drone_yaw``   – direct compass heading (most accurate).
    2. ``speed_north`` / ``speed_east`` – velocity vector (good when moving).

    Returns ``None`` when neither source is available or reliable.
    """
    # 1. Yaw field (DJI Air 3, Mini 3 Pro, etc.)
    if row.get("drone_yaw") is not None:
        return float(row["drone_yaw"]) % 360.0

    # 2. Velocity vector
    vn = row.get("speed_north")
    ve = row.get("speed_east")
    if vn is not None and ve is not None:
        speed = math.hypot(vn, ve)
        if speed >= _MIN_SPEED_MS:
            return (math.degrees(math.atan2(ve, vn)) + 360.0) % 360.0

    return None


def estimate_center_ground_coordinate(
    row: dict, bearing: float, camera_pitch_deg: float
) -> tuple[float, float]:
    """
    Estimate the WGS-84 coordinate of the ground point visible at the image centre.

    horizontal_ground_offset = altitude / tan(camera_pitch_deg)

    The drone position is moved forward along *bearing* by that offset.

    Altitude fallback priority
    --------------------------
    1. ``rel_alt`` – barometric altitude relative to takeoff (preferred).
    2. ``abs_alt`` – absolute altitude above sea level.
    3. ``0.0``     – last resort; a warning is printed so the caller is aware.

    Pitch validation
    ----------------
    camera_pitch_deg must be strictly between 0° and 90°.  Values outside
    this range make tan() undefined or produce nonsensical ground distances.
    """
    # ---- Altitude (Fix: rel_alt → abs_alt → warn + 0) ----
    if row.get("rel_alt") is not None:
        altitude = float(row["rel_alt"])
    elif row.get("abs_alt") is not None:
        altitude = float(row["abs_alt"])
        import warnings
        warnings.warn(
            "rel_alt missing — falling back to abs_alt. "
            "Ground distance may be less accurate.",
            stacklevel=2,
        )
    else:
        altitude = 0.0
        import warnings
        warnings.warn(
            "Both rel_alt and abs_alt are missing — using altitude=0. "
            "Ground coordinate will be incorrect.",
            stacklevel=2,
        )

    # ---- Pitch validation (Fix: guard against degenerate values) ----
    if not (0.0 < camera_pitch_deg < 90.0):
        raise ValueError(
            f"camera_pitch_deg must be strictly between 0 and 90, got {camera_pitch_deg}."
        )

    ground_distance_m = altitude / math.tan(math.radians(camera_pitch_deg))
    return destination_point(row["latitude"], row["longitude"], bearing, ground_distance_m)


def estimate_bearing_from_gnss(
    rows: list[dict], t: float, window_s: float = 1.0
) -> float:
    """
    Estimate the drone's forward-viewing bearing at time *t*.

    Strategy
    --------
    1. Try ``estimate_bearing_from_row`` on the nearest telemetry row.
       This uses drone_yaw or the speed vector when they are available
       in the SRT – far more accurate than GPS differencing.
    2. Fall back to position-difference bearing over a ±window_s window.
       If the drone barely moved, the window is automatically widened to
       ±3×window_s before giving up and returning 0°.
    """
    nearest = nearest_row_by_time(rows, t)
    direct = estimate_bearing_from_row(nearest)
    if direct is not None:
        return direct

    # --- GPS-differencing fallback ---
    def _gps_bearing(w: float) -> tuple[float, float]:
        r1 = nearest_row_by_time(rows, max(0.0, t - w))
        r2 = nearest_row_by_time(rows, min(rows[-1]["start_s"], t + w))
        dist = haversine_m(r1["latitude"], r1["longitude"], r2["latitude"], r2["longitude"])
        brng = bearing_deg(r1["latitude"], r1["longitude"], r2["latitude"], r2["longitude"])
        return dist, brng

    dist, brng = _gps_bearing(window_s)
    if dist < _MIN_DIST_M:
        dist, brng = _gps_bearing(3 * window_s)

    # If the drone is still nearly stationary, return 0° rather than garbage.
    return brng if dist >= _MIN_DIST_M else 0.0
