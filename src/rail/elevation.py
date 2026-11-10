"""Terrain elevation along the real track, from SRTM via the public OpenTopoData API.

With `[realistic] elevation_file`, the first `--realistic` run samples the track every
`elevation_step_km`, asks OpenTopoData for the elevation of each sample (100 per request, one
request per second, as the public API asks) and saves km,lat,lon,elevation_m to the CSV; later
runs only read it. SRTM is the terrain surface: under a bridge it gives the bottom of the gorge
and over a tunnel the top of the hill, so the profile is smoothed with a running median before
use. Elevation data: SRTM (NASA), via OpenTopoData.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np

from rail.route import Route
from rail.track import USER_AGENT, TrackError, haversine_km

OPENTOPODATA_URL = "https://api.opentopodata.org/v1/{dataset}"
ATTRIBUTION = "SRTM (NASA) via OpenTopoData"
MAX_LOCATIONS = 100  # per request (public API limit)
PAUSE_S = 1.1  # between requests (public API: 1 request per second)
MATCH_KM = 0.3  # a cached sample farther than this from the track means the file belongs to another track
SMOOTH_KM = 5.0  # running-median window: removes gorges under bridges and hills over tunnels


def sample_track(route: Route, step_km: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(km, lat, lon) every step_km along the route's track, including both ends."""
    track = route.track
    n = max(2, int(np.ceil(track.km[-1] / step_km)) + 1)
    km = np.linspace(0.0, track.km[-1], n)
    return km, np.interp(km, track.km, track.lat), np.interp(km, track.km, track.lon)


def download_elevations(lat: np.ndarray, lon: np.ndarray, dataset: str, log=print) -> np.ndarray:
    """Elevation (m) of each location from OpenTopoData, in chunks of MAX_LOCATIONS."""
    out = np.empty(len(lat))
    chunks = range(0, len(lat), MAX_LOCATIONS)
    for k, start in enumerate(chunks):
        if k:
            time.sleep(PAUSE_S)
        end = min(start + MAX_LOCATIONS, len(lat))
        out[start:end] = _request(lat[start:end], lon[start:end], dataset)
        log(f"  elevation: {end}/{len(lat)} points")
    return out


def _request(lat: np.ndarray, lon: np.ndarray, dataset: str) -> np.ndarray:
    locations = "|".join(f"{a:.6f},{b:.6f}" for a, b in zip(lat, lon))
    url = OPENTOPODATA_URL.format(dataset=dataset) + "?" + urllib.parse.urlencode({"locations": locations})
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        data = json.load(response)
    if data.get("status") != "OK":
        raise OSError(f"OpenTopoData: {data.get('error', data.get('status'))}")
    values = np.array([np.nan if r["elevation"] is None else r["elevation"] for r in data["results"]], dtype=float)
    if np.isnan(values).all():
        raise OSError("OpenTopoData returned no elevations (outside the dataset?)")
    if np.isnan(values).any():  # a few voids: fill from the neighbours
        ok = ~np.isnan(values)
        values = np.interp(np.arange(len(values)), np.flatnonzero(ok), values[ok])
    return values


def save_elevation(path: Path, km, lat, lon, elevation, source: str) -> None:
    header = f"# {source}; downloaded {date.today():%Y-%m-%d}\nkm,lat,lon,elevation_m\n"
    rows = "\n".join(f"{a:.3f},{b:.7f},{c:.7f},{d:.1f}" for a, b, c, d in zip(km, lat, lon, elevation))
    path.write_text(header + rows + "\n", encoding="utf-8")


def load_elevation(path: Path) -> tuple[np.ndarray, str]:
    """(n, 4) array of km,lat,lon,elevation_m and the source comment."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    source = lines[0].lstrip("# ").strip() if lines and lines[0].startswith("#") else str(path)
    rows = [line for line in lines if line.strip() and not line.startswith("#")]
    try:
        if rows[0].replace(" ", "") != "km,lat,lon,elevation_m":
            raise ValueError
        data = np.array([[float(v) for v in row.split(",")] for row in rows[1:]])
    except (IndexError, ValueError) as exc:
        raise TrackError(f"{path}: expected a 'km,lat,lon,elevation_m' header and numeric rows") from exc
    if data.ndim != 2 or data.shape[0] < 2 or data.shape[1] != 4 or np.any(np.diff(data[:, 0]) <= 0):
        raise TrackError(f"{path}: expected at least two rows with increasing km")
    return data, source


def running_median(values: np.ndarray, window: int) -> np.ndarray:
    """Running median over `window` samples (odd); near the ends the window shrinks symmetrically, so the end
    values are not pulled towards the inside of the line."""
    n, half = len(values), max(window // 2, 0)
    return np.array([np.median(values[i - h:i + h + 1]) for i in range(n) for h in [min(half, i, n - 1 - i)]])


def elevation_path(route: Route) -> Path:
    base = route.path.parent if route.path is not None else Path.cwd()
    return base / route.realistic.elevation_file


def ensure_elevation(route: Route, log=print) -> Path:
    """Path of the elevation CSV, downloading it first if it does not exist yet (needs the real track)."""
    path = elevation_path(route)
    if path.is_file():
        return path
    spec = route.realistic
    log(f"Downloading terrain elevation ({spec.elevation_dataset}) from OpenTopoData into {path} ...")
    km, lat, lon = sample_track(route, spec.elevation_step_km)
    try:
        elevation = download_elevations(lat, lon, spec.elevation_dataset, log)
    except OSError as exc:
        raise TrackError(f"could not download the elevation: {exc}") from exc
    save_elevation(path, km, lat, lon, elevation, f"{spec.elevation_dataset}; {ATTRIBUTION}")
    return path


def apply_elevation(route: Route, data: np.ndarray, source: str) -> Route:
    """Route whose track and points take their elevation from the (smoothed) SRTM samples."""
    km, lat, lon, raw = data.T
    track = route.track
    off = haversine_km(lat, lon, np.interp(km, track.km, track.lat), np.interp(km, track.km, track.lon))
    if km[-1] < track.km[-1] - MATCH_KM or off.max() > MATCH_KM:
        raise TrackError(f"{elevation_path(route)} does not match the track (delete it to download it again)")
    step = float(np.median(np.diff(km)))
    window = int(round(SMOOTH_KM / step)) | 1  # odd number of samples
    smooth = running_median(raw, window)
    elevation = np.interp(track.km, km, smooth)
    points = tuple(replace(p, elevation_m=float(np.interp(p.km, km, smooth))) for p in route.points)
    return replace(route, points=points, track=replace(track, elevation_m=elevation, elevation_source=source))
