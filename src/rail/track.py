"""Real track geometry from OpenStreetMap, cached as a CSV next to the route file.

The first `--realistic` run downloads the route relation from the Overpass API, chains its ways
into one polyline and saves it; later runs only read the CSV (no internet needed). Each route
point is then snapped onto the polyline, so km becomes the real distance along the track.
Track data: (c) OpenStreetMap contributors, ODbL.
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

from rail.route import Route, RouteError, Track

# Public Overpass servers, tried in order (they are sometimes busy).
OVERPASS_URLS = ("https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter")
USER_AGENT = "rail-sim (python urllib)"
ATTEMPTS = 3  # rounds over OVERPASS_URLS
RETRY_PAUSE_S = 5.0
ATTRIBUTION = "(c) OpenStreetMap contributors, ODbL"
MAX_GAP_KM = 0.5  # largest gap between consecutive ways that is bridged with a straight segment
MAX_SNAP_KM = 10.0  # a route point farther than this from the track is an error
SNAP_WINDOW_KM = 40.0  # search window ahead of the previous point, so loops do not snap backwards


class TrackError(RouteError):
    """The track could not be built or does not match the route."""


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km (arrays broadcast)."""
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2))
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(h))


def download_relation(relation_id: int, timeout_s: float = 180.0) -> dict:
    """Overpass JSON of a relation with the geometry of its members."""
    query = f"[out:json][timeout:{int(timeout_s)}];relation({relation_id});out geom;"
    body = urllib.parse.urlencode({"data": query}).encode()
    errors = []
    for attempt, url in enumerate(OVERPASS_URLS * ATTEMPTS):
        if attempt:
            time.sleep(RETRY_PAUSE_S)
        request = urllib.request.Request(url, data=body, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                return json.load(response)
        except (OSError, ValueError) as exc:  # HTTP/network errors, or an incomplete JSON answer
            errors.append(f"{url}: {exc}")
    raise OSError("; ".join(errors))


def chain_ways(data: dict) -> np.ndarray:
    """Join the relation's ways, in member order, into one (n, 2) lat/lon polyline (flipping ways as needed)."""
    relations = [e for e in data.get("elements", []) if e.get("type") == "relation"]
    if not relations:
        raise TrackError("the OpenStreetMap answer has no relation")
    ways = [np.array([(g["lat"], g["lon"]) for g in m["geometry"]]) for m in relations[0]["members"]
            if m["type"] == "way" and m.get("geometry")]  # fmt: skip
    if not ways:
        raise TrackError("the relation has no ways with geometry")
    if len(ways) > 1:  # orient the first way towards the second one
        start, end = ways[0][0], ways[0][-1]
        if min(_dist(start, ways[1][0]), _dist(start, ways[1][-1])) < min(_dist(end, ways[1][0]), _dist(end, ways[1][-1])):
            ways[0] = ways[0][::-1]
    parts = [ways[0]]
    for i, way in enumerate(ways[1:], start=2):
        tail = parts[-1][-1]
        if _dist(tail, way[-1]) < _dist(tail, way[0]):
            way = way[::-1]
        gap = _dist(tail, way[0])
        if gap > MAX_GAP_KM:
            raise TrackError(f"gap of {gap:.1f} km before way #{i} of the relation")
        parts.append(way[1:] if gap < 1e-3 else way)
    return np.concatenate(parts)


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(haversine_km(a[0], a[1], b[0], b[1]))


def save_track(path: Path, latlon: np.ndarray, source: str) -> None:
    header = f"# {source}; downloaded {date.today():%Y-%m-%d}\nlat,lon\n"
    rows = "\n".join(f"{lat:.7f},{lon:.7f}" for lat, lon in latlon)
    path.write_text(header + rows + "\n", encoding="utf-8")


def load_track(path: Path) -> tuple[np.ndarray, str]:
    """(n, 2) lat/lon array and the source comment of a track CSV."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    source = lines[0].lstrip("# ").strip() if lines and lines[0].startswith("#") else str(path)
    rows = [line for line in lines if line.strip() and not line.startswith("#")]
    try:
        if rows[0].replace(" ", "") != "lat,lon":
            raise ValueError
        data = np.array([[float(v) for v in row.split(",")] for row in rows[1:]])
    except (IndexError, ValueError) as exc:
        raise TrackError(f"{path}: expected a 'lat,lon' header and numeric rows") from exc
    if data.ndim != 2 or data.shape[0] < 2 or data.shape[1] != 2:
        raise TrackError(f"{path}: expected at least two 'lat,lon' rows")
    return data, source


def track_path(route: Route) -> Path:
    if route.realistic is None:
        raise TrackError("the route has no [realistic] section (track_file, osm_relation, ...)")
    base = route.path.parent if route.path is not None else Path.cwd()
    return base / route.realistic.track_file


def ensure_track(route: Route, log=print) -> Path:
    """Path of the track CSV, downloading it from OpenStreetMap first if it does not exist yet."""
    path = track_path(route)
    if path.is_file():
        return path
    relation = route.realistic.osm_relation
    if relation is None:
        raise TrackError(f"{path} does not exist and [realistic] osm_relation is not set to download it")
    log(f"Downloading the track from OpenStreetMap (relation {relation}) into {path} ...")
    try:
        data = download_relation(relation)
    except OSError as exc:
        raise TrackError(f"could not download relation {relation}: {exc}") from exc
    save_track(path, chain_ways(data), f"OpenStreetMap relation {relation}; {ATTRIBUTION}")
    return path


def snap_route(route: Route, latlon: np.ndarray, source: str = "") -> Route:
    """Route whose points sit on the track, with km measured along it (from the first point)."""
    lat, lon = latlon[:, 0], latlon[:, 1]
    first, last = route.points[0], route.points[-1]
    # Orient the polyline from the first route point to the last one.
    if _dist((lat[0], lon[0]), (first.lat, first.lon)) > _dist((lat[-1], lon[-1]), (first.lat, first.lon)):
        lat, lon = lat[::-1], lon[::-1]
    s = np.concatenate(([0.0], np.cumsum(haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:]))))
    snapped, idx = [], []
    lo, prev_km = 0, first.km
    for p in route.points:
        ahead = np.arange(lo, len(s))
        ahead = ahead[s[ahead] <= s[lo] + (p.km - prev_km) + SNAP_WINDOW_KM]
        d = haversine_km(p.lat, p.lon, lat[ahead], lon[ahead])
        i = int(ahead[np.argmin(d)])
        if d.min() > MAX_SNAP_KM:
            raise TrackError(f"{p.name} is {d.min():.1f} km away from the track")
        idx.append(i)
        snapped.append(p)
        lo, prev_km = min(i + 1, len(s) - 1), p.km
    if len(set(idx)) != len(idx):
        raise TrackError("two route points snap to the same track vertex")
    i0, i1 = idx[0], idx[-1]
    points = tuple(replace(p, km=float(s[i] - s[i0]), lat=float(lat[i]), lon=float(lon[i]))
                   for p, i in zip(snapped, idx))  # fmt: skip
    track = Track(km=s[i0:i1 + 1] - s[i0], lat=lat[i0:i1 + 1], lon=lon[i0:i1 + 1], source=source)
    return replace(route, points=points, track=track)


def realistic_route(route: Route, log=print) -> Route:
    """The route on its real track (downloaded once and cached)."""
    path = ensure_track(route, log)
    latlon, source = load_track(path)
    return snap_route(route, latlon, source)
