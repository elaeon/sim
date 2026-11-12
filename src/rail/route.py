"""Train route: curated points along the line, interpolated by distance (km)."""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

KINDS = ("station", "landmark", "waypoint")


class RouteError(ValueError):
    """Invalid route file."""


@dataclass(frozen=True, slots=True)
class Point:
    name: str
    kind: str
    km: float
    lat: float
    lon: float
    elevation_m: float
    max_speed_kmh: float  # limit from this point to the next one
    dwell_min: float = 0.0
    note: str = ""
    label_offset: tuple[float, float] = (8.0, 0.0)
    highlight: bool = False  # drawn in a distinct colour (e.g. the Chepe Express stops)


@dataclass(frozen=True, slots=True)
class TrainSpec:
    accel_ms2: float = 0.3
    brake_ms2: float = 0.5
    dt_s: float = 1.0


@dataclass(frozen=True, slots=True)
class RealisticSpec:
    """[realistic] section: real track geometry and slow motion through a scenic section."""

    track_file: str  # CSV with the track polyline, relative to the route file
    osm_relation: int | None = None  # OpenStreetMap relation used to download track_file when missing
    slow_from: str | None = None  # name of the point where slow motion starts
    slow_to: str | None = None  # name of the point where it ends
    slow_factor: float = 1.0  # how many times slower the video plays in that section
    elevation_file: str | None = None  # CSV with terrain elevation along the track (SRTM); None keeps elevation_m
    elevation_dataset: str = "srtm90m"  # OpenTopoData dataset used to download elevation_file when missing
    elevation_step_km: float = 1.0  # sample spacing along the track for the download


@dataclass(frozen=True, eq=False)
class Track:
    """Real track polyline; km is the distance along it, aligned with the route points."""

    km: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    source: str = ""
    elevation_m: np.ndarray | None = None  # terrain elevation at each vertex (smoothed SRTM), if loaded
    elevation_source: str = ""


@dataclass(frozen=True, slots=True)
class Route:
    name: str
    points: tuple[Point, ...]
    train: TrainSpec
    text: str = ""  # original TOML, copied to the results folder
    highlight_label: str = ""  # legend text for highlighted stations
    author: str = ""  # shown in the credits line of the video and route.png
    path: Path | None = None  # route file, to resolve relative paths
    realistic: RealisticSpec | None = None
    track: Track | None = None  # set by rail.track.realistic_route
    ends: tuple[str, str] = ("start", "end")  # short names of the first and last station, for --route
    reversed: bool = False  # True when travelling from the last point of the file to the first

    @property
    def route_id(self) -> str:
        """Direction of travel as '<from>-<to>', e.g. 'mochis-chihuahua'."""
        return f"{self.ends[0]}-{self.ends[1]}"

    @property
    def length_km(self) -> float:
        return self.points[-1].km

    @property
    def stations(self) -> tuple[Point, ...]:
        return tuple(p for p in self.points if p.kind == "station")

    @property
    def landmarks(self) -> tuple[Point, ...]:
        return tuple(p for p in self.points if p.kind == "landmark")

    def point(self, name: str) -> Point:
        return next(p for p in self.points if p.name == name)

    def slow_section(self) -> tuple[float, float, float] | None:
        """(km_from, km_to, factor) of the slow-motion section; None unless the real track is loaded (--realistic)."""
        spec = self.realistic
        if self.track is None or spec is None or spec.slow_from is None or spec.slow_factor == 1:
            return None
        return self.point(spec.slow_from).km, self.point(spec.slow_to).km, spec.slow_factor

    def line(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(km, lon, lat) vertices of the line drawn on the map: the real track or the straight segments."""
        if self.track is not None:
            return self.track.km, self.track.lon, self.track.lat
        return self._column("km"), self._column("lon"), self._column("lat")

    def _column(self, attr: str) -> np.ndarray:
        return np.array([getattr(p, attr) for p in self.points], dtype=float)

    def locate(self, km):
        """(lon, lat, elevation_m) at distance km, linearly interpolated (scalar or array)."""
        line_km, lon, lat = self.line()
        if self.track is not None and self.track.elevation_m is not None:
            elevation = np.interp(km, line_km, self.track.elevation_m)
        else:
            elevation = np.interp(km, self._column("km"), self._column("elevation_m"))
        return np.interp(km, line_km, lon), np.interp(km, line_km, lat), elevation

    def speed_limit(self, km: float) -> float:
        """Speed limit (km/h) of the segment that contains km."""
        x = self._column("km")
        i = int(np.searchsorted(x, km, side="right")) - 1
        return self.points[min(max(i, 0), len(self.points) - 1)].max_speed_kmh


def load_route(path: Path) -> Route:
    """Read and validate a route TOML file."""
    text = Path(path).read_text(encoding="utf-8")
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise RouteError(f"{path}: {exc}") from exc
    return parse_route(data, text, Path(path))


def parse_route(data: dict, text: str = "", path: Path | None = None) -> Route:
    raw = data.get("point")
    if not isinstance(raw, list) or len(raw) < 2:
        raise RouteError("the route needs at least two [[point]] entries")
    points: list[Point] = []
    limit = None
    for i, item in enumerate(raw):
        where = f"point #{i + 1} ({item.get('name', '?')})"
        for key in ("name", "kind", "km", "lat", "lon", "elevation_m"):
            if key not in item:
                raise RouteError(f"{where}: missing '{key}'")
        if item["kind"] not in KINDS:
            raise RouteError(f"{where}: kind must be one of {KINDS}, got {item['kind']!r}")
        limit = item.get("max_speed_kmh", limit)
        if limit is None or limit <= 0:
            raise RouteError(f"{where}: the first point must set a positive max_speed_kmh")
        if points and item["km"] <= points[-1].km:
            raise RouteError(f"{where}: km must be strictly increasing ({item['km']} after {points[-1].km})")
        dwell = float(item.get("dwell_min", 0.0))
        if dwell < 0 or (dwell > 0 and item["kind"] != "station"):
            raise RouteError(f"{where}: dwell_min must be >= 0 and only set on stations")
        offset = item.get("label_offset", (8.0, 0.0))
        highlight = item.get("highlight", False)
        if not isinstance(highlight, bool) or (highlight and item["kind"] != "station"):
            raise RouteError(f"{where}: highlight must be true or false, and only set on stations")
        points.append(Point(name=str(item["name"]), kind=item["kind"], km=float(item["km"]), lat=float(item["lat"]),
                            lon=float(item["lon"]), elevation_m=float(item["elevation_m"]),
                            max_speed_kmh=float(limit), dwell_min=dwell, note=str(item.get("note", "")),
                            label_offset=(float(offset[0]), float(offset[1])), highlight=highlight))  # fmt: skip
    if points[0].kind != "station" or points[-1].kind != "station":
        raise RouteError("the first and last points must be stations")
    if points[0].km != 0:
        raise RouteError("the first point must be at km 0")
    try:
        train = TrainSpec(**{k: float(v) for k, v in data.get("train", {}).items()})
    except TypeError as exc:
        raise RouteError(f"[train]: {exc}") from exc
    if min(train.accel_ms2, train.brake_ms2, train.dt_s) <= 0:
        raise RouteError("[train] accel_ms2, brake_ms2 and dt_s must be positive")
    realistic = _parse_realistic(data.get("realistic"), points)
    ends = data.get("ends", (slug(points[0].name), slug(points[-1].name)))
    if (not isinstance(ends, (list, tuple)) or len(ends) != 2 or not all(isinstance(e, str) and e for e in ends)
            or ends[0] == ends[1] or any("-" in e for e in ends)):
        raise RouteError("ends must be two different short names without '-', e.g. [\"mochis\", \"chihuahua\"]")
    return Route(name=str(data.get("name", "route")), points=tuple(points), train=train, text=text, path=path,
                 realistic=realistic, highlight_label=str(data.get("highlight_label", "highlighted stop")),
                 author=str(data.get("author", "")),
                 ends=(ends[0], ends[1]))  # fmt: skip


def slug(name: str) -> str:
    """Short lowercase name for a route end: 'Los Mochis' -> 'los_mochis'."""
    return re.sub(r"[^0-9a-z]+", "_", name.lower()).strip("_") or "end"


def directions(route: Route) -> tuple[str, str]:
    """The two values --route accepts: the file's order and the opposite one."""
    return route.route_id, f"{route.ends[1]}-{route.ends[0]}"


def orient(route: Route, route_id: str | None) -> Route:
    """The route travelled in the direction route_id ('<from>-<to>'); None keeps the file's order."""
    forward, backward = directions(route)
    if route_id is None or route_id == forward:
        return route
    if route_id == backward:
        return reversed_route(route)
    raise RouteError(f"unknown --route {route_id!r}; choose {forward!r} or {backward!r}")


def reversed_route(route: Route) -> Route:
    """The same line travelled the other way: km measured from the other end, limits and slow section flipped."""
    length = route.length_km
    pts = route.points
    # The limit set at a point covers the segment towards the next one; travelling backwards, the segment that
    # starts at point k (towards k-1) is that same segment, so it takes the limit set at point k-1.
    points = tuple(replace(pts[k], km=length - pts[k].km,
                           max_speed_kmh=pts[k - 1].max_speed_kmh if k > 0 else pts[k].max_speed_kmh)
                   for k in range(len(pts) - 1, -1, -1))  # fmt: skip
    track = route.track
    if track is not None:
        track = replace(track, km=length - track.km[::-1], lat=track.lat[::-1], lon=track.lon[::-1],
                        elevation_m=None if track.elevation_m is None else track.elevation_m[::-1])  # fmt: skip
    realistic = route.realistic
    if realistic is not None and realistic.slow_from is not None:
        realistic = replace(realistic, slow_from=realistic.slow_to, slow_to=realistic.slow_from)
    first, last = pts[0].name, pts[-1].name
    name = route.name.replace(first, "\0").replace(last, first).replace("\0", last)  # swap the ends in the title
    return replace(route, name=name, points=points, track=track, realistic=realistic,
                   ends=(route.ends[1], route.ends[0]), reversed=not route.reversed)  # fmt: skip


def _parse_realistic(raw: dict | None, points: list[Point]) -> RealisticSpec | None:
    if raw is None:
        return None
    try:
        spec = RealisticSpec(**raw)
    except TypeError as exc:
        raise RouteError(f"[realistic]: {exc}") from exc
    if not isinstance(spec.track_file, str) or not spec.track_file:
        raise RouteError("[realistic] track_file must be a file name")
    if spec.slow_factor < 1:
        raise RouteError("[realistic] slow_factor must be >= 1")
    if spec.elevation_file is not None and (not isinstance(spec.elevation_file, str) or not spec.elevation_file):
        raise RouteError("[realistic] elevation_file must be a file name")
    if not spec.elevation_step_km > 0:
        raise RouteError("[realistic] elevation_step_km must be > 0")
    if (spec.slow_from is None) != (spec.slow_to is None):
        raise RouteError("[realistic] slow_from and slow_to go together")
    if spec.slow_from is not None:
        km = {p.name: p.km for p in points}
        for key in ("slow_from", "slow_to"):
            if getattr(spec, key) not in km:
                raise RouteError(f"[realistic] {key}: no point named {getattr(spec, key)!r}")
        if km[spec.slow_from] >= km[spec.slow_to]:
            raise RouteError("[realistic] slow_from must come before slow_to")
    return spec
