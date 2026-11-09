"""Train route: curated points along the line, interpolated by distance (km)."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
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


@dataclass(frozen=True, eq=False)
class Track:
    """Real track polyline; km is the distance along it, aligned with the route points."""

    km: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    source: str = ""


@dataclass(frozen=True, slots=True)
class Route:
    name: str
    points: tuple[Point, ...]
    train: TrainSpec
    text: str = ""  # original TOML, copied to the results folder
    path: Path | None = None  # route file, to resolve relative paths
    realistic: RealisticSpec | None = None
    track: Track | None = None  # set by rail.track.realistic_route

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
        points.append(Point(name=str(item["name"]), kind=item["kind"], km=float(item["km"]), lat=float(item["lat"]),
                            lon=float(item["lon"]), elevation_m=float(item["elevation_m"]),
                            max_speed_kmh=float(limit), dwell_min=dwell, note=str(item.get("note", "")),
                            label_offset=(float(offset[0]), float(offset[1]))))  # fmt: skip
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
    return Route(name=str(data.get("name", "route")), points=tuple(points), train=train, text=text, path=path,
                 realistic=realistic)  # fmt: skip


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
