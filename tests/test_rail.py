from pathlib import Path

import numpy as np
import pytest

from rail.cli import run
from rail.route import RouteError, load_route, parse_route
from rail.sim import clock, simulate, timetable
from rail.track import (
    TrackError, chain_ways, ensure_track, haversine_km, load_track, realistic_route, save_track, snap_route,
)  # fmt: skip
from rail.video import frame_times, render_journey

CHEPE = Path(__file__).resolve().parents[1] / "examples" / "route_chepe.toml"

TINY = {
    "name": "tiny",
    "train": {"accel_ms2": 0.5, "brake_ms2": 0.5, "dt_s": 1.0},
    "point": [
        {"name": "A", "kind": "station", "km": 0, "lat": 0.0, "lon": 0.0, "elevation_m": 0, "max_speed_kmh": 60},
        {"name": "Bridge", "kind": "landmark", "km": 2, "lat": 0.0, "lon": 0.02, "elevation_m": 50,
         "max_speed_kmh": 20, "note": "slow bridge"},
        {"name": "B", "kind": "station", "km": 3, "lat": 0.0, "lon": 0.03, "elevation_m": 80, "dwell_min": 2},
        {"name": "C", "kind": "station", "km": 6, "lat": 0.01, "lon": 0.05, "elevation_m": 100, "max_speed_kmh": 40},
    ],
}


@pytest.fixture(scope="module")
def chepe():
    route = load_route(CHEPE)
    return route, simulate(route)


def test_chepe_route_loads_with_increasing_km_to_chihuahua(chepe):
    route, _ = chepe
    km = [p.km for p in route.points]
    assert all(b > a for a, b in zip(km, km[1:]))
    assert route.points[0].name == "Los Mochis" and route.points[-1].name == "Chihuahua"
    assert 640 <= route.length_km <= 670


def test_locate_at_a_station_returns_its_coordinates(chepe):
    route, _ = chepe
    p = next(s for s in route.stations if s.name == "Divisadero")
    assert route.locate(p.km) == pytest.approx((p.lon, p.lat, p.elevation_m))


def test_train_stops_at_every_station_with_its_dwell(chepe):
    route, journey = chepe
    assert [s.name for s in journey.stops] == [s.name for s in route.stations]
    for stop, station in zip(journey.stops[1:-1], route.stations[1:-1]):
        assert stop.departure_s - stop.arrival_s == pytest.approx(station.dwell_min * 60)
        assert journey.at(stop.arrival_s)[0] == pytest.approx(station.km)
    assert journey.km[-1] == pytest.approx(route.length_km)
    assert journey.speed_kmh[-1] == 0


def test_train_never_exceeds_the_segment_limit(chepe):
    route, journey = chepe
    limits = np.array([route.speed_limit(k) for k in journey.km])
    assert np.all(journey.speed_kmh <= limits + 1e-9)
    assert np.all(np.diff(journey.t_s) >= 0) and np.all(np.diff(journey.km) >= 0)


def test_chepe_times_match_the_published_schedule(chepe):
    """Calibration: Los Mochis -> Creel ~9 h 40 min (Chepe Express); whole line in a plausible range."""
    _, journey = chepe
    assert 9.0 <= journey.arrival("Creel") / 3600 <= 10.5
    assert 14.0 <= journey.duration_s / 3600 <= 18.0


def test_train_brakes_before_a_lower_limit():
    route = parse_route(TINY)
    journey = simulate(route)
    on_bridge = (journey.km >= 2) & (journey.km < 3)
    assert journey.speed_kmh[on_bridge].max() <= 20 + 1e-9
    assert journey.speed_kmh.max() > 20  # it does go faster before and after


def test_timetable_uses_the_departure_time():
    journey = simulate(parse_route(TINY))
    rows = timetable(journey, depart_s=23.95 * 3600)
    assert rows[0]["departure"] == "23:57"
    assert rows[-1]["arrival"].endswith("(+1d)")
    assert clock(3600 * 25) == "01:00 (+1d)"


@pytest.mark.parametrize(
    ("index", "change", "message"),
    [
        (2, {"km": 1.5}, "strictly increasing"),
        (1, {"kind": "tunnel"}, "kind"),
        (1, {"dwell_min": 3}, "dwell_min"),
        (0, {"max_speed_kmh": None}, "max_speed_kmh"),
    ],
)
def test_invalid_route_raises(index, change, message):
    data = {**TINY, "point": [dict(p) for p in TINY["point"]]}
    for key, value in change.items():
        if value is None:
            del data["point"][index][key]
        else:
            data["point"][index][key] = value
    with pytest.raises(RouteError, match=message):
        parse_route(data)


def test_video_smoke_gif(tmp_path):
    route = parse_route(TINY)
    path = render_journey(route, simulate(route), tmp_path / "journey", duration_s=1, fps=4, fmt="gif", dpi=30)
    assert path.suffix == ".gif" and path.stat().st_size > 0


def test_cli_writes_timetable_summary_and_png(tmp_path, monkeypatch):
    monkeypatch.setattr("rail.cli.project_root", lambda: tmp_path)
    route_file = tmp_path / "route_tiny.toml"
    route_file.write_text(CHEPE.read_text(encoding="utf-8"), encoding="utf-8")
    run_dir = run([str(route_file), "--no-video"])
    assert run_dir.parent == tmp_path / "resultados" and run_dir.name.endswith("_tiny")
    for name in ("summary.txt", "timetable.csv", "route.png", "route_tiny.toml"):
        assert (run_dir / name).is_file()
    assert "to Creel" in (run_dir / "summary.txt").read_text(encoding="utf-8")


# ------------------------------------------------------------------ --realistic

REALISTIC = {"track_file": "track_tiny.csv", "slow_from": "Bridge", "slow_to": "C", "slow_factor": 3.0}


def _tiny_with_track(tmp_path):
    """TINY plus a [realistic] section and a wiggly synthetic track next to the route file."""
    lon = np.linspace(-0.01, 0.06, 141)
    lat = 0.002 * np.sin(lon * 300)  # ~200 m wiggles, so the track is longer than the straight line
    save_track(tmp_path / "track_tiny.csv", np.column_stack([lat, lon]), "synthetic")
    data = {**TINY, "realistic": REALISTIC}
    return parse_route(data, path=tmp_path / "route_tiny.toml")


def test_slow_motion_needs_the_real_track(chepe):
    route, _ = chepe
    assert route.realistic is not None and route.slow_section() is None  # default run: no slow motion


def test_snap_route_measures_km_along_the_track(tmp_path):
    route = realistic_route(_tiny_with_track(tmp_path), log=lambda _: None)
    km = [p.km for p in route.points]
    assert km[0] == 0 and all(b > a for a, b in zip(km, km[1:]))
    a, c = route.point("A"), route.point("C")
    assert route.length_km > 1.05 * float(haversine_km(a.lat, a.lon, c.lat, c.lon))  # wiggles: longer than straight
    assert route.track.km[0] == 0 and route.track.km[-1] == pytest.approx(route.length_km)
    for p in route.points:  # every point sits on the track
        assert route.locate(p.km)[:2] == pytest.approx((p.lon, p.lat))
    km_from, km_to, factor = route.slow_section()
    assert (km_from, km_to, factor) == (route.point("Bridge").km, route.point("C").km, 3.0)


def test_snap_route_rejects_points_far_from_the_track(tmp_path):
    route = _tiny_with_track(tmp_path)
    latlon = np.column_stack([np.full(10, 1.0), np.linspace(0, 0.05, 10)])  # ~110 km north
    with pytest.raises(TrackError, match="away from the track"):
        snap_route(route, latlon)


def test_track_csv_round_trip(tmp_path):
    latlon = np.array([[25.5, -108.9], [25.6, -108.8], [25.7, -108.7]])
    save_track(tmp_path / "t.csv", latlon, "OpenStreetMap relation 1")
    loaded, source = load_track(tmp_path / "t.csv")
    assert loaded == pytest.approx(latlon) and source.startswith("OpenStreetMap relation 1")


def test_missing_track_without_relation_is_an_error(tmp_path):
    route = _tiny_with_track(tmp_path)
    (tmp_path / "track_tiny.csv").unlink()
    with pytest.raises(TrackError, match="osm_relation"):
        ensure_track(route, log=lambda _: None)


def test_chain_ways_flips_reversed_ways_and_rejects_gaps():
    def way(*pts):
        return {"type": "way", "geometry": [{"lat": a, "lon": b} for a, b in pts]}

    data = {"elements": [{"type": "relation", "members": [
        way((0, 0.02), (0, 0.01), (0, 0)),  # first way backwards
        way((0, 0.02), (0, 0.03)),
        way((0, 0.05), (0, 0.04), (0, 0.03)),  # reversed
    ]}]}  # fmt: skip
    line = chain_ways(data)
    assert line[:, 1] == pytest.approx([0, 0.01, 0.02, 0.03, 0.04, 0.05])
    data["elements"][0]["members"].append(way((0, 0.2), (0, 0.3)))  # ~16 km gap
    with pytest.raises(TrackError, match="gap"):
        chain_ways(data)


def test_frame_times_are_denser_in_the_slow_section():
    journey = simulate(parse_route(TINY))
    times = frame_times(journey, 400, slow=(2.0, 3.0, 4.0))
    assert times[0] == 0 and times[-1] == pytest.approx(journey.duration_s)
    assert np.all(np.diff(times) >= 0)
    km = journey.at(times)[0]
    t_slow = np.ptp(journey.t_s[(journey.km >= 2.0) & (journey.km <= 3.0)])
    share_frames = np.mean((km >= 2.0) & (km <= 3.0))
    assert share_frames > 1.5 * t_slow / journey.duration_s  # more frames than its share of time


def test_realistic_video_smoke_gif(tmp_path):
    route = realistic_route(_tiny_with_track(tmp_path), log=lambda _: None)
    path = render_journey(route, simulate(route), tmp_path / "journey", duration_s=1, fps=4, fmt="gif", dpi=30)
    assert path.stat().st_size > 0


def test_cli_realistic_copies_track_and_reports_slow_motion(tmp_path, monkeypatch):
    monkeypatch.setattr("rail.cli.project_root", lambda: tmp_path)
    _tiny_with_track(tmp_path)
    lines = ['name = "tiny"', "[realistic]", *(f"{k} = {v!r}".replace("'", '"') for k, v in REALISTIC.items())]
    for p in TINY["point"]:
        lines.append("[[point]]")
        lines += [f"{k} = {v!r}".replace("'", '"') for k, v in p.items()]
    (tmp_path / "route_tiny.toml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    run_dir = run([str(tmp_path / "route_tiny.toml"), "--realistic", "--no-video"])
    assert run_dir.name.endswith("_tiny_realistic")
    assert (run_dir / "track_tiny.csv").is_file()
    summary = (run_dir / "summary.txt").read_text(encoding="utf-8")
    assert "slow motion         x3, Bridge - C" in summary and "track               synthetic" in summary


@pytest.mark.skipif(not (CHEPE.parent / "track_chepe.csv").is_file(), reason="Chepe track not downloaded")
def test_chepe_on_the_real_track_keeps_its_calibration():
    route = realistic_route(load_route(CHEPE), log=lambda _: None)
    assert 645 <= route.length_km <= 660  # official: 653 km
    journey = simulate(route)
    assert 9.0 <= journey.arrival("Creel") / 3600 <= 10.5
