from pathlib import Path

import numpy as np
import pytest

from rail.cli import run
from rail.route import RouteError, directions, load_route, orient, parse_route, reversed_route
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
    assert "Los Mochis - Creel" in (run_dir / "summary.txt").read_text(encoding="utf-8")


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


def test_chepe_express_stops_are_highlighted(chepe):
    route, _ = chepe
    assert [p.name for p in route.stations if p.highlight] == ["Los Mochis", "El Fuerte", "Bahuichivo", "Divisadero",
                                                               "Creel"]  # fmt: skip
    assert route.highlight_label == "Chepe Express stop"


@pytest.mark.parametrize(("index", "value"), [(1, True), (0, "yes")])
def test_highlight_only_on_stations_and_boolean(index, value):
    data = {**TINY, "point": [dict(p) for p in TINY["point"]]}
    data["point"][index]["highlight"] = value  # index 1 is a landmark
    with pytest.raises(RouteError, match="highlight"):
        parse_route(data)


# ------------------------------------------------------------------ SRTM elevation


def _fake_elevation(monkeypatch):
    """Offline OpenTopoData: elevation = 1000 * lon + 500, recording the size of each request."""
    calls = []

    def request(lat, lon, dataset):
        calls.append(len(lat))
        return 1000 * np.asarray(lon) + 500

    monkeypatch.setattr("rail.elevation._request", request)
    monkeypatch.setattr("rail.elevation.PAUSE_S", 0.0)
    return calls


def _tiny_with_elevation(tmp_path):
    _tiny_with_track(tmp_path)
    data = {**TINY, "realistic": {**REALISTIC, "elevation_file": "elevation_tiny.csv", "elevation_step_km": 0.05}}
    return parse_route(data, path=tmp_path / "route_tiny.toml")


def test_elevation_is_downloaded_in_chunks_and_applied(tmp_path, monkeypatch):
    calls = _fake_elevation(monkeypatch)
    route = realistic_route(_tiny_with_elevation(tmp_path), log=lambda _: None)
    assert (tmp_path / "elevation_tiny.csv").is_file()
    assert len(calls) > 1 and max(calls) <= 100  # public API limit per request
    for p in route.points:  # points take the terrain elevation at their place on the track
        assert p.elevation_m == pytest.approx(1000 * p.lon + 500, abs=2)
        assert route.locate(p.km)[2] == pytest.approx(p.elevation_m, abs=2)
    assert route.track.elevation_source.startswith("srtm90m")
    calls.clear()
    realistic_route(_tiny_with_elevation(tmp_path), log=lambda _: None)
    assert calls == []  # second run reads the cached CSV


def test_elevation_file_of_another_track_is_rejected(tmp_path, monkeypatch):
    _fake_elevation(monkeypatch)
    route = _tiny_with_elevation(tmp_path)
    realistic_route(route, log=lambda _: None)
    path = tmp_path / "elevation_tiny.csv"
    lines = path.read_text(encoding="utf-8").splitlines()
    shifted = [lines[0], lines[1]] + [",".join([r.split(",")[0], str(float(r.split(",")[1]) + 0.1), *r.split(",")[2:]])
                                      for r in lines[2:]]  # ~11 km north  # fmt: skip
    path.write_text("\n".join(shifted) + "\n", encoding="utf-8")
    with pytest.raises(TrackError, match="does not match the track"):
        realistic_route(route, log=lambda _: None)


def test_running_median_removes_a_tunnel_spike():
    from rail.elevation import running_median

    profile = np.linspace(100, 200, 21)
    profile[10] += 300  # SRTM measures the hill above a tunnel
    smooth = running_median(profile, 5)
    assert smooth[10] == pytest.approx(150, abs=6) and smooth[0] == pytest.approx(100, abs=6)


def test_chepe_includes_sufragio_loreto_and_pitorreal(chepe):
    route, journey = chepe
    names = [s.name for s in route.stations]
    for name in ("Sufragio", "Loreto", "Pitorreal"):
        assert name in names and name in [s.name for s in journey.stops]
    assert names.index("Sufragio") < names.index("El Fuerte") < names.index("Loreto") < names.index("Témoris")
    assert names.index("Divisadero") < names.index("Pitorreal") < names.index("Creel")


@pytest.mark.skipif(not (CHEPE.parent / "elevation_chepe.csv").is_file(), reason="Chepe elevation not downloaded")
def test_chepe_srtm_elevations_match_known_stations():
    route = realistic_route(load_route(CHEPE), log=lambda _: None)
    known = {"Los Mochis": 10, "El Fuerte": 90, "Divisadero": 2240, "Creel": 2340, "Chihuahua": 1440}
    for name, elevation in known.items():
        assert route.point(name).elevation_m == pytest.approx(elevation, abs=60)


# ------------------------------------------------------------------ --route (direction of travel)


def test_chepe_directions_come_from_ends(chepe):
    route, _ = chepe
    assert directions(route) == ("mochis-chihuahua", "chihuahua-mochis")
    assert orient(route, None) is route and orient(route, "mochis-chihuahua") is route
    with pytest.raises(RouteError, match="chihuahua-mochis"):
        orient(route, "chihuahua-creel")


def test_reversed_route_runs_from_the_other_end(chepe):
    route, _ = chepe
    back = orient(route, "chihuahua-mochis")
    assert back.reversed and back.route_id == "chihuahua-mochis"
    assert back.name == "Chepe: Chihuahua - Los Mochis"
    assert [p.name for p in back.points] == [p.name for p in reversed(route.points)]
    assert back.points[0].km == 0 and back.length_km == pytest.approx(route.length_km)
    for p in back.points:  # same place on the line, measured from the other end
        assert p.km == pytest.approx(route.length_km - route.point(p.name).km)
    # Each segment keeps its own limit: the one set at its western point in the file.
    for km in np.linspace(1, route.length_km - 1, 200):
        assert back.speed_limit(route.length_km - km) == route.speed_limit(km)


def test_reversing_twice_gives_the_original(chepe):
    route, _ = chepe
    twice = reversed_route(reversed_route(route))
    assert twice.points == route.points and twice.name == route.name and twice.ends == route.ends
    assert not twice.reversed


def test_reversed_journey_stops_in_the_opposite_order(chepe):
    route, journey = chepe
    back = simulate(orient(route, "chihuahua-mochis"))
    assert [s.name for s in back.stops] == [s.name for s in reversed(journey.stops)]
    assert back.duration_s == pytest.approx(journey.duration_s, rel=0.03)  # same line, almost the same time
    limits = np.array([orient(route, "chihuahua-mochis").speed_limit(k) for k in back.km])
    assert np.all(back.speed_kmh <= limits + 1e-9)


def test_reversed_realistic_route_flips_track_elevation_and_slow_section(tmp_path, monkeypatch):
    _fake_elevation(monkeypatch)
    route = realistic_route(_tiny_with_elevation(tmp_path), log=lambda _: None)
    back = reversed_route(route)
    assert back.track.km[0] == 0 and np.all(np.diff(back.track.km) > 0)
    for p in back.points:
        assert back.locate(p.km)[:2] == pytest.approx((p.lon, p.lat))
        assert back.locate(p.km)[2] == pytest.approx(route.locate(route.point(p.name).km)[2])
    assert (back.realistic.slow_from, back.realistic.slow_to) == ("C", "Bridge")
    km_from, km_to, _ = back.slow_section()
    assert km_from == pytest.approx(0.0) and km_to == pytest.approx(back.point("Bridge").km)


def test_ends_default_to_station_slugs_and_are_validated():
    route = parse_route(TINY)
    assert directions(route) == ("a-c", "c-a")
    with pytest.raises(RouteError, match="ends"):
        parse_route({**TINY, "ends": ["same", "same"]})


def test_cli_route_option_names_the_folder_after_the_direction(tmp_path, monkeypatch):
    monkeypatch.setattr("rail.cli.project_root", lambda: tmp_path)
    route_file = tmp_path / "route_tiny.toml"
    route_file.write_text(CHEPE.read_text(encoding="utf-8"), encoding="utf-8")
    run_dir = run([str(route_file), "--route", "chihuahua-mochis", "--no-video"])
    assert run_dir.name.endswith("_tiny_chihuahua-mochis")
    rows = (run_dir / "timetable.csv").read_text(encoding="utf-8").splitlines()
    assert rows[1].startswith("Chihuahua,0.0") and rows[-1].startswith("Los Mochis,")
    assert "Creel - Los Mochis" in (run_dir / "summary.txt").read_text(encoding="utf-8")


def test_credits_start_with_the_author_then_the_data_sources(chepe, tmp_path, monkeypatch):
    from rail.video import credits_text

    route, _ = chepe
    assert route.author == "Cream Crimson"
    assert credits_text(route) == "Author: Cream Crimson · Data: curated route file (approximate)"
    _fake_elevation(monkeypatch)
    real = realistic_route(_tiny_with_elevation(tmp_path), log=lambda _: None)
    assert credits_text(real) == ("Track: © OpenStreetMap contributors (ODbL) · "
                                  "Elevation: SRTM (NASA) via OpenTopoData")  # no author in TINY  # fmt: skip


# ------------------------------------------------------------------ styles

def test_cream_crimson_style_colours_route_png(tmp_path, monkeypatch):
    from PIL import Image

    from rail.theme import THEMES

    monkeypatch.setattr("rail.cli.project_root", lambda: tmp_path)
    route_file = tmp_path / "route_tiny.toml"
    route_file.write_text(CHEPE.read_text(encoding="utf-8"), encoding="utf-8")
    pngs = {style: run([str(route_file), "--no-video", "--style", style]) / "route.png"
            for style in ("default", "cream_crimson")}  # fmt: skip
    for style, png in pngs.items():
        corner = Image.open(png).convert("RGB").getpixel((3, 3))
        assert "#{:02x}{:02x}{:02x}".format(*corner) == THEMES[style].surface.lower()


def test_unknown_style_is_rejected(tmp_path, monkeypatch):
    from rail.theme import get_theme

    with pytest.raises(ValueError, match="cream_crimson"):
        get_theme("crimson")
    with pytest.raises(SystemExit):
        run(["--no-video", "--style", "crimson"])


def test_cream_crimson_style_colours_the_video(tmp_path):
    from PIL import Image

    from rail.theme import THEMES

    route = parse_route(TINY)
    theme = THEMES["cream_crimson"]
    path = render_journey(route, simulate(route), tmp_path / "journey", duration_s=1, fps=4, fmt="gif", dpi=30,
                          theme=theme)  # fmt: skip
    corner = Image.open(path).convert("RGB").getpixel((2, 2))
    expected = tuple(int(theme.surface[i:i + 2], 16) for i in (1, 3, 5))
    assert all(abs(a - b) <= 8 for a, b in zip(corner, expected))  # the GIF palette may round colours
