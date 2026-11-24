"""`traffic-emissions --light-spacing` and the offset of the traffic lights (`offset`)."""

import json
from pathlib import Path

import numpy as np
import pytest

from traffic.bump_spacing import (
    DATA_NAME, LIGHT_CYCLE, META_NAME, PROFILE_CSV, SUMMARY_NAME, Spacing, analyze, build_configs, first_position,
    travel_time,
)  # fmt: skip
from traffic.cli import CONFIG_NAME, emissions
from traffic.config import DT, GREEN, POLLUTANTS, RED, YELLOW, Light, SimConfig, SpeedBump
from traffic.emissions import EMIS_BIN
from traffic.pedestrians import fixed_phases
from traffic.results import read_results
from traffic.settings import ConfigError, parse_settings

# ---------------------------------------------------------------- offset


def test_offset_delays_the_whole_cycle():
    plain = Light(100.0, red=20, green=25, yellow=3)
    late = Light(100.0, red=20, green=25, yellow=3, offset=12.5)
    shift = round(12.5 / DT)
    ticks = np.arange(0, 2000)
    assert [late.phase(int(t)) for t in ticks] == [plain.phase(int(t) - shift) for t in ticks]
    assert np.array_equal(fixed_phases(late, 2000), np.array([late.phase(int(t)) for t in ticks], np.int8))
    assert {late.phase(int(t)) for t in ticks} == {RED, GREEN, YELLOW}
    # The red intervals match the phases, also with an offset larger than the red and at t = 0.
    for light in (late, Light(100.0, red=20, green=25, yellow=3, offset=40.0, start_phase="green")):
        red = np.array([light.phase(t) == RED for t in range(3000)])
        edges = np.flatnonzero(np.diff(np.r_[False, red, False].astype(int)))
        expected = [(a * DT, b * DT) for a, b in zip(edges[::2], edges[1::2])]
        assert light.red_intervals(300.0) == pytest.approx(expected)
    assert "offset 12.5 s" in late.label and "offset" not in plain.label


def _settings(light: str) -> str:
    return f"[road]\nlength = 200.0\n[[traffic_light]]\nposition = 100.0\n{light}\n"


def test_offset_in_the_configuration():
    sim = parse_settings(_settings("red = 20.0\ngreen = 25.0\noffset = 10.0"), Path("x.toml")).sim
    assert sim.extra_lights[0].offset == 10.0 and sim.exit_light.offset == 0.0
    sim = parse_settings("[road]\nlength = 200.0\n[traffic_light]\nred = 20.0\ngreen = 25.0\noffset = 4.0\n", Path("x.toml")).sim
    assert sim.light_offset == 4.0 and sim.exit_light.offset == 4.0
    for text, match in [
        ("red = 20.0\ngreen = 25.0\noffset = 45.0", "offset must be between 0 and the cycle"),
        ("red = 20.0\ngreen = 25.0\noffset = 0.05", "offset must be a multiple"),
        ("pedestrian = true\noffset = 3.0", "offset does not apply"),
    ]:
        with pytest.raises(ConfigError, match=match):
            parse_settings(_settings(text), Path("x.toml"))


# ---------------------------------------------------------------- scenarios


def _base(**kw) -> SimConfig:
    bump = SpeedBump(40.0, None)
    exit_and_inner = dict(traffic_light=True, extra_lights=(Light(60.0, pedestrian=True),), speed_bumps=(bump,))
    return SimConfig(length=400, lanes=2, rates=(10, 0, 0), run=3, **{**exit_and_inner, **kw})


def _spacing(**kw) -> Spacing:
    args = dict(first=100.0, distances=(50.0, 150.0), bump_lanes=None, light=None, length=None, tolerance=0.05,
                threshold=0.1, replicas=1, run=3.0, kind="semaforo", cycle=Light(0.0, red=30, green=30, yellow=3))  # fmt: skip
    return Spacing(**{**args, **kw})


def test_build_configs_puts_only_the_sweep_lights_with_red_on_arrival():
    configs, skipped = build_configs(_base(), _spacing(distances=(50.0, 150.0, 300.0)))
    assert skipped == [300.0]
    assert [n for n, _ in configs][:3] == ["no traffic light", "one traffic light at 100 m", "traffic lights at 100 and 150 m (d = 50 m)"]
    for _, cfg in configs:  # without the light at the end or those of the configuration, and without bumps
        assert not cfg.has_light and cfg.speed_bumps == ()
    assert configs[0][1].inner_lights == ()
    pair = configs[3][1].inner_lights
    assert [lt.position for lt in pair] == [100.0, 250.0] and all(lt.start_phase == "red" for lt in pair)
    # The second turns red when the head of the platoon that left the first one on green arrives (t = 30 s).
    car = configs[0][1].specs[0]
    v, a = min(car.speed_kmh, 1e9) / 3.6, car.accel
    trip = travel_time(configs[0][1], 150.0)
    assert trip == pytest.approx(150 / v + v / (2 * a)) if a else trip == pytest.approx(150 / v)
    assert pair[1].offset == pytest.approx(round((30.0 + trip) / DT) * DT)
    t = round((30.0 + trip) / DT)
    assert pair[0].phase(round(30.0 / DT) - 1) == RED and pair[0].phase(round(30.0 / DT)) == GREEN
    assert pair[1].phase(t - 2) != RED and pair[1].phase(t + 1) == RED
    # With a short d the head does not reach its speed: it travels while accelerating.
    if a:
        assert travel_time(configs[0][1], 5.0) == pytest.approx(np.sqrt(2 * 5.0 / a))


def test_build_configs_chain_offsets_same_phase_and_fixed_bumps():
    configs, _ = build_configs(_base(), _spacing(distances=(50.0,), count=3, bumps=(40.0,)))
    chain = configs[2][1].inner_lights
    step = 30.0 + travel_time(configs[0][1], 50.0)
    assert [lt.offset for lt in chain] == pytest.approx([0.0, round(step, 1), round((2 * step) % 63.0, 1)], abs=0.051)
    assert [b.position for b in configs[2][1].bumps] == [40.0]
    same, _ = build_configs(_base(), _spacing(distances=(50.0,), offset_mode="igual"))
    assert all(lt.offset == 0.0 for lt in same[2][1].inner_lights)
    with pytest.raises(ConfigError, match="--first-light must be between 0 and 400 m"):
        build_configs(_base(), _spacing(first=400.0))
    with pytest.raises(ConfigError, match="at least 2 lights"):
        build_configs(_base(), _spacing(count=1))


def test_green_wave_and_mixed_chain():
    configs, _ = build_configs(_base(), _spacing(distances=(50.0,), offset_mode="verde"))
    a, b = configs[2][1].inner_lights
    trip = travel_time(configs[0][1], 50.0)
    assert b.offset == pytest.approx(round(trip / DT) * DT)
    # B turns green when the head of the platoon that left A on green arrives (t = 30 s).
    t = round((30.0 + trip) / DT)
    assert b.phase(t - 2) == RED and b.phase(t + 1) == GREEN
    # Mixed chain: A and B (d) in a green wave, C 150 m after B, red on arrival.
    s = _spacing(distances=(40.0, 60.0), count=3, offset_mode="verde,rojo", fixed_gaps=(150.0,))
    configs, skipped = build_configs(_base(), s)
    assert skipped == [] and configs[2][0] == "traffic lights at 100, 140 and 290 m (d = 40 m)"
    a, b, c = configs[2][1].inner_lights
    base = configs[0][1]
    expected_b = travel_time(base, 40.0)
    expected_c = expected_b + 30.0 + travel_time(base, 150.0)
    assert b.offset == pytest.approx(expected_b, abs=0.051) and c.offset == pytest.approx(expected_c % 63.0, abs=0.051)
    assert [lt.position for lt in configs[3][1].inner_lights] == [100.0, 160.0, 310.0]
    for bad in (dict(fixed_gaps=(150.0, 20.0)), dict(offset_mode="verde,rojo,rojo"), dict(offset_mode="azul")):
        with pytest.raises(ConfigError, match="--fixed-spacings|--offset"):
            build_configs(_base(), _spacing(distances=(40.0,), count=3, **{"offset_mode": "rojo", **bad}))


# ---------------------------------------------------------------- analysis

BINS = 120  # 600 m
X = (np.arange(BINS) + 0.5) * EMIS_BIN
FIRST = 100.0
BEFORE, AFTER = 30.0, 40.0  # m of queue and of wake


def _footprint(at: float, height: float = 600.0) -> np.ndarray:
    """Excess of a traffic light: a plateau for the queue (BEFORE m before) and a triangular wake of AFTER m after."""
    queue = np.where((X >= at - BEFORE) & (X < at), 0.5 * height, 0.0)
    wake = height * np.clip(1 - (X - at) / AFTER, 0, 1) * (X >= at)
    return queue + wake


def _synthetic(distances, far_factor: float = 1.1, near_extra: dict | None = None, entry: tuple = ()) -> tuple[dict, dict]:
    """The second light emits `far_factor` × an isolated one (the whole platoon stops); with d in `near_extra`, the
    pair also emits that uniform excess; with d in `entry`, the queue reaches the entry."""
    base = np.full(BINS, 100.0)
    profs = [base, base + _footprint(FIRST)]
    for d in distances:
        p = base + _footprint(FIRST) + far_factor * _footprint(FIRST + d)
        p += (near_extra or {}).get(d, 0.0) * (X < FIRST + d)
        if d in entry:
            p[0] += 1000.0
        profs.append(p)
    pos = np.zeros((len(profs), len(POLLUTANTS), 1, BINS))
    for i, prof in enumerate(profs):
        pos[i, POLLUTANTS.index("co2"), 0] = prof * EMIS_BIN
    meta = {"contaminantes": ["co2"], "primer_semaforo_m": FIRST, "distancias_m": list(distances), "tolerancia": 0.05,
            "umbral": 0.1, "s_simulados": 3600.0, "topes_por_cadena": 2, "elemento": "semaforo"}  # fmt: skip
    tt = np.array([[20.0], [30.0], *([[40.0 + 0 * d] for d in distances])])
    return meta, {"emissions_pos": pos, "travel_time": tt, "crossed_veh": np.ones_like(tt)}


def test_two_sided_footprint_and_stable_separation():
    dist = [50.0, 80.0, 120.0, 200.0]
    meta, data = _synthetic(dist, near_extra={50.0: 150.0})
    assert first_position(meta) == FIRST
    r = analyze(meta, data)
    co2 = r["pollutants"]["co2"]
    assert co2["huella_antes_m"] == pytest.approx(BEFORE + EMIS_BIN / 2)  # centre of the previous free bin
    assert co2["huella_m"] == pytest.approx(37.5)
    # Outside the footprint: d ≥ before + after; the distant pair emits 1.05 × two isolated ones (not 1): it is compared with it.
    assert co2["d_huella"] == 80.0
    assert co2["aditividad"][-1] == pytest.approx(1.05, abs=0.01)
    assert co2["estabilidad"][0] > 1.05 and co2["estabilidad"][1:] == pytest.approx(1.0, abs=0.01)
    assert co2["d_aditiva"] == 80.0 and co2["d_minima"] == 80.0 and r["recomendada_m"] == 80.0
    # Delay: (40 − 20) / (2 × (30 − 20)) = 1.
    assert np.allclose(r["demora"], 1.0) and not r["cola_en_entrada"]


def test_entry_blocking_and_no_plateau():
    meta, data = _synthetic([80.0, 120.0, 200.0], entry=(80.0,))
    r = analyze(meta, data)
    assert list(r["cola_entrada_d"]) == [True, False, False]
    assert r["pollutants"]["co2"]["d_aditiva"] == 120.0  # the one with the queue at the entry does not count
    # If only the widest spacing qualifies (there is no plateau), there is no stable spacing.
    meta, data = _synthetic([80.0, 120.0], near_extra={80.0: 150.0, 120.0: 150.0})
    assert analyze(meta, data)["pollutants"]["co2"]["d_aditiva"] is None


# ---------------------------------------------------------------- command


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr("traffic.settings.project_root", lambda: tmp_path)
    (tmp_path / CONFIG_NAME).write_text(
        "[road]\nlength = 150.0\nmax_line_speed = [50, 50]\n[traffic_light]\nenabled = false\n"
        "[[speed_bump]]\nposition = 40.0\n"
        "[demand]\ncar_rate = {min = 12, max = 12, mean = 12, std = 0}\nbike_rate = {min = 0, max = 0, mean = 0, std = 0}\n"
        "bus_rate = {min = 0, max = 0, mean = 0, std = 0}\n"
        "[vehicles.car]\naccel = 2.5\ndecel = 4.5\nspeed_bump_kmh = 10.0\n"
        "[execution]\nrun = 6\nreplicas = 2\nworkers = 2\nseed = 7\n[output]\nprogress = false\n",
        encoding="utf-8",
    )
    return tmp_path


def test_command_runs_the_light_sweep_and_redraws(root, capsys):
    folder = emissions(["--light-spacing", "40", "100", "--replicas", "1", "--cycle", "20/20/2"])
    names = {p.name for p in folder.iterdir()}
    assert {"light_spacing.png", "light_spacing.csv", PROFILE_CSV, DATA_NAME, META_NAME, SUMMARY_NAME,
            "base_config.toml", "results.json"} <= names  # fmt: skip
    meta = json.loads((folder / META_NAME).read_text())
    assert meta["elemento"] == "semaforo" and meta["primer_semaforo_m"] == 100.0 and meta["desfase"] == "rojo"
    assert meta["ciclo"] == {"rojo_s": 20.0, "verde_s": 20.0, "amarillo_s": 2.0} and meta["topes_fijos_m"] == []
    assert meta["tipo_peloton"] and len(meta["viaje_s"]) == 2 and meta["viaje_s"][0] < meta["viaje_s"][1]
    assert meta["largo_m"] == 350.0  # 100 + 100 + 150, extended from 150 m
    assert meta["escenarios"][:2] == ["no traffic light", "one traffic light at 100 m"]
    out = capsys.readouterr().out
    assert "the segment is extended from 150 to 350 m" in out and "Minimum spacing between traffic lights" in out
    header = (folder / "light_spacing.csv").read_text().splitlines()[0]
    assert "excess_between_lights_pct_peak" in header and header.endswith("delay_x_isolated")
    doc = read_results(folder)
    assert doc["modo"] == "separacion" and doc["metricas"]["elemento"] == "semaforo"
    assert "huella_antes_m" in doc["metricas"] and len(doc["metricas"]["cola_hasta_la_entrada"]) == 2
    emissions(["--redraw", str(folder)])
    assert (folder / "light_spacing-2.png").stat().st_size > 10_000


def test_command_mixed_chain(root, capsys):
    folder = emissions(["--light-spacing", "40", "--chain", "3", "--fixed-spacings", "120", "--offset",
                        "green,red", "--replicas", "1", "--cycle", "20/20/2"])  # fmt: skip
    meta = json.loads((folder / META_NAME).read_text())
    assert meta["desfase"] == "verde,rojo" and meta["separaciones_fijas_m"] == [120.0]
    assert meta["escenarios"][2] == "traffic lights at 100, 140 and 260 m (d = 40 m)" and meta["largo_m"] == 410.0
    summary = (folder / SUMMARY_NAME).read_text()
    assert "2nd green when the platoon arrives (green wave), 3rd red when the platoon arrives" in summary
    assert "≡ isolated" in summary and "The chain of 3 emits like" in summary and "Minimum spacing" not in summary


def test_command_light_sweep_with_fixed_bumps_and_given_length(root):
    folder = emissions(["--light-spacing", "40", "200", "--replicas", "1", "--bumps", "40", "--length", "250",
                        "--offset", "same"])  # fmt: skip
    meta = json.loads((folder / META_NAME).read_text())
    assert meta["topes_fijos_m"] == [40.0] and meta["largo_m"] == 250.0 and meta["omitidas_m"] == [200.0]
    assert meta["desfase"] == "igual"
    assert meta["ciclo"]["rojo_s"] == LIGHT_CYCLE.red


@pytest.mark.parametrize(
    "args",
    [
        ["--spacing", "20", "--light-spacing", "50"],
        ["--cycle", "30/30"],
        ["--offset", "red"],
        ["--first-light", "50"],
        ["--light-spacing", "50", "--light", "yes"],
        ["--light-spacing", "50", "--cycle", "30"],
        ["--light-spacing", "50", "--bumps", "40", "none"],
        ["--light-spacing", "0"],
        ["--light-spacing", "50", "--offset", "azul"],
        ["--light-spacing", "50", "--fixed-spacings", "100"],
        ["--light-spacing", "50", "--chain", "3", "--offset", "green,red,red"],
        ["--fixed-spacings", "100"],
    ],
)
def test_command_rejects_invalid_light_options(root, args):
    with pytest.raises(SystemExit):
        emissions(args)
