import json

import numpy as np
import pytest

from traffic.bump_spacing import (
    DATA_NAME, META_NAME, PLOT_NAME, PROFILE_CSV, SUMMARY_NAME, TABLE_CSV, Spacing, analyze, build_configs,
)  # fmt: skip
from traffic.cli import CONFIG_NAME, emissions
from traffic.config import POLLUTANTS, SimConfig, SpeedBump
from traffic.emissions import EMIS_BIN
from traffic.settings import ConfigError

BINS = 40  # 200 m of street
X = (np.arange(BINS) + 0.5) * EMIS_BIN
FIRST = 20.0
WAKE = 40.0  # m of the wake


def _wake(at: float) -> np.ndarray:
    """Triangular excess of 600 g/(m·h) that falls to 0 within WAKE m after the bump."""
    return 600 * np.clip(1 - (X - at) / WAKE, 0, 1) * (X >= at)


def _synthetic(distances, overlap: str = "max", count: int = 2) -> tuple[dict, dict]:
    """Synthetic profiles: base of 100 g/(m·h); one bump and, for each d, a chain of `count` bumps whose wakes
    overlap (max) or add up."""
    base = np.full(BINS, 100.0)

    def chain(d: float) -> np.ndarray:
        waves = [_wake(FIRST + i * d) for i in range(count)]
        return base + (np.maximum.reduce(waves) if overlap == "max" else np.add.reduce(waves))

    profs = [base, base + _wake(FIRST), *(chain(d) for d in distances)]
    pos = np.zeros((len(profs), len(POLLUTANTS), 1, BINS))
    for i, prof in enumerate(profs):
        pos[i, POLLUTANTS.index("co2"), 0] = prof * EMIS_BIN  # g in 1 h
    meta = {"contaminantes": ["co2"], "primer_tope_m": FIRST, "distancias_m": list(distances), "tolerancia": 0.02,
            "umbral": 0.1, "s_simulados": 3600.0, "topes_por_cadena": count}  # fmt: skip
    return meta, {"emissions_pos": pos}


def test_analysis_finds_the_wake_and_the_minimum_separation():
    meta, data = _synthetic([10.0, 30.0, 60.0])
    r = analyze(meta, data)
    co2 = r["pollutants"]["co2"]
    assert co2["huella_m"] == pytest.approx(37.5)  # first bin below 10 % of the peak, counted from the bump
    assert co2["pico"] == pytest.approx(562.5)
    # With overlapping wakes the pair emits less than two isolated bumps; without overlap, the same.
    assert co2["aditividad"][0] < co2["aditividad"][1] < 0.98 and co2["aditividad"][2] == pytest.approx(1.0)
    assert co2["d_aditiva"] == 60.0 and co2["d_huella"] == 60.0 and co2["d_minima"] == 60.0
    assert r["recomendada_m"] == 60.0 and r["huella_m"] == pytest.approx(37.5)
    # The mean over the stretch falls with the spacing; with no stretch between bumps (d = 10) there is no excess between them.
    assert co2["media_tramo"][0] > co2["media_tramo"][2] > 1.0
    assert np.isnan(co2["exceso_entre"][0])
    assert co2["exceso_entre"][2] < co2["exceso_entre"][1]
    # The gradient is the mean |dE/dx| over the whole stretch: it is defined for every spacing and the triangular wake
    # (600 g/(m·h) over 40 m: 15 per m, with the drop when the second bump arrives) leaves it between 15 and the maximum slope.
    grad = np.array(co2["gradiente_tramo"])
    assert np.isfinite(grad).all() and grad[0] == pytest.approx(15.0, rel=0.05)
    assert (grad >= 14.9).all() and (grad <= np.array(co2["pendiente_max"])).all()


def test_chain_cost_of_each_added_bump():
    # Without overlap (wakes that add up) each added bump costs like an isolated one: R = 1 and cost 1 (100 / d % per metre).
    meta, data = _synthetic([60.0], overlap="sum", count=3)
    co2 = analyze(meta, data)["pollutants"]["co2"]
    assert co2["aditividad"][0] == pytest.approx(1.0) and co2["costo_tope"][0] == pytest.approx(1.0)
    assert co2["costo_por_m"][0] == pytest.approx(100 / 60)
    # With overlapping wakes the added bump costs less, and the cost grows with the spacing.
    meta, data = _synthetic([10.0, 20.0, 30.0], count=4)
    r = analyze(meta, data)
    cost = np.array(r["pollutants"]["co2"]["costo_tope"])
    assert (cost < 1).all() and cost[0] < cost[1] < cost[2]
    assert r["pollutants"]["co2"]["costo_por_m"][0] == pytest.approx(100 * cost[0] / 10)
    # In a pair (n = 2) the cost of the second bump is 2R − 1.
    meta, data = _synthetic([10.0, 30.0])
    co2 = analyze(meta, data)["pollutants"]["co2"]
    assert np.allclose(co2["costo_tope"], 2 * np.array(co2["aditividad"]) - 1)


def test_analysis_without_a_sufficient_separation_gives_none():
    meta, data = _synthetic([10.0, 20.0, 30.0])
    r = analyze(meta, data)
    assert r["pollutants"]["co2"]["d_minima"] is None and r["recomendada_m"] is None
    # Adding wakes (no overlap to subtract), every spacing is additive; even so the second bump must be outside the footprint.
    meta, data = _synthetic([30.0, 60.0], overlap="sum")
    r = analyze(meta, data)["pollutants"]["co2"]
    assert r["d_aditiva"] == 30.0 and r["d_huella"] == 60.0 and r["d_minima"] == 60.0


# ---------------------------------------------------------------- scenarios


def _base(**kw) -> SimConfig:
    return SimConfig(length=150, lanes=2, rates=(10, 0, 0), run=3, traffic_light=False, **kw)


def _spacing(**kw) -> Spacing:
    args = dict(first=30.0, distances=(20.0, 60.0), bump_lanes=None, light=None, length=None, tolerance=0.05,
                threshold=0.1, replicas=1, run=3.0)  # fmt: skip
    return Spacing(**{**args, **kw})


def test_build_configs_orders_scenarios_and_copies_the_configured_bump():
    bump = SpeedBump(30.0, (0,), pedestrian=True, pedestrian_time=6.0)
    configs, skipped = build_configs(_base(speed_bumps=(bump,)), _spacing(distances=(20.0, 60.0, 140.0)))
    assert skipped == [140.0]  # 30 + 140 is less than one bin from the end of the segment
    assert [n for n, _ in configs][:2] == ["no bump", "one bump at 30 m"]
    assert [c.speed_bumps for _, c in configs][0] == () and configs[1][1].speed_bumps == (bump,)
    second = configs[2][1].bumps
    assert [b.position for b in second] == [30.0, 50.0] and second[1] == SpeedBump(50.0, (0,), True, bump.pedestrian_crossing, 6.0)
    assert [b.position for b in configs[3][1].bumps] == [30.0, 90.0] and "d = 60 m" in configs[3][0]
    # --bump-lanes replaces the lanes of all the bumps
    laned, _ = build_configs(_base(speed_bumps=(bump,)), _spacing(bump_lanes=(1,)))
    assert all(b.lanes == (1,) for b in laned[2][1].bumps)


def test_build_configs_chain():
    configs, skipped = build_configs(_base(), _spacing(first=20.0, distances=(10.0, 30.0, 60.0), count=4))
    assert skipped == [60.0]  # 20 + 3 × 60 = 200 does not fit in 150 m
    assert [n for n, _ in configs][:3] == ["no bump", "one bump at 20 m", "4 bumps from 20 m every 10 m"]
    assert [b.position for b in configs[2][1].bumps] == [20.0, 30.0, 40.0, 50.0]
    assert [b.position for b in configs[3][1].bumps] == [20.0, 50.0, 80.0, 110.0]
    with pytest.raises(ConfigError, match="at least 2 bumps"):
        build_configs(_base(), _spacing(count=1))
    with pytest.raises(ConfigError, match="the last of 3 bumps"):
        build_configs(_base(), _spacing(first=100.0, distances=(30.0,), count=3))


def test_build_configs_errors():
    with pytest.raises(ConfigError, match="no spacing fits on the segment"):
        build_configs(_base(), _spacing(distances=(200.0,)))
    with pytest.raises(ConfigError, match="--first-bump must be between 0 and 150 m"):
        build_configs(_base(), _spacing(first=150.0))
    configs, _ = build_configs(_base(), _spacing(light=True, length=100.0, distances=(20.0,)))  # --length and --light
    assert configs[0][1].length == 100.0 and configs[0][1].traffic_light


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


def test_command_runs_the_sweep_and_redraws(root, capsys):
    folder = emissions(["--spacing", "20", "50", "--replicas", "1"])
    names = {p.name for p in folder.iterdir()}
    assert {PLOT_NAME, TABLE_CSV, PROFILE_CSV, DATA_NAME, META_NAME, SUMMARY_NAME, "base_config.toml"} <= names
    meta = json.loads((folder / META_NAME).read_text())
    assert meta["modo"] == "separacion" and meta["primer_tope_m"] == 40.0 and meta["distancias_m"] == [20.0, 50.0]
    assert meta["escenarios"][:2] == ["no bump", "one bump at 40 m"] and len(meta["escenarios"]) == 4
    header = (folder / TABLE_CSV).read_text().splitlines()[0]
    assert header.startswith("pollutant,spacing_m,additivity") and len((folder / TABLE_CSV).read_text().splitlines()) > 4
    out = capsys.readouterr().out
    assert "4 scenarios × 1 replicas" in out and "Minimum spacing between bumps" in out
    assert (folder / PLOT_NAME).stat().st_size > 10_000
    emissions(["--redraw", str(folder)])
    assert (folder / "bump_spacing-2.png").stat().st_size > 10_000


def test_command_runs_a_chain(root, capsys):
    folder = emissions(["--spacing", "20", "--chain", "3", "--replicas", "1"])
    meta = json.loads((folder / META_NAME).read_text())
    assert meta["topes_por_cadena"] == 3 and meta["escenarios"][2] == "3 bumps from 40 m every 20 m"
    assert "chain of 3 bumps" in (folder / SUMMARY_NAME).read_text()
    header = (folder / TABLE_CSV).read_text().splitlines()[0]
    assert "added_bump_cost" in header and "cost_pct_per_m" in header
    assert (folder / PLOT_NAME).stat().st_size > 10_000
    emissions(["--redraw", str(folder)])
    assert (folder / "bump_spacing-2.png").stat().st_size > 10_000


@pytest.mark.parametrize(
    "args",
    [
        ["--chain", "3"],
        ["--spacing", "20", "--chain", "1"],
        ["--spacing", "20", "--bumps", "30"],
        ["--spacing", "20", "--light", "both"],
        ["--spacing", "0"],
        ["--spacing", "20", "20"],
        ["--spacing", "20", "--first-bump", "500"],
        ["--spacing", "300"],
        ["--spacing", "20", "--tolerance", "2"],
    ],
)
def test_command_rejects_invalid_options(root, args):
    with pytest.raises(SystemExit):
        emissions(args)


def test_default_first_bump_is_a_quarter_of_the_street_without_configured_bumps(root):
    text = (root / CONFIG_NAME).read_text().replace("[[speed_bump]]\nposition = 40.0\n", "")
    (root / CONFIG_NAME).write_text(text, encoding="utf-8")
    folder = emissions(["--spacing", "30", "--replicas", "1"])
    meta = json.loads((folder / META_NAME).read_text())
    assert meta["primer_tope_m"] == 150.0 / 4
