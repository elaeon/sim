import json

import numpy as np
import pytest

from trafico.bump_spacing import (
    DATA_NAME, META_NAME, PLOT_NAME, PROFILE_CSV, SUMMARY_NAME, TABLE_CSV, Spacing, analyze, build_configs,
)  # fmt: skip
from trafico.cli import CONFIG_NAME, emissions
from trafico.config import POLLUTANTS, SimConfig, SpeedBump
from trafico.emissions import EMIS_BIN
from trafico.settings import ConfigError

BINS = 40  # 200 m de calle
X = (np.arange(BINS) + 0.5) * EMIS_BIN
FIRST = 20.0
WAKE = 40.0  # m de la estela


def _wake(at: float) -> np.ndarray:
    """Exceso triangular de 600 g/(m·h) que baja a 0 en WAKE m tras el tope."""
    return 600 * np.clip(1 - (X - at) / WAKE, 0, 1) * (X >= at)


def _synthetic(distances, overlap: str = "max", count: int = 2) -> tuple[dict, dict]:
    """Perfiles sintéticos: base de 100 g/(m·h); un tope y, para cada d, una cadena de `count` topes cuyas estelas se
    solapan (max) o suman."""
    base = np.full(BINS, 100.0)

    def chain(d: float) -> np.ndarray:
        waves = [_wake(FIRST + i * d) for i in range(count)]
        return base + (np.maximum.reduce(waves) if overlap == "max" else np.add.reduce(waves))

    profs = [base, base + _wake(FIRST), *(chain(d) for d in distances)]
    pos = np.zeros((len(profs), len(POLLUTANTS), 1, BINS))
    for i, prof in enumerate(profs):
        pos[i, POLLUTANTS.index("co2"), 0] = prof * EMIS_BIN  # g en 1 h
    meta = {"contaminantes": ["co2"], "primer_tope_m": FIRST, "distancias_m": list(distances), "tolerancia": 0.02,
            "umbral": 0.1, "s_simulados": 3600.0, "topes_por_cadena": count}  # fmt: skip
    return meta, {"emissions_pos": pos}


def test_analysis_finds_the_wake_and_the_minimum_separation():
    meta, data = _synthetic([10.0, 30.0, 60.0])
    r = analyze(meta, data)
    co2 = r["pollutants"]["co2"]
    assert co2["huella_m"] == pytest.approx(37.5)  # primer intervalo bajo el 10 % del pico, contado desde el tope
    assert co2["pico"] == pytest.approx(562.5)
    # Con estelas solapadas la pareja emite menos que dos topes aislados; sin solape, lo mismo.
    assert co2["aditividad"][0] < co2["aditividad"][1] < 0.98 and co2["aditividad"][2] == pytest.approx(1.0)
    assert co2["d_aditiva"] == 60.0 and co2["d_huella"] == 60.0 and co2["d_minima"] == 60.0
    assert r["recomendada_m"] == 60.0 and r["huella_m"] == pytest.approx(37.5)
    # La media en el tramo baja con la separación; sin tramo entre topes (d = 10) no hay exceso entre ellos.
    assert co2["media_tramo"][0] > co2["media_tramo"][2] > 1.0
    assert np.isnan(co2["exceso_entre"][0])
    assert co2["exceso_entre"][2] < co2["exceso_entre"][1]
    # El gradiente es el |dE/dx| medio de todo el tramo: se define para toda separación y la estela triangular
    # (600 g/(m·h) en 40 m: 15 por m, con la caída al llegar el segundo tope) lo deja entre 15 y la pendiente máxima.
    grad = np.array(co2["gradiente_tramo"])
    assert np.isfinite(grad).all() and grad[0] == pytest.approx(15.0, rel=0.05)
    assert (grad >= 14.9).all() and (grad <= np.array(co2["pendiente_max"])).all()


def test_chain_cost_of_each_added_bump():
    # Sin solape (estelas que suman) cada tope añadido cuesta como uno aislado: R = 1 y costo 1 (100 / d % por metro).
    meta, data = _synthetic([60.0], overlap="sum", count=3)
    co2 = analyze(meta, data)["pollutants"]["co2"]
    assert co2["aditividad"][0] == pytest.approx(1.0) and co2["costo_tope"][0] == pytest.approx(1.0)
    assert co2["costo_por_m"][0] == pytest.approx(100 / 60)
    # Con estelas solapadas el tope añadido cuesta menos, y el costo crece con la separación.
    meta, data = _synthetic([10.0, 20.0, 30.0], count=4)
    r = analyze(meta, data)
    cost = np.array(r["pollutants"]["co2"]["costo_tope"])
    assert (cost < 1).all() and cost[0] < cost[1] < cost[2]
    assert r["pollutants"]["co2"]["costo_por_m"][0] == pytest.approx(100 * cost[0] / 10)
    # En una pareja (n = 2) el costo del segundo tope es 2R − 1.
    meta, data = _synthetic([10.0, 30.0])
    co2 = analyze(meta, data)["pollutants"]["co2"]
    assert np.allclose(co2["costo_tope"], 2 * np.array(co2["aditividad"]) - 1)


def test_analysis_without_a_sufficient_separation_gives_none():
    meta, data = _synthetic([10.0, 20.0, 30.0])
    r = analyze(meta, data)
    assert r["pollutants"]["co2"]["d_minima"] is None and r["recomendada_m"] is None
    # Sumando estelas (no hay solape que reste), toda separación es aditiva; aun así el segundo tope debe quedar fuera de la huella.
    meta, data = _synthetic([30.0, 60.0], overlap="sum")
    r = analyze(meta, data)["pollutants"]["co2"]
    assert r["d_aditiva"] == 30.0 and r["d_huella"] == 60.0 and r["d_minima"] == 60.0


# ---------------------------------------------------------------- escenarios


def _base(**kw) -> SimConfig:
    return SimConfig(length=150, lanes=2, rates=(10, 0, 0), run=3, traffic_light=False, **kw)


def _spacing(**kw) -> Spacing:
    args = dict(first=30.0, distances=(20.0, 60.0), bump_lanes=None, light=None, length=None, tolerance=0.05,
                threshold=0.1, replicas=1, run=3.0)  # fmt: skip
    return Spacing(**{**args, **kw})


def test_build_configs_orders_scenarios_and_copies_the_configured_bump():
    bump = SpeedBump(30.0, (0,), pedestrian=True, pedestrian_time=6.0)
    configs, skipped = build_configs(_base(speed_bumps=(bump,)), _spacing(distances=(20.0, 60.0, 140.0)))
    assert skipped == [140.0]  # 30 + 140 queda a menos de un intervalo del final del tramo
    assert [n for n, _ in configs][:2] == ["sin tope", "un tope a 30 m"]
    assert [c.speed_bumps for _, c in configs][0] == () and configs[1][1].speed_bumps == (bump,)
    second = configs[2][1].bumps
    assert [b.position for b in second] == [30.0, 50.0] and second[1] == SpeedBump(50.0, (0,), True, bump.pedestrian_crossing, 6.0)
    assert [b.position for b in configs[3][1].bumps] == [30.0, 90.0] and "d = 60 m" in configs[3][0]
    # --carriles-tope reemplaza los carriles de todos los topes
    laned, _ = build_configs(_base(speed_bumps=(bump,)), _spacing(bump_lanes=(1,)))
    assert all(b.lanes == (1,) for b in laned[2][1].bumps)


def test_build_configs_chain():
    configs, skipped = build_configs(_base(), _spacing(first=20.0, distances=(10.0, 30.0, 60.0), count=4))
    assert skipped == [60.0]  # 20 + 3 × 60 = 200 no cabe en 150 m
    assert [n for n, _ in configs][:3] == ["sin tope", "un tope a 20 m", "4 topes desde 20 m cada 10 m"]
    assert [b.position for b in configs[2][1].bumps] == [20.0, 30.0, 40.0, 50.0]
    assert [b.position for b in configs[3][1].bumps] == [20.0, 50.0, 80.0, 110.0]
    with pytest.raises(ConfigError, match="al menos 2 topes"):
        build_configs(_base(), _spacing(count=1))
    with pytest.raises(ConfigError, match="el último de 3 topes"):
        build_configs(_base(), _spacing(first=100.0, distances=(30.0,), count=3))


def test_build_configs_errors():
    with pytest.raises(ConfigError, match="ninguna separación cabe en el tramo"):
        build_configs(_base(), _spacing(distances=(200.0,)))
    with pytest.raises(ConfigError, match="--primer-tope debe estar entre 0 y 150 m"):
        build_configs(_base(), _spacing(first=150.0))
    configs, _ = build_configs(_base(), _spacing(light=True, length=100.0, distances=(20.0,)))  # --largo y --semaforo
    assert configs[0][1].length == 100.0 and configs[0][1].traffic_light


# ---------------------------------------------------------------- comando


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr("trafico.settings.project_root", lambda: tmp_path)
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
    folder = emissions(["--separacion", "20", "50", "--replicas", "1"])
    names = {p.name for p in folder.iterdir()}
    assert {PLOT_NAME, TABLE_CSV, PROFILE_CSV, DATA_NAME, META_NAME, SUMMARY_NAME, "config_base.toml"} <= names
    meta = json.loads((folder / META_NAME).read_text())
    assert meta["modo"] == "separacion" and meta["primer_tope_m"] == 40.0 and meta["distancias_m"] == [20.0, 50.0]
    assert meta["escenarios"][:2] == ["sin tope", "un tope a 40 m"] and len(meta["escenarios"]) == 4
    header = (folder / TABLE_CSV).read_text().splitlines()[0]
    assert header.startswith("contaminante,separacion_m,aditividad") and len((folder / TABLE_CSV).read_text().splitlines()) > 4
    out = capsys.readouterr().out
    assert "4 escenarios × 1 réplicas" in out and "Separación mínima entre topes" in out
    assert (folder / PLOT_NAME).stat().st_size > 10_000
    emissions(["--redibujar", str(folder)])
    assert (folder / "separacion_topes-2.png").stat().st_size > 10_000


def test_command_runs_a_chain(root, capsys):
    folder = emissions(["--separacion", "20", "--cadena", "3", "--replicas", "1"])
    meta = json.loads((folder / META_NAME).read_text())
    assert meta["topes_por_cadena"] == 3 and meta["escenarios"][2] == "3 topes desde 40 m cada 20 m"
    assert "cadena de 3 topes" in (folder / SUMMARY_NAME).read_text()
    header = (folder / TABLE_CSV).read_text().splitlines()[0]
    assert "costo_tope_anadido" in header and "costo_pct_por_m" in header
    assert (folder / PLOT_NAME).stat().st_size > 10_000
    emissions(["--redibujar", str(folder)])
    assert (folder / "separacion_topes-2.png").stat().st_size > 10_000


@pytest.mark.parametrize(
    "args",
    [
        ["--cadena", "3"],
        ["--separacion", "20", "--cadena", "1"],
        ["--separacion", "20", "--topes", "30"],
        ["--separacion", "20", "--semaforo", "ambos"],
        ["--separacion", "0"],
        ["--separacion", "20", "20"],
        ["--separacion", "20", "--primer-tope", "500"],
        ["--separacion", "300"],
        ["--separacion", "20", "--tolerancia", "2"],
    ],
)
def test_command_rejects_invalid_options(root, args):
    with pytest.raises(SystemExit):
        emissions(args)


def test_default_first_bump_is_a_quarter_of_the_street_without_configured_bumps(root):
    text = (root / CONFIG_NAME).read_text().replace("[[speed_bump]]\nposition = 40.0\n", "")
    (root / CONFIG_NAME).write_text(text, encoding="utf-8")
    folder = emissions(["--separacion", "30", "--replicas", "1"])
    meta = json.loads((folder / META_NAME).read_text())
    assert meta["primer_tope_m"] == 150.0 / 4
