import json

import numpy as np
import pytest

from trafico.cli import CONFIG_NAME, emissions, latest_run_dir
from trafico.config import CAR, POLLUTANTS
from trafico.emission_scenarios import (
    BASE_CONFIG_NAME, DATA_NAME, META_NAME, PLOT_NAME, POSITION_CSV, SUMMARY_NAME, TABLE_CSV,
)  # fmt: skip


@pytest.fixture
def root(tmp_path, monkeypatch):
    """Raíz de proyecto temporal: calle de 2 carriles, autos que emiten (con accel y decel) y un tope en su
    configuración."""
    monkeypatch.setattr("trafico.settings.project_root", lambda: tmp_path)
    (tmp_path / CONFIG_NAME).write_text(
        "[road]\nlength = 150.0\nmax_line_speed = [50, 50]\n[traffic_light]\nenabled = false\n"
        "[speed_bump]\nposition = 80.0\n"
        "[demand]\ncar_rate = {min = 12, max = 12, mean = 12, std = 0}\nbike_rate = {min = 0, max = 0, mean = 0, std = 0}\n"
        "bus_rate = {min = 0, max = 0, mean = 0, std = 0}\n"
        "[vehicles.car]\naccel = 2.5\ndecel = 4.5\nspeed_bump_kmh = 10.0\n"
        "[execution]\nrun = 3\nreplicas = 2\nworkers = 2\nseed = 7\n[output]\nprogress = false\n",
        encoding="utf-8",
    )
    return tmp_path


def test_command_compares_bumps(root, capsys):
    """Por defecto compara sin tope contra el tope de la configuración, con la misma semilla: los mismos autos
    llegan, pero con tope tardan más y emiten más CO2 por km. Escribe tablas, datos, parámetros y la gráfica, y
    redibuja sin simular ni sobrescribir."""
    folder = emissions(["prueba"])
    assert folder.parent == root / "resultados" and folder.name.endswith("_prueba")
    files = {p.name for p in folder.iterdir()}
    assert {PLOT_NAME, DATA_NAME, META_NAME, TABLE_CSV, POSITION_CSV, SUMMARY_NAME, BASE_CONFIG_NAME} <= files
    assert CONFIG_NAME not in files and latest_run_dir(folder.parent) is None  # no es una corrida de `trafico`
    meta = json.loads((folder / META_NAME).read_text())
    assert meta["escenarios"] == ["sin tope", "tope a 80 m"] and meta["topes_m"] == [[], [80.0]]
    assert meta["contaminantes"] == ["co2", "nox", "voc"] and meta["semilla"] == 7
    with np.load(folder / DATA_NAME) as d:
        arrived = d["crossed_veh"][:, CAR]
        co2 = d["emissions"][:, CAR, POLLUTANTS.index("co2")] / d["veh_km"][:, CAR]
        travel = d["travel_time"][:, CAR]
        assert d["emissions_pos"].shape[:2] == (2, len(POLLUTANTS)) and d["flow"].shape[0] == 2
    assert abs(arrived[0] - arrived[1]) <= 2 and travel[1] > travel[0] and co2[1] > 1.2 * co2[0]
    out = capsys.readouterr().out
    assert "2 escenarios × 2 réplicas" in out and "auto: CO2 (g/km)" in out
    assert "co2_cambio_pct" in (folder / TABLE_CSV).read_text().splitlines()[0]

    emissions(["--redibujar", str(folder)])
    assert (folder / "comparacion_emisiones-2.png").stat().st_size > 10_000
    assert (folder / PLOT_NAME).stat().st_size > 10_000


def test_command_with_and_without_light(root):
    folder = emissions(["--semaforo", "ambos", "--topes", "sin", "60", "--replicas", "1"])
    meta = json.loads((folder / META_NAME).read_text())
    assert meta["escenarios"] == ["sin tope · con semáforo", "tope a 60 m · con semáforo",
                                  "sin tope · sin semáforo", "tope a 60 m · sin semáforo"]  # fmt: skip
    assert meta["semaforo"] == [True, True, False, False]


def test_command_errors(root):
    with pytest.raises(SystemExit):
        emissions(["--topes", "sin", "cien"])
    with pytest.raises(SystemExit):
        emissions(["--topes", "500"])  # fuera del tramo
    with pytest.raises(SystemExit):
        emissions(["--redibujar", str(root)])  # no es una comparación
    (root / CONFIG_NAME).write_text((root / CONFIG_NAME).read_text().replace("accel = 2.5\n", ""), encoding="utf-8")
    with pytest.raises(SystemExit):
        emissions([])  # sin accel, los autos no emiten


def test_scenarios_with_several_bumps(root):
    """--topes 30,60 es un escenario con dos topes (los de [speed_bump] también pueden ser una lista)."""
    from trafico.cli import _bumps
    from trafico.emission_scenarios import scenario_label

    assert _bumps(["sin", "30,60", "45"], None, 100.0) == ((), (30.0, 60.0), (45.0,))
    assert _bumps(None, (70.0, 20.0), 100.0) == ((), (20.0, 70.0))
    assert _bumps(None, None, 100.0) == ((), (50.0,))
    assert scenario_label((60.0, 30.0), None) == "topes a 30 y 60 m" and scenario_label((), True) == "sin tope · con semáforo"
    assert scenario_label(45.0, False) == "tope a 45 m · sin semáforo" and scenario_label(None, None) == "sin tope"
    with pytest.raises(SystemExit):
        emissions(["--topes", "30;60"])
    folder = emissions(["--topes", "sin", "30,60", "--replicas", "1"])
    meta = json.loads((folder / META_NAME).read_text())
    assert meta["escenarios"] == ["sin tope", "topes a 30 y 60 m"] and meta["topes_m"] == [[], [30.0, 60.0]]
    with np.load(folder / DATA_NAME) as d:
        co2 = d["emissions"][:, CAR, POLLUTANTS.index("co2")] / d["veh_km"][:, CAR]
    assert co2[1] > 1.2 * co2[0]


def test_semaforo_option_switches_every_light(root):
    """--semaforo no/si apaga o enciende todos los semáforos, también los intermedios."""
    from dataclasses import replace

    from trafico.config import Light, SimConfig
    from trafico.emission_scenarios import Scenarios, build_configs

    base = SimConfig(length=150, lanes=2, rates=(10, 0, 0), run=3, traffic_light=True, extra_lights=(Light(60.0),),
                     specs=tuple(replace(s, accel=2.5, decel=4.5) if s.key == "car" else s
                                 for s in SimConfig().specs))  # fmt: skip
    s = Scenarios(bumps=(None,), bump_lanes=None, lights=(True, False), length=None, lanes=None, without=(),
                  replicas=1, run=3.0)  # fmt: skip
    (on_name, on), (off_name, off) = build_configs(base, s)
    assert on.has_light and on.inner_lights and not off.has_light and not off.inner_lights and not off.any_light
    assert on_name == "sin tope · con semáforo" and off_name == "sin tope · sin semáforo"
