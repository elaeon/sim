import json
from dataclasses import replace

import numpy as np
import pytest

from trafico.cli import CONFIG_NAME, _lights, latest_run_dir, variants
from trafico.config import BUS, CAR, DEFAULT_SPECS, Bottleneck, SimConfig, SpeedBump
from trafico.settings import ConfigError
from trafico.variants import (
    BASE_CONFIG_NAME, DATA_NAME, META_NAME, PLOT_NAME, QUEUE_CSV, SPEED_CSV, SUMMARY_NAME, Variants, light_name,
    reduce_config, run_variant_replica, scenario,
)  # fmt: skip


def _base() -> SimConfig:
    """4 carriles como en config.toml: bicis (0) y autobuses (3) exclusivos; autos en 1 y 2."""
    car, bike, bus = DEFAULT_SPECS
    return SimConfig(
        length=300, lanes=4, lane_speed_limit=(20.0, 30.0, 50.0, 80.0), rates=(18, 8, 1),
        specs=(replace(car, bottleneck_prob=0.3), replace(bike, lane=0, exclusive=True),
               replace(bus, lane=3, exclusive=True, stop_position=250.0)),
        initial_occupancy=(0.0, 0.3, 0.2, 0.0),
        bottleneck=Bottleneck(stop_lanes=(1,)),
    )  # fmt: skip


def test_reduce_config_keeps_the_chosen_lanes():
    cfg = reduce_config(_base(), (1, 2), ("bike", "bus"))
    assert cfg.lanes == 2
    assert cfg.lane_max_kmh == (30.0, 50.0)
    assert cfg.lane_initial_occupancy == (0.3, 0.2)
    assert cfg.bottleneck_lanes() == (0,)  # el carril 1 de la base es el 0
    assert cfg.rates == (18, 0.0, 0.0)
    # Los tipos que no participan no dejan carriles reservados ni paradas.
    assert cfg.reserved_lanes == frozenset() and cfg.specs[BUS].stop_position is None
    assert cfg.specs[CAR].bottleneck_prob == 0.3


def test_reduce_config_keeps_exit_queue_and_free_lanes_of_the_chosen_lanes():
    base = replace(_base(), exit_capacity=(0.0, 6.0, 8.0, 2.0), exit_storage=(10.0, 55.0, 60.0, 20.0),
                   free_lanes=(0, 3))  # carriles exclusivos de bicis y autobuses  # fmt: skip
    cfg = reduce_config(base, (3, 1), ("bike",))
    assert cfg.lane_exit_capacity == (2.0, 6.0) and cfg.lane_exit_storage == (20.0, 55.0)
    assert cfg.free_lanes == (0,)  # el 3 de la base (autobuses) es el 0; el de las bicis no quedó
    # Sin los autobuses, su carril deja de ser exclusivo y deja de estar sin semáforo.
    assert reduce_config(base, (3, 1), ("bike", "bus")).free_lanes == ()


def test_reduce_config_keeps_the_speed_bump_of_the_chosen_lanes():
    base = replace(_base(), speed_bump=SpeedBump(position=100.0, lanes=(1, 2)))
    assert reduce_config(base, (2, 3), ("bike",)).speed_bump == SpeedBump(position=100.0, lanes=(0,))
    assert reduce_config(base, (0, 3), ("car",)).speed_bump_lanes() == ()  # ninguno de sus carriles: sin tope
    everywhere = replace(_base(), speed_bump=SpeedBump(position=100.0))
    assert reduce_config(everywhere, (1, 2), ("bike", "bus")).speed_bump_lanes() == (0, 1)


def test_reduce_config_errors():
    with pytest.raises(ConfigError, match="--sin: no existe el tipo 'truck'"):
        reduce_config(_base(), None, ("truck",))
    with pytest.raises(ConfigError, match="--carriles: cada carril debe estar entre 0 y 3"):
        reduce_config(_base(), (1, 7), ())
    with pytest.raises(ConfigError, match=r"\[vehicles.bike\] lane = 0 no está en --carriles"):
        reduce_config(_base(), (1, 2), ("bus",))  # las bicis participan con su carril 0


def test_scenario_is_validated():
    v = Variants(lengths=(40.0,), lights=((30.0, 30.0),), lanes=(1, 2), without=(), queue_types=("car",),
                 replicas=2, run=10.0)  # fmt: skip
    reduced = reduce_config(_base(), (1, 2, 0, 3), ())
    with pytest.raises(ConfigError, match="tramo de 40 m, semáforo 30/30: .*stop_position"):
        scenario(reduced, v, 40.0, 30.0, 30.0)  # la parada del autobús (250 m) no cabe


def test_light_names_and_default_splits():
    assert [light_name(r, g) for r, g in ((40, 20), (30, 30), (20, 40))] == ["rojo > verde", "rojo = verde", "verde > rojo"]
    assert _lights(None, 30.0, 30.0) == ((40.0, 20.0), (30.0, 30.0), (20.0, 40.0))
    assert _lights(["45/15", "15.5/44.5"], 0, 0) == ((45.0, 15.0), (15.5, 44.5))
    with pytest.raises(ConfigError, match="ROJO/VERDE"):
        _lights(["45-15"], 0, 0)


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr("trafico.settings.project_root", lambda: tmp_path)
    (tmp_path / CONFIG_NAME).write_text(
        "[road]\nmax_line_speed = [20, 30, 50]\n[demand]\ncar_rate = {min = 30, max = 30, mean = 30, std = 0}\nbike_rate = {min = 8, max = 8, mean = 8, std = 0}\nbus_rate = {min = 1, max = 1, mean = 1, std = 0}\n"
        "[vehicles.bike]\nlane = 0\nexclusive = true\n[execution]\nrun = 2\nreplicas = 3\nworkers = 2\nseed = 7\n"
        "[output]\nprogress = false\n",
        encoding="utf-8",
    )
    return tmp_path


def test_command_writes_the_comparison(root, capsys):
    argv = ["prueba", "--largos", "60", "100", "--semaforos", "20/10", "10/20", "--carriles", "1", "2",
            "--sin", "bike", "bus", "--replicas", "2"]  # fmt: skip
    folder = variants(argv)
    assert folder.parent == root / "resultados" and folder.name.endswith("_prueba")
    files = {p.name for p in folder.iterdir()}
    assert {PLOT_NAME, DATA_NAME, META_NAME, SPEED_CSV, QUEUE_CSV, SUMMARY_NAME, BASE_CONFIG_NAME} <= files
    assert CONFIG_NAME not in files and latest_run_dir(folder.parent) is None  # no es una corrida de `trafico`
    meta = json.loads((folder / META_NAME).read_text())
    assert meta["largos_m"] == [60.0, 100.0] and meta["carriles"] == ["carril 0 (30 km/h)", "carril 1 (50 km/h)"]
    assert meta["semilla"] == 7 and meta["replicas"] == 2 and meta["cola_tipos"] == [CAR]
    with np.load(folder / DATA_NAME) as d:
        assert d["speed"].shape == (2, 2, 2, 2) and d["queue"].shape[:3] == (2, 2, 2)
        speed = d["speed"]
    # La réplica 2 del tramo de 100 m con 10/20 es la misma simulación que con esa semilla.
    v = Variants((100.0,), ((10.0, 20.0),), (1, 2), ("bike", "bus"), ("car",), 2, 2.0)
    from trafico.settings import load_settings

    base = load_settings(root / CONFIG_NAME).sim
    cfg = scenario(reduce_config(base, v.lanes, v.without), v, 100.0, 10.0, 20.0)
    one, _, _ = run_variant_replica(cfg, np.random.SeedSequence(7).spawn(2)[1])
    np.testing.assert_allclose(speed[1, 1, 1], one)
    out = capsys.readouterr().out
    assert "4 escenarios × 2 réplicas" in out and "rojo > verde (20/10)" in out

    # Redibujar no simula ni sobrescribe.
    variants(["--redibujar", str(folder)])
    assert (folder / "variantes_semaforo-2.png").stat().st_size > 10_000
    assert (folder / PLOT_NAME).stat().st_size > 10_000


def test_command_errors(root):
    with pytest.raises(SystemExit):
        variants(["--carriles", "1", "2", "--sin", "bus"])  # las bicis tienen carril 0
    with pytest.raises(SystemExit):
        variants(["--redibujar", str(root)])  # no es una comparación
