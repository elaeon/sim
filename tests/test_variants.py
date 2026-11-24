import json
from dataclasses import replace

import numpy as np
import pytest

from traffic.cli import CONFIG_NAME, _lights, latest_run_dir, variants
from traffic.config import BUS, CAR, DEFAULT_SPECS, Bottleneck, Light, SimConfig, SpeedBump
from traffic.settings import ConfigError
from traffic.variants import (
    BASE_CONFIG_NAME, DATA_NAME, META_NAME, PLOT_NAME, QUEUE_CSV, SPEED_CSV, SUMMARY_NAME, Variants, light_name,
    reduce_config, run_variant_replica, scenario,
)  # fmt: skip


def _base() -> SimConfig:
    """4 lanes as in config.toml: bikes (0) and buses (3) exclusive; cars in 1 and 2."""
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
    assert cfg.bottleneck_lanes() == (0,)  # lane 1 of the base is 0
    assert cfg.rates == (18, 0.0, 0.0)
    # Types that do not take part leave no reserved lanes or stops.
    assert cfg.reserved_lanes == frozenset() and cfg.specs[BUS].stop_position is None
    assert cfg.specs[CAR].bottleneck_prob == 0.3


def test_reduce_config_keeps_exit_queue_and_free_lanes_of_the_chosen_lanes():
    base = replace(_base(), exit_capacity=(0.0, 6.0, 8.0, 2.0), exit_storage=(10.0, 55.0, 60.0, 20.0),
                   free_lanes=(0, 3))  # exclusive lanes of bikes and buses  # fmt: skip
    cfg = reduce_config(base, (3, 1), ("bike",))
    assert cfg.lane_exit_capacity == (2.0, 6.0) and cfg.lane_exit_storage == (20.0, 55.0)
    assert cfg.free_lanes == (0,)  # lane 3 of the base (buses) is 0; the bikes' lane is gone
    # Without the buses, their lane is no longer exclusive and no longer without a traffic light.
    assert reduce_config(base, (3, 1), ("bike", "bus")).free_lanes == ()


def test_reduce_config_keeps_the_speed_bump_of_the_chosen_lanes():
    base = replace(_base(), speed_bumps=(SpeedBump(100.0, (1, 2)), SpeedBump(150.0, (0,))))
    assert reduce_config(base, (2, 3), ("bike",)).speed_bumps == (SpeedBump(100.0, (0,)),)  # the one at 150 m loses its only lane
    assert reduce_config(base, (3,), ("car", "bike")).speed_bumps == ()  # none of its lanes: no bump
    everywhere = replace(_base(), speed_bumps=(SpeedBump(100.0),))
    reduced = reduce_config(everywhere, (1, 2), ("bike", "bus"))
    assert reduced.speed_bumps == (SpeedBump(100.0),) and reduced.bump_lanes(reduced.speed_bumps[0]) == (0, 1)


def test_reduce_config_errors():
    with pytest.raises(ConfigError, match="--without: there is no vehicle type 'truck'"):
        reduce_config(_base(), None, ("truck",))
    with pytest.raises(ConfigError, match="--lanes: each lane must be between 0 and 3"):
        reduce_config(_base(), (1, 7), ())
    with pytest.raises(ConfigError, match=r"\[vehicles.bike\] lane = 0 is not in --lanes"):
        reduce_config(_base(), (1, 2), ("bus",))  # the bikes take part with their lane 0


def test_scenario_is_validated():
    v = Variants(lengths=(40.0,), lights=((30.0, 30.0),), lanes=(1, 2), without=(), queue_types=("car",),
                 replicas=2, run=10.0)  # fmt: skip
    reduced = reduce_config(_base(), (1, 2, 0, 3), ())
    with pytest.raises(ConfigError, match="40 m segment, light 30/30: .*stop_position"):
        scenario(reduced, v, 40.0, 30.0, 30.0)  # the bus stop (250 m) does not fit


def test_light_names_and_default_splits():
    assert [light_name(r, g) for r, g in ((40, 20), (30, 30), (20, 40))] == ["red > green", "red = green", "green > red"]
    assert _lights(None, 30.0, 30.0) == ((40.0, 20.0), (30.0, 30.0), (20.0, 40.0))
    assert _lights(["45/15", "15.5/44.5"], 0, 0) == ((45.0, 15.0), (15.5, 44.5))
    with pytest.raises(ConfigError, match="RED/GREEN"):
        _lights(["45-15"], 0, 0)


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr("traffic.settings.project_root", lambda: tmp_path)
    (tmp_path / CONFIG_NAME).write_text(
        "[road]\nmax_line_speed = [20, 30, 50]\n[demand]\ncar_rate = {min = 30, max = 30, mean = 30, std = 0}\nbike_rate = {min = 8, max = 8, mean = 8, std = 0}\nbus_rate = {min = 1, max = 1, mean = 1, std = 0}\n"
        "[vehicles.bike]\nlane = 0\nexclusive = true\n[execution]\nrun = 2\nreplicas = 3\nworkers = 2\nseed = 7\n"
        "[output]\nprogress = false\n",
        encoding="utf-8",
    )
    return tmp_path


def test_command_writes_the_comparison(root, capsys):
    argv = ["prueba", "--lengths", "60", "100", "--lights", "20/10", "10/20", "--lanes", "1", "2",
            "--without", "bike", "bus", "--replicas", "2"]  # fmt: skip
    folder = variants(argv)
    assert folder.parent == root / "results" and folder.name.endswith("_prueba")
    files = {p.name for p in folder.iterdir()}
    assert {PLOT_NAME, DATA_NAME, META_NAME, SPEED_CSV, QUEUE_CSV, SUMMARY_NAME, BASE_CONFIG_NAME} <= files
    assert CONFIG_NAME not in files and latest_run_dir(folder.parent) is None  # not a `traffic` run
    meta = json.loads((folder / META_NAME).read_text())
    assert meta["largos_m"] == [60.0, 100.0] and meta["carriles"] == ["lane 0 (30 km/h)", "lane 1 (50 km/h)"]
    assert meta["semilla"] == 7 and meta["replicas"] == 2 and meta["cola_tipos"] == [CAR]
    with np.load(folder / DATA_NAME) as d:
        assert d["speed"].shape == (2, 2, 2, 2) and d["queue"].shape[:3] == (2, 2, 2)
        speed = d["speed"]
    # Replica 2 of the 100 m segment with 10/20 is the same simulation as with that seed.
    v = Variants((100.0,), ((10.0, 20.0),), (1, 2), ("bike", "bus"), ("car",), 2, 2.0)
    from traffic.settings import load_settings

    base = load_settings(root / CONFIG_NAME).sim
    cfg = scenario(reduce_config(base, v.lanes, v.without), v, 100.0, 10.0, 20.0)
    one, _, _ = run_variant_replica(cfg, np.random.SeedSequence(7).spawn(2)[1])
    np.testing.assert_allclose(speed[1, 1, 1], one)
    out = capsys.readouterr().out
    assert "4 scenarios × 2 replicas" in out and "red > green (20/10)" in out

    # Redrawing neither simulates nor overwrites.
    variants(["--redraw", str(folder)])
    assert (folder / "light_variants-2.png").stat().st_size > 10_000
    assert (folder / PLOT_NAME).stat().st_size > 10_000


def test_command_errors(root):
    with pytest.raises(SystemExit):
        variants(["--lanes", "1", "2", "--without", "bus"])  # the bikes have lane 0
    with pytest.raises(SystemExit):
        variants(["--redraw", str(root)])  # not a comparison


def test_reduce_config_keeps_the_free_lanes_of_each_light():
    base = replace(_base(), extra_lights=(Light(100.0, free_lanes=(0, 3)),))
    cfg = reduce_config(base, (3, 1), ("bike",))
    assert cfg.extra_lights[0].free_lanes == (0,)  # lane 3 (buses) is 0; the bikes' lane is gone
    assert reduce_config(base, (3, 1), ("bike", "bus")).extra_lights[0].free_lanes == ()


def test_scenario_varies_only_the_light_at_the_end():
    v = Variants(lengths=(200.0,), lights=((20.0, 40.0),), lanes=(1, 2), without=(), queue_types=("car",),
                 replicas=1, run=10.0)  # fmt: skip
    base = replace(reduce_config(_base(), (1, 2, 0, 3), ("bus",)), extra_lights=(Light(90.0, red=12, green=8),))
    cfg = scenario(base, v, 200.0, 20.0, 40.0)
    assert (cfg.red, cfg.green) == (20.0, 40.0) and cfg.extra_lights == (Light(90.0, red=12, green=8),)
    with pytest.raises(ConfigError, match="60 m segment, light 20/40: .*each intermediate light"):
        scenario(base, replace(v, lengths=(60.0,)), 60.0, 20.0, 40.0)  # the 90 m one no longer fits


def test_scenario_replaces_a_pedestrian_light_at_the_end_with_a_fixed_cycle_one():
    """traffic-variants varies the red/green split of the end of the segment: there the light is fixed-cycle, even if it is
    a pedestrian light in the configuration; the intermediate pedestrian lights are kept."""
    from traffic.config import Rate

    pedestrian = Light(100.0, red=12.0, green=15.0, yellow=3.0, pedestrian=True, pedestrian_crossing=Rate(2, 2, 2, 0))
    base = replace(_base(), light_pedestrian=True, extra_lights=(pedestrian,), length=300)
    reduced = reduce_config(base, None, ())
    assert reduced.extra_lights == (pedestrian,) and reduced.exit_light.pedestrian
    v = Variants(lengths=(300.0,), lights=((20.0, 40.0),), lanes=None, without=(), queue_types=(), replicas=1, run=2.0)
    cfg = scenario(reduced, v, 300.0, 20.0, 40.0)
    assert not cfg.exit_light.pedestrian and cfg.ref_light.position == 300.0 and cfg.cycle == 60.0
    assert cfg.extra_lights == (pedestrian,)
