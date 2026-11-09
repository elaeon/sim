"""El kernel compilado (kernel.py) visto desde Simulation: paso a paso o de corrido da lo mismo, los arreglos crecen
cuando hace falta y una corrida corta da exactamente los números del motor anterior, escrito con numpy."""

from dataclasses import replace

import numpy as np
import pytest

from trafico import kernel
from trafico.config import CAR, DEFAULT_SPECS, Light, SimConfig, SpeedBump
from trafico.engine import Simulation

CAR_SPEC, BIKE_SPEC, BUS_SPEC = DEFAULT_SPECS
GRADUAL = replace(CAR_SPEC, accel=2.0, decel=3.0, speed_bump_kmh=15.0, speed_std=5.0)
# 1 200 pasos (cruza el límite de un bloque de llegadas), con dinámica gradual, condición inicial, cola de salida, un
# semáforo intermedio con desfase y un tope con paso peatonal.
STREET = SimConfig(length=200, lanes=2, rates=(20, 4, 1), specs=(GRADUAL, BIKE_SPEC, BUS_SPEC), red=20, green=25,
                   yellow=3, run=12, time_scale=10, initial_occupancy=0.3, exit_capacity=15.0, exit_storage=30.0,
                   extra_lights=(Light(80.0, red=15, green=20, yellow=3, offset=5.0),),
                   speed_bumps=(SpeedBump(40.0, pedestrian=True),))  # fmt: skip


def test_the_street_spans_more_than_one_block_of_arrivals():
    assert STREET.n_ticks > kernel.ARRIVAL_CHUNK


def test_step_by_step_is_the_same_as_a_whole_run():
    whole = Simulation(STREET, np.random.default_rng(3))
    whole.run()
    steps = Simulation(STREET, np.random.default_rng(3))
    for _ in range(STREET.n_ticks):
        steps.step()
    for key, value in whole.summary().items():
        np.testing.assert_array_equal(value, steps.summary()[key], err_msg=key)
    np.testing.assert_array_equal(whole.recorder.emis, steps.recorder.emis)
    np.testing.assert_array_equal(whole.x[: whole.n], steps.x[: steps.n])


def test_a_short_run_matches_the_numpy_engine():
    """Valores de la misma corrida con el motor anterior (en numpy, antes de kernel.py): son idénticos bit a bit."""
    sim = Simulation(STREET, np.random.default_rng(11))
    sim.run()
    s = sim.summary()
    assert s["crossed_veh"].tolist() == [21.0, 3.0, 2.0]
    assert s["crossed_pax"].tolist() == [38.0, 3.0, 87.0]
    assert s["travel_time"].tolist() == [69.56666666666668, 79.10000000000001, 73.8]
    assert s["veh_km"].tolist() == [5.572621075498055, 1.0112064809482293, 0.5774901225704506]
    assert s["lane_crossed"].tolist() == [14.0, 12.0]
    assert s["queue_wait"].tolist() == [0.010000000000000002, 0.03333333333333333, 0.5666666666666668]
    assert s["emissions"][CAR, 0] == 4076.3924916189603
    assert sim.lane_changes == 26
    assert sim.n == 33 and sim.x[13] == 148.61511908288892


def test_entry_queues_grow_and_keep_the_arrival_order():
    """Con más llegadas de las que caben, los anillos de las colas de entrada crecen sin perder a nadie ni el orden."""
    cfg = SimConfig(length=100, lanes=2, rates=(200, 20, 0), red=30, green=5, run=6, time_scale=10)
    sim = Simulation(cfg, np.random.default_rng(1))
    first_capacity = sim.q_vt.shape[1]
    sim.run()
    assert sim.q_vt.shape[1] > first_capacity
    s = sim.summary()
    np.testing.assert_array_equal(s["arrived_veh"], s["queued"] + s["entered_veh"])
    for queue in sim.queues:
        arrived = [tick for *_, tick in queue]
        assert arrived == sorted(arrived)
    assert sim.queue_len.sum() == s["queued"].sum() == sum(len(q) for q in sim.queues)


def test_vehicle_arrays_grow_keeping_their_contents():
    cfg = SimConfig(length=100, lanes=1, rates=(0, 0, 0), run=1)
    sim = Simulation(cfg, np.random.default_rng(0))
    capacity = sim.x.size
    for k in range(capacity + 5):
        sim._add(CAR, 1, 0, x=99.0 - 0.1 * k)
    assert sim.x.size > capacity and sim.n == capacity + 5
    np.testing.assert_allclose(sim.x[: sim.n], 99.0 - 0.1 * np.arange(sim.n))
    np.testing.assert_array_equal(sim.vid[: sim.n], np.arange(sim.n))
    sim.step()  # el kernel recibe los arreglos nuevos
    assert sim.tick == 1


@pytest.mark.parametrize("phase_light", [Light(0.0, red=3, green=4, yellow=1), Light(0.0, red=3, green=4, offset=2.0)])
def test_line_phases_follow_the_light_cycle(phase_light):
    """Las fases que usa el kernel son las de `Light.phase`, también con amarillo, desfase y pasos negativos."""
    cfg = SimConfig(length=100, lanes=1, rates=(0, 0, 0), run=1,
                    extra_lights=(replace(phase_light, position=50.0),))  # fmt: skip
    sim = Simulation(cfg, np.random.default_rng(0))
    light = cfg.inner_lights[0]
    for tick in range(-20, 200):
        assert sim.line_phases(tick) == (light.phase(tick),)
        assert sim.exit_phase(tick) == cfg.phase(tick)
