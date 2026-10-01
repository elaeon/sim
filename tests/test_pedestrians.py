from dataclasses import replace

import numpy as np
import pytest

from trafico.config import (
    CAR, DEFAULT_SPECS, DT, GREEN, RED, YELLOW, Behavior, Light, Rate, SimConfig, SpeedBump,
)  # fmt: skip
from trafico.engine import Simulation
from trafico.pedestrians import Schedule, arrivals, bump_schedule, fixed_phases, light_schedule

N = 1000


def _counts(*events: tuple[int, int], n: int = N) -> np.ndarray:
    counts = np.zeros(n, np.int64)
    for tick, m in events:
        counts[tick] = m
    return counts


def _at(phases: np.ndarray, a: int, b: int) -> bool:
    return bool((phases[a:b] == phases[a]).all())


# ---------------------------------------------------------------- paso peatonal de un tope


def test_bump_schedule_pedestrians_during_the_crossing_extend_it():
    """El peatón de t=10 espera 30 pasos (amarillo) y cruza 100 (rojo, hasta 140); el de t=50 se une y lo alarga hasta
    150; el de t=400 abre otro cierre."""
    s = bump_schedule(_counts((10, 1), (50, 1), (400, 1)), cross_ticks=100, yield_ticks=30, n=N)
    ph = s.phases
    assert ph.size == N + 1 and ph.dtype == np.int8
    assert _at(ph, 0, 10) and ph[0] == GREEN
    assert _at(ph, 10, 40) and ph[10] == YELLOW
    assert _at(ph, 40, 150) and ph[40] == RED
    assert _at(ph, 150, 400) and ph[150] == GREEN
    assert ph[400] == YELLOW and ph[430] == RED and ph[529] == RED and ph[530] == GREEN
    assert (s.crossed, s.crossings) == (3, 2)
    assert s.closed_ticks == 110 + 100
    assert s.wait_ticks == 30 + 0 + 30  # el de t=50 ya cruza al llegar


def test_bump_schedule_pedestrians_waiting_at_the_curb_cross_together():
    s = bump_schedule(_counts((10, 2), (20, 1)), cross_ticks=100, yield_ticks=30, n=N)
    assert (s.crossed, s.crossings) == (3, 1)
    assert s.phases[40] == RED and s.phases[139] == RED and s.phases[140] == GREEN  # el fin no se mueve
    assert s.wait_ticks == 2 * 30 + 1 * 20


def test_bump_schedule_without_pedestrians_is_always_green():
    s = bump_schedule(_counts(), cross_ticks=100, yield_ticks=30, n=N)
    assert (s.phases == GREEN).all() and (s.crossed, s.crossings, s.closed_ticks, s.wait_ticks) == (0, 0, 0, 0)


def test_bump_schedule_clips_at_the_horizon():
    s = bump_schedule(_counts((90, 1), n=100), cross_ticks=100, yield_ticks=30, n=100)
    assert s.phases.size == 101 and s.phases[89] == GREEN and s.phases[90] == YELLOW and s.phases[100] == YELLOW
    assert s.closed_ticks == 0 and s.crossed == 1  # cuenta el peatón; el rojo empezaría después del final


# ---------------------------------------------------------------- semáforo peatonal


def _light(red=15.0, green=20.0, yellow=3.0) -> Light:
    return Light(100.0, red=red, green=green, yellow=yellow, pedestrian=True)


def test_fixed_phases_match_light_phase():
    for start in ("red", "green"):
        light = Light(50.0, red=5.0, green=7.0, yellow=2.0, start_phase=start)
        expected = [light.phase(t) for t in range(500)]
        np.testing.assert_array_equal(fixed_phases(light, 500), expected)
    assert (fixed_phases(Light(50.0, enabled=False), 20) == GREEN).all()


def test_light_schedule_without_pedestrians_never_turns_red():
    s = light_schedule(_counts(), _light(), None, N)
    assert (s.phases == GREEN).all() and s.crossings == 0


def test_light_schedule_without_a_previous_light_only_waits_for_the_yellow():
    s = light_schedule(_counts((10, 2)), _light(), None, 2000)
    ph = s.phases
    assert ph[9] == GREEN and ph[10] == YELLOW and ph[39] == YELLOW and ph[40] == RED and ph[189] == RED
    assert ph[190] == GREEN and (ph[190:] == GREEN).all()
    assert (s.crossed, s.crossings, s.closed_ticks) == (2, 1, 150)
    assert s.wait_ticks == 2 * 30  # los dos esperan hasta que empieza el rojo (paso 40)


def test_light_schedule_waits_for_the_previous_light_to_be_red():
    prev_red = np.zeros(2001, bool)
    prev_red[300:400] = True
    s = light_schedule(_counts((10, 1), (100, 1)), _light(), prev_red, 2000)
    ph = s.phases
    assert (ph[:300] == GREEN).all()  # los dos esperan el rojo del anterior
    assert ph[300] == YELLOW and ph[330] == RED and ph[479] == RED and ph[480] == GREEN
    assert (s.crossed, s.crossings) == (2, 1) and s.wait_ticks == (330 - 10) + (330 - 100)


def test_light_schedule_never_activates_if_the_previous_never_turns_red():
    s = light_schedule(_counts((10, 3)), _light(), np.zeros(2001, bool), 2000)
    assert (s.phases == GREEN).all() and s.crossed == 0


def test_light_schedule_respects_the_minimum_green_between_crossings():
    """El del paso 60 llega con el rojo en curso (hasta 190): espera al siguiente cruce, que empieza al terminar el
    verde mínimo (200 pasos = 20 s)."""
    s = light_schedule(_counts((10, 1), (60, 1)), _light(), None, 2000)
    ph = s.phases
    assert ph[40] == RED and ph[189] == RED and ph[190] == GREEN and ph[389] == GREEN
    assert ph[390] == YELLOW and ph[420] == RED and ph[569] == RED and ph[570] == GREEN
    assert (s.crossed, s.crossings) == (2, 2)
    assert s.wait_ticks == 30 + (420 - 60)


def test_light_schedule_pedestrians_arriving_during_the_yellow_cross():
    s = light_schedule(_counts((10, 1), (25, 1), (45, 1)), _light(), None, 2000)
    assert s.crossed == 3 and s.crossings == 2  # el de t=45 llega con el rojo en curso: otro cruce
    assert s.wait_ticks == 30 + 15 + (390 + 30 - 45)


def test_light_schedule_with_zero_minimum_green_can_chain_crossings():
    s = light_schedule(_counts((10, 1), (60, 1)), _light(green=0.0), None, 2000)
    assert s.phases[190] == YELLOW  # sin verde mínimo, vuelve a empezar al terminar el rojo
    assert s.crossings == 2


# ---------------------------------------------------------------- llegadas


def test_arrivals_follow_the_rate_and_are_reproducible():
    cfg = SimConfig(run=100.0, time_scale=10.0)  # 1000 s simulados
    rate = Rate(6.0, 6.0, 6.0, 0.0)  # 6 peatones/min
    a = arrivals(*np.random.default_rng(1).spawn(2), rate, cfg)
    b = arrivals(*np.random.default_rng(1).spawn(2), rate, cfg)
    np.testing.assert_array_equal(a, b)
    assert a.size == cfg.n_ticks and abs(a.sum() - 100) < 35  # ~100 peatones (Poisson)
    assert arrivals(*np.random.default_rng(2).spawn(2), Rate(0, 0, 0, 0), cfg).sum() == 0


# ---------------------------------------------------------------- motor


GRADUAL = replace(DEFAULT_SPECS[CAR], accel=2.5, decel=4.5)
INSTANT = DEFAULT_SPECS[CAR]


def _sim(spec=GRADUAL, lanes=1, bump_lanes=None, red=(30, 700), yellow_from=10, **kw) -> Simulation:
    """Calle sin llegadas con un paso peatonal a 60 m y un calendario fijo: amarillo desde el paso `yellow_from` y
    rojo en los pasos [red[0], red[1])."""
    cfg = SimConfig(length=150, lanes=lanes, rates=(0, 0, 0), traffic_light=False, run=5.0, time_scale=20.0,
                    specs=(spec, *DEFAULT_SPECS[1:]),
                    behavior=Behavior(reaction_min=1.0, reaction_max=1.0, reaction_mean=1.0, reaction_std=0.0),
                    speed_bumps=(SpeedBump(60.0, bump_lanes, pedestrian=True),), **kw)  # fmt: skip
    sim = Simulation(cfg, np.random.default_rng(0))
    phases = np.full(cfg.n_ticks + 1, GREEN, np.int8)
    phases[yellow_from : red[0]] = YELLOW
    phases[red[0] : red[1]] = RED
    schedule = Schedule(phases, 1, 1, red[1] - red[0], 0)
    sim.ped_schedules[("tope", 60.0)] = schedule
    sim._line_sched[0] = schedule
    return sim


def _place(sim: Simulation, x: float, lane: int = 0) -> int:
    sim._add(CAR, 1, lane)
    sim.x[sim.n - 1] = x
    return sim.n - 1


@pytest.mark.parametrize("spec", [GRADUAL, INSTANT], ids=["gradual", "instantáneo"])
def test_vehicles_stop_completely_at_the_crossing_and_go_when_it_ends(spec):
    sim = _sim(spec)
    i = _place(sim, 20.0)
    vid = sim.vid[i]
    stopped_at_line = False
    while sim.tick < 700:
        sim.step()
        assert sim.x[0] <= 60.0 + 1e-9  # no cruza mientras el paso está cerrado
        if sim.tick > 300 and sim.x[0] > 59.0 and sim.v_last[0] == 0.0:
            stopped_at_line = True
    assert stopped_at_line, "debe quedar detenido con el frente en el paso"
    while sim.n and sim.tick < sim.cfg.n_ticks:
        sim.step()
    assert sim.vid[0] == vid or sim.n == 0
    assert sim.cum_veh[CAR] == 1  # al terminar el cruce sigue y sale del tramo


def test_a_vehicle_that_cannot_stop_in_the_yellow_goes_through():
    """Con el frente a 3 m del paso y a 50 km/h, el amarillo no le da para frenar a decel: pasa."""
    sim = _sim(red=(60, 700), yellow_from=0)
    i = _place(sim, 56.5)
    sim.v_last[i] = 50 / 3.6 * DT
    while sim.tick < 60:
        sim.step()
    assert sim.x[0] > 60.0


def test_the_crossing_only_rules_in_the_bump_lanes():
    sim = _sim(lanes=2, bump_lanes=(1,))
    _place(sim, 20.0, lane=0)
    left = _place(sim, 20.0, lane=1)
    waiting = sim.vid[left]
    while sim.tick < 600:
        sim.step()
    assert sim.cum_veh[CAR] == 1 and sim.n == 1  # el del carril sin tope siguió y salió
    assert sim.vid[0] == waiting and sim.x[0] <= 60.0  # el del carril del tope espera


def test_pedestrians_do_not_change_vehicle_arrivals():
    """Los peatones tienen sus propios generadores: con la misma semilla llegan los mismos vehículos."""
    base = dict(length=150, lanes=2, rates=(20, 6, 1), red=10, green=10, run=10.0, time_scale=10.0)
    plain = Simulation(SimConfig(**base, speed_bumps=(SpeedBump(60.0),)), np.random.default_rng(3))
    ped = Simulation(SimConfig(**base, speed_bumps=(SpeedBump(60.0, pedestrian=True),)), np.random.default_rng(3))
    plain.run()
    ped.run()
    for key in ("arrived_veh", "arrived_pax", "arrived_cargo"):
        np.testing.assert_array_equal(plain.summary()[key], ped.summary()[key])
    assert "pedestrians" not in plain.summary() and ped.summary()["pedestrians"].shape == (1, 4)


def test_pedestrian_light_without_pedestrians_is_always_green():
    light = Light(60.0, red=10.0, green=10.0, yellow=3.0, pedestrian=True, pedestrian_crossing=Rate(0, 0, 0, 0))
    cfg = SimConfig(length=150, lanes=1, rates=(10, 0, 0), traffic_light=False, extra_lights=(light,), run=5.0)
    sim = Simulation(cfg, np.random.default_rng(0))
    assert all(sim.line_phases(t) == (GREEN,) for t in range(cfg.n_ticks))
    assert cfg.stop_lines[0].pedestrian and cfg.inner_phases(0) == (GREEN,)
    sim.run()
    assert sim.summary()["pedestrians"].tolist() == [[0.0, 0.0, 0.0, 0.0]]


def test_pedestrian_light_at_the_end_of_the_street_stops_the_line():
    cfg = SimConfig(length=150, lanes=1, rates=(10, 0, 0), red=10.0, green=5.0, yellow=2.0, light_pedestrian=True,
                    light_pedestrian_crossing=Rate(6, 6, 6, 0), run=10.0, time_scale=10.0)
    assert cfg.has_light and cfg.ref_light is None and cfg.pedestrian_spots == (("semáforo", 150.0),)
    sim = Simulation(cfg, np.random.default_rng(1))
    reds = [sim.exit_phase(t) == RED for t in range(cfg.n_ticks)]
    assert any(reds) and not all(reds)
    for _ in range(cfg.n_ticks):
        red = sim.exit_phase(sim.tick) == RED
        before = sim.cum_veh.copy()
        sim.step()
        if red:
            assert (sim.cum_veh == before).all()  # nadie cruza la línea en rojo
    row = sim.summary()["pedestrians"][0]
    assert row[0] > 0 and row[1] > 0 and row[2] > 0


def test_summary_rows_follow_pedestrian_spots_in_order():
    light = Light(100.0, red=10.0, green=10.0, pedestrian=True, pedestrian_crossing=Rate(4, 4, 4, 0))
    cfg = SimConfig(length=150, lanes=1, rates=(5, 0, 0), traffic_light=False, extra_lights=(light,), run=5.0,
                    speed_bumps=tuple(SpeedBump(x, pedestrian=True, pedestrian_crossing=Rate(8, 8, 8, 0))
                                      for x in (30.0, 120.0)))
    assert cfg.pedestrian_spots == (("tope", 30.0), ("semáforo", 100.0), ("tope", 120.0))
    assert [s.position for s in cfg.stop_lines] == [30.0, 100.0, 120.0]
    sim = Simulation(cfg, np.random.default_rng(2))
    sim.run()
    assert sim.summary()["pedestrians"].shape == (3, 4)
