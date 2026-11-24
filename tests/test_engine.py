from dataclasses import replace

import numpy as np
import pytest

from traffic.config import (
    BIKE, BUS, CAR, DEFAULT_SPECS, DT, GREEN, POLLUTANTS, RED, YELLOW, Behavior, Bottleneck, Rate, Light, SimConfig,
    SpeedBump, VehicleSpec, as_positions,
)  # fmt: skip
from traffic.engine import Simulation


def _gaps(sim: Simulation):
    """(gap to the leader, minimum gap allowed to the follower) for each entry with a leader, not counting
    a vehicle beside its leader or approaching that place (side-by-side rows, which are checked with
    _check_rows)."""
    occ = sim.occupancy()
    beside = sim.rows(occ)[0]
    keep = occ.has_leader & ~beside
    if sim.any_abreast:
        keep &= ~sim._side_slot(occ)
    lead = occ.leader[keep]
    fol = occ.veh[keep]
    gap = sim.x[lead] - sim.vlen[lead] - sim.x[fol]
    min_gap = sim.gap_stop_t[sim.vtype[fol]]
    # Between vehicles of the same type with abreast > 1 not overlapping is enough: one that was going to move alongside
    # may be closer than the gap if the one ahead starts in that same step.
    same_row_type = (sim.vtype[lead] == sim.vtype[fol]) & (sim.abreast[sim.vtype[fol]] > 1)
    return gap, np.where(same_row_type, 0.0, min_gap), fol


def _check_rows(sim: Simulation):
    """Side-by-side rows: only of the same type with abreast > 1, stopped, aligned and not exceeding abreast."""
    occ = sim.occupancy()
    beside, rid, _ = sim.rows(occ)
    if not beside.any():
        return
    svt = sim.vtype[occ.veh]
    lvt = sim.vtype[occ.leader]
    assert np.all(svt[beside] == lvt[beside])
    assert np.all(sim.abreast[svt[beside]] > 1)
    assert np.all(np.abs(sim.x[occ.leader[beside]] - occ.x[beside]) < sim.vlen[occ.veh[beside]])
    sizes = np.bincount(rid)
    assert np.all(sizes[rid] <= sim.abreast[svt]), "fila con más vehículos que abreast"


def _check_conservation(sim: Simulation):
    n = sim.n
    inside = ~sim.crossed[:n]
    s = sim.summary()
    queued_pax = np.zeros(sim.n_types)
    for q in sim.queues:
        for vt, p, *_ in q:
            queued_pax[vt] += p
    on_road_pax = np.bincount(sim.vtype[:n][inside], weights=sim.pax[:n][inside], minlength=sim.n_types)
    np.testing.assert_array_equal(s["arrived_veh"] + s["initial_veh"], s["queued"] + s["entered_veh"])
    np.testing.assert_array_equal(s["entered_veh"], s["on_road"] + s["crossed_veh"])
    np.testing.assert_allclose(s["arrived_pax"] + s["initial_pax"], queued_pax + on_road_pax + s["crossed_pax"])


# Freight vehicle: freight only, length normal(10, 2) truncated to [8, 16] m.
CARGA = VehicleSpec("carga", "carga", 40.0, 10.0, 4.0, 1.5, 1, 1, 1.0, 0.0, True,
                    cargo_prob=1.0, length_std=2.0, length_min=8.0)  # fmt: skip

# Types the code does not know: one that changes lanes and one that does not.
EXTRA_TYPES = (
    VehicleSpec("motorbike", "moto", 40.0, 2.0, 2.0, 1.0, 1, 2, 1.2, 0.2, True),
    VehicleSpec("tram", "tranvía", 30.0, 20.0, 6.0, 2.0, 20, 200, 120.0, 30.0, False),
)


@pytest.mark.parametrize(
    ("lanes", "extra", "separate", "bus_stop", "abreast", "overtake"),
    [
        (1, False, False, False, 1, False), (3, False, False, False, 1, False),
        (3, True, False, False, 1, False), (3, False, True, False, 1, False),
        (3, True, True, False, 1, False), (1, False, False, True, 1, False),
        (3, True, True, True, 1, False), (1, False, False, False, 2, False),
        (3, True, True, True, 2, False), (2, False, False, False, 3, False),
        (3, True, False, False, 1, True), (4, True, True, True, 2, True),
    ],
)  # fmt: skip
def test_invariants_every_step(lanes, extra, separate, bus_stop, abreast, overtake):
    specs, rates = DEFAULT_SPECS, (30, 8, 2)
    if extra:
        specs, rates = specs + EXTRA_TYPES, rates + (10, 1)
    limits = None
    if overtake:  # overtaking motorcycles, lanes with different limits (there are move-ups) and halts
        specs = tuple(replace(s, overtake=True, lookahead=50.0, min_advantage=1.0, lane_change_cooldown=1.0)
                      if s.key == "motorbike" else s for s in specs)  # fmt: skip
        specs = tuple(replace(s, bottleneck_prob=0.3, bottleneck_time_mean=8.0, bottleneck_time_std=3.0,
                              speed_bump_kmh=15.0)
                      if s.key == "car" else s for s in specs)  # fmt: skip
        limits = (30.0, 50.0, 40.0, 80.0)[:lanes]
        specs, rates = specs + (CARGA,), rates + (4,)  # variable length and freight only
    if separate:  # bikes in the right lane and buses in the next one
        free = overtake and lanes == 4  # with a lane without a traffic light: the bikes' one, exclusive
        specs = (specs[CAR], replace(specs[BIKE], lane=0, exclusive=free), replace(specs[BUS], lane=1),
                 *specs[BUS + 1 :])
    if abreast > 1:  # bikes side by side when stopping
        specs = tuple(replace(s, abreast=abreast) if k == BIKE else s for k, s in enumerate(specs))
    if bus_stop:  # bus stop at 100 m (the segment is 150 long)
        specs = tuple(replace(s, stop_position=100.0, stop_time_mean=8.0, stop_time_std=3.0) if k == BUS else s
                      for k, s in enumerate(specs))  # fmt: skip
    cfg = SimConfig(
        length=150, lanes=lanes, lane_speed_limit=limits, rates=rates, red=20, green=15, yellow=3.0 if overtake else 0.0,
        initial_occupancy=0.4 if overtake else 0.0,
        bottleneck=Bottleneck(stop_lanes=(1, 2)) if overtake else Bottleneck(),
        free_lanes=(0,) if overtake and lanes == 4 else (),  # lane without a traffic light
        speed_bumps=(SpeedBump(80.0, (1, 2)),) if overtake else (),  # bump for the cars
        run=30, specs=specs,
    )  # fmt: skip
    sim = Simulation(cfg, np.random.default_rng(7))
    ruled = [lane for lane in range(lanes) if lane not in cfg.free_lanes]
    free_in_red = 0
    for _ in range(cfg.n_ticks):
        red = cfg.is_red(sim.tick)
        crossed_before = sim.lane_crossed[ruled].sum()
        free_before = sim.lane_crossed[list(cfg.free_lanes)].sum()
        x_before = dict(zip(sim.vid[: sim.n].tolist(), sim.x[: sim.n].tolist()))
        sim.step()
        n = sim.n
        gap, min_gap, fol = _gaps(sim)
        # When a side-by-side row starts, the one alongside is briefly closer than the gap
        # while the one ahead pulls away: it is only an error if the follower advanced to end up like that.
        moved = sim.x[fol] > np.array([x_before.get(v, -np.inf) for v in sim.vid[fol].tolist()]) + 1e-12
        assert np.all((gap >= min_gap - 1e-9) | (~moved & sim.any_abreast)), "traslape o gap menor al comprimido"
        _check_rows(sim)
        if red:
            assert sim.lane_crossed[ruled].sum() == crossed_before, "cruzó en rojo"
            free_in_red += sim.lane_crossed[list(cfg.free_lanes)].sum() - free_before
        slow = ~sim.can_change[sim.vtype[:n]]
        assert np.all(sim.lc_target[:n][slow] == -1)
        # Without lane: the rightmost free lane (without the exclusive lanes of other types).
        fixed = np.array([cfg.entry_lanes(k)[0] if cfg.allowed_lanes(k) else -1 for k in range(len(specs))])
        assert np.all(sim.lane[:n][slow] == fixed[sim.vtype[:n][slow]])
    _check_conservation(sim)
    assert sim.cum_veh.sum() > 0
    if extra:
        assert sim.cum_veh[3] > 0 and sim.arrived_veh[4] > 0  # the new types move
    if separate:
        assert sim.cum_veh[BUS] > 0 and sim.cum_veh[BIKE] > 0
    if bus_stop:
        assert sim.stops[BUS] > 0 and sim.stops[BUS] >= sim.cum_veh[BUS]  # every bus that crossed stopped
        assert sim.stops[[CAR, BIKE]].sum() == 0
    if lanes == 1:
        assert sim.lane_changes == 0
    else:
        assert sim.lane_changes > 0
    assert (free_in_red > 0) == bool(cfg.free_lanes)  # the lane without a traffic light is crossed on red


def _empty_sim(seed: int = 0, **kw) -> Simulation:
    """Segment without random arrivals; vehicles are placed by hand."""
    base = dict(length=100, lanes=1, rates=(0, 0, 0), red=10, green=30, run=5)
    base.update(kw)
    return Simulation(SimConfig(**base), np.random.default_rng(seed))


def _place(sim: Simulation, vt: int, x: float, lane: int = 0, pax: int = 1) -> int:
    sim._add(vt, pax, lane)
    i = sim.n - 1
    sim.x[i] = x
    return i


def _start_ticks(sim: Simulation, until: int) -> dict[int, int]:
    """First step after the green in which each vehicle advances, identified by its number of passengers."""
    starts: dict[int, int] = {}
    while sim.tick < until:
        before = dict(zip(sim.pax[: sim.n].tolist(), sim.x[: sim.n].tolist()))
        tick = sim.tick
        sim.step()
        after = dict(zip(sim.pax[: sim.n].tolist(), sim.x[: sim.n].tolist()))
        if tick >= sim.cfg.red_ticks:
            for vid, x in before.items():
                if vid not in starts and after.get(vid, np.inf) > x:
                    starts[vid] = tick
    return starts


@pytest.mark.parametrize("seed", range(5))
def test_reaction_time_after_green(seed):
    sim = _empty_sim(seed=seed)
    _place(sim, CAR, sim.L - 0.5, pax=1)
    starts = _start_ticks(sim, sim.cfg.red_ticks + 60)
    delay = (starts[1] - sim.cfg.red_ticks) * DT
    assert 1.0 <= delay <= 5.0 + DT


def test_reaction_time_follows_the_configured_normal():
    """With reaction_mean = 2 s and a small σ, the car stopped at the line starts ~2 s after the green."""
    delays = []
    for seed in range(20):
        sim = _empty_sim(seed=seed, behavior=Behavior(reaction_min=1.0, reaction_max=5.0, reaction_mean=2.0,
                                                      reaction_std=0.1))  # fmt: skip
        _place(sim, CAR, sim.L - 0.5, pax=1)
        starts = _start_ticks(sim, sim.cfg.red_ticks + 60)
        delays.append((starts[1] - sim.cfg.red_ticks) * DT)
    assert 1.7 <= min(delays) and max(delays) <= 2.4
    assert abs(np.mean(delays) - 2.05) < 0.15


@pytest.mark.parametrize("seed", range(5))
def test_queue_compresses_and_stretches(seed):
    sim = _empty_sim(seed=seed, length=120)
    spec = sim.cfg.specs[CAR]
    # Three moving cars separated by the moving gap; when reaching the red they compress.
    xs = [sim.L - 20, sim.L - 20 - spec.length - spec.gap_run, sim.L - 20 - 2 * (spec.length + spec.gap_run)]
    for vid, x in enumerate(xs, start=1):
        _place(sim, CAR, x, pax=vid)
    while sim.tick < sim.cfg.red_ticks:
        sim.step()
    assert np.all(sim.stopped[:3])
    gap, _, _ = _gaps(sim)
    np.testing.assert_allclose(gap, spec.gap_stop, atol=1e-9)  # compression: 1 m < 3 m

    starts = _start_ticks(sim, sim.cfg.red_ticks + 200)
    assert sorted(starts) == [1, 2, 3]
    # Stretching: each car starts at least one reaction time after its leader.
    assert starts[2] - starts[1] >= round(1.0 / DT)
    assert starts[3] - starts[2] >= round(1.0 / DT)


def test_lane_change_passes_bike_with_cost():
    sim = _empty_sim(length=600, lanes=2, red=0, green=60, run=60, seed=3)
    bike = _place(sim, BIKE, 60.0, lane=0)
    car = _place(sim, CAR, 60.0 - 1.8 - 12.0, lane=0)
    v_car = sim.v_step[CAR]
    duration = 0
    while sim.tick < 300:
        was_changing = sim.lc_target[car] >= 0
        x0 = sim.x[car]
        sim.step()
        if was_changing:
            duration += 1
            # Cost: during the manoeuvre it occupies both lanes at reduced speed.
            assert sim.x[car] - x0 <= 0.5 * v_car + 1e-9
            if sim.lc_target[car] < 0:
                break
    assert sim.lane[car] == 1 and sim.lane[bike] == 0
    assert round(1.0 / DT) <= duration <= round(4.0 / DT)
    for _ in range(200):
        sim.step()
    assert sim.x[car] > sim.x[bike]  # it overtook it through the left lane


def test_queue_does_not_slow_far_vehicles():
    """With 4 cars stopped at the red (a queue of 4 × 5.5 m on a 100 m segment), a car far from the queue
    advances at its normal speed, as in the lane without a queue: only the queue slows it down when it reaches it."""
    sim = _empty_sim(length=100, lanes=2, red=30, green=30)
    spec = sim.cfg.specs[CAR]
    for k in range(4):
        _place(sim, CAR, sim.L - k * (spec.length + spec.gap_stop), pax=k + 1)
    sim.stopped[:4] = True
    other = _place(sim, CAR, 10.0, lane=1, pax=5)  # lane without a queue: normal speed
    follower = _place(sim, CAR, 10.0, lane=0, pax=6)
    x0 = sim.x[: sim.n].copy()
    sim.step()
    assert sim.x[follower] - x0[follower] == pytest.approx(sim.v_step[CAR])
    assert sim.x[other] - x0[other] == pytest.approx(sim.v_step[CAR])


def test_lane_saturation_is_recorded():
    """4 cars stopped at the red in lane 0 of a 100 m segment: saturation 0.22 and 0."""
    sim = _empty_sim(length=100, lanes=2, red=30, green=30)
    spec = sim.cfg.specs[CAR]
    for k in range(4):
        _place(sim, CAR, sim.L - k * (spec.length + spec.gap_stop), pax=k + 1)
    sim.stopped[:4] = True
    for _ in range(sim.cfg.sample_ticks):
        sim.step()
    assert sim.recorder.count == 1
    np.testing.assert_allclose(sim.recorder.lane_sat[0], [4 * (spec.length + spec.gap_stop) / sim.L, 0.0], rtol=1e-6)


def _free_lane_sim(gradual: bool = False, **kw) -> Simulation:
    """Two lanes; bikes with exclusive lane 0 and without a traffic light; cars in lane 1."""
    car, bike, bus = DEFAULT_SPECS
    bike = replace(bike, lane=0, exclusive=True)
    specs = tuple(_gradual(s) for s in (car, bike, bus)) if gradual else (car, bike, bus)
    return _empty_sim(length=100, lanes=2, free_lanes=(0,), specs=specs, **kw)


@pytest.mark.parametrize("gradual", [False, True])
def test_free_lane_ignores_red(gradual):
    """On red, the bike in its exclusive lane without a traffic light (0) crosses the line without stopping; the car in
    lane 1 stops."""
    sim = _free_lane_sim(gradual, red=30, green=30)
    _place(sim, BIKE, 80.0, lane=0, pax=1)
    _place(sim, CAR, 60.0, lane=1, pax=2)
    for _ in range(round(10 / DT)):
        sim.step()
        free = sim.pax[: sim.n] == 1
        assert not sim.stopped[: sim.n][free].any(), "la bici del carril sin semáforo se detuvo"
    assert sim.lane_crossed[0] == 1 and sim.lane_crossed[1] == 0
    i = int(np.flatnonzero(sim.pax[: sim.n] == 2)[0])
    assert sim.stopped[i] and sim.x[i] <= sim.L + 1e-9


def test_free_lane_ignores_yellow():
    """On yellow, 40 m from the line: the bike in the exclusive lane without a traffic light advances at its speed and the
    car in lane 1 at yellow_speed_factor of its own."""
    sim = _free_lane_sim(red=30, green=1, yellow=5, start_phase="green")
    while sim.cfg.phase(sim.tick) != YELLOW:
        sim.step()
    free = _place(sim, BIKE, 60.0, lane=0, pax=1)
    ruled = _place(sim, CAR, 60.0, lane=1, pax=2)
    x0 = sim.x[: sim.n].copy()
    sim.step()
    assert sim.x[free] - x0[free] == pytest.approx(sim.v_step[BIKE])
    factor = sim.cfg.behavior.yellow_speed_factor
    assert sim.x[ruled] - x0[ruled] == pytest.approx(sim.v_step[CAR] * factor)


def test_free_lane_only_frees_the_exclusive_type():
    """A bus with lane = 0 but without exclusive shares the bikes' lane without a traffic light: it does stop on
    red, and the bike behind it waits as behind any stopped leader."""
    car, bike, bus = DEFAULT_SPECS
    specs = (car, replace(bike, lane=0, exclusive=True), replace(bus, lane=0, stop_position=None))
    sim = _empty_sim(length=100, lanes=2, red=30, green=30, free_lanes=(0,), specs=specs)
    assert not sim.cfg.ignores_light(BUS) and sim.cfg.ignores_light(BIKE)
    _place(sim, BUS, 70.0, lane=0, pax=1)
    _place(sim, BIKE, 40.0, lane=0, pax=2)
    for _ in range(round(15 / DT)):
        sim.step()
    assert sim.lane_crossed[0] == 0
    assert sim.stopped[: sim.n].all() and sim.x[: sim.n].max() <= sim.L + 1e-9


def _bump_run(gradual: bool, lanes: int = 1, bump_lanes=None, lane: int = 0, vt: int = CAR,
              position=60.0) -> tuple[list, Simulation]:
    """A single vehicle, always on green, with a bump at 60 m (or several, `position`) in `bump_lanes` (car at
    10 km/h on it). Returns, per step, (x before, x after, speed in m/step) and the simulation."""
    car = replace(DEFAULT_SPECS[CAR], speed_bump_kmh=10.0)
    specs = (car, *DEFAULT_SPECS[1:])
    if gradual:
        specs = tuple(_gradual(s) for s in specs)
    sim = _empty_sim(length=150, lanes=lanes, red=0, green=60, run=10, specs=specs,
                     speed_bumps=tuple(SpeedBump(p, bump_lanes) for p in as_positions(position)))  # fmt: skip
    i = _place(sim, vt, 20.0, lane=lane)
    vid = sim.vid[i]
    steps = []
    while sim.n and sim.vid[0] == vid and sim.tick < 400:
        x0 = float(sim.x[0])
        sim.step()
        if sim.n and sim.vid[0] == vid:
            steps.append((x0, float(sim.x[0]), float(sim.x[0] - x0)))
    return steps, sim


@pytest.mark.parametrize("gradual", [False, True])
def test_speed_bump_limits_speed_while_on_it(gradual):
    """While the car is on the bump (front past it, rear not) it advances at most at 10 km/h; before and after, at its
    speed. With gradual braking, it never brakes harder than decel in one step."""
    steps, sim = _bump_run(gradual)
    vb, vlen = 10 / 3.6 * DT, sim.cfg.specs[CAR].length
    on = [adv for x0, x1, adv in steps if x0 - vlen < 60.0 and x1 > 60.0]
    assert on and max(on) <= vb + 1e-9
    assert len(on) >= vlen / vb - 1  # it goes over at that speed: ~1.6 s for 4.5 m
    after = [adv for x0, x1, adv in steps if x0 - vlen > 60.0]
    assert max(after) == pytest.approx(sim.v_step[CAR])  # afterwards it recovers its speed
    if gradual:
        advs = [adv for *_, adv in steps]
        dec = sim.dec_t[CAR]
        assert all(b >= a - dec - 1e-9 for a, b in zip(advs, advs[1:])), "frenado brusco antes del tope"
        assert min(adv for x0, x1, adv in steps if x0 < 60.0) >= vb - dec - 1e-9  # it does not stop before


@pytest.mark.parametrize("gradual", [False, True])
def test_several_speed_bumps_each_slow_the_vehicle(gradual):
    """With two bumps (40 and 100 m) the car goes over each one at most at 10 km/h, recovers its speed between them and
    takes longer than with a single one."""
    steps, sim = _bump_run(gradual, position=(100.0, 40.0))  # unordered
    assert [b.position for b in sim.cfg.bumps] == [40.0, 100.0]
    vb, vlen = 10 / 3.6 * DT, sim.cfg.specs[CAR].length
    for at in (40.0, 100.0):
        on = [adv for x0, x1, adv in steps if x0 - vlen < at and x1 > at]
        assert on and max(on) <= vb + 1e-9
    between = [adv for x0, x1, adv in steps if 40.0 + vlen + 3 < x0 < 100.0 - 25]
    # It recovers speed between them (with gradual braking, without reaching the maximum: it is already braking for the second).
    assert max(between) == pytest.approx(sim.v_step[CAR]) if not gradual else max(between) > 1.5 * vb
    if gradual:
        advs = [adv for *_, adv in steps]
        dec = sim.dec_t[CAR]
        assert all(b >= a - dec - 1e-9 for a, b in zip(advs, advs[1:])), "frenado brusco antes del tope"
    assert len(steps) > len(_bump_run(gradual, position=40.0)[0]) + 5


def test_speed_bump_only_in_its_lanes_and_for_types_with_speed():
    """The bump in lane 1 does not slow down the car in lane 0, nor a bike (without speed_bump_kmh) in lane 1."""
    other_lane, sim = _bump_run(False, lanes=2, bump_lanes=(1,), lane=0)
    assert min(adv for *_, adv in other_lane) == pytest.approx(sim.v_step[CAR])
    bike, sim = _bump_run(False, lanes=2, bump_lanes=(1,), lane=1, vt=BIKE)
    assert min(adv for *_, adv in bike) == pytest.approx(sim.v_step[BIKE])
    car, _ = _bump_run(False, lanes=2, bump_lanes=(1,), lane=1)
    assert min(adv for *_, adv in car) == pytest.approx(10 / 3.6 * DT)


def test_no_traffic_light_is_the_same_with_enabled_false_or_all_phases_zero():
    """Without a traffic light (enabled = false, or red, green and yellow at 0) the line is always open: both forms give
    the same run, like a permanent green, and nobody stops in front of the line."""
    base = dict(length=150, lanes=2, rates=(20, 4, 1), run=20)
    configs = [
        SimConfig(red=30, green=30, yellow=3, traffic_light=False, **base),
        SimConfig(red=0, green=0, yellow=0, **base),
        SimConfig(red=0, green=60, **base),
    ]
    assert not configs[0].has_light and not configs[1].has_light and configs[2].has_light
    assert configs[0].red_intervals() == [] and configs[0].cycle == 0 and configs[0].flow_window == 60.0
    results = []
    for cfg in configs:
        sim = Simulation(cfg, np.random.default_rng(3))
        assert all(cfg.phase(t) == GREEN for t in range(0, cfg.n_ticks, 97))
        sim.run()
        s = sim.summary()
        results.append((s["crossed_veh"].tolist(), s["travel_time"].tolist()))
    assert results[0] == results[1] == results[2]
    assert sum(results[0][0]) > 0


def _emission_run(**kw) -> dict:
    """A car with gradual acceleration and braking (it emits CO2, NOx and VOC) alone in its lane of a 200 m segment."""
    car = replace(DEFAULT_SPECS[CAR], accel=2.5, decel=4.5, speed_bump_kmh=10.0)
    base = dict(length=200, lanes=2, rates=(0, 0, 0), red=0, green=60, run=6, specs=(car, *DEFAULT_SPECS[1:]))
    base.update(kw)
    sim = Simulation(SimConfig(**base), np.random.default_rng(0))
    _place(sim, BIKE, 50.0, lane=0)  # a bike in the other lane: it does not emit
    sim._add(CAR, 1, 1)
    sim.run()
    return sim.summary()


def test_emissions_excess_over_free_flow():
    """At free flow the car emits the same as at constant speed (excess 0); a bump and a stop on red make it
    emit more. What is emitted by position adds up to the total, and the bike does not emit."""
    free = _emission_run()
    co2 = POLLUTANTS.index("co2")
    assert free["emissions"][CAR, co2] > 0 and free["emissions"][BIKE].sum() == 0
    assert free["emissions"][CAR, co2] == pytest.approx(free["emissions_free"][CAR, co2], rel=1e-6)
    np.testing.assert_allclose(free["emissions_pos"].sum(axis=(1, 2)), free["emissions"].sum(axis=0))
    excess = {}
    for name, kw in (("tope", dict(speed_bumps=(SpeedBump(100.0),))), ("rojo", dict(red=20, green=40))):
        s = _emission_run(**kw)
        excess[name] = s["emissions"][CAR, co2] / s["emissions_free"][CAR, co2] - 1
    assert excess["tope"] > 0.3 and excess["rojo"] > 0.3
    assert free["veh_km"][CAR] == pytest.approx(0.2 - 0.0045, abs=0.01)  # enters with its rear at 0


def test_lane_speed_series():
    """A single car in lane 1 at free flow: 50 km/h; the empty lane 0 has no speed (NaN)."""
    from traffic.metrics import LANE_SPEED, derive_series

    sim = _empty_sim(length=300, lanes=2, red=0, green=60, run=2)
    _place(sim, CAR, 10.0, lane=1)
    sim.run()
    speed = derive_series(sim.recorder, sim.cfg)[LANE_SPEED]
    assert np.all(np.isnan(speed[:, 0]))
    np.testing.assert_allclose(speed[:, 1], 50.0, rtol=1e-4)


def test_lane_speed_limit_is_never_exceeded():
    """Limits of 40 and 20 km/h: no vehicle advances more than its lane's limit (nor the lower
    of its two lanes if it is changing), even if its maximum speed is higher."""
    specs = DEFAULT_SPECS + EXTRA_TYPES[:1]
    cfg = SimConfig(length=200, lanes=2, rates=(30, 6, 2, 10), red=20, green=20, run=20, specs=specs,
                    lane_speed_limit=(40.0, 20.0))  # fmt: skip
    sim = Simulation(cfg, np.random.default_rng(3))
    limit = np.array([40.0, 20.0]) / 3.6 * DT

    def snapshot():
        """Vehicles by (entry tick, passengers, type); repeated keys are ambiguous and are left out."""
        n = sim.n
        keys = list(zip(sim.entry_tick[:n].tolist(), sim.pax[:n].tolist(), sim.vtype[:n].tolist()))
        vals = zip(sim.x[:n].tolist(), sim.lane[:n].tolist(), sim.lc_target[:n].tolist())
        return {k: v for k, v in zip(keys, vals) if keys.count(k) == 1}

    checked = 0
    for _ in range(cfg.n_ticks):
        before = snapshot()
        sim.step()
        for key, (x, _, _) in snapshot().items():
            if key in before:
                x0, lane, tgt = before[key]
                cap = min(limit[lane], limit[tgt]) if tgt >= 0 else limit[lane]
                assert x - x0 <= cap + 1e-9
                checked += 1
    assert checked > 1000
    assert sim.cum_veh.sum() > 0


def test_free_flow_speed_respects_lane_limits():
    specs = (DEFAULT_SPECS[CAR], replace(DEFAULT_SPECS[BIKE], lane=1), DEFAULT_SPECS[BUS])
    cfg = SimConfig(lanes=3, lane_speed_limit=(30.0, 10.0, 45.0), specs=specs)
    assert cfg.free_flow_kmh(CAR) == 45.0  # it can change: the best lane (45 < 50)
    assert cfg.free_flow_kmh(BIKE) == 10.0  # fixed in lane 1
    assert cfg.free_flow_kmh(BUS) == 30.0  # does not change lanes and has no lane: lane 0


def test_exclusive_lanes_are_only_used_by_their_types():
    """Exclusive bikes (lane 0) and buses (lane 1): cars and motorcycles never occupy them, neither on
    entering nor during a lane change; the tram without a fixed lane only uses the free ones."""
    car, bike, bus = DEFAULT_SPECS
    specs = (car, replace(bike, lane=0, exclusive=True), replace(bus, lane=1, exclusive=True), *EXTRA_TYPES)
    cfg = SimConfig(
        length=150, lanes=4, rates=(30, 8, 2, 10, 1), red=20, green=15, run=30, specs=specs,
    )  # fmt: skip
    assert cfg.allowed_lanes(0) == (2, 3) and cfg.allowed_lanes(1) == (0,) and cfg.allowed_lanes(4) == (2, 3)
    sim = Simulation(cfg, np.random.default_rng(3))
    used = np.zeros((cfg.n_types, cfg.lanes), np.bool_)
    for _ in range(cfg.n_ticks):
        sim.step()
        n = sim.n
        vt, lane, tgt = sim.vtype[:n], sim.lane[:n], sim.lc_target[:n]
        used[vt, lane] = True
        changing = tgt >= 0
        used[vt[changing], tgt[changing]] = True
        for ln, q in enumerate(sim.queues):
            for t, *_ in q:
                used[t, ln] = True
    assert not (used & ~sim.allowed).any()
    assert used[0, 2] and used[0, 3] and sim.lane_changes > 0
    assert not used[4, 3]  # the tram enters through the rightmost free lane


def test_non_exclusive_fixed_lane_is_shared():
    car, bike, bus = DEFAULT_SPECS
    cfg = SimConfig(lanes=2, specs=(car, replace(bike, lane=0), bus))
    assert cfg.allowed_lanes(0) == (0, 1) and cfg.reserved_lanes == frozenset()


MOTO = VehicleSpec("motorbike", "moto", 70.0, 2.0, 2.0, 1.0, 1, 2, 1.2, 0.2, True,
                   overtake=True, lookahead=50.0, min_advantage=1.0, lane_change_cooldown=1.0)  # fmt: skip
MOTO_IDX = 3


def _moto_sim(moto: VehicleSpec = MOTO, **kw) -> Simulation:
    """Empty segment with cars, bikes, buses and motorcycles (in that order), without arrivals."""
    return _empty_sim(specs=DEFAULT_SPECS + (moto,), rates=(0, 0, 0, 0), **kw)


@pytest.mark.parametrize("overtake", [True, False])
def test_overtaker_drops_to_slower_lane_behind_a_slow_leader(overtake):
    """A car that in the last step advanced less than the motorcycle can makes it move down to the
    30 km/h lane (a lower limit than its own 60), because it advances more there. Without `overtake` slowness is
    judged by the nominal speed (the same as its own in that lane) and it does not change."""
    sim = _moto_sim(replace(MOTO, overtake=overtake), length=300, lanes=2, lane_speed_limit=(30.0, 60.0))
    car = _place(sim, CAR, 100.0, lane=1)
    moto = _place(sim, MOTO_IDX, 100.0 - 4.5 - 10.0, lane=1)
    sim.v_last[car] = 0.5 * sim.lane_vmax[0]  # e.g. braked by its own leader
    sim._lane_changes()
    assert sim.lc_target[moto] == (0 if overtake else -1)
    assert sim.lane_changes_t[MOTO_IDX] == int(overtake)


@pytest.mark.parametrize(("vtype", "changes"), [(MOTO_IDX, True), (CAR, False)])
def test_lookahead_per_type(vtype, changes):
    """With a stopped leader at 40 m, the motorcycle (lookahead = 50 m) changes lanes and the car (30 m) does not."""
    sim = _moto_sim(length=300, lanes=2)
    leader = _place(sim, CAR, 150.0, lane=0)
    sim.stopped[leader] = True
    follower = _place(sim, vtype, 150.0 - 4.5 - 40.0, lane=0)
    sim._lane_changes()
    assert (sim.lc_target[follower] == 1) == changes


def test_no_lane_change_without_a_vehicle_ahead():
    """Alone in the 30 km/h lane, with the 50 one free, it does not change: it has no leader slowing it down."""
    sim = _empty_sim(length=600, lanes=2, lane_speed_limit=(30.0, 50.0), red=0, green=60, run=60)
    car = _place(sim, CAR, 20.0, lane=0)
    for _ in range(100):
        sim.step()
    assert sim.lane[car] == 0 and sim.lane_changes == 0


@pytest.mark.parametrize(("ratio", "changes"), [(0.80, False), (0.70, True), (0.0, True)])
def test_changes_lane_only_if_leader_is_25_percent_below_the_lane_limit(ratio, changes):
    """With a leader closer than lookahead going at `ratio` of the lane's limit (50 km/h): it changes
    only if it is at least 25 % below (leader_slowdown = 0.25). The 50 lane is the fastest,
    so the change is an overtake or a move up to the 80 one."""
    sim = _moto_sim(length=300, lanes=2, lane_speed_limit=(50.0, 80.0))
    leader = _place(sim, CAR, 100.0, lane=0)
    moto = _place(sim, MOTO_IDX, 100.0 - 4.5 - 10.0, lane=0)
    sim.v_last[leader] = ratio * sim.lane_vmax[0]
    sim._lane_changes()
    assert (sim.lc_target[moto] == 1) == changes


def test_does_not_rise_without_room_or_speed_gain():
    # Follower in the fast lane closer than its moving gap: it does not move up.
    sim = _empty_sim(length=300, lanes=2, lane_speed_limit=(30.0, 50.0))
    car = _place(sim, CAR, 100.0, lane=0)
    _place(sim, CAR, 100.0 - 4.5 - 1.0, lane=1)
    sim._lane_changes()
    assert sim.lc_target[car] == -1
    # Stopped leader close by in the fast lane: it does not move up.
    sim = _empty_sim(length=300, lanes=2, lane_speed_limit=(30.0, 50.0))
    car = _place(sim, CAR, 100.0, lane=0)
    leader = _place(sim, CAR, 100.0 + 4.5 + 15.0, lane=1)
    sim.stopped[leader] = True
    sim._lane_changes()
    assert sim.lc_target[car] == -1
    # Without different limits there is nothing to move up to.
    sim = _empty_sim(length=600, lanes=2, red=0, green=60, run=60)
    _place(sim, CAR, 20.0, lane=0)
    for _ in range(100):
        sim.step()
    assert sim.lane_changes == 0


def test_motorbikes_change_lanes_more_than_cars():
    """config.toml scenario: bikes and buses in exclusive lanes; cars and motorcycles in the
    30 and 50 km/h ones. Overtaking motorcycles change lanes more per vehicle than cars, and more
    than the same motorcycles without `overtake`."""
    car, bike, bus = DEFAULT_SPECS
    base = dict(
        length=300, lanes=4, lane_speed_limit=(20.0, 30.0, 50.0, 80.0), rates=(18, 8, 0.25, 8), red=25,
        green=35, run=30,
    )  # fmt: skip
    fixed = (replace(car, speed_kmh=80.0), replace(bike, lane=0, exclusive=True), replace(bus, lane=3, exclusive=True))
    per_veh = {}
    for overtake in (True, False):
        moto = MOTO if overtake else replace(MOTO, overtake=False, lookahead=None, min_advantage=None,
                                             lane_change_cooldown=None)  # fmt: skip
        cfg = SimConfig(specs=fixed + (moto,), **base)
        changes, entered = np.zeros(4), np.zeros(4)
        for seed in range(3):
            sim = Simulation(cfg, np.random.default_rng(seed))
            sim.run()
            changes += sim.lane_changes_t
            entered += sim.entered_veh
        per_veh[overtake] = changes / entered
    assert per_veh[True][MOTO_IDX] > 1.3 * per_veh[True][CAR]
    assert per_veh[True][MOTO_IDX] > 1.3 * per_veh[False][MOTO_IDX]


def _bus_with_stop(mean: float = 12.0, std: float = 0.0, **kw) -> Simulation:
    bus = replace(DEFAULT_SPECS[BUS], stop_position=60.0, stop_time_mean=mean, stop_time_std=std)
    return _empty_sim(red=0, green=30, run=20, specs=(DEFAULT_SPECS[CAR], DEFAULT_SPECS[BIKE], bus), **kw)


def test_bus_stops_at_its_stop_for_the_dwell_time():
    sim = _bus_with_stop(mean=12.0)
    bus = _place(sim, BUS, 20.0, pax=40)
    stopped_at, stopped_ticks = [], 0
    while sim.n and sim.tick < 400:
        sim.step()
        if sim.n and sim.stopped[0]:
            stopped_ticks += 1
            stopped_at.append(float(sim.x[0]))
    assert stopped_ticks == 120  # 12 s of passengers getting off and on, without extra reaction
    assert np.allclose(stopped_at, 60.0)  # stopped with its front at the stop
    assert sim.stops[BUS] == 1 and sim.summary()["stop_time"][BUS] == pytest.approx(12.0)
    assert sim.cum_veh[BUS] == 1  # afterwards it continued until crossing
    assert bus == 0


def test_followers_wait_behind_the_stopped_bus():
    sim = _bus_with_stop(mean=10.0)
    _place(sim, BUS, 58.0, pax=40)
    car = _place(sim, CAR, 58.0 - 12.0 - 6.0, pax=2)
    for _ in range(60):  # the bus arrives and has been at the stop for a few seconds
        sim.step()
    assert sim.stopped[0] and sim.stopped[1]
    gap = sim.x[0] - sim.vlen[0] - sim.x[1]
    assert DEFAULT_SPECS[CAR].gap_stop - 1e-9 <= gap < DEFAULT_SPECS[CAR].gap_run  # compressed behind
    assert car == 1


def test_stop_time_distribution():
    from traffic.distributions import stop_ticks

    rng = np.random.default_rng(3)
    spec = replace(DEFAULT_SPECS[BUS], stop_position=60.0, stop_time_mean=4.0, stop_time_std=4.0)
    ticks = stop_ticks(rng, spec, 100_000)
    assert ticks.min() >= 0
    assert ticks.mean() * DT == pytest.approx(spec.expected_stop_time, abs=0.03)  # truncated: ~5.2 s, not 4
    assert spec.expected_stop_time > 5.0
    assert replace(spec, stop_time_std=0.0).expected_stop_time == 4.0
    assert DEFAULT_SPECS[BUS].expected_stop_time == 0.0  # no stop


def _run(cfg: SimConfig, seed: int = 11) -> Simulation:
    sim = Simulation(cfg, np.random.default_rng(seed))
    sim.run()
    return sim


def test_bus_stop_changes_only_the_bus():
    """Generators per process and per type: with the bus in its exclusive lane, enabling its stop
    leaves the other types identical (arrivals, passengers, reactions and lane changes)."""
    car, bike, bus = DEFAULT_SPECS
    specs = (car, replace(bike, lane=0, exclusive=True), replace(bus, lane=2, exclusive=True))
    base = dict(length=200, lanes=3, rates=(25, 6, 3), red=20, green=25, run=30, specs=specs)
    with_stop = SimConfig(**{**base, "specs": (*specs[:2], replace(specs[2], stop_position=120.0))})
    a, b = _run(SimConfig(**base)), _run(with_stop)
    others = [CAR, BIKE]
    for field in ("arrived_veh", "cum_veh", "cum_pax", "travel_ticks", "pax_m"):
        np.testing.assert_array_equal(getattr(a, field)[others], getattr(b, field)[others], err_msg=field)
    assert a.lane_changes == b.lane_changes
    np.testing.assert_array_equal(a.arrived_veh[BUS], b.arrived_veh[BUS])  # the same bus arrivals
    np.testing.assert_array_equal(a.pax_hist, b.pax_hist)  # and the same passengers
    assert b.stops[BUS] > 0 and b.travel_ticks[BUS] / b.cum_veh[BUS] > a.travel_ticks[BUS] / a.cum_veh[BUS]


def test_demand_of_one_type_does_not_shift_the_others():
    base = dict(length=200, lanes=2, red=20, green=25, run=30)
    a = _run(SimConfig(rates=(25, 6, 3), **base))
    b = _run(SimConfig(rates=(25, 6, 0.5), **base))  # another bus rate
    c = _run(SimConfig(rates=(25, 6, 3, 10), specs=DEFAULT_SPECS + EXTRA_TYPES[:1], **base))  # a new type
    for other in (b, c):
        np.testing.assert_array_equal(a.arrived_veh[[CAR, BIKE]], other.arrived_veh[[CAR, BIKE]])
        np.testing.assert_array_equal(a.pax_hist[[CAR, BIKE]], other.pax_hist[[CAR, BIKE]])
    np.testing.assert_array_equal(a.pax_hist[BUS], c.pax_hist[BUS])


def _bikes_at_red(abreast: int, count: int) -> Simulation:
    bike = replace(DEFAULT_SPECS[BIKE], abreast=abreast)
    sim = _empty_sim(red=60, green=30, run=15, specs=(DEFAULT_SPECS[CAR], bike, DEFAULT_SPECS[BUS]))
    for k in range(count):  # in a row, separated, approaching the line on red
        _place(sim, BIKE, 60.0 - k * 4.0, pax=k + 1)
    for _ in range(round(40 / DT)):  # 40 s on red: they all arrive and stop
        sim.step()
    return sim


def test_bikes_stop_side_by_side_in_pairs():
    sim = _bikes_at_red(abreast=2, count=5)
    assert sim.stopped[: sim.n].all()
    front = np.sort(sim.x[: sim.n])[::-1]
    # Two rows of two with aligned fronts and the fifth queued behind the second pair.
    assert front[0] == pytest.approx(sim.L) and front[1] == pytest.approx(sim.L - 0.01)
    bike = DEFAULT_SPECS[BIKE]
    second = sim.L - 0.01 - bike.length - bike.gap_stop
    assert front[2] == pytest.approx(second) and front[3] == pytest.approx(second - 0.01)
    assert front[4] == pytest.approx(second - 0.01 - bike.length - bike.gap_stop)
    occ = sim.occupancy()
    assert sorted(np.bincount(sim.rows(occ)[1])) == [1, 2, 2]
    # The queue takes 3 places (not 5) in the lane's saturation.
    assert sim._lane_saturation(occ)[0] == pytest.approx(3 * (bike.length + bike.gap_stop) / sim.L)
    # On green they all leave, one after another.
    while sim.tick < sim.cfg.n_ticks:
        sim.step()
    assert sim.cum_veh[BIKE] == 5


def test_without_abreast_bikes_queue_single_file():
    sim = _bikes_at_red(abreast=1, count=3)
    front = np.sort(sim.x[: sim.n])[::-1]
    slot = DEFAULT_SPECS[BIKE].length + DEFAULT_SPECS[BIKE].gap_stop
    np.testing.assert_allclose(front, [sim.L, sim.L - slot, sim.L - 2 * slot])


def test_cars_do_not_pull_beside_bikes():
    bike = replace(DEFAULT_SPECS[BIKE], abreast=2)
    sim = _empty_sim(red=60, green=30, run=15, specs=(DEFAULT_SPECS[CAR], bike, DEFAULT_SPECS[BUS]))
    _place(sim, BIKE, 60.0, pax=1)
    _place(sim, CAR, 50.0, pax=2)
    for _ in range(round(40 / DT)):
        sim.step()
    car = int(np.flatnonzero(sim.vtype[: sim.n] == CAR)[0])
    assert sim.x[car] == pytest.approx(sim.L - bike.length - DEFAULT_SPECS[CAR].gap_stop)


GRADUAL = {"car": (2.5, 4.5), "bike": (1.0, 3.0), "bus": (1.2, 3.5), "motorbike": (3.0, 5.0), "carga": (1.0, 3.0),
           "tram": (1.0, 3.0)}  # (accel, decel) in m/s²


def _gradual(spec: VehicleSpec) -> VehicleSpec:
    accel, decel = GRADUAL[spec.key]
    return replace(spec, accel=accel, decel=decel)


def test_gradual_braking_before_red_and_acceleration_after_green():
    """A car at 50 km/h brakes for the red without exceeding its decel and stops right at the line; on
    green, after its reaction, it accelerates at most at its accel up to its speed."""
    car, bike, bus = DEFAULT_SPECS
    car = _gradual(car)
    b = Behavior(reaction_min=1.0, reaction_max=1.0, reaction_mean=1.0, reaction_std=0.0)
    cfg = SimConfig(length=400, lanes=1, rates=(0, 0, 0), red=30, green=60, start_phase="red", run=8,
                    specs=(car, bike, bus), behavior=b)  # fmt: skip
    sim = Simulation(cfg, np.random.default_rng(0))
    _place(sim, CAR, 200.0)
    v = []
    while sim.n:  # until it leaves the segment
        sim.step()
        if sim.n:
            v.append(sim.v_last[0] / DT)
        if sim.tick == round(30 / DT):
            assert sim.stopped[0] and sim.x[0] == pytest.approx(cfg.length)  # stopped at the line
    a = np.diff(np.array(v)) / DT
    assert -a.min() <= car.decel + 1e-6 and a.max() <= car.accel + 1e-6
    assert max(v) == pytest.approx(car.speed)  # it was going at its speed before braking
    assert np.count_nonzero(a < -1e-9) > 20  # it braked over several seconds, not all at once
    assert np.count_nonzero(a > 1e-9) > 10  # and it started accelerating, not all at once


def test_gradual_dynamics_lengthen_queue_discharge():
    """Six cars stopped at the red: with gradual acceleration they cross more spaced out (the first also
    loses time when starting) than with instantaneous speed changes."""
    b = Behavior(reaction_min=1.0, reaction_max=1.0, reaction_mean=1.0, reaction_std=0.0)

    def headways(car: VehicleSpec) -> np.ndarray:
        cfg = SimConfig(length=400, lanes=1, rates=(0, 0, 0), red=20, green=60, start_phase="red", run=5,
                        specs=(car, *DEFAULT_SPECS[1:]), behavior=b)  # fmt: skip
        sim = Simulation(cfg, np.random.default_rng(0))
        for k in range(6):
            _place(sim, CAR, 380.0 - 8 * k)
        crossed = []
        while len(crossed) < 6:
            before = sim.cum_veh[CAR]
            sim.step()
            crossed += [sim.tick * DT] * int(sim.cum_veh[CAR] - before)
        return np.diff(crossed)

    instant, gradual = headways(DEFAULT_SPECS[CAR]), headways(_gradual(DEFAULT_SPECS[CAR]))
    assert gradual.mean() > instant.mean() + 0.2
    assert gradual[0] > gradual[-1]  # the first loses more time when starting from 0


def test_invariants_with_gradual_dynamics():
    """All types with gradual acceleration and braking, with in-lane overtaking, overtaking motorcycles,
    a bus stop, bottlenecks, yellow, congestion, an entry queue with reaction and a limited exit
    queue in two lanes: no
    overlaps, no crossing on red and vehicles are conserved."""
    car, bike, bus = DEFAULT_SPECS
    specs = [car, bike, bus, *EXTRA_TYPES, CARGA]
    specs[CAR] = replace(car, bottleneck_prob=0.3, bottleneck_time_mean=8.0, bottleneck_time_std=3.0,
                         speed_std=10.0, speed_min=30.0, speed_max=70.0)  # fmt: skip
    specs[BIKE] = replace(bike, lane=0, exclusive=True, abreast=2, pass_in_lane=True, speed_kmh=15.0, speed_std=5.0,
                          speed_min=6.0, speed_max=25.0)  # fmt: skip
    specs[BUS] = replace(bus, lane=3, exclusive=True, stop_position=100.0, stop_time_mean=8.0, stop_time_std=3.0)
    specs[3] = replace(specs[3], overtake=True, lookahead=50.0, min_advantage=1.0, lane_change_cooldown=1.0)
    specs = tuple(_gradual(s) for s in specs)
    cfg = SimConfig(
        length=150, lanes=4, lane_speed_limit=(30.0, 50.0, 40.0, 80.0), rates=(30, 8, 2, 10, 1, 3), red=20,
        green=15, yellow=3.0, run=30, specs=specs, initial_occupancy=0.4, bottleneck=Bottleneck(stop_lanes=(1, 2)),
        behavior=Behavior(queue_reaction=True), exit_capacity=(0, 12, 8, 0), exit_storage=12,
    )  # fmt: skip
    sim = Simulation(cfg, np.random.default_rng(3))
    for _ in range(cfg.n_ticks):
        red = cfg.is_red(sim.tick)
        crossed_before = sim.cum_veh.sum()
        x_before = dict(zip(sim.vid[: sim.n].tolist(), sim.x[: sim.n].tolist()))
        sim.step()
        gap, min_gap, fol = _gaps(sim)
        moved = sim.x[fol] > np.array([x_before.get(v, -np.inf) for v in sim.vid[fol].tolist()]) + 1e-12
        assert np.all((gap >= min_gap - 1e-9) | ~moved), "traslape o gap menor al comprimido"
        _check_rows(sim)
        if red:
            assert sim.cum_veh.sum() == crossed_before, "cruzó en rojo"
    _check_conservation(sim)
    assert sim.cum_veh.sum() > 50 and sim.lane_changes > 0 and sim.in_lane_passes[BIKE] > 0


CAR_EXIT_M = 5.5  # what a car takes in the exit queue: length 4.5 + stopped gap 1


def _exit_run(capacity: float, storage: float = 3 * CAR_EXIT_M, **kw) -> Simulation:
    """One lane on green almost all the time, 30 cars/min (more than the exit accepts); the exit queue
    holds 3 cars."""
    base = dict(length=150, lanes=1, rates=(30, 0, 0), red=2, green=120, start_phase="green", run=30,
                exit_capacity=capacity, exit_storage=storage)  # fmt: skip
    base.update(kw)
    cfg = SimConfig(**base)
    sim = Simulation(cfg, np.random.default_rng(2))
    for _ in range(cfg.n_ticks):
        before = sim.lane_crossed[0]
        sim.step()
        now = sim.lane_crossed[0] - before
        if now and capacity > 0:
            # A car only crosses if it fitted in the exit queue, after subtracting what left in the step (or
            # the queue was empty).
            before_m = sim.exit_q[0] - now * CAR_EXIT_M
            assert before_m < 1e-9 or before_m + CAR_EXIT_M <= sim.exit_storage[0] + 1e-9
        # At most one extra car if two cross in the same step.
        assert sim.exit_q[0] <= sim.exit_storage[0] + CAR_EXIT_M + 1e-9
    return sim


def test_exit_queue_limits_what_crosses_and_saturates_the_lane():
    """With an exit that accepts 10 veh/min, ~10 veh/min cross the line even if 30 arrive (without it,
    more than 20): the line closes with the exit full and the lane saturates."""
    free, limited = _exit_run(0.0), _exit_run(10.0)
    minutes = free.cfg.sim_seconds / 60
    assert free.lane_crossed[0] / minutes > 20
    assert limited.lane_crossed[0] / minutes == pytest.approx(10.0, abs=0.5)
    assert limited.exit_blocked_ticks[0] > 0.3 * limited.tick  # the line was closed a good part of the time
    assert free.exit_blocked_ticks[0] == 0
    sat_free = free.recorder.lane_sat[: free.recorder.count, 0].mean()
    sat_limited = limited.recorder.lane_sat[: limited.recorder.count, 0].mean()
    assert sat_limited > sat_free + 0.3
    summary = limited.summary()
    assert summary["exit_blocked"][0] == pytest.approx(limited.exit_blocked_ticks[0] / limited.tick)
    # Saturation propagates backwards: the recorded entry queue grows and ends up as the actual one.
    entry = limited.recorder.entry_q[: limited.recorder.count, 0]
    assert entry[-1] == len(limited.queues[0]) and entry[-1] > free.recorder.entry_q[free.recorder.count - 1, 0]
    assert limited.recorder.exit_q[: limited.recorder.count, 0].max() == pytest.approx(3 * CAR_EXIT_M)
    assert len(limited.exit_items[0]) * CAR_EXIT_M == pytest.approx(limited.exit_q[0])


def test_exit_queue_storage_is_in_meters():
    """The exit queue measures m: a bus (12 m + 1.5) does not fit where there is already a car in 16.5 m, and waits until
    the queue empties; an empty queue accepts anyone even if it is shorter than the vehicle."""
    sim = _empty_sim(length=100, lanes=1, red=2, green=200, start_phase="green", run=20, exit_capacity=6.0,
                     exit_storage=3 * CAR_EXIT_M)  # fmt: skip
    sim._queue_exit(0, CAR_EXIT_M)
    bus = _place(sim, BUS, 95.0, pax=10)
    left_at = None
    while not sim.lane_crossed[0] and sim.tick < 400:
        sim.step()
        if sim.exit_items[0] and left_at is None:
            assert sim.x[bus] <= sim.L + 1e-9 and sim.exit_closed[0]  # the car continues: the bus does not fit
        elif left_at is None:
            left_at = sim.tick
    assert left_at == pytest.approx(10 / DT, abs=1)  # at 6 veh/min the car leaves after 10 s
    assert sim.lane_crossed[0] == 1 and sim.exit_q[0] == pytest.approx(12.0 + 1.5)  # the bus enters alone
    tiny = _empty_sim(length=100, lanes=1, red=2, green=200, start_phase="green", run=20, exit_capacity=6.0,
                      exit_storage=2.0)  # fmt: skip
    _place(tiny, CAR, 95.0, pax=1)
    for _ in range(20):
        tiny.step()
    assert tiny.lane_crossed[0] == 1 and tiny.exit_q[0] == pytest.approx(CAR_EXIT_M)


def test_exit_queue_with_gradual_braking_stops_at_the_line():
    """With gradual braking, the line closed by a full exit is respected just like the red."""
    car = _gradual(DEFAULT_SPECS[CAR])
    sim = _exit_run(10.0, specs=(car, *DEFAULT_SPECS[1:]))
    assert sim.lane_crossed[0] / (sim.cfg.sim_seconds / 60) == pytest.approx(10.0, abs=0.5)


def _two_bikes(pass_in_lane: bool) -> tuple[Simulation, int, int]:
    """A slow bike (10 km/h) 10 m ahead of a fast one (20 km/h), on green and in the same lane."""
    bike = replace(DEFAULT_SPECS[BIKE], abreast=2, pass_in_lane=pass_in_lane)
    sim = _empty_sim(length=200, red=10, green=200, start_phase="green", run=20,
                     specs=(DEFAULT_SPECS[CAR], bike, DEFAULT_SPECS[BUS]))  # fmt: skip
    slow = _place(sim, BIKE, 30.0, pax=1)
    fast = _place(sim, BIKE, 20.0, pax=2)
    sim.vmax[slow], sim.vmax[fast] = 10 / 3.6 * DT, 20 / 3.6 * DT
    return sim, slow, fast


def _queue_entries(queue_reaction: bool) -> tuple[Simulation, list[int]]:
    """Cars on a 50 m single-lane segment, with more demand than fits: the queue reaches the entry.
    Fixed reaction of 2 s. Returns the simulation and the steps in which a car entered during the first green."""
    b = Behavior(reaction_min=2.0, reaction_max=2.0, reaction_mean=2.0, reaction_std=0.0, queue_reaction=queue_reaction)
    cfg = SimConfig(length=50, lanes=1, rates=(40, 0, 0), red=40, green=30, run=7, behavior=b)
    sim = Simulation(cfg, np.random.default_rng(5))
    entries, before = [], 0
    while sim.tick < cfg.n_ticks:
        sim.step()
        entered = int(sim.entered_veh.sum())
        if entered > before and cfg.red_ticks <= sim.tick - 1 < cfg.red_ticks + cfg.green_ticks:
            entries.append(sim.tick - 1)
        before = entered
    return sim, entries


def test_stopped_entry_queue_starts_with_reaction_time():
    """With queue_reaction, the stopped entry queue starts like the rest of the queue: each car enters
    at least its reaction (2 s) after the previous one; without the option they enter as soon as there is room."""
    on, entries_on = _queue_entries(True)
    off, entries_off = _queue_entries(False)
    assert len(entries_on) >= 3 and np.diff(entries_on).min() >= round(2.0 / DT)
    assert np.diff(entries_off).min() < round(2.0 / DT)
    assert len(entries_on) < len(entries_off) and on.cum_veh[CAR] < off.cum_veh[CAR]


def test_faster_bike_passes_a_slower_one_within_the_lane():
    sim, slow, fast = _two_bikes(pass_in_lane=True)
    for _ in range(round(20 / DT)):  # in 20 s the fast one advances 111 m and the slow one 56 m
        sim.step()
        assert (sim.lane[: sim.n] == 0).all() and sim.lane_changes == 0
    # Neither braked: each advanced at its speed and the fast one ended up ahead.
    assert sim.x[fast] == pytest.approx(20.0 + 20 / 3.6 * 20, rel=1e-6)
    assert sim.x[slow] == pytest.approx(30.0 + 10 / 3.6 * 20, rel=1e-6)
    assert sim.in_lane_passes[BIKE] == 1 and sim.summary()["in_lane_passes"][BIKE] == 1


def test_without_pass_in_lane_the_faster_bike_follows():
    sim, slow, fast = _two_bikes(pass_in_lane=False)
    for _ in range(round(20 / DT)):
        sim.step()
    # It follows at gap_run from where the slow one was one step before (parallel update).
    gap_run = DEFAULT_SPECS[BIKE].gap_run
    assert sim.x[fast] == pytest.approx(sim.x[slow] - 10 / 3.6 * DT - DEFAULT_SPECS[BIKE].length - gap_run)
    assert sim.in_lane_passes.sum() == 0


def test_in_lane_passing_invariants_every_step():
    """Many bikes with different speeds in their exclusive lane: no overlaps except side by side
    of the same type, at most two at a time, no crossing on red, and there are overtakes."""
    car, bike, bus = DEFAULT_SPECS
    bike = replace(bike, lane=0, exclusive=True, abreast=2, pass_in_lane=True,
                   speed_kmh=15.0, speed_std=5.0, speed_min=6.0, speed_max=25.0)  # fmt: skip
    cfg = SimConfig(length=150, lanes=2, rates=(10, 30, 0), red=20, green=15, yellow=3.0, run=30,
                    specs=(car, bike, bus))  # fmt: skip
    sim = Simulation(cfg, np.random.default_rng(4))
    for _ in range(cfg.n_ticks):
        red = cfg.is_red(sim.tick)
        crossed_before = sim.cum_veh.sum()
        x_before = dict(zip(sim.vid[: sim.n].tolist(), sim.x[: sim.n].tolist()))
        sim.step()
        gap, min_gap, fol = _gaps(sim)
        moved = sim.x[fol] > np.array([x_before.get(v, -np.inf) for v in sim.vid[fol].tolist()]) + 1e-12
        assert np.all((gap >= min_gap - 1e-9) | ~moved), "traslape o gap menor al comprimido"
        _check_rows(sim)
        if red:
            assert sim.cum_veh.sum() == crossed_before, "cruzó en rojo"
    _check_conservation(sim)
    assert sim.in_lane_passes[BIKE] > 10 and sim.cum_veh[BIKE] > 0


@pytest.mark.parametrize("start", ["red", "green"])
def test_light_cycle_is_green_yellow_red(start):
    cfg = SimConfig(red=10, green=20, yellow=3, start_phase=start, run=20)
    phases = [cfg.phase(t) for t in range(cfg.n_ticks)]
    # Consecutive phases without repeating: always green → yellow → red.
    order = [p for k, p in enumerate(phases) if k == 0 or p != phases[k - 1]]
    first = order.index(GREEN)
    assert order[first:first + 4] == [GREEN, YELLOW, RED, GREEN]
    for intervals, phase in ((cfg.red_intervals(), RED), (cfg.yellow_intervals(), YELLOW)):
        ticks = {t for a, b in intervals for t in range(round(a / DT), round(b / DT))}
        assert ticks == {t for t, p in enumerate(phases) if p == phase}


def test_yellow_slows_vehicles_approaching_the_light():
    """On yellow, a vehicle closer than yellow_approach (50 m) to the line goes at half its
    speed and can cross; one farther away does not change. When the red comes, it stops."""
    sim = _empty_sim(length=300, lanes=2, red=10, green=0.1, yellow=10, start_phase="green", run=30)
    near = _place(sim, CAR, sim.L - 20.0, lane=0)
    far = _place(sim, CAR, sim.L - 100.0, lane=1)
    far_vid = sim.vid[far]
    sim.step()  # green: full speed
    assert sim.x[near] == pytest.approx(sim.L - 20.0 + sim.v_step[CAR])
    assert sim.cfg.phase(sim.tick) == YELLOW
    x_near, x_far = sim.x[near], sim.x[far]
    sim.step()
    assert sim.x[near] - x_near == pytest.approx(0.5 * sim.v_step[CAR])
    assert sim.x[far] - x_far == pytest.approx(sim.v_step[CAR])
    while sim.cfg.phase(sim.tick) == YELLOW:
        sim.step()
    assert sim.cum_veh[CAR] == 1  # the close one crossed on yellow (and left: the indices are compacted)
    far = int(np.flatnonzero(sim.vid[: sim.n] == far_vid)[0])
    # The far one enters the zone during the yellow, brakes and does not manage to cross before the red.
    assert 0 < sim.L - sim.x[far] < 50.0
    while sim.cfg.phase(sim.tick) == RED:
        sim.step()
    assert sim.cum_veh[CAR] == 1 and sim.stopped[far]


def test_without_yellow_nothing_changes():
    a = Simulation(SimConfig(length=150, lanes=2, red=20, green=15, run=10), np.random.default_rng(1))
    b = Simulation(SimConfig(length=150, lanes=2, red=20, green=15, yellow=0, run=10,
                             behavior=Behavior(yellow_approach=80, yellow_speed_factor=0.2)),
                   np.random.default_rng(1))  # fmt: skip
    a.run()
    b.run()
    np.testing.assert_array_equal(a.x[: a.n], b.x[: b.n])


def _cargo_sim(car_cargo: float, seed: int = 4) -> Simulation:
    car, bike, bus = DEFAULT_SPECS
    cfg = SimConfig(
        length=200, lanes=2, rates=(30, 8, 0, 4), red=20, green=20, run=30,
        specs=(replace(car, cargo_prob=car_cargo), bike, bus, CARGA),
    )  # fmt: skip
    sim = Simulation(cfg, np.random.default_rng(seed))
    sim.run()
    return sim


def test_cargo_vehicles_occupy_the_road_but_carry_no_passengers():
    sim = _cargo_sim(0.3)
    carga = 3
    s = sim.summary()
    # The freight type moves, but adds no passengers and does not appear in the histogram.
    assert s["crossed_veh"][carga] > 0 and s["arrived_cargo"][carga] == s["arrived_veh"][carga]
    assert s["arrived_pax"][carga] == 0 and s["crossed_pax"][carga] == 0 and sim.pax_hist[carga].sum() == 0
    assert np.isnan(s["pax_per_veh"][carga])
    # Cars: ~30 % with freight; the passenger average is only of those carrying passengers.
    share = s["arrived_cargo"][CAR] / s["arrived_veh"][CAR]
    assert 0.2 < share < 0.4
    assert sim.pax_hist[CAR, 0] == 0 and sim.pax_hist[CAR].sum() == s["arrived_veh"][CAR] - s["arrived_cargo"][CAR]
    assert s["pax_per_veh"][CAR] == pytest.approx(s["arrived_pax"][CAR] / sim.pax_hist[CAR].sum())
    assert s["pax_per_veh"][CAR] >= 1.0
    _check_conservation(sim)


def test_cargo_does_not_change_the_other_random_draws():
    """The bikes' arrivals and passengers do not change with the cars' freight probability."""
    a, b = _cargo_sim(0.0), _cargo_sim(0.5)
    np.testing.assert_array_equal(a.arrived_veh, b.arrived_veh)
    np.testing.assert_array_equal(a.pax_hist[BIKE], b.pax_hist[BIKE])
    assert a.arrived_cargo[CAR] == 0 and b.arrived_cargo[CAR] > 0


def test_each_cargo_vehicle_has_its_own_length():
    sim = _cargo_sim(0.0)
    lengths = set()
    for _ in range(200):
        sim.step()
        n = sim.n
        lengths |= set(sim.vlen[:n][sim.vtype[:n] == 3].round(6).tolist())
        np.testing.assert_array_equal(sim.vlen[:n][sim.vtype[:n] == CAR], 4.5)
    assert len(lengths) > 3 and 8.0 <= min(lengths) and max(lengths) <= 16.0


def test_each_vehicle_has_its_own_max_speed():
    """With variable speed, each car has its own and can advance at it without exceeding the lane's
    limit; types with a fixed speed do not change."""
    car, bike, bus = DEFAULT_SPECS
    fast = replace(car, speed_kmh=60.0, speed_std=10.0, speed_min=40.0, speed_max=80.0)
    cfg = SimConfig(length=300, lanes=1, lane_speed_limit=70, rates=(20, 5, 0), red=0.1, green=100,
                    start_phase="green", run=6, specs=(fast, bike, bus))  # fmt: skip
    sim = Simulation(cfg, np.random.default_rng(3))
    speeds = set()
    for _ in range(300):
        sim.step()
        n = sim.n
        cars = sim.vtype[:n] == CAR
        kmh = sim.vmax[:n] / DT * 3.6
        speeds |= set(kmh[cars].round(6).tolist())
        np.testing.assert_allclose(kmh[sim.vtype[:n] == BIKE], bike.speed_kmh)
        # One lane, without congestion or yellow: it can advance at its maximum, without exceeding the limit.
        np.testing.assert_allclose(sim.vcap_last[:n][cars], np.minimum(sim.vmax[:n][cars], 70 / 3.6 * DT))
    assert len(speeds) > 5 and 40.0 <= min(speeds) and max(speeds) <= 80.0


def test_free_flow_time_averages_over_the_speed_distribution():
    """With variable speed, the free-flow time is the mean of length / min(v, limit), greater than
    length divided by the mean speed; with std = 0 it is length / speed."""
    car, bike, bus = DEFAULT_SPECS
    cfg = SimConfig(length=300, lanes=2, lane_speed_limit=(70, 70), specs=(car, bike, bus))
    assert cfg.free_flow_time(CAR) == pytest.approx(300 / (min(car.speed_kmh, 70) / 3.6))
    varied = replace(car, speed_kmh=50.0, speed_std=10.0, speed_min=30.0, speed_max=90.0)
    cfg = replace(cfg, specs=(varied, bike, bus))
    rng = np.random.default_rng(0)
    from traffic.distributions import sample_speeds

    v = sample_speeds(rng, varied, 400_000)
    expected = (300 / (np.minimum(v, 70) / 3.6)).mean()
    assert cfg.free_flow_time(CAR) == pytest.approx(expected, rel=2e-3)
    assert cfg.free_flow_time(CAR) > 300 / (50 / 3.6)


def _initial_cfg(occupancy, **kw) -> SimConfig:
    """Cars and motorcycles; bikes in their exclusive lane (0) and buses in theirs (3)."""
    car, bike, bus = DEFAULT_SPECS
    base = dict(
        length=300, lanes=4, rates=(18, 8, 1, 8), red=20, green=20, run=10, initial_occupancy=occupancy,
        specs=(car, replace(bike, lane=0, exclusive=True), replace(bus, lane=3, exclusive=True), MOTO),
    )  # fmt: skip
    base.update(kw)
    return SimConfig(**base)


def test_initial_occupancy_fills_each_lane():
    occupancy = (0.2, 0.5, 0.8, 0.3)
    sim = Simulation(_initial_cfg(occupancy), np.random.default_rng(2))
    n = sim.n
    assert n > 0 and sim.tick == 0
    np.testing.assert_array_equal(sim.initial_veh.sum(), n)
    rear = sim.x[:n] - sim.vlen[:n]
    assert rear.min() >= -1e-9 and sim.x[:n].max() <= sim.L + 1e-9
    fp = sim.vlen[:n] + sim.gap_stop_t[sim.vtype[:n]]
    for lane, target in enumerate(occupancy):
        on = sim.lane[:n] == lane
        filled = fp[on].sum() / sim.L
        assert target - 17.0 / sim.L < filled <= target + 1e-9  # at most one bus fewer
        # Only the types that can use the lane.
        assert set(sim.vtype[:n][on].tolist()) <= {k for k in range(4) if sim.allowed[k, lane]}
    gap, _, fol = _gaps(sim)
    assert np.all(gap >= sim.gap_stop_t[sim.vtype[fol]] - 1e-9)
    # A vehicle closer than its gap_run starts stopped (jam); the others, moving.
    np.testing.assert_array_equal(sim.stopped[fol], gap < sim.gap_run_t[sim.vtype[fol]] - 1e-6)
    assert set(sim.vtype[:n][sim.lane[:n] == 1].tolist()) == {CAR, MOTO_IDX}  # mix according to the rates
    _check_conservation(sim)


@pytest.mark.parametrize("start", ["red", "green"])
def test_full_occupancy_is_a_compact_queue(start):
    """With 1, the lanes are left as a compact queue: each vehicle almost at its gap_stop from the one
    ahead, all stopped except the first, and the queue stretches without overlaps."""
    sim = Simulation(_initial_cfg(1.0, start_phase=start, run=3), np.random.default_rng(3))
    gap, min_gap, fol = _gaps(sim)
    stop_gap = sim.gap_stop_t[sim.vtype[fol]]
    assert np.all(gap >= stop_gap - 1e-9) and np.all(gap < stop_gap + 1.0)
    assert np.all(sim.stopped[fol])
    for _ in range(sim.cfg.n_ticks):
        sim.step()
        gap, min_gap, _ = _gaps(sim)
        assert np.all(gap >= min_gap - 1e-9)
    assert sim.cum_veh.sum() > 0


def test_initial_vehicles_have_no_travel_time():
    """Those of the initial condition cross and add passengers, but do not count in the travel time;
    in 10 s nobody who entered covers 300 m."""
    sim = Simulation(_initial_cfg(0.5, red=0, green=20, run=1), np.random.default_rng(1))  # 10 simulated s
    sim.run()
    s = sim.summary()
    assert s["crossed_veh"].sum() > 0 and s["crossed_pax"].sum() > 0
    assert sim.timed_veh.sum() == 0 and np.all(np.isnan(s["travel_time"]))
    _check_conservation(sim)


def test_initial_occupancy_does_not_change_the_arrivals():
    a = Simulation(_initial_cfg(0.0), np.random.default_rng(5))
    b = Simulation(_initial_cfg(0.6), np.random.default_rng(5))
    a.run()
    b.run()
    np.testing.assert_array_equal(a.arrived_veh, b.arrived_veh)
    np.testing.assert_array_equal(a.pax_hist, b.pax_hist)
    assert a.initial_veh.sum() == 0 and b.initial_veh.sum() > 0


def _bottleneck_sim(**bn) -> Simulation:
    car, bike, bus = DEFAULT_SPECS
    specs = (replace(car, bottleneck_prob=1.0, bottleneck_time_mean=20.0), bike, bus)
    return _empty_sim(length=400, lanes=2, red=0, green=60, run=60, seed=1, specs=specs, bottleneck=Bottleneck(**bn))


def _place_stopping(sim: Simulation, vt: int, x: float, lane: int, at: float) -> int:
    sim._add(vt, 1, lane, bottleneck_at=at)
    sim.x[sim.n - 1] = x
    return sim.n - 1


def test_bottleneck_stop_makes_followers_change_lane():
    """A car stops for 20 s at x = 100 in lane 0; the one behind, without special code, treats it
    as a stopped leader: it changes to lane 1 and overtakes it. Then the stopped one continues."""
    sim = _bottleneck_sim()
    stopper = _place_stopping(sim, CAR, 60.0, 0, at=100.0)
    follower = _place(sim, CAR, 60.0 - 4.5 - 20.0, lane=0)
    ids = sim.vid[[stopper, follower]].tolist()
    stopped_ticks, passed = 0, False
    for _ in range(round(40 / DT)):
        sim.step()
        idx = {v: i for i, v in enumerate(sim.vid[: sim.n].tolist())}
        a, b = idx.get(ids[0]), idx.get(ids[1])
        if a is not None and sim.stopped[a]:
            assert sim.x[a] == pytest.approx(100.0)
            stopped_ticks += 1
        if a is not None and b is not None and sim.x[b] > sim.x[a]:
            passed = sim.lane[b] == 1
    assert sim.bn_stops[CAR] == 1 and sim.bn_ticks[CAR] == round(20 / DT)
    assert abs(stopped_ticks - round(20 / DT)) <= 1
    assert passed and sim.lane_changes >= 1
    assert sim.x[{v: i for i, v in enumerate(sim.vid[: sim.n].tolist())}[ids[0]]] > 100.0  # it went on its way


def test_bottleneck_only_in_its_lanes():
    sim = _bottleneck_sim(stop_lanes=(1,))
    car = _place_stopping(sim, CAR, 60.0, 0, at=100.0)
    for _ in range(200):
        sim.step()
        assert not sim.stopped[car]
    assert sim.bn_stops.sum() == 0 and sim.x[car] > 100.0


def test_bottleneck_probability_and_types():
    """With bottleneck_prob = 0.3 for cars, ~30 % of the cars that cross stopped; the bikes
    (without bottleneck_prob) did not."""
    car, bike, bus = DEFAULT_SPECS
    cfg = SimConfig(length=200, lanes=2, rates=(30, 8, 0), red=20, green=20, run=60,
                    specs=(replace(car, bottleneck_prob=0.3, bottleneck_time_mean=5.0), bike, bus),
                    bottleneck=Bottleneck(stop_zone=(20.0, 60.0)))  # fmt: skip
    sim = Simulation(cfg, np.random.default_rng(0))
    sim.run()
    assert 0.2 < sim.bn_stops[CAR] / sim.cum_veh[CAR] < 0.4
    assert sim.bn_stops[BIKE] == 0
    s = sim.summary()
    assert s["bottleneck_time"][CAR] == pytest.approx(5.0)
    _check_conservation(sim)


def test_without_bottleneck_nothing_changes():
    base = dict(length=150, lanes=2, rates=(30, 8, 2), red=20, green=15, run=10)
    a = Simulation(SimConfig(**base), np.random.default_rng(1))
    specs = tuple(replace(s, bottleneck_time_mean=60.0) for s in DEFAULT_SPECS)  # without a probability there are no halts
    b = Simulation(SimConfig(**base, specs=specs, bottleneck=Bottleneck(stop_lanes=(0,))), np.random.default_rng(1))
    a.run()
    b.run()
    np.testing.assert_array_equal(a.x[: a.n], b.x[: b.n])
    assert b.bn_stops.sum() == 0


def test_bottleneck_per_type_probability_and_duration():
    """Each type with its probability and duration: freight stops more often (0.8) and for longer
    (30 s) than cars (0.2, 5 s); bikes, without bottleneck_prob, never."""
    car, bike, bus = DEFAULT_SPECS
    specs = (replace(car, bottleneck_prob=0.2, bottleneck_time_mean=5.0), bike, bus,
             replace(CARGA, bottleneck_prob=0.8, bottleneck_time_mean=30.0))  # fmt: skip
    cfg = SimConfig(length=300, lanes=2, rates=(30, 8, 0, 6), red=0, green=60, run=60, specs=specs,
                    bottleneck=Bottleneck(stop_zone=(30.0, 100.0)))  # fmt: skip
    assert [cfg.bottleneck_prob(k) for k in range(4)] == [0.2, 0.0, 0.0, 0.8]
    assert cfg.bottleneck_time(3) == (30.0, 0.0) and cfg.bottleneck_time(CAR) == (5.0, 0.0)
    sim = Simulation(cfg, np.random.default_rng(3))
    sim.run()
    s = sim.summary()
    assert 0.1 < sim.bn_stops[CAR] / sim.cum_veh[CAR] < 0.3
    assert 0.65 < sim.bn_stops[3] / sim.cum_veh[3] <= 1.0
    assert s["bottleneck_time"][CAR] == pytest.approx(5.0) and s["bottleneck_time"][3] == pytest.approx(30.0)
    assert sim.bn_stops[BIKE] == 0


def test_entry_queue_waiting_time():
    """The accumulated wait in the entry queue matches the vehicle·steps counted in the queue: those that
    already entered plus what those still in it have been waiting. With the road saturated the wait grows."""
    cfg = SimConfig(length=150, lanes=2, rates=(90, 20, 2), red=25, green=10, run=10)
    sim = Simulation(cfg, np.random.default_rng(6))
    in_queue = np.zeros(cfg.n_types)
    for _ in range(cfg.n_ticks):
        sim.step()
        for q in sim.queues:
            for vt, *_ in q:
                in_queue[vt] += 1
    still = np.zeros(cfg.n_types)
    for q in sim.queues:
        for vt, *_, arrived in q:
            still[vt] += sim.tick - arrived
    np.testing.assert_allclose(sim.queue_wait_ticks + still, in_queue)
    s = sim.summary()
    entered = s["entered_veh"] - s["initial_veh"]
    np.testing.assert_allclose(s["queue_wait"][CAR], sim.queue_wait_ticks[CAR] * DT / entered[CAR])
    assert s["queue_wait"][CAR] > 5.0 and s["queued"].sum() > 0


def test_no_entry_queue_wait_on_an_empty_road():
    sim = Simulation(SimConfig(length=300, lanes=2, rates=(6, 2, 0), red=10, green=50, run=5), np.random.default_rng(0))
    sim.run()
    assert sim.queue_wait_ticks.sum() == 0 and sim.summary()["queue_wait"][CAR] == 0


def test_mean_speed_is_free_flow_on_an_empty_road_in_green():
    """Without traffic and on green, the mean speed of each type is the free-flow one; a type that does not
    take part has no value."""
    cfg = SimConfig(length=300, lanes=2, rates=(3, 0, 0), red=0.1, green=100, start_phase="green", run=9)
    sim = Simulation(cfg, np.random.default_rng(1))
    sim.run()
    speed = sim.summary()["mean_speed"]
    assert speed[CAR] == pytest.approx(cfg.free_flow_kmh(CAR), rel=1e-6)
    assert np.isnan(speed[BIKE]) and np.isnan(speed[BUS])


def test_mean_speed_counts_stopped_time():
    """The mean speed is distance / time within the segment, including the time stopped at the red."""
    cfg = SimConfig(length=150, lanes=2, rates=(20, 0, 0), red=40, green=10, run=10)
    sim = Simulation(cfg, np.random.default_rng(2))
    sim.run()
    speed = sim.summary()["mean_speed"][CAR]
    assert speed == pytest.approx(sim.type_dist[CAR] / sim.type_time[CAR] * 3.6)
    assert 0 < speed < 0.5 * cfg.free_flow_kmh(CAR)


def test_fixed_rate_dicts_give_the_same_run_as_plain_rates():
    """A rate with std = 0 consumes no draws: the run is identical to that of the fixed rate."""
    base = SimConfig(length=150, lanes=2, rates=(20, 5, 1), red=20, green=20, run=5)
    dists = replace(base, rate_dists=(Rate.fixed(20), Rate(3, 9, 5, 0), Rate.fixed(1)))
    assert dists.rates == base.rates
    a, b = Simulation(base, np.random.default_rng(4)), Simulation(dists, np.random.default_rng(4))
    a.run()
    b.run()
    for key, value in a.summary().items():
        np.testing.assert_array_equal(value, b.summary()[key], err_msg=key)


def test_variable_rate_is_redrawn_every_interval():
    """Each interval draws its rate in [min, max]; the arrivals of the interval follow that rate."""
    rate = Rate(min=10, max=110, mean=60, std=40)
    cfg = SimConfig(length=150, lanes=2, rates=(0, 0, 0), rate_dists=(rate, Rate.fixed(0), Rate.fixed(0)),
                    rate_interval=30, red=10, green=50, run=24)  # fmt: skip
    sim = Simulation(cfg, np.random.default_rng(5))
    per_min = sim.rate_by_interval[CAR] * 60.0 / DT
    assert per_min.size == 8 and np.all((per_min >= 10) & (per_min <= 110)) and np.ptp(per_min) > 20
    assert BIKE not in sim.rate_by_interval
    counts = []
    for _ in range(per_min.size):
        before = sim.arrived_veh[CAR]
        for _ in range(cfg.rate_ticks):
            sim.step()
        counts.append(sim.arrived_veh[CAR] - before)
    expected = per_min * cfg.rate_interval / 60.0
    assert np.corrcoef(counts, expected)[0, 1] > 0.9


def test_rate_expected_is_the_truncated_mean():
    assert Rate.fixed(4).expected == 4
    assert Rate(0, 20, 10, 5).expected == pytest.approx(10)  # symmetric
    draws = np.random.default_rng(0).normal(15, 5, 2_000_000)
    draws = draws[(draws >= 1) & (draws <= 20)]
    assert Rate(1, 20, 15, 5).expected == pytest.approx(draws.mean(), abs=0.01)


def _front_stays_behind(sim: Simulation, line: float, phase_of, until: int) -> tuple[int, bool]:
    """Run `until` steps; returns (steps on red, whether someone passed `line` with its front while it was red)."""
    in_red, crossed_in_red = 0, False
    crossed_before = lambda: bool((sim.x[: sim.n] > line + 1e-9).any())  # noqa: E731
    for _ in range(until):
        was = crossed_before()
        red = phase_of(sim.tick) == RED
        sim.step()
        in_red += red
        if red and not was and crossed_before():
            crossed_in_red = True
    return in_red, crossed_in_red


@pytest.mark.parametrize("gradual", [False, True])
def test_inner_light_stops_vehicles_at_its_line_in_red_and_releases_them_in_green(gradual):
    """A traffic light at 60 m (red 20 s, green 20 s, starts on red) stops at its line whoever arrives on red, and on
    green lets them through; the one at the end of the segment (disabled) does not intervene."""
    spec = _gradual(DEFAULT_SPECS[CAR]) if gradual else DEFAULT_SPECS[CAR]
    light = Light(60.0, red=20, green=20, start_phase="red")
    sim = _empty_sim(length=150, red=10, green=30, traffic_light=False, extra_lights=(light,), run=10,
                     specs=(spec, *DEFAULT_SPECS[1:]))  # fmt: skip
    _place(sim, CAR, 20.0)
    in_red, crossed = _front_stays_behind(sim, 60.0, light.phase, round(20 / DT))
    assert in_red == round(20 / DT) and not crossed
    assert sim.n == 1 and sim.stopped[0] and sim.x[0] <= 60.0 + 1e-9 and sim.x[0] > 50.0  # waiting at the line
    for _ in range(round(25 / DT)):
        sim.step()
    assert sim.n == 0 or sim.x[0] > 60.0  # on green, it went through


def test_each_light_has_its_own_phases_and_initial_phase():
    """Two intermediate traffic lights with different cycles and initial phases: each one stops a single car only during its reds."""
    lights = (Light(40.0, red=10, green=30, start_phase="red"), Light(100.0, red=15, green=5, start_phase="green"))
    cfg = SimConfig(length=150, lanes=1, rates=(0, 0, 0), traffic_light=False, extra_lights=lights, run=5)
    assert [lt.phase(0) for lt in lights] == [RED, GREEN] and cfg.inner_phases(0) == (RED, GREEN)
    assert cfg.inner_phases(round(22 / DT)) == (GREEN, GREEN) and cfg.inner_phases(round(27 / DT)) == (GREEN, RED)
    sim = Simulation(cfg, np.random.default_rng(0))
    _place(sim, CAR, 10.0)
    xs = []
    for _ in range(round(120 / DT)):
        sim.step()
        if sim.n:
            xs.append(float(sim.x[0]))
    # It waits at the first (initial red) until it opens; the second (initial green) lets it through or slows it down.
    assert max(x for x in xs[: round(9 / DT)]) <= 40.0 + 1e-9
    assert not sim.n or sim.x[0] > 100.0


def test_light_starting_in_green_lets_the_first_vehicle_through():
    light = Light(60.0, red=20, green=20, start_phase="green")
    sim = _empty_sim(length=150, traffic_light=False, extra_lights=(light,), run=5)
    _place(sim, CAR, 20.0)
    for _ in range(round(8 / DT)):
        sim.step()
    assert sim.n == 1 and sim.x[0] > 60.0 and not sim.stopped[0]  # it crossed without stopping


def test_inner_light_free_lane_ignores_red_and_yellow():
    """The bikes, with an exclusive lane in the intermediate light's free_lanes, cross it on red; the car, which does not
    have an exclusive lane, stops."""
    bike = replace(DEFAULT_SPECS[BIKE], lane=0, exclusive=True)
    light = Light(60.0, red=30, green=10, start_phase="red", free_lanes=(0,))
    sim = _empty_sim(length=150, lanes=2, traffic_light=False, extra_lights=(light,), run=5,
                     specs=(DEFAULT_SPECS[CAR], bike, DEFAULT_SPECS[BUS]))  # fmt: skip
    _place(sim, BIKE, 20.0, lane=0)
    _place(sim, CAR, 20.0, lane=1)
    for _ in range(round(15 / DT)):
        sim.step()
    x = {int(v): float(sim.x[j]) for j, v in enumerate(sim.vtype[: sim.n])}
    assert BIKE not in x or x[BIKE] > 60.0  # it crossed (or already left)
    assert x[CAR] <= 60.0 + 1e-9


def test_lights_properties_and_labels():
    lights = (Light(100.0, red=20, green=20, start_phase="green"), Light(40.0, red=10, green=30, yellow=3))
    cfg = SimConfig(length=150, red=25, green=35, extra_lights=lights)
    assert [lt.position for lt in cfg.inner_lights] == [40.0, 100.0]  # from the entry to the exit
    assert [lt.position for lt in cfg.lights] == [40.0, 100.0, 150.0] and cfg.any_light
    assert cfg.ref_light.position == 150.0 and cfg.cycle == 60.0 and cfg.flow_window == 60.0
    assert cfg.lights_label == ("at 40 m: red 10 s / green 30 s / yellow 3 s, starts red · "
                                "at 100 m: red 20 s / green 20 s, starts green · "
                                "at 150 m: red 25 s / green 35 s, starts red")  # fmt: skip
    # Without a light at the end, the last intermediate one sets the window and the shaded phases.
    cfg = SimConfig(length=150, traffic_light=False, extra_lights=lights)
    assert not cfg.has_light and cfg.ref_light.position == 100.0 and cfg.cycle == 40.0 and cfg.flow_window == 40.0
    assert cfg.red_intervals()[0] == (0.0, 20.0) or cfg.red_intervals()[0][1] > 0  # those of the one at 100 m
    off = cfg.with_lights(False)
    assert not off.any_light and off.flow_window == 60.0
    assert SimConfig(length=150, red=0, green=0, extra_lights=(Light(50.0, enabled=False),)).lights == ()


def test_time_headway_spacing_and_capacity():
    """With time_headway, behind a moving leader each car keeps gap_stop + T·v: without gradual dynamics, at 50 km/h and
    T = 1.5 s, the headway between cars at capacity is ≈ T + (length + gap_stop) / v (≈ 1.9 s), not that of gap_run."""
    from traffic.calibration import measure

    car = replace(DEFAULT_SPECS[CAR], speed_std=0.0, length_std=0.0, accel=None, decel=None, time_headway=1.5)
    cfg = SimConfig(specs=(car, *DEFAULT_SPECS[1:]), rates=(10, 0, 0), lane_speed_limit=50.0)
    m = measure(cfg, "car", queues=2, rates=(120,), flow_seconds=300.0, workers=1)
    v = 50 / 3.6
    ideal = 1.5 + (car.length + car.gap_stop) / v
    assert ideal <= m["intervalo_marcha_s"] <= ideal + 0.2  # entering the segment, in steps, adds at most a little


def test_time_headway_with_gradual_dynamics_keeps_order_and_gaps():
    """Cars and buses with gradual acceleration and braking and time_headway, with a traffic light, initial condition and lane
    changes: nobody overlaps with its leader and, at speed, a close follower keeps more than 1 s."""
    car, bike, bus = DEFAULT_SPECS
    specs = tuple(replace(_gradual(s), time_headway=1.2) for s in (car, bike, bus))
    cfg = SimConfig(length=300, lanes=2, rates=(30, 0, 3), red=20, green=25, yellow=3.0, run=30, specs=specs,
                    initial_occupancy=0.4)  # fmt: skip
    sim = Simulation(cfg, np.random.default_rng(4))
    far = []
    for _ in range(cfg.n_ticks):
        sim.step()
        occ = sim.occupancy()
        gap = sim.x[occ.leader] - sim.vlen[occ.leader] - occ.x
        assert (gap[occ.has_leader] > -1e-6).all()
        moving = occ.has_leader & ~sim.stopped[occ.leader] & (sim.v_last[occ.veh] / DT > 10.0)  # > 36 km/h
        far += (gap[moving] / (sim.v_last[occ.veh][moving] / DT)).tolist()
    # At speed, a close follower is more than 1 s behind (5th percentile ≈ 1.46 s); with a fixed gap_run, ≈ 0.33 s.
    assert far and np.percentile(far, 5) > 1.0


def test_lights_short_names_a_single_intermediate_light():
    from traffic.config import Light, SimConfig

    cfg = SimConfig(length=250, traffic_light=False, extra_lights=(Light(200.0, pedestrian=True, red=5.0),))
    assert cfg.lights_short == "1 traffic light (at 200 m)"
