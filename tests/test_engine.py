from dataclasses import replace

import numpy as np
import pytest

from trafico.config import BIKE, BUS, CAR, DEFAULT_SPECS, DT, Behavior, SimConfig, VehicleSpec
from trafico.engine import Simulation


def _gaps(sim: Simulation):
    """(gap al líder, gap mínimo permitido al seguidor) para cada entrada con líder, sin contar a
    quien está al lado de su líder o acercándose a ese lugar (filas lado a lado, que se revisan con
    _check_rows)."""
    occ = sim.occupancy()
    beside = sim.rows(occ)[0]
    keep = occ.has_leader & ~beside
    if sim.any_abreast:
        keep &= ~sim._side_slot(occ)
    lead = occ.leader[keep]
    fol = occ.veh[keep]
    gap = sim.x[lead] - sim.length_t[sim.vtype[lead]] - sim.x[fol]
    min_gap = sim.gap_stop_t[sim.vtype[fol]]
    # Entre vehículos del mismo tipo con abreast > 1 basta no traslaparse: quien iba a ponerse al lado
    # puede quedar a menos del gap si el de adelante arranca en ese mismo paso.
    same_row_type = (sim.vtype[lead] == sim.vtype[fol]) & (sim.abreast[sim.vtype[fol]] > 1)
    return gap, np.where(same_row_type, 0.0, min_gap), fol


def _check_rows(sim: Simulation):
    """Filas lado a lado: solo del mismo tipo con abreast > 1, detenidos, alineados y sin exceder abreast."""
    occ = sim.occupancy()
    beside, rid, _ = sim.rows(occ)
    if not beside.any():
        return
    svt = sim.vtype[occ.veh]
    lvt = sim.vtype[occ.leader]
    assert np.all(svt[beside] == lvt[beside])
    assert np.all(sim.abreast[svt[beside]] > 1)
    assert np.all(np.abs(sim.x[occ.leader[beside]] - occ.x[beside]) < sim.length_t[svt[beside]])
    sizes = np.bincount(rid)
    assert np.all(sizes[rid] <= sim.abreast[svt]), "fila con más vehículos que abreast"


def _check_conservation(sim: Simulation):
    n = sim.n
    inside = ~sim.crossed[:n]
    s = sim.summary()
    queued_pax = np.zeros(sim.n_types)
    for q in sim.queues:
        for vt, p in q:
            queued_pax[vt] += p
    on_road_pax = np.bincount(sim.vtype[:n][inside], weights=sim.pax[:n][inside], minlength=sim.n_types)
    np.testing.assert_array_equal(s["arrived_veh"], s["queued"] + s["entered_veh"])
    np.testing.assert_array_equal(s["entered_veh"], s["on_road"] + s["crossed_veh"])
    np.testing.assert_allclose(s["arrived_pax"], queued_pax + on_road_pax + s["crossed_pax"])


# Tipos que el código no conoce: uno que cambia de carril y otro que no.
EXTRA_TYPES = (
    VehicleSpec("motorbike", "moto", 40.0, 2.0, 2.0, 1.0, 1, 2, 1.2, 0.2, True),
    VehicleSpec("tram", "tranvía", 30.0, 20.0, 6.0, 2.0, 20, 200, 120.0, 30.0, False),
)


@pytest.mark.parametrize(
    ("lanes", "extra", "separate", "congestion", "bus_stop", "abreast"),
    [
        (1, False, False, 0.0, False, 1), (3, False, False, 0.0, False, 1), (3, True, False, 0.0, False, 1),
        (3, False, True, 0.0, False, 1), (3, True, True, 0.6, False, 1), (1, False, False, 0.0, True, 1),
        (3, True, True, 0.3, True, 1), (1, False, False, 0.0, False, 2), (3, True, True, 0.3, True, 2),
        (2, False, False, 0.0, False, 3),
    ],
)  # fmt: skip
def test_invariants_every_step(lanes, extra, separate, congestion, bus_stop, abreast):
    specs, rates = DEFAULT_SPECS, (30, 8, 2)
    if extra:
        specs, rates = specs + EXTRA_TYPES, rates + (10, 1)
    if separate:  # bicis en el carril derecho y autobuses en el siguiente
        specs = (specs[CAR], replace(specs[BIKE], lane=0), replace(specs[BUS], lane=1), *specs[BUS + 1 :])
    if abreast > 1:  # bicis lado a lado al detenerse
        specs = tuple(replace(s, abreast=abreast) if k == BIKE else s for k, s in enumerate(specs))
    if bus_stop:  # parada de autobús a 100 m (el tramo mide 150)
        specs = tuple(replace(s, stop_position=100.0, stop_time_mean=8.0, stop_time_std=3.0) if k == BUS else s
                      for k, s in enumerate(specs))  # fmt: skip
    cfg = SimConfig(
        length=150, lanes=lanes, rates=rates, red=20, green=15, run=30, specs=specs,
        behavior=Behavior(congestion_factor=congestion),
    )  # fmt: skip
    sim = Simulation(cfg, np.random.default_rng(7))
    for _ in range(cfg.n_ticks):
        green = cfg.is_green(sim.tick)
        crossed_before = sim.cum_veh.sum()
        x_before = dict(zip(sim.vid[: sim.n].tolist(), sim.x[: sim.n].tolist()))
        sim.step()
        n = sim.n
        gap, min_gap, fol = _gaps(sim)
        # Al arrancar una fila lado a lado, quien estaba al lado queda un momento a menos del gap
        # mientras el de adelante se aleja: solo es error si el seguidor avanzó para quedar así.
        moved = sim.x[fol] > np.array([x_before.get(v, -np.inf) for v in sim.vid[fol].tolist()]) + 1e-12
        assert np.all((gap >= min_gap - 1e-9) | (~moved & sim.any_abreast)), "traslape o gap menor al comprimido"
        _check_rows(sim)
        if not green:
            assert sim.cum_veh.sum() == crossed_before, "cruzó en rojo"
        slow = ~sim.can_change[sim.vtype[:n]]
        assert np.all(sim.lc_target[:n][slow] == -1)
        fixed = np.array([s.lane if s.lane is not None else 0 for s in specs])  # slow_lane = "right"
        assert np.all(sim.lane[:n][slow] == fixed[sim.vtype[:n][slow]])
    _check_conservation(sim)
    assert sim.cum_veh.sum() > 0
    if extra:
        assert sim.cum_veh[-2] > 0 and sim.arrived_veh[-1] > 0  # los tipos nuevos circulan
    if separate:
        assert sim.cum_veh[BUS] > 0 and sim.cum_veh[BIKE] > 0
    if bus_stop:
        assert sim.stops[BUS] > 0 and sim.stops[BUS] >= sim.cum_veh[BUS]  # todo autobús que cruzó paró
        assert sim.stops[[CAR, BIKE]].sum() == 0
    if lanes == 1:
        assert sim.lane_changes == 0
    else:
        assert sim.lane_changes > 0


def _empty_sim(seed: int = 0, **kw) -> Simulation:
    """Tramo sin llegadas aleatorias; los vehículos se colocan a mano."""
    base = dict(length=100, lanes=1, rates=(0, 0, 0), red=10, green=30, run=5)
    base.update(kw)
    return Simulation(SimConfig(**base), np.random.default_rng(seed))


def _place(sim: Simulation, vt: int, x: float, lane: int = 0, pax: int = 1) -> int:
    sim._add(vt, pax, lane)
    i = sim.n - 1
    sim.x[i] = x
    return i


def _start_ticks(sim: Simulation, until: int) -> dict[int, int]:
    """Primer paso tras el verde en que avanza cada vehículo, identificado por su nº de pasajeros."""
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


@pytest.mark.parametrize("seed", range(5))
def test_queue_compresses_and_stretches(seed):
    sim = _empty_sim(seed=seed, length=120)
    spec = sim.cfg.specs[CAR]
    # Tres autos en marcha separados por el gap en marcha; al llegar al rojo se comprimen.
    xs = [sim.L - 20, sim.L - 20 - spec.length - spec.gap_run, sim.L - 20 - 2 * (spec.length + spec.gap_run)]
    for vid, x in enumerate(xs, start=1):
        _place(sim, CAR, x, pax=vid)
    while sim.tick < sim.cfg.red_ticks:
        sim.step()
    assert np.all(sim.stopped[:3])
    gap, _, _ = _gaps(sim)
    np.testing.assert_allclose(gap, spec.gap_stop, atol=1e-9)  # compresión: 1 m < 3 m

    starts = _start_ticks(sim, sim.cfg.red_ticks + 200)
    assert sorted(starts) == [1, 2, 3]
    # Estiramiento: cada auto arranca al menos un tiempo de reacción después de su líder.
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
            # Costo: durante la maniobra ocupa ambos carriles a velocidad reducida.
            assert sim.x[car] - x0 <= 0.5 * v_car + 1e-9
            if sim.lc_target[car] < 0:
                break
    assert sim.lane[car] == 1 and sim.lane[bike] == 0
    assert round(1.0 / DT) <= duration <= round(4.0 / DT)
    for _ in range(200):
        sim.step()
    assert sim.x[car] > sim.x[bike]  # lo rebasó por el carril izquierdo


@pytest.mark.parametrize("factor", [0.0, 0.5])
def test_congestion_slows_lane_with_queue(factor):
    """Con 4 autos detenidos en rojo (cola de 4 × 5.5 m en un tramo de 100 m), un auto
    que se acerca por el mismo carril avanza a 1 − factor × 0.22 de su velocidad."""
    sim = _empty_sim(length=100, lanes=2, red=30, green=30, behavior=Behavior(congestion_factor=factor))
    spec = sim.cfg.specs[CAR]
    for k in range(4):
        _place(sim, CAR, sim.L - k * (spec.length + spec.gap_stop), pax=k + 1)
    sim.stopped[:4] = True
    other = _place(sim, CAR, 10.0, lane=1, pax=5)  # carril sin cola: velocidad normal
    follower = _place(sim, CAR, 10.0, lane=0, pax=6)
    x0 = sim.x[: sim.n].copy()
    sim.step()
    expected = 1.0 - factor * 4 * (spec.length + spec.gap_stop) / sim.L
    assert sim.x[follower] - x0[follower] == pytest.approx(sim.v_step[CAR] * expected)
    assert sim.x[other] - x0[other] == pytest.approx(sim.v_step[CAR])


def test_lane_saturation_is_recorded():
    """4 autos detenidos en rojo en el carril 0 de un tramo de 100 m: saturación 0.22 y 0."""
    sim = _empty_sim(length=100, lanes=2, red=30, green=30)
    spec = sim.cfg.specs[CAR]
    for k in range(4):
        _place(sim, CAR, sim.L - k * (spec.length + spec.gap_stop), pax=k + 1)
    sim.stopped[:4] = True
    for _ in range(sim.cfg.sample_ticks):
        sim.step()
    assert sim.recorder.count == 1
    np.testing.assert_allclose(sim.recorder.lane_sat[0], [4 * (spec.length + spec.gap_stop) / sim.L, 0.0], rtol=1e-6)


def test_congestion_factor_per_lane():
    """Misma cola en los dos carriles, factores distintos: cada carril se frena según el suyo."""
    sim = _empty_sim(length=100, lanes=2, red=30, green=30, behavior=Behavior(congestion_factor=(0.2, 0.6)))
    spec = sim.cfg.specs[CAR]
    for lane in (0, 1):
        for k in range(4):
            _place(sim, CAR, sim.L - k * (spec.length + spec.gap_stop), lane=lane, pax=lane * 4 + k + 1)
    sim.stopped[:8] = True
    right = _place(sim, CAR, 10.0, lane=0, pax=20)
    left = _place(sim, CAR, 10.0, lane=1, pax=21)
    x0 = sim.x[: sim.n].copy()
    sim.step()
    sat = 4 * (spec.length + spec.gap_stop) / sim.L
    assert sim.x[right] - x0[right] == pytest.approx(sim.v_step[CAR] * (1 - 0.2 * sat))
    assert sim.x[left] - x0[left] == pytest.approx(sim.v_step[CAR] * (1 - 0.6 * sat))


def test_lane_speed_series():
    """Un auto solo en el carril 1 a flujo libre: 50 km/h; el carril 0 vacío no tiene velocidad (NaN)."""
    from trafico.metrics import LANE_SPEED, derive_series

    sim = _empty_sim(length=300, lanes=2, red=0, green=60, run=2)
    _place(sim, CAR, 10.0, lane=1)
    sim.run()
    speed = derive_series(sim.recorder, sim.cfg)[LANE_SPEED]
    assert np.all(np.isnan(speed[:, 0]))
    np.testing.assert_allclose(speed[:, 1], 50.0, rtol=1e-4)


def test_lane_speed_limit_is_never_exceeded():
    """Límites de 40 y 20 km/h: ningún vehículo avanza más que el límite de su carril (ni el menor
    de sus dos carriles si está cambiando), aunque su velocidad máxima sea mayor."""
    specs = DEFAULT_SPECS + EXTRA_TYPES[:1]
    cfg = SimConfig(length=200, lanes=2, rates=(30, 6, 2, 10), red=20, green=20, run=20, specs=specs,
                    lane_speed_limit=(40.0, 20.0))  # fmt: skip
    sim = Simulation(cfg, np.random.default_rng(3))
    limit = np.array([40.0, 20.0]) / 3.6 * DT

    def snapshot():
        """Vehículos por (tick de entrada, pasajeros, tipo); las claves repetidas son ambiguas y se omiten."""
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
    assert cfg.free_flow_kmh(CAR) == 45.0  # puede cambiar: el mejor carril (45 < 50)
    assert cfg.free_flow_kmh(BIKE) == 10.0  # fijo en el carril 1
    assert cfg.free_flow_kmh(BUS) == 30.0  # slow_lane right: carril 0


@pytest.mark.parametrize("slow_lane", ["right", "random"])
def test_exclusive_lanes_are_only_used_by_their_types(slow_lane):
    """Bicis (carril 0) y autobuses (carril 1) exclusivos: autos y motos nunca los ocupan, ni al
    entrar ni durante un cambio de carril; el tranvía sin carril fijo usa solo los libres."""
    car, bike, bus = DEFAULT_SPECS
    specs = (car, replace(bike, lane=0, exclusive=True), replace(bus, lane=1, exclusive=True), *EXTRA_TYPES)
    cfg = SimConfig(
        length=150, lanes=4, rates=(30, 8, 2, 10, 1), red=20, green=15, run=30, specs=specs,
        slow_lane=slow_lane,
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
            for t, _ in q:
                used[t, ln] = True
    assert not (used & ~sim.allowed).any()
    assert used[0, 2] and used[0, 3] and sim.lane_changes > 0
    if slow_lane == "right":
        assert not used[4, 3]  # el tranvía entra por el carril libre más a la derecha


def test_non_exclusive_fixed_lane_is_shared():
    car, bike, bus = DEFAULT_SPECS
    cfg = SimConfig(lanes=2, specs=(car, replace(bike, lane=0), bus))
    assert cfg.allowed_lanes(0) == (0, 1) and cfg.reserved_lanes == frozenset()


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
    assert stopped_ticks == 120  # 12 s de descenso y ascenso, sin reacción extra
    assert np.allclose(stopped_at, 60.0)  # detenido con el frente en la parada
    assert sim.stops[BUS] == 1 and sim.summary()["stop_time"][BUS] == pytest.approx(12.0)
    assert sim.cum_veh[BUS] == 1  # después siguió hasta cruzar
    assert bus == 0


def test_followers_wait_behind_the_stopped_bus():
    sim = _bus_with_stop(mean=10.0)
    _place(sim, BUS, 58.0, pax=40)
    car = _place(sim, CAR, 58.0 - 12.0 - 6.0, pax=2)
    for _ in range(60):  # el autobús llega y lleva unos segundos en la parada
        sim.step()
    assert sim.stopped[0] and sim.stopped[1]
    gap = sim.x[0] - sim.length_t[BUS] - sim.x[1]
    assert DEFAULT_SPECS[CAR].gap_stop - 1e-9 <= gap < DEFAULT_SPECS[CAR].gap_run  # comprimido detrás
    assert car == 1


def test_stop_time_distribution():
    from trafico.distributions import stop_ticks

    rng = np.random.default_rng(3)
    spec = replace(DEFAULT_SPECS[BUS], stop_position=60.0, stop_time_mean=4.0, stop_time_std=4.0)
    ticks = stop_ticks(rng, spec, 100_000)
    assert ticks.min() >= 0
    assert ticks.mean() * DT == pytest.approx(spec.expected_stop_time, abs=0.03)  # truncada: ~5.2 s, no 4
    assert spec.expected_stop_time > 5.0
    assert replace(spec, stop_time_std=0.0).expected_stop_time == 4.0
    assert DEFAULT_SPECS[BUS].expected_stop_time == 0.0  # sin parada


def _run(cfg: SimConfig, seed: int = 11) -> Simulation:
    sim = Simulation(cfg, np.random.default_rng(seed))
    sim.run()
    return sim


def test_bus_stop_changes_only_the_bus():
    """Generadores por proceso y por tipo: con el autobús en su carril exclusivo, activar su parada
    deja idénticos a los demás tipos (llegadas, pasajeros, reacciones y cambios de carril)."""
    car, bike, bus = DEFAULT_SPECS
    specs = (car, replace(bike, lane=0, exclusive=True), replace(bus, lane=2, exclusive=True))
    base = dict(length=200, lanes=3, rates=(25, 6, 3), red=20, green=25, run=30, specs=specs)
    with_stop = SimConfig(**{**base, "specs": (*specs[:2], replace(specs[2], stop_position=120.0))})
    a, b = _run(SimConfig(**base)), _run(with_stop)
    others = [CAR, BIKE]
    for field in ("arrived_veh", "cum_veh", "cum_pax", "travel_ticks", "pax_m"):
        np.testing.assert_array_equal(getattr(a, field)[others], getattr(b, field)[others], err_msg=field)
    assert a.lane_changes == b.lane_changes
    np.testing.assert_array_equal(a.arrived_veh[BUS], b.arrived_veh[BUS])  # mismas llegadas de autobuses
    np.testing.assert_array_equal(a.pax_hist, b.pax_hist)  # y mismos pasajeros
    assert b.stops[BUS] > 0 and b.travel_ticks[BUS] / b.cum_veh[BUS] > a.travel_ticks[BUS] / a.cum_veh[BUS]


def test_demand_of_one_type_does_not_shift_the_others():
    base = dict(length=200, lanes=2, red=20, green=25, run=30)
    a = _run(SimConfig(rates=(25, 6, 3), **base))
    b = _run(SimConfig(rates=(25, 6, 0.5), **base))  # otra tasa de autobuses
    c = _run(SimConfig(rates=(25, 6, 3, 10), specs=DEFAULT_SPECS + EXTRA_TYPES[:1], **base))  # un tipo nuevo
    for other in (b, c):
        np.testing.assert_array_equal(a.arrived_veh[[CAR, BIKE]], other.arrived_veh[[CAR, BIKE]])
        np.testing.assert_array_equal(a.pax_hist[[CAR, BIKE]], other.pax_hist[[CAR, BIKE]])
    np.testing.assert_array_equal(a.pax_hist[BUS], c.pax_hist[BUS])


def _bikes_at_red(abreast: int, count: int) -> Simulation:
    bike = replace(DEFAULT_SPECS[BIKE], abreast=abreast)
    sim = _empty_sim(red=60, green=30, run=15, specs=(DEFAULT_SPECS[CAR], bike, DEFAULT_SPECS[BUS]))
    for k in range(count):  # en fila, separadas, acercándose a la línea en rojo
        _place(sim, BIKE, 60.0 - k * 4.0, pax=k + 1)
    for _ in range(round(40 / DT)):  # 40 s en rojo: todas llegan y se detienen
        sim.step()
    return sim


def test_bikes_stop_side_by_side_in_pairs():
    sim = _bikes_at_red(abreast=2, count=5)
    assert sim.stopped[: sim.n].all()
    front = np.sort(sim.x[: sim.n])[::-1]
    # Dos filas de dos con los frentes alineados y la quinta en fila detrás de la segunda pareja.
    assert front[0] == pytest.approx(sim.L) and front[1] == pytest.approx(sim.L - 0.01)
    bike = DEFAULT_SPECS[BIKE]
    second = sim.L - 0.01 - bike.length - bike.gap_stop
    assert front[2] == pytest.approx(second) and front[3] == pytest.approx(second - 0.01)
    assert front[4] == pytest.approx(second - 0.01 - bike.length - bike.gap_stop)
    occ = sim.occupancy()
    assert sorted(np.bincount(sim.rows(occ)[1])) == [1, 2, 2]
    # La cola ocupa 3 lugares (no 5) en la saturación del carril.
    assert sim._lane_saturation(occ)[0] == pytest.approx(3 * (bike.length + bike.gap_stop) / sim.L)
    # En verde salen todas, una tras otra.
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
