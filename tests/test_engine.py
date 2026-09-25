from dataclasses import replace

import numpy as np
import pytest

from trafico.config import (
    BIKE, BUS, CAR, DEFAULT_SPECS, DT, GREEN, RED, YELLOW, Behavior, Bottleneck, SimConfig, VehicleSpec,
)  # fmt: skip
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
    gap = sim.x[lead] - sim.vlen[lead] - sim.x[fol]
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


# Vehículo de carga: solo mercancía, largo normal(10, 2) truncado a [8, 16] m.
CARGA = VehicleSpec("carga", "carga", 40.0, 10.0, 4.0, 1.5, 1, 1, 1.0, 0.0, True,
                    cargo_prob=1.0, length_std=2.0, length_min=8.0)  # fmt: skip

# Tipos que el código no conoce: uno que cambia de carril y otro que no.
EXTRA_TYPES = (
    VehicleSpec("motorbike", "moto", 40.0, 2.0, 2.0, 1.0, 1, 2, 1.2, 0.2, True),
    VehicleSpec("tram", "tranvía", 30.0, 20.0, 6.0, 2.0, 20, 200, 120.0, 30.0, False),
)


@pytest.mark.parametrize(
    ("lanes", "extra", "separate", "congestion", "bus_stop", "abreast", "overtake"),
    [
        (1, False, False, 0.0, False, 1, False), (3, False, False, 0.0, False, 1, False),
        (3, True, False, 0.0, False, 1, False), (3, False, True, 0.0, False, 1, False),
        (3, True, True, 0.6, False, 1, False), (1, False, False, 0.0, True, 1, False),
        (3, True, True, 0.3, True, 1, False), (1, False, False, 0.0, False, 2, False),
        (3, True, True, 0.3, True, 2, False), (2, False, False, 0.0, False, 3, False),
        (3, True, False, 0.0, False, 1, True), (4, True, True, 0.4, True, 2, True),
    ],
)  # fmt: skip
def test_invariants_every_step(lanes, extra, separate, congestion, bus_stop, abreast, overtake):
    specs, rates = DEFAULT_SPECS, (30, 8, 2)
    if extra:
        specs, rates = specs + EXTRA_TYPES, rates + (10, 1)
    limits = None
    if overtake:  # motos que rebasan, carriles con límites distintos (hay subidas) y detenciones
        specs = tuple(replace(s, overtake=True, lookahead=50.0, min_advantage=1.0, lane_change_cooldown=1.0)
                      if s.key == "motorbike" else s for s in specs)  # fmt: skip
        specs = tuple(replace(s, bottleneck_prob=0.3, bottleneck_time_mean=8.0, bottleneck_time_std=3.0)
                      if s.key == "car" else s for s in specs)  # fmt: skip
        limits = (30.0, 50.0, 40.0, 80.0)[:lanes]
        specs, rates = specs + (CARGA,), rates + (4,)  # largo variable y solo mercancía
    if separate:  # bicis en el carril derecho y autobuses en el siguiente
        specs = (specs[CAR], replace(specs[BIKE], lane=0), replace(specs[BUS], lane=1), *specs[BUS + 1 :])
    if abreast > 1:  # bicis lado a lado al detenerse
        specs = tuple(replace(s, abreast=abreast) if k == BIKE else s for k, s in enumerate(specs))
    if bus_stop:  # parada de autobús a 100 m (el tramo mide 150)
        specs = tuple(replace(s, stop_position=100.0, stop_time_mean=8.0, stop_time_std=3.0) if k == BUS else s
                      for k, s in enumerate(specs))  # fmt: skip
    cfg = SimConfig(
        length=150, lanes=lanes, lane_speed_limit=limits, rates=rates, red=20, green=15, yellow=3.0 if overtake else 0.0,
        initial_occupancy=0.4 if overtake else 0.0,
        bottleneck=Bottleneck(stop_lanes=(1, 2)) if overtake else Bottleneck(),
        run=30, specs=specs,
        behavior=Behavior(congestion_factor=congestion),
    )  # fmt: skip
    sim = Simulation(cfg, np.random.default_rng(7))
    for _ in range(cfg.n_ticks):
        red = cfg.is_red(sim.tick)
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
        if red:
            assert sim.cum_veh.sum() == crossed_before, "cruzó en rojo"
        slow = ~sim.can_change[sim.vtype[:n]]
        assert np.all(sim.lc_target[:n][slow] == -1)
        fixed = np.array([s.lane if s.lane is not None else 0 for s in specs])  # slow_lane = "right"
        assert np.all(sim.lane[:n][slow] == fixed[sim.vtype[:n][slow]])
    _check_conservation(sim)
    assert sim.cum_veh.sum() > 0
    if extra:
        assert sim.cum_veh[3] > 0 and sim.arrived_veh[4] > 0  # los tipos nuevos circulan
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


def test_reaction_time_follows_the_configured_normal():
    """Con reaction_mean = 2 s y σ pequeña, el auto detenido en la línea arranca ~2 s después del verde."""
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
            for t, *_ in q:
                used[t, ln] = True
    assert not (used & ~sim.allowed).any()
    assert used[0, 2] and used[0, 3] and sim.lane_changes > 0
    if slow_lane == "right":
        assert not used[4, 3]  # el tranvía entra por el carril libre más a la derecha


def test_non_exclusive_fixed_lane_is_shared():
    car, bike, bus = DEFAULT_SPECS
    cfg = SimConfig(lanes=2, specs=(car, replace(bike, lane=0), bus))
    assert cfg.allowed_lanes(0) == (0, 1) and cfg.reserved_lanes == frozenset()


MOTO = VehicleSpec("motorbike", "moto", 70.0, 2.0, 2.0, 1.0, 1, 2, 1.2, 0.2, True,
                   overtake=True, lookahead=50.0, min_advantage=1.0, lane_change_cooldown=1.0)  # fmt: skip
MOTO_IDX = 3


def _moto_sim(moto: VehicleSpec = MOTO, **kw) -> Simulation:
    """Tramo vacío con autos, bicis, autobuses y motos (en ese orden), sin llegadas."""
    return _empty_sim(specs=DEFAULT_SPECS + (moto,), rates=(0, 0, 0, 0), **kw)


@pytest.mark.parametrize("overtake", [True, False])
def test_overtaker_drops_to_slower_lane_behind_a_slow_leader(overtake):
    """Un auto que en el último paso avanzó menos de lo que puede la moto la hace bajar al carril de
    30 km/h (menor límite que el suyo de 60), porque ahí avanza más. Sin `overtake` la lentitud se
    juzga por la velocidad nominal (la misma que la suya en ese carril) y no cambia."""
    sim = _moto_sim(replace(MOTO, overtake=overtake), length=300, lanes=2, lane_speed_limit=(30.0, 60.0))
    car = _place(sim, CAR, 100.0, lane=1)
    moto = _place(sim, MOTO_IDX, 100.0 - 4.5 - 10.0, lane=1)
    sim.v_last[car] = 0.5 * sim.lane_vmax[0]  # p. ej. frenado por su propio líder
    sim._lane_changes()
    assert sim.lc_target[moto] == (0 if overtake else -1)
    assert sim.lane_changes_t[MOTO_IDX] == int(overtake)


@pytest.mark.parametrize(("vtype", "changes"), [(MOTO_IDX, True), (CAR, False)])
def test_lookahead_per_type(vtype, changes):
    """Con un líder detenido a 40 m, la moto (lookahead = 50 m) cambia de carril y el auto (30 m) no."""
    sim = _moto_sim(length=300, lanes=2)
    leader = _place(sim, CAR, 150.0, lane=0)
    sim.stopped[leader] = True
    follower = _place(sim, vtype, 150.0 - 4.5 - 40.0, lane=0)
    sim._lane_changes()
    assert (sim.lc_target[follower] == 1) == changes


def test_no_lane_change_without_a_vehicle_ahead():
    """Solo en el carril de 30 km/h, con el de 50 libre, no cambia: no tiene líder que lo frene."""
    sim = _empty_sim(length=600, lanes=2, lane_speed_limit=(30.0, 50.0), red=0, green=60, run=60)
    car = _place(sim, CAR, 20.0, lane=0)
    for _ in range(100):
        sim.step()
    assert sim.lane[car] == 0 and sim.lane_changes == 0


@pytest.mark.parametrize(("ratio", "changes"), [(0.80, False), (0.70, True), (0.0, True)])
def test_changes_lane_only_if_leader_is_25_percent_below_the_lane_limit(ratio, changes):
    """Con un líder a menos de lookahead que va a `ratio` del límite del carril (50 km/h): cambia
    solo si va al menos 25 % por debajo (leader_slowdown = 0.25). El carril de 50 es el más rápido,
    así que el cambio es un rebase o una subida al de 80."""
    sim = _moto_sim(length=300, lanes=2, lane_speed_limit=(50.0, 80.0))
    leader = _place(sim, CAR, 100.0, lane=0)
    moto = _place(sim, MOTO_IDX, 100.0 - 4.5 - 10.0, lane=0)
    sim.v_last[leader] = ratio * sim.lane_vmax[0]
    sim._lane_changes()
    assert (sim.lc_target[moto] == 1) == changes


def test_does_not_rise_without_room_or_speed_gain():
    # Seguidor en el carril rápido a menos de su gap en marcha: no sube.
    sim = _empty_sim(length=300, lanes=2, lane_speed_limit=(30.0, 50.0))
    car = _place(sim, CAR, 100.0, lane=0)
    _place(sim, CAR, 100.0 - 4.5 - 1.0, lane=1)
    sim._lane_changes()
    assert sim.lc_target[car] == -1
    # Líder detenido cerca en el carril rápido: no sube.
    sim = _empty_sim(length=300, lanes=2, lane_speed_limit=(30.0, 50.0))
    car = _place(sim, CAR, 100.0, lane=0)
    leader = _place(sim, CAR, 100.0 + 4.5 + 15.0, lane=1)
    sim.stopped[leader] = True
    sim._lane_changes()
    assert sim.lc_target[car] == -1
    # Sin límites distintos no hay a qué subir.
    sim = _empty_sim(length=600, lanes=2, red=0, green=60, run=60)
    _place(sim, CAR, 20.0, lane=0)
    for _ in range(100):
        sim.step()
    assert sim.lane_changes == 0


def test_motorbikes_change_lanes_more_than_cars():
    """Escenario de config.toml: bicis y autobuses en carriles exclusivos; autos y motos en los de
    30 y 50 km/h. Las motos que rebasan cambian de carril más por vehículo que los autos, y más
    que las mismas motos sin `overtake`."""
    car, bike, bus = DEFAULT_SPECS
    base = dict(
        length=300, lanes=4, lane_speed_limit=(20.0, 30.0, 50.0, 80.0), rates=(18, 8, 0.25, 8), red=25,
        green=35, run=30, behavior=Behavior(congestion_factor=(0.05, 0.5, 0.4, 0.6)),
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
    gap = sim.x[0] - sim.vlen[0] - sim.x[1]
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


@pytest.mark.parametrize("start", ["red", "green"])
def test_light_cycle_is_green_yellow_red(start):
    cfg = SimConfig(red=10, green=20, yellow=3, start_phase=start, run=20)
    phases = [cfg.phase(t) for t in range(cfg.n_ticks)]
    # Fases consecutivas sin repetir: siempre verde → amarillo → rojo.
    order = [p for k, p in enumerate(phases) if k == 0 or p != phases[k - 1]]
    first = order.index(GREEN)
    assert order[first:first + 4] == [GREEN, YELLOW, RED, GREEN]
    for intervals, phase in ((cfg.red_intervals(), RED), (cfg.yellow_intervals(), YELLOW)):
        ticks = {t for a, b in intervals for t in range(round(a / DT), round(b / DT))}
        assert ticks == {t for t, p in enumerate(phases) if p == phase}


def test_yellow_slows_vehicles_approaching_the_light():
    """En amarillo, quien está a menos de yellow_approach (50 m) de la línea va a la mitad de su
    velocidad y puede cruzar; quien está más lejos no cambia. Al llegar el rojo, se detiene."""
    sim = _empty_sim(length=300, lanes=2, red=10, green=0.1, yellow=10, start_phase="green", run=30)
    near = _place(sim, CAR, sim.L - 20.0, lane=0)
    far = _place(sim, CAR, sim.L - 100.0, lane=1)
    far_vid = sim.vid[far]
    sim.step()  # verde: velocidad completa
    assert sim.x[near] == pytest.approx(sim.L - 20.0 + sim.v_step[CAR])
    assert sim.cfg.phase(sim.tick) == YELLOW
    x_near, x_far = sim.x[near], sim.x[far]
    sim.step()
    assert sim.x[near] - x_near == pytest.approx(0.5 * sim.v_step[CAR])
    assert sim.x[far] - x_far == pytest.approx(sim.v_step[CAR])
    while sim.cfg.phase(sim.tick) == YELLOW:
        sim.step()
    assert sim.cum_veh[CAR] == 1  # el cercano cruzó en amarillo (y salió: los índices se compactan)
    far = int(np.flatnonzero(sim.vid[: sim.n] == far_vid)[0])
    # El lejano entra a la zona durante el amarillo, frena y no alcanza a cruzar antes del rojo.
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
    # El tipo de carga circula, pero no aporta pasajeros ni aparece en el histograma.
    assert s["crossed_veh"][carga] > 0 and s["arrived_cargo"][carga] == s["arrived_veh"][carga]
    assert s["arrived_pax"][carga] == 0 and s["crossed_pax"][carga] == 0 and sim.pax_hist[carga].sum() == 0
    assert np.isnan(s["pax_per_veh"][carga])
    # Autos: ~30 % con mercancía; el promedio de pasajeros es solo de los que llevan pasajeros.
    share = s["arrived_cargo"][CAR] / s["arrived_veh"][CAR]
    assert 0.2 < share < 0.4
    assert sim.pax_hist[CAR, 0] == 0 and sim.pax_hist[CAR].sum() == s["arrived_veh"][CAR] - s["arrived_cargo"][CAR]
    assert s["pax_per_veh"][CAR] == pytest.approx(s["arrived_pax"][CAR] / sim.pax_hist[CAR].sum())
    assert s["pax_per_veh"][CAR] >= 1.0
    _check_conservation(sim)


def test_cargo_does_not_change_the_other_random_draws():
    """Las llegadas y los pasajeros de las bicis no cambian con la probabilidad de mercancía de los autos."""
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


def _initial_cfg(occupancy, **kw) -> SimConfig:
    """Autos y motos; bicis en su carril exclusivo (0) y autobuses en el suyo (3)."""
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
        assert target - 17.0 / sim.L < filled <= target + 1e-9  # a lo más un autobús de menos
        # Solo los tipos que pueden circular por el carril.
        assert set(sim.vtype[:n][on].tolist()) <= {k for k in range(4) if sim.allowed[k, lane]}
    gap, _, fol = _gaps(sim)
    assert np.all(gap >= sim.gap_stop_t[sim.vtype[fol]] - 1e-9)
    # Quien queda a menos de su gap_run empieza detenido (embotellamiento); los demás, en marcha.
    np.testing.assert_array_equal(sim.stopped[fol], gap < sim.gap_run_t[sim.vtype[fol]] - 1e-6)
    assert set(sim.vtype[:n][sim.lane[:n] == 1].tolist()) == {CAR, MOTO_IDX}  # mezcla según las tasas
    _check_conservation(sim)


@pytest.mark.parametrize("start", ["red", "green"])
def test_full_occupancy_is_a_compact_queue(start):
    """Con 1, los carriles quedan como una cola compacta: cada vehículo a casi su gap_stop del de
    adelante, todos detenidos salvo el primero, y la cola se estira sin traslapes."""
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
    """Los de la condición inicial cruzan y suman pasajeros, pero no entran en el tiempo de recorrido;
    en 10 s nadie que entró recorre 300 m."""
    sim = Simulation(_initial_cfg(0.5, red=0, green=20, run=1), np.random.default_rng(1))  # 10 s simulados
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
    """Un auto se detiene 20 s en x = 100 del carril 0; el de atrás, sin código especial, lo trata
    como un líder detenido: se cambia al carril 1 y lo rebasa. Luego el detenido sigue."""
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
    assert sim.x[{v: i for i, v in enumerate(sim.vid[: sim.n].tolist())}[ids[0]]] > 100.0  # siguió su camino


def test_bottleneck_only_in_its_lanes():
    sim = _bottleneck_sim(stop_lanes=(1,))
    car = _place_stopping(sim, CAR, 60.0, 0, at=100.0)
    for _ in range(200):
        sim.step()
        assert not sim.stopped[car]
    assert sim.bn_stops.sum() == 0 and sim.x[car] > 100.0


def test_bottleneck_probability_and_types():
    """Con bottleneck_prob = 0.3 para autos, ~30 % de los autos que cruzan se detuvieron; las bicis
    (sin bottleneck_prob) no."""
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
    specs = tuple(replace(s, bottleneck_time_mean=60.0) for s in DEFAULT_SPECS)  # sin probabilidad no hay detenciones
    b = Simulation(SimConfig(**base, specs=specs, bottleneck=Bottleneck(stop_lanes=(0,))), np.random.default_rng(1))
    a.run()
    b.run()
    np.testing.assert_array_equal(a.x[: a.n], b.x[: b.n])
    assert b.bn_stops.sum() == 0


def test_bottleneck_per_type_probability_and_duration():
    """Cada tipo con su probabilidad y duración: la carga se detiene más seguido (0.8) y por más
    tiempo (30 s) que los autos (0.2, 5 s); las bicis, sin bottleneck_prob, nunca."""
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
    """La espera acumulada en la cola de entrada cuadra con los vehículo·paso contados en la cola: los que
    ya entraron más lo que llevan esperando los que siguen en ella. Con la vía saturada la espera crece."""
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
