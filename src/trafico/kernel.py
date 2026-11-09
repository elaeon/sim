"""Paso de la simulación compilado con numba: el ciclo de pasos de `Simulation` (engine.py) sobre su estado en arreglos.

El modelo es el que describe engine.py; aquí está su cálculo, vehículo por vehículo, en ciclos simples que numba
compila (`@njit(cache=True)`: la primera corrida compila y guarda el resultado en `__pycache__`). Con
`NUMBA_DISABLE_JIT=1` el mismo código corre como Python puro, para depurarlo.

Todo el estado vive en tres `NamedTuple` de arreglos que arma `Simulation`:

  * `Params`: tablas de solo lectura (por tipo, por carril, líneas de alto y su calendario de fases).
  * `State`: los vehículos (estructura de arreglos compacta en [0, n)), las colas de entrada y de salida (anillos por
    carril), los acumuladores y las series del `Recorder`. Los escalares (n, paso, …) van en `State.ints`.
  * `Rngs`: los generadores de numpy de cada proceso aleatorio y tipo (MAX_TYPES por tupla, los que sobran sin uso).
    numba los usa con los mismos algoritmos que numpy, así que los sorteos son idénticos a los del motor en numpy.

Al kernel no se le pasan las tuplas sino una referencia a ellas (`as_ref`): numba pasa una tupla por valor, campo por
campo, y con ~90 arreglos eso hace que compilar tarde minutos. Con la referencia se usan igual (`s.x[i]`, `p.L`).

Las llegadas Poisson y lo que se sortea al llegar (pasajeros, largo, velocidad, detención) los calcula Python por
bloques de ARRIVAL_CHUNK pasos (`Arrivals`); `advance` regresa a pedir el siguiente bloque, o más lugar cuando un
arreglo se llena. Las sumas se hacen en el mismo orden que en numpy (un `bincount` suma en orden de índice), así que
los resultados son idénticos bit a bit a los del motor anterior con la misma semilla.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numba
import numpy as np
from numba import njit, types
from numba.core import cgutils
from numba.experimental import structref
from numba.experimental.jitclass.base import imp_dtor
from numba.extending import intrinsic

from trafico.config import DT, EMISSION_DECEL, GREEN, RED, YELLOW
from trafico.emissions import EMIS_BIN

# Las funciones del paso se compilan sin envoltorio para Python (no se llaman desde Python); las que usa engine.py
# están al final (`py_*` y `advance`) y numba compila cada una la primera vez que se usa.
jit = njit(cache=True, no_cpython_wrapper=True)
api = njit(cache=True)

EPS = 1e-6
INF = np.inf
NAN = np.nan  # «sin valor» en los argumentos de punto flotante (el None de Python)
SIDE_EPS = 0.01  # m que queda atrás el frente de quien se detiene al lado (conserva el orden)
ARRIVAL_CHUNK = 1024  # pasos por bloque de sorteo de llegadas
# Estados de parada: sin parada pendiente, antes de la parada, detenido en ella.
NO_STOP, STOP_AHEAD, AT_STOP = 0, 1, 2
# Clase de parada: la propia del tipo (p. ej. el autobús) o una detención de [bottleneck].
TYPE_STOP, BOTTLENECK_STOP = 0, 1
# Cómo se sortea el tiempo de reacción: uniforme, fijo o normal truncada.
REACT_UNIFORM, REACT_FIXED, REACT_NORMAL = 0, 1, 2
# Clase de cada línea de alto (calendario de fases): siempre verde, ciclo fijo o calendario precalculado (peatones).
LINE_OPEN, LINE_CYCLE, LINE_SCHEDULE = 0, 1, 2
# Por qué regresa `advance`: llegó al paso pedido, necesita el siguiente bloque de llegadas o más lugar.
DONE, NEED_ARRIVALS, NEED_ROOM = 0, 1, 2
# Índices de State.ints.
N, TICK, LANE_CHANGES, SAMPLES, ARRIVAL_POS = range(5)


class Params(NamedTuple):
    """Tablas de solo lectura de una réplica."""

    L: float
    n_lanes: int
    n_types: int
    n_ticks: int
    lc_every: int  # pasos entre rondas de cambio de carril
    sample_ticks: int
    key_stride: float  # separa los carriles en la clave de orden
    any_headway: bool
    any_stop: bool
    any_abreast: bool
    any_pass_in_lane: bool
    any_dynamics: bool
    any_bump: bool
    any_emissions: bool
    any_exit: bool
    any_rise: bool
    # [behavior]
    yellow_approach: float
    yellow_speed_factor: float
    release_space: float
    lane_change_speed_factor: float
    leader_slowdown: float
    queue_reaction: bool
    react_mode: int
    react_lo: int  # pasos (uniforme)
    react_hi: int
    react_mean: float  # s (fijo o normal)
    react_std: float
    react_min: float
    react_max: float
    lc_min: int  # pasos que dura un cambio de carril
    lc_max: int
    # Por tipo de vehículo.
    v_step: np.ndarray
    length_t: np.ndarray
    gap_run_t: np.ndarray
    gap_stop_t: np.ndarray
    has_headway_t: np.ndarray
    headway_t: np.ndarray
    can_change: np.ndarray
    stop_x: np.ndarray
    has_stop: np.ndarray
    stop_mean: np.ndarray  # s, duración de la parada propia
    stop_std: np.ndarray
    stop_fixed: np.ndarray  # pasos si stop_std ≤ 0
    bn_mean: np.ndarray  # s, duración de la detención de [bottleneck]
    bn_std: np.ndarray
    bn_fixed: np.ndarray
    free_type: np.ndarray
    abreast: np.ndarray
    pass_in_lane: np.ndarray
    allowed: np.ndarray  # (tipos, carriles)
    entry_lanes: np.ndarray  # (tipos, carriles): carriles de entrada de cada tipo, los primeros entry_count
    entry_count: np.ndarray
    lookahead_t: np.ndarray
    min_adv_t: np.ndarray
    cooldown_ticks_t: np.ndarray
    overtake: np.ndarray
    acc_t: np.ndarray
    dec_t: np.ndarray
    dyn_t: np.ndarray
    dec_fin_t: np.ndarray
    dec_safe_t: np.ndarray
    bump_v_t: np.ndarray
    emits_t: np.ndarray
    emis_on: np.ndarray  # (tipos, contaminantes)
    coef_sets: np.ndarray  # (2·tipos, contaminantes, 6): fila 2·tipo para a ≥ −0.5, 2·tipo + 1 para a < −0.5
    emis_amax: np.ndarray
    emis_dmax: np.ndarray
    # Por carril.
    lane_vmax: np.ndarray
    bn_lane: np.ndarray
    exit_rate: np.ndarray
    exit_limited: np.ndarray
    exit_add: np.ndarray
    exit_storage: np.ndarray
    # Topes.
    bump_x: np.ndarray
    bump_lane: np.ndarray  # (topes, carriles)
    # Líneas de alto intermedias (cfg.stop_lines): posición, tipos que no las obedecen, carriles donde rigen.
    inner_x: np.ndarray
    inner_free: np.ndarray  # (líneas, tipos)
    inner_lane: np.ndarray  # (líneas, carriles)
    # Fases: fila 0 = semáforo del final del tramo, fila 1 + j = línea intermedia j.
    line_kind: np.ndarray  # LINE_OPEN | LINE_CYCLE | LINE_SCHEDULE
    line_red: np.ndarray  # pasos
    line_green: np.ndarray
    line_yellow: np.ndarray
    line_start_red: np.ndarray
    line_offset: np.ndarray  # pasos
    line_sched: np.ndarray  # (líneas + 1, n_ticks + 1) fase de cada paso de las LINE_SCHEDULE


class State(NamedTuple):
    """Estado mutable de una réplica (ver `Simulation`)."""

    ints: np.ndarray  # N, TICK, LANE_CHANGES, SAMPLES, ARRIVAL_POS
    # Vehículos en el tramo, [0, n).
    vtype: np.ndarray
    pax: np.ndarray
    lane: np.ndarray
    lc_target: np.ndarray
    lc_timer: np.ndarray
    cooldown: np.ndarray
    react: np.ndarray
    x: np.ndarray
    stopped: np.ndarray
    crossed: np.ndarray
    entry_tick: np.ndarray
    vid: np.ndarray
    stop_state: np.ndarray
    dwell: np.ndarray
    v_last: np.ndarray
    vcap_last: np.ndarray
    vlen: np.ndarray
    vmax: np.ndarray
    stop_pos: np.ndarray
    stop_kind: np.ndarray
    # Cola de entrada de cada carril (anillo): tipo, pax, largo, detención, velocidad km/h, paso de llegada.
    q_vt: np.ndarray
    q_pax: np.ndarray
    q_len: np.ndarray
    q_at: np.ndarray
    q_speed: np.ndarray
    q_arrived: np.ndarray
    q_head: np.ndarray
    q_count: np.ndarray
    queue_stopped: np.ndarray
    queue_react: np.ndarray
    # Cola de salida de cada carril (anillo con lo que ocupa cada vehículo).
    exit_ring: np.ndarray
    exit_head: np.ndarray
    exit_count: np.ndarray
    exit_q: np.ndarray
    exit_credit: np.ndarray
    exit_closed: np.ndarray
    exit_blocked_ticks: np.ndarray
    # Acumuladores.
    arrived_veh: np.ndarray
    arrived_pax: np.ndarray
    arrived_cargo: np.ndarray
    entered_veh: np.ndarray
    cum_veh: np.ndarray
    cum_pax: np.ndarray
    travel_ticks: np.ndarray
    pax_m: np.ndarray
    lane_dist: np.ndarray
    lane_time: np.ndarray
    type_dist: np.ndarray
    type_time: np.ndarray
    lane_changes_t: np.ndarray
    in_lane_passes: np.ndarray
    stops: np.ndarray
    stop_time_ticks: np.ndarray
    queue_wait_ticks: np.ndarray
    bn_stops: np.ndarray
    bn_ticks: np.ndarray
    pax_hist: np.ndarray
    initial_veh: np.ndarray
    initial_pax: np.ndarray
    timed_veh: np.ndarray
    veh_m: np.ndarray
    lane_cross: np.ndarray
    lane_crossed: np.ndarray
    emis_g: np.ndarray
    emis_free_g: np.ndarray
    emis_interval: np.ndarray
    emis_pos: np.ndarray
    # Series del Recorder (metrics.py).
    r_cum_pax: np.ndarray
    r_pax_m: np.ndarray
    r_pax_on: np.ndarray
    r_footprint: np.ndarray
    r_lane_sat: np.ndarray
    r_lane_dist: np.ndarray
    r_lane_time: np.ndarray
    r_lane_cross: np.ndarray
    r_exit_q: np.ndarray
    r_entry_q: np.ndarray
    r_emis: np.ndarray


class Rngs(NamedTuple):
    """Generadores de los procesos aleatorios del paso: una tupla de MAX_TYPES por proceso (uno por tipo)."""

    react: tuple  # tiempos de reacción
    stop: tuple  # duración de la parada
    bottleneck_time: tuple  # duración de la detención de [bottleneck]
    queue_react: tuple  # reacción de quien arranca en la cola de entrada detenida
    entry: tuple  # desempate del carril de entrada
    lane_change: np.random.Generator  # orden de decisión y duración de los cambios de carril


class Arrivals(NamedTuple):
    """Llegadas de un bloque de ARRIVAL_CHUNK pasos que empieza en `start`, en orden de paso y de tipo."""

    start: int
    tick: np.ndarray
    vt: np.ndarray
    pax: np.ndarray
    length: np.ndarray
    at: np.ndarray  # posición de su detención de [bottleneck]; inf = no se detiene
    speed: np.ndarray  # km/h


@structref.register
class Ref(types.StructRef):
    """Referencia a los campos de un NamedTuple (ver `as_ref`)."""

    def preprocess_fields(self, fields):
        return tuple((name, types.unliteral(typ)) for name, typ in fields)


class RefProxy(structref.StructRefProxy):
    """Lo que ve Python de una `Ref`: solo se pasa de vuelta al kernel."""


structref.define_boxing(Ref, RefProxy)


@intrinsic
def _as_ref(typingctx, tup):
    """Una `Ref` nueva con los campos de la tupla con nombre `tup` (los arreglos se comparten, no se copian)."""
    ref_type = Ref(list(zip(tup.fields, tup.types)))

    def codegen(context, builder, signature, args):
        # Como numba.experimental.structref.new, llenando cada campo con el de la tupla.
        alloc_type = context.data_model_manager[ref_type.get_data_type()].get_value_type()
        meminfo = context.nrt.meminfo_alloc_dtor(
            builder, context.get_constant(types.uintp, context.get_abi_sizeof(alloc_type)),
            imp_dtor(context, builder.module, ref_type),
        )  # fmt: skip
        data_pointer = builder.bitcast(context.nrt.meminfo_data(builder, meminfo), alloc_type.as_pointer())
        data = cgutils.create_struct_proxy(ref_type.get_data_type())(context, builder, ref=data_pointer)
        for i, (name, typ) in enumerate(zip(tup.fields, tup.types)):
            value = builder.extract_value(args[0], i)
            context.nrt.incref(builder, typ, value)
            setattr(data, name, value)
        ref = context.make_helper(builder, ref_type)
        ref.meminfo = meminfo
        return ref._getvalue()

    return ref_type(tup), codegen


@njit(cache=True)
def _new_ref(tup):
    return _as_ref(tup)


def as_ref(tup):
    """Referencia a una tupla con nombre para pasarla al kernel; con NUMBA_DISABLE_JIT=1, la misma tupla."""
    return tup if numba.config.DISABLE_JIT else _new_ref(tup)


class Occupancy(NamedTuple):
    """Entradas carril-vehículo ordenadas por (carril, x). Quien cambia de carril aparece dos veces."""

    order: np.ndarray  # posición ordenada -> índice de entrada (primarias [0,n), secundarias [n,m))
    veh: np.ndarray  # índice de vehículo de cada entrada ordenada
    lane: np.ndarray  # carril de cada entrada ordenada
    x: np.ndarray  # posición del frente de cada entrada ordenada
    has_leader: np.ndarray  # la siguiente entrada está en el mismo carril
    leader: np.ndarray  # índice de vehículo del líder (válido si has_leader)
    changing: np.ndarray  # vehículos con cambio de carril en curso (orden de las secundarias)


# ------------------------------------------------------------------ fórmulas


@jit
def gipps(space, lead, dec, finite):
    """Velocidad máxima (m/paso) con la que aún se detiene frenando a `dec` (m/paso²), con un paso de reacción, sin
    rebasar `space` m más lo que el de adelante recorre al detenerse (`lead` = v_lead²·dec/max(dec_lead, dec), o 0 si
    no cuenta): v + v²/(2·dec) ≤ space + v_lead²/(2·dec_lead) (regla de Gipps). INF si el frenado no es `finite`.

    Se supone que el de adelante frena al menos tan fuerte como él (dec_lead ≥ dec): así, siguiendo a un líder a
    velocidad constante, el equilibrio es space = v_lead, el mismo que impone el límite duro de espacio
    (actualización en paralelo), y se acerca sin frenazos. Con un líder que frena menos (p. ej. un auto detrás de una
    bici), la regla lo dejaría acercarse más y el límite duro lo frenaría de golpe."""
    if not finite:
        return INF
    return math.sqrt(dec * dec + 2 * dec * max(space, 0.0) + lead) - dec


@jit
def safe_speed_1(space, v_lead, dec, dec_lead):
    """`gipps` con el frenado del de adelante: INF si `dec` es infinito (frenado instantáneo, sin anticipación); un
    líder de frenado instantáneo se detiene en seco."""
    if not math.isfinite(dec):
        return INF
    lead = v_lead * v_lead * dec / max(dec_lead, dec) if math.isfinite(dec_lead) else 0.0
    return -dec + math.sqrt(dec * dec + 2 * dec * max(space, 0.0) + lead)


@jit
def run_gap(p, k, v_step):
    """Gap (m) que guarda un vehículo del tipo k detrás de un líder en marcha yendo a `v_step` m/paso: gap_run o, con
    time_headway, gap_stop más lo que recorre en ese tiempo."""
    if not p.has_headway_t[k]:
        return p.gap_run_t[k]
    return p.gap_stop_t[k] + p.headway_t[k] * v_step / DT


@jit
def bump_speed_1(p, k, lane, front, rear):
    """Velocidad máxima (m/paso) de un vehículo del tipo k en `lane` por los topes: la del tipo si pisa uno; antes de
    él, con frenado gradual, la que aún le permite llegar a ella frenando a `decel`; INF si no aplica (sin tope en el
    carril, ya los pasó o el tipo pasa sin frenar). Con varios, la menor."""
    vb = p.bump_v_t[k]
    if not math.isfinite(vb):
        return INF
    dec = p.dec_t[k]
    speed = INF
    for b in range(p.bump_x.size):
        bump = p.bump_x[b]
        if rear >= bump or not p.bump_lane[b, lane]:
            continue
        if front >= bump:
            speed = min(speed, vb)
        elif math.isfinite(dec):
            speed = min(speed, safe_speed_1(bump - front, vb, dec, dec))
    return speed


@jit
def line_phase(p, j, tick):
    """Fase (RED, GREEN o YELLOW) de la línea j en el paso `tick`: 0 = el semáforo del final del tramo, 1 + i = la
    línea intermedia i. Un ciclo fijo va verde → amarillo → rojo, retrasado `offset` pasos."""
    kind = p.line_kind[j]
    if kind == LINE_OPEN:
        return GREEN
    if kind == LINE_SCHEDULE:
        return p.line_sched[j, min(tick, p.n_ticks)]
    red, green, yellow = p.line_red[j], p.line_green[j], p.line_yellow[j]
    cycle = red + green + yellow
    pos = (tick - p.line_offset[j]) % cycle
    if p.line_start_red[j]:
        pos = (pos - red) % cycle  # el ciclo empieza en el rojo
    if pos < green:
        return GREEN
    return YELLOW if pos < green + yellow else RED


# ------------------------------------------------------------------ sorteos (como distributions.py)


@jit
def reaction_ticks(g, p, size):
    """Tiempo de reacción en pasos (al menos uno), como distributions.reaction_ticks: uniforme en [reaction_min,
    reaction_max] o normal(reaction_mean, reaction_std) truncada a ese rango por remuestreo."""
    if p.react_mode == REACT_UNIFORM:
        ticks = g.integers(p.react_lo, p.react_hi, size, endpoint=True)
    else:
        out = np.empty(size)
        if p.react_mode == REACT_FIXED:
            out[:] = p.react_mean
        else:
            filled = 0
            while filled < size:
                draw = g.normal(p.react_mean, p.react_std, 2 * (size - filled) + 8)
                for v in draw:
                    if filled < size and p.react_min <= v <= p.react_max:
                        out[filled] = v
                        filled += 1
        ticks = np.rint(out / DT).astype(np.int64)
    return np.maximum(ticks, 1)  # con 0 pasos nunca arrancaría


@jit
def normal_tick(g, mean, std, fixed):
    """Una duración en pasos, como distributions.normal_ticks(…, 1): normal(mean, std) truncada a ≥ 0 por remuestreo,
    cuantizada a DT; `fixed` si std ≤ 0."""
    if std <= 0:
        return fixed
    while True:
        draw = g.normal(mean, std, 10)  # 2·1 + 8, como normal_ticks
        for v in draw:
            if v >= 0:
                return int(np.rint(v / DT))


# ------------------------------------------------------------------ orden en los carriles


@jit
def occupancy(p, s):
    n = s.ints[N]
    m = n
    for i in range(n):
        if s.lc_target[i] >= 0:
            m += 1
    veh = np.empty(m, np.int64)
    lane = np.empty(m, np.int64)
    j = n
    for i in range(n):
        veh[i] = i
        lane[i] = s.lane[i]
        if s.lc_target[i] >= 0:
            veh[j] = i
            lane[j] = s.lc_target[i]
            j += 1
    # La clave de orden es la de numpy (carril · separación + x), así que el orden es el mismo.
    key = np.empty(m)
    for e in range(m):
        key[e] = lane[e] * p.key_stride + s.x[veh[e]]
    order = np.argsort(key, kind="mergesort")
    s_veh = veh[order]
    s_lane = lane[order]
    s_x = np.empty(m)
    has_leader = np.zeros(m, np.bool_)
    leader = np.zeros(m, np.int64)
    for k in range(m):
        s_x[k] = s.x[s_veh[k]]
        if k + 1 < m:
            has_leader[k] = s_lane[k + 1] == s_lane[k]
            leader[k] = s_veh[k + 1]
    return Occupancy(order, s_veh, s_lane, s_x, has_leader, leader, veh[n:].copy())


@jit
def rows(s, occ):
    """Filas lado a lado en el orden de `occ`: (junto, id de fila, índice del último de su fila).

    `junto` marca a quien se traslapa con su líder, es decir, está a su lado en la misma fila. Una fila es un tramo
    consecutivo de entradas unidas así; sin `abreast`, cada vehículo es su propia fila."""
    m = occ.veh.size
    beside = np.zeros(m, np.bool_)
    for k in range(m):
        ld = occ.leader[k]
        beside[k] = occ.has_leader[k] and s.x[ld] - s.vlen[ld] - occ.x[k] < 0
    rid = np.empty(m, np.int64)
    row = -1
    for k in range(m):
        if k == 0 or not beside[k - 1]:  # la entrada k sigue en la fila de k-1 si k-1 está junto a ella
            row += 1
        rid[k] = row
    row_last = np.empty(m, np.int64)
    last = m - 1
    for k in range(m - 1, -1, -1):
        if k < m - 1 and rid[k + 1] != rid[k]:
            last = k
        row_last[k] = last
    return beside, rid, row_last


@jit
def side_slot(p, s, occ, beside, rid, row_last):
    """Entradas que pueden detenerse al lado de su líder: su tipo admite `abreast` > 1, el líder está detenido, es de
    su mismo tipo (como toda su fila) y la fila tiene lugar. Solo quien llega en marcha toma el lugar (o quien ya lo
    ocupa): un vehículo ya detenido en fila, o que tiene a otro a su lado, no se adelanta a otra fila. No aplica a
    quien cambia de carril ni a quien ya cruzó."""
    m = occ.veh.size
    out = np.zeros(m, np.bool_)
    if m < 2:
        return out
    # Tipos mínimo y máximo de cada fila.
    row_lo = np.empty(m, np.int64)
    row_hi = np.empty(m, np.int64)
    for k in range(m):
        t = s.vtype[occ.veh[k]]
        r = rid[k]
        if k == 0 or rid[k - 1] != r:
            row_lo[r] = t
            row_hi[r] = t
        else:
            row_lo[r] = min(row_lo[r], t)
            row_hi[r] = max(row_hi[r], t)
    for k in range(m):
        i = occ.veh[k]
        ld = occ.leader[k]
        t = s.vtype[i]
        nxt = min(k + 1, m - 1)  # entrada del líder (la siguiente en el orden)
        ahead = row_last[nxt] - k  # vehículos de la fila del líder que van delante
        companion = k > 0 and beside[k - 1]  # alguien se detuvo a su lado (la entrada anterior lo tiene de líder)
        out[k] = (
            occ.has_leader[k]
            and p.abreast[t] > 1
            and s.stopped[ld]
            and row_lo[rid[nxt]] == t and row_hi[rid[nxt]] == t
            and ahead < p.abreast[t]
            and (not s.stopped[i] or beside[k])
            and not companion
            and s.lc_target[i] < 0 and s.lc_target[ld] < 0
            and not s.crossed[i] and not s.crossed[ld]
        )  # fmt: skip
    return out


@jit
def in_lane_pass(p, s, occ, beside, side):
    """Rebase dentro del carril entre vehículos en marcha del mismo tipo con `pass_in_lane`.

    Devuelve, por entrada, (no cuenta a su líder, lo está rebasando). Rebasa quien tiene una velocidad máxima mayor que
    su líder (ambas acotadas por el límite del carril), siempre que el líder no tenga ya a otro a su lado ni a él lo
    esté alcanzando alguien por un lado: así nunca van más de dos lado a lado. El líder no cuenta tampoco para quien es
    rebasado mientras el que se le adelanta, más rápido, va a su lado o se aleja sin sacarle aún `gap_run` más un paso.

    `beside` (se traslapa con su líder) y `side` (puede detenerse al lado de su líder) son los de rows y side_slot."""
    m = occ.veh.size
    base = np.zeros(m, np.bool_)
    for k in range(m):
        me, ld = occ.veh[k], occ.leader[k]
        t = s.vtype[me]
        base[k] = (
            occ.has_leader[k]
            and p.pass_in_lane[t]
            and s.vtype[ld] == t
            and not s.stopped[me] and not s.stopped[ld]
            and s.lc_target[me] < 0 and s.lc_target[ld] < 0
        )  # fmt: skip
    if not base.any():
        return base, base.copy()
    passing = np.zeros(m, np.bool_)
    yielding = np.zeros(m, np.bool_)
    for k in range(m):
        if not base[k]:
            continue
        me, ld = occ.veh[k], occ.leader[k]
        t = s.vtype[me]
        lane_v = p.lane_vmax[occ.lane[k]]
        v_me = min(s.vmax[me], lane_v)
        v_lead = min(s.vmax[ld], lane_v)
        # El líder ya tiene a otro a su lado o va a detenerse al lado del suyo; alguien lo alcanza por un lado.
        lead_beside = k + 1 < m and (beside[k + 1] or side[k + 1])
        behind_beside = k > 0 and beside[k - 1]
        faster = v_me > v_lead + EPS
        if p.any_dynamics:
            # Con dinámica gradual además debe ir ya al menos tan rápido como él (no rebasa mientras acelera).
            faster = faster and (not p.dyn_t[t] or s.v_last[me] >= s.v_last[ld] - EPS)
        passing[k] = faster and not lead_beside and not behind_beside
        gap = s.x[ld] - s.vlen[ld] - occ.x[k]
        # Mientras están lado a lado y, después, mientras el que se adelanta de verdad se aleja (avanzó en el último
        # paso al menos lo que él puede), hasta que el espacio le permita avanzar sin frenar. Con dinámica gradual se
        # compara con su velocidad real (no puede acelerar de golpe a la deseada).
        mine = s.v_last[me] if p.any_dynamics and p.dyn_t[t] else s.vcap_last[me]
        pulling_away = s.v_last[ld] >= mine - EPS and gap < p.gap_run_t[t] + v_me
        yielding[k] = v_lead > v_me + EPS and (gap < 0 or pulling_away)
    # No rebasa a quien está rebasando: tres quedarían lado a lado.
    for k in range(m - 1):
        if passing[k + 1]:
            passing[k] = False
    return passing | yielding, passing


@jit
def lane_saturation(p, s, occ):
    """Saturación de cada carril: largo de su cola de detenidos dentro del tramo (largo + gap detenido de cada uno)
    entre la longitud del tramo, acotada a 1. Quien está cambiando de carril cuenta en sus dos carriles; una fila lado
    a lado cuenta una vez."""
    if p.any_abreast:
        beside = rows(s, occ)[0]
    else:
        beside = np.zeros(occ.veh.size, np.bool_)
    queue_m = np.zeros(p.n_lanes)
    for k in range(occ.veh.size):
        i = occ.veh[k]
        if s.stopped[i] and not s.crossed[i] and not beside[k]:
            queue_m[occ.lane[k]] += s.vlen[i] + p.gap_stop_t[s.vtype[i]]
    return np.minimum(queue_m / p.L, 1.0)


@jit
def lane_tails(p, s):
    """Parte trasera del último vehículo de cada carril, si está detenido y su índice (-1 = carril vacío)."""
    rear_min = np.full(p.n_lanes, INF)
    tail_stopped = np.zeros(p.n_lanes, np.bool_)
    tail = np.full(p.n_lanes, -1, np.int64)
    n = s.ints[N]
    for ln in range(p.n_lanes):
        for i in range(n):
            if s.lane[i] == ln or s.lc_target[i] == ln:
                rear = s.x[i] - s.vlen[i]
                if tail[ln] < 0 or rear < rear_min[ln]:
                    rear_min[ln] = rear
                    tail[ln] = i
        if tail[ln] >= 0:
            tail_stopped[ln] = s.stopped[tail[ln]]
    return rear_min, tail_stopped, tail


# ------------------------------------------------------------------ altas y bajas


@jit
def add_vehicle(p, s, vt, pax, lane, length, x, bottleneck_at, speed_kmh, v0):
    """Agrega un vehículo al inicio del tramo; pax = 0 si lleva mercancía. Sin `length` (NAN), mide el largo medio de
    su tipo; sin `speed_kmh`, su velocidad máxima es la media del tipo. Con `x` (frente), es un vehículo de la
    condición inicial: ya está en el tramo, no cuenta como llegada y no tiene tiempo de recorrido. `bottleneck_at` es
    la posición de su detención de [bottleneck] (inf = no se detiene). `v0` es su velocidad al entrar (m/paso); sin
    ella, entra a su máxima. Debe haber lugar en los arreglos."""
    i = s.ints[N]
    initial = not math.isnan(x)
    s.vtype[i] = vt
    s.pax[i] = pax
    s.lane[i] = lane
    s.lc_target[i] = -1
    s.lc_timer[i] = 0
    s.cooldown[i] = 0
    s.react[i] = -1
    s.vlen[i] = p.length_t[vt] if math.isnan(length) else length
    s.x[i] = x if initial else s.vlen[i]  # entra con la parte trasera en x = 0
    s.vmax[i] = p.v_step[vt] if math.isnan(speed_kmh) else speed_kmh / 3.6 * DT
    s.stopped[i] = False
    s.crossed[i] = False
    s.entry_tick[i] = -1 if initial else s.ints[TICK]
    s.vid[i] = int(s.entered_veh.sum())
    if p.has_stop[vt]:
        s.stop_pos[i], s.stop_kind[i] = p.stop_x[vt], TYPE_STOP
    else:
        s.stop_pos[i], s.stop_kind[i] = bottleneck_at, BOTTLENECK_STOP
    ahead = s.x[i] < s.stop_pos[i] - EPS  # si empieza pasada su parada, no para
    s.stop_state[i] = STOP_AHEAD if ahead else NO_STOP
    s.dwell[i] = 0
    s.vcap_last[i] = min(s.vmax[i], p.lane_vmax[lane])  # entra en marcha
    s.v_last[i] = s.vcap_last[i] if math.isnan(v0) else min(v0, s.vcap_last[i])
    s.ints[N] = i + 1
    s.entered_veh[vt] += 1
    if initial:
        s.initial_veh[vt] += 1
        s.initial_pax[vt] += pax
    else:
        s.pax_m[vt] += pax * s.vlen[i]


@jit
def compact(s, gone):
    """Quita los vehículos marcados en `gone`, conservando el orden de los demás."""
    n = s.ints[N]
    k = 0
    for i in range(n):
        if gone[i]:
            continue
        if k != i:
            s.vtype[k] = s.vtype[i]
            s.pax[k] = s.pax[i]
            s.lane[k] = s.lane[i]
            s.lc_target[k] = s.lc_target[i]
            s.lc_timer[k] = s.lc_timer[i]
            s.cooldown[k] = s.cooldown[i]
            s.react[k] = s.react[i]
            s.x[k] = s.x[i]
            s.stopped[k] = s.stopped[i]
            s.crossed[k] = s.crossed[i]
            s.entry_tick[k] = s.entry_tick[i]
            s.vid[k] = s.vid[i]
            s.stop_state[k] = s.stop_state[i]
            s.dwell[k] = s.dwell[i]
            s.v_last[k] = s.v_last[i]
            s.vcap_last[k] = s.vcap_last[i]
            s.vlen[k] = s.vlen[i]
            s.vmax[k] = s.vmax[i]
            s.stop_pos[k] = s.stop_pos[i]
            s.stop_kind[k] = s.stop_kind[i]
        k += 1
    s.ints[N] = k


# ------------------------------------------------------------------ llegadas y entrada


@jit
def entry_lane(p, s, rngs, vt):
    if p.n_lanes == 1:
        return 0
    count = p.entry_count[vt]  # sin los carriles exclusivos de otros tipos
    lanes = p.entry_lanes[vt, :count]
    if count == 1:  # carril fijo: los que no cambian de carril
        return lanes[0]
    # Los que cambian de carril: cola de entrada más corta; desempate por más espacio libre a la entrada.
    shortest = s.q_count[lanes].min()
    best = lanes[s.q_count[lanes] == shortest]
    if best.size > 1:
        tail = lane_tails(p, s)[0][best]
        best = best[tail == tail.max()]
    if best.size == 1:
        return best[0]
    return best[rngs.entry[vt].integers(0, best.size)]  # lo mismo que Generator.choice(best)


@jit
def arrivals(p, s, rngs, arr):
    """Llegadas del paso actual (del bloque `arr`): cuentan como llegadas y esperan en la cola de entrada."""
    tick = s.ints[TICK]
    k = s.ints[ARRIVAL_POS]
    while k < arr.tick.size and arr.tick[k] == tick:
        vt = arr.vt[k]
        pax = arr.pax[k]
        s.arrived_veh[vt] += 1
        if pax == 0:
            s.arrived_cargo[vt] += 1
        else:
            s.arrived_pax[vt] += pax
            s.pax_hist[vt, pax] += 1
        ln = entry_lane(p, s, rngs, vt)
        cap = s.q_vt.shape[1]
        j = (s.q_head[ln] + s.q_count[ln]) % cap
        s.q_vt[ln, j] = vt
        s.q_pax[ln, j] = pax
        s.q_len[ln, j] = arr.length[k]
        s.q_at[ln, j] = arr.at[k]
        s.q_speed[ln, j] = arr.speed[k]
        s.q_arrived[ln, j] = tick
        s.q_count[ln] += 1
        k += 1
    s.ints[ARRIVAL_POS] = k


@jit
def entry_speed(p, s, vt, speed_kmh, from_stop, space, tail):
    """Velocidad (m/paso) con la que entra un vehículo con aceleración o frenado graduales: 0 si sale de la cola de
    entrada detenida y acelera gradualmente; si no, a lo más la que le permite frenar detrás del último del carril.
    NAN = a su máxima, como sin dinámica."""
    if from_stop and math.isfinite(p.acc_t[vt]):
        return 0.0
    if tail < 0 or not math.isfinite(p.dec_t[vt]):
        return NAN
    v_lead = 0.0 if s.stopped[tail] else s.v_last[tail]
    safe = safe_speed_1(space, v_lead, p.dec_t[vt], p.dec_t[s.vtype[tail]])
    return min(speed_kmh / 3.6 * DT, safe)


@jit
def queue_ready(p, s, rngs, ln, vt, fits, tail_stopped):
    """El primero de la cola de entrada detenida del carril `ln` ya reaccionó y puede entrar.

    Empieza a reaccionar cuando el de adelante se mueve (o ya hay lugar para él) y entra cuando termina su reacción,
    en cuanto quepa."""
    left = s.queue_react[ln]
    if left < 0:
        if tail_stopped and not fits:
            return False
        s.queue_react[ln] = reaction_ticks(rngs.queue_react[vt], p, 1)[0]
        return False
    if left > 0:
        left -= 1
        s.queue_react[ln] = left
    return left == 0


@jit
def spawn(p, s, rngs):
    if not (s.q_count > 0).any():
        return
    rear_min, tail_stopped, tail = lane_tails(p, s)
    cap = s.q_vt.shape[1]
    for ln in range(p.n_lanes):
        if s.q_count[ln] == 0:
            continue
        h = s.q_head[ln]
        vt, pax, length = s.q_vt[ln, h], s.q_pax[ln, h], s.q_len[ln, h]
        at, speed, arrived = s.q_at[ln, h], s.q_speed[ln, h], s.q_arrived[ln, h]
        if tail_stopped[ln]:
            gap = p.gap_stop_t[vt]
        elif p.has_headway_t[vt]:  # entra a su velocidad deseada: guarda el gap de time_headway a ella
            gap = run_gap(p, vt, min(speed / 3.6 * DT, p.lane_vmax[ln]))
        else:
            gap = p.gap_run_t[vt]
        fits = rear_min[ln] >= length + gap
        from_stop = p.queue_reaction and s.queue_stopped[ln]
        if from_stop and not queue_ready(p, s, rngs, ln, vt, fits, tail_stopped[ln]):
            continue
        if fits:
            s.q_head[ln] = (h + 1) % cap
            s.q_count[ln] -= 1
            s.queue_wait_ticks[vt] += s.ints[TICK] - arrived
            v0 = NAN
            if p.any_dynamics:
                v0 = entry_speed(p, s, vt, speed, from_stop, rear_min[ln] - length - gap, tail[ln])
                bump = bump_speed_1(p, vt, ln, length, 0.0) if p.dyn_t[vt] else INF
                if math.isfinite(bump):  # entra a lo más a la velocidad con la que aún llega al tope
                    v0 = min(speed / 3.6 * DT if math.isnan(v0) else v0, bump)
            add_vehicle(p, s, vt, pax, ln, length, NAN, at, speed, v0)
            s.queue_react[ln] = -1
            # Quien sigue esperaba detrás de un detenido: también arranca con reacción.
            s.queue_stopped[ln] = s.queue_stopped[ln] and s.q_count[ln] > 0
        elif p.queue_reaction and tail_stopped[ln]:
            s.queue_stopped[ln] = True  # la cola del tramo llega detenida hasta la entrada


@jit
def drain_exit(p, s):
    """Salen de cada cola de salida los vehículos que acepta su capacidad en este paso. Sin cola, la fracción
    acumulada no pasa de una salida (no se ahorran salidas para después)."""
    s.exit_credit[:] += p.exit_add
    cap = s.exit_ring.shape[1]
    for ln in range(p.n_lanes):
        if not (p.exit_limited[ln] and s.exit_credit[ln] >= 1):
            continue
        while s.exit_count[ln] > 0 and s.exit_credit[ln] >= 1:
            s.exit_q[ln] -= s.exit_ring[ln, s.exit_head[ln]]
            s.exit_head[ln] = (s.exit_head[ln] + 1) % cap
            s.exit_count[ln] -= 1
            s.exit_credit[ln] -= 1
        if s.exit_count[ln] == 0:
            s.exit_q[ln] = 0.0  # sin residuos de redondeo
            s.exit_credit[ln] = min(s.exit_credit[ln], 1.0)


# ------------------------------------------------------------------ avance


@jit
def move(p, s, rngs, phase, tick):
    n = s.ints[N]
    dynamics = p.any_dynamics
    occ = occupancy(p, s)
    m = occ.veh.size

    # Espacio disponible de cada entrada respecto a su líder.
    lead = occ.leader.copy()
    has_lead = occ.has_leader.copy()
    beside = np.zeros(m, np.bool_)
    side = np.zeros(m, np.bool_)
    if p.any_abreast or p.any_pass_in_lane:
        beside, rid, row_last = rows(s, occ)
        if p.any_abreast:
            side = side_slot(p, s, occ, beside, rid, row_last)
    skip = np.zeros(m, np.bool_)
    passing = np.zeros(m, np.bool_)
    if p.any_pass_in_lane:
        # Rebase dentro del carril: el líder que se rebasa (o que se adelanta) no cuenta; manda el siguiente.
        skip, passing = in_lane_pass(p, s, occ, beside, side)
        for k in range(m):
            if skip[k]:
                lead[k] = occ.veh[min(k + 2, m - 1)]
                has_lead[k] = occ.has_leader[k] and k + 1 < m and occ.has_leader[k + 1]
    lead_stop = np.zeros(m, np.bool_)
    stop_e = np.empty(m)
    move_e = np.empty(m)
    safe_e = np.empty(m)
    for k in range(m):
        i = occ.veh[k]
        t = s.vtype[i]
        ld = lead[k]
        ls = has_lead[k] and s.stopped[ld]
        lead_stop[k] = ls
        gap_need = p.gap_stop_t[t] if ls else p.gap_run_t[t]
        if dynamics and p.dyn_t[t] and not ls:
            # Con dinámica gradual, el gap detrás de un líder en marcha crece con la velocidad propia: gap_stop casi
            # detenido, gap_run a su velocidad deseada. Así, cuando la cola arranca, no le exige de golpe gap_run a
            # quien estaba a gap_stop.
            own = s.v_last[i] / min(s.vmax[i], p.lane_vmax[occ.lane[k]])
            gap_need = p.gap_stop_t[t] + (p.gap_run_t[t] - p.gap_stop_t[t]) * min(max(own, 0.0), 1.0)
        if p.any_headway and p.has_headway_t[t] and not ls:
            # Con time_headway, detrás de un líder en marcha guarda gap_stop más lo que recorre en ese tiempo a su
            # velocidad: la real del paso anterior con dinámica gradual; sin ella, la deseada (que alcanza de golpe).
            v_own = s.v_last[i] if p.dyn_t[t] else min(s.vmax[i], p.lane_vmax[occ.lane[k]])
            gap_need = p.gap_stop_t[t] + p.headway_t[t] * v_own / DT
        sp = s.x[ld] - s.vlen[ld] - gap_need - occ.x[k]
        yielding = skip[k] and not passing[k]
        if yielding:
            # Quien deja pasar puede quedar al lado del que se le adelanta, pero no volver a rebasarlo; si ambos se
            # detienen detrás del mismo vehículo, queda SIDE_EPS atrás (como en una fila lado a lado).
            front = s.x[occ.leader[k]] - SIDE_EPS - occ.x[k]
            if dynamics and p.dyn_t[t]:
                # Con dinámica gradual cuenta lo que el que se adelanta avanzó en el último paso: si apenas lo
                # rebasó, no lo obliga a frenar en seco.
                front = front + s.v_last[occ.leader[k]]
            sp = min(sp - SIDE_EPS, front)
        if passing[k]:
            # Si el siguiente vehículo frena a los dos, quien rebasa queda SIDE_EPS detrás del rebasado: van lado a
            # lado en vez de empatados en la misma posición.
            sp = sp - SIDE_EPS
        if side[k]:
            # Lugar libre al lado del líder detenido: se detiene con el frente alineado al suyo.
            sp = s.x[occ.leader[k]] - SIDE_EPS - occ.x[k]
        e = occ.order[k]
        stop_e[e] = sp if ls else INF
        move_e[e] = sp if has_lead[k] and not ls else INF
        if dynamics:
            # Velocidad segura respecto al de adelante (el que manda en cada entrada).
            v_lead = 0.0 if ls else s.v_last[ld]
            if yielding:
                v_lead = min(v_lead, s.v_last[occ.leader[k]])
            dec_e = p.dec_safe_t[t]
            vt_lead = s.vtype[ld]
            lead_term = v_lead * v_lead * dec_e / max(p.dec_t[vt_lead], dec_e) if p.dec_fin_t[vt_lead] else 0.0
            safe_e[e] = gipps(sp, lead_term, dec_e, p.dec_fin_t[t]) if has_lead[k] else INF

    # Por vehículo: mínimo entre su carril y, si está cambiando, el carril destino.
    stop_space = stop_e[:n].copy()
    move_space = move_e[:n].copy()
    vsafe = safe_e[:n].copy()
    for j in range(occ.changing.size):
        i = occ.changing[j]
        stop_space[i] = min(stop_space[i], stop_e[n + j])
        move_space[i] = min(move_space[i], move_e[n + j])
        if dynamics:
            vsafe[i] = min(vsafe[i], safe_e[n + j])

    # Línea del final del tramo. Fuera del semáforo: quien ya cruzó y los tipos con carril exclusivo sin semáforo.
    ruled_out = np.empty(n, np.bool_)
    safe_line = np.empty(n)
    for i in range(n):
        t = s.vtype[i]
        ruled_out[i] = s.crossed[i] or p.free_type[t]
        if dynamics:
            to_line = INF if s.crossed[i] else p.L - s.x[i]
            safe_line[i] = gipps(to_line, 0.0, p.dec_safe_t[t], p.dec_fin_t[t])
            if phase == YELLOW:
                # Amarillo: quien todavía puede detenerse antes de la línea (frenando a decel) se detiene.
                dec_v = p.dec_t[t]
                if math.isfinite(dec_v) and not ruled_out[i] and s.v_last[i] - dec_v <= safe_line[i] + EPS:
                    stop_space[i] = min(stop_space[i], to_line)
                    vsafe[i] = min(vsafe[i], safe_line[i])
            elif phase == RED and not ruled_out[i]:
                vsafe[i] = min(vsafe[i], safe_line[i])
        if phase == RED and not ruled_out[i]:
            stop_space[i] = min(stop_space[i], p.L - s.x[i])

    # Líneas de alto intermedias (semáforos y pasos peatonales): en rojo, quien no las ha pasado (su frente) se
    # detiene en la línea; en amarillo, se detiene si aún puede (con frenado gradual) o baja la velocidad cerca de
    # ella, como en el semáforo del final. Un paso peatonal solo rige en los carriles de su tope.
    n_lines = p.inner_x.size
    inner_near = np.zeros((n_lines, n), np.bool_)
    for j in range(n_lines):
        ph = line_phase(p, 1 + j, tick)
        if ph == GREEN:
            continue
        pos = p.inner_x[j]
        for i in range(n):
            t = s.vtype[i]
            tgt = s.lc_target[i]
            before = (
                s.x[i] <= pos
                and not p.inner_free[j, t]
                and (p.inner_lane[j, s.lane[i]] or (tgt >= 0 and p.inner_lane[j, max(tgt, 0)]))
            )  # fmt: skip
            if not before:
                continue
            to_inner = pos - s.x[i]
            dec_v = p.dec_t[t]
            if dynamics:
                safe_inner = gipps(to_inner, 0.0, p.dec_safe_t[t], p.dec_fin_t[t])
                if ph == YELLOW:
                    if math.isfinite(dec_v) and s.v_last[i] - dec_v <= safe_inner + EPS:
                        stop_space[i] = min(stop_space[i], to_inner)
                        vsafe[i] = min(vsafe[i], safe_inner)
                else:
                    vsafe[i] = min(vsafe[i], safe_inner)
            if ph == RED:
                stop_space[i] = min(stop_space[i], to_inner)
            else:
                inner_near[j, i] = pos - s.x[i] <= p.yellow_approach and not (dynamics and math.isfinite(dec_v))

    if p.any_exit:
        # Cola de salida sin lugar: quien no cabe en los m libres de la cola de su carril (o del destino, si está
        # cambiando) no cruza la línea. Una cola vacía acepta a cualquiera, aunque no quepa entero.
        closed = np.zeros(n, np.bool_)
        front = np.full(p.n_lanes, -1, np.int64)  # el de adelante que no ha cruzado, por carril
        for i in range(n):
            if s.crossed[i]:
                continue
            need = s.vlen[i] + p.gap_stop_t[s.vtype[i]]
            ln, tgt = s.lane[i], s.lc_target[i]
            no_room = s.exit_q[ln] > 0 and s.exit_q[ln] + need > p.exit_storage[ln] + EPS
            if tgt >= 0:
                no_room = no_room or (s.exit_q[tgt] > 0 and s.exit_q[tgt] + need > p.exit_storage[tgt] + EPS)
            closed[i] = no_room
            if front[ln] < 0 or s.x[i] >= s.x[front[ln]]:
                front[ln] = i
        # Línea cerrada en el carril si no cabe el siguiente en cruzar.
        for ln in range(p.n_lanes):
            s.exit_closed[ln] = front[ln] >= 0 and closed[front[ln]]
            if s.exit_closed[ln]:
                s.exit_blocked_ticks[ln] += 1.0
        for i in range(n):
            if closed[i]:
                stop_space[i] = min(stop_space[i], p.L - s.x[i])
                if dynamics:
                    vsafe[i] = min(vsafe[i], safe_line[i])

    bn_skip = np.zeros(n, np.bool_)
    if p.any_stop:
        for i in range(n):
            # Detenciones de [bottleneck] que no aplican ahora: fuera de sus carriles o cambiando de carril.
            bn_lane_ok = p.bn_lane[s.lane[i]] and s.lc_target[i] < 0
            bn_skip[i] = s.stop_kind[i] == BOTTLENECK_STOP and not bn_lane_ok
            # Parada: cuenta regresiva de quien está en ella; al terminar arranca (ya estaba listo).
            if s.stop_state[i] == AT_STOP:
                s.dwell[i] -= 1
                if s.dwell[i] <= 0:
                    s.stop_state[i] = NO_STOP
                    s.stopped[i] = False
                    s.react[i] = -1
            # Mientras la parada esté pendiente o en curso, es un alto en su posición.
            pending = s.stop_state[i] == AT_STOP or (s.stop_state[i] == STOP_AHEAD and not bn_skip[i])
            if pending:
                t = s.vtype[i]
                stop_space[i] = min(stop_space[i], s.stop_pos[i] - s.x[i])
                if dynamics:
                    vsafe[i] = min(vsafe[i], gipps(s.stop_pos[i] - s.x[i], 0.0, p.dec_safe_t[t], p.dec_fin_t[t]))

    # Máquina de estados: reacción de arranque de los detenidos.
    space = np.empty(n)
    binding_stop = np.empty(n, np.bool_)
    released = np.zeros(n, np.bool_)
    any_released = False
    for i in range(n):
        space[i] = min(stop_space[i], move_space[i])
        binding_stop[i] = stop_space[i] <= move_space[i]
        if s.stopped[i] and s.react[i] > 0:
            s.react[i] -= 1
            if s.react[i] == 0:
                s.stopped[i] = False
                s.react[i] = -1
        released[i] = s.stopped[i] and s.react[i] < 0 and (not binding_stop[i] or space[i] > p.release_space)
        any_released = any_released or released[i]
    if any_released:
        for t in range(p.n_types):  # cada tipo con su generador, en orden de llegada
            count = 0
            for i in range(n):
                if released[i] and s.vtype[i] == t:
                    count += 1
            if count == 0:
                continue
            ticks = reaction_ticks(rngs.react[t], p, count)
            c = 0
            for i in range(n):
                if released[i] and s.vtype[i] == t:
                    s.react[i] = ticks[c]
                    c += 1

    # Avance: velocidad máxima dentro del límite del carril (el menor de sus dos carriles si está cambiando),
    # reducida durante la maniobra, en los topes y en amarillo cerca de la línea de alto.
    vcap = np.empty(n)
    adv = np.empty(n)
    for i in range(n):
        t = s.vtype[i]
        tgt = s.lc_target[i]
        dec_v = p.dec_t[t]
        cap = min(s.vmax[i], p.lane_vmax[s.lane[i]])
        if tgt >= 0:
            cap = min(cap, p.lane_vmax[tgt])
        cap = cap * (p.lane_change_speed_factor if tgt >= 0 else 1.0)
        if p.any_bump:
            # Tope: quien lo pisa, o lo alcanzaría en este paso, no rebasa la velocidad de su tipo. Con frenado
            # gradual, además frena antes a decel: con la distancia al tope, la velocidad segura (Gipps, un paso de
            # reacción) solo deja alcanzarlo en un paso a la velocidad del tope o menos. Cuenta el carril de origen
            # y el destino si está cambiando. Cada tope por separado; queda la menor velocidad.
            vb = p.bump_v_t[t]
            for b in range(p.bump_x.size):
                bump_x = p.bump_x[b]
                in_lane = p.bump_lane[b, s.lane[i]] or (tgt >= 0 and p.bump_lane[b, max(tgt, 0)])
                before = in_lane and math.isfinite(vb) and s.x[i] - s.vlen[i] < bump_x  # aún no lo pasa entero
                if before and s.x[i] + cap > bump_x:
                    cap = min(cap, vb)
                if dynamics and before and s.x[i] < bump_x:
                    dec_vf = p.dec_safe_t[t]
                    lead_term = vb * vb * dec_vf / dec_vf  # el de adelante es el tope: frena como él
                    vsafe[i] = min(vsafe[i], gipps(bump_x - s.x[i], lead_term, dec_vf, p.dec_fin_t[t]))
        # Quien se aproxima a la línea de alto en amarillo baja la velocidad; con frenado gradual decide: se detiene
        # si aún puede (arriba) o sigue sin frenar.
        near = phase == YELLOW and not ruled_out[i] and p.L - s.x[i] <= p.yellow_approach
        if near and not (dynamics and math.isfinite(dec_v)):
            cap = cap * p.yellow_speed_factor
        for j in range(n_lines):
            if inner_near[j, i]:
                cap = cap * p.yellow_speed_factor
        vcap[i] = cap
        if dynamics:
            # Hacia la velocidad deseada a lo más accel·DT por paso (y bajando a lo más decel·DT), sin rebasar la
            # velocidad segura; el espacio al de adelante sigue siendo un límite duro.
            v_prev = s.v_last[i]
            v_new = min(max(min(v_prev + p.acc_t[t], cap), v_prev - dec_v), vsafe[i])
        else:
            v_new = cap
        moving = not s.stopped[i]
        adv[i] = max(min(v_new, space[i]), 0.0) if moving else 0.0
        if moving and adv[i] < EPS and binding_stop[i]:
            s.stopped[i] = True
    if p.any_emissions:
        emit(p, s, adv)  # antes de actualizar v_last: la aceleración usa la del paso anterior

    # Recorrido del paso: pasajeros·m y m por tipo; distancia y tiempo por carril y por tipo de los que están
    # dentro del tramo (con los detenidos, en su carril actual). En el paso en que cruza la línea solo cuenta la
    # fracción del paso que pasó dentro del tramo. Cada suma por tipo o carril se acumula aparte y luego se agrega,
    # como un bincount.
    pax_m = np.zeros(p.n_types)
    veh_m = np.zeros(p.n_types)
    lane_dist = np.zeros(p.n_lanes)
    lane_count = np.zeros(p.n_lanes)
    type_dist = np.zeros(p.n_types)
    type_frac = np.zeros(p.n_types)
    for i in range(n):
        t = s.vtype[i]
        s.v_last[i] = adv[i]
        s.vcap_last[i] = vcap[i]
        inside = min(adv[i], max(p.L - s.x[i], 0.0))
        pax_m[t] += s.pax[i] * inside
        veh_m[t] += inside
        if not s.crossed[i]:
            lane_dist[s.lane[i]] += inside
            lane_count[s.lane[i]] += 1
            type_dist[t] += inside
            type_frac[t] += inside / max(adv[i], EPS) if adv[i] > EPS else 1.0
        s.x[i] += adv[i]
    s.pax_m[:] += pax_m
    s.veh_m[:] += veh_m
    s.lane_dist[:] += lane_dist
    s.lane_time[:] += lane_count * DT
    s.type_dist[:] += type_dist
    s.type_time[:] += type_frac * DT
    for k in range(m):
        if passing[k]:  # quien rebasa terminó de adelantarse al que rebasaba
            me, other = occ.veh[k], occ.leader[k]
            if s.x[me] > s.x[other] and s.x[me] - adv[me] <= s.x[other] - adv[other]:
                s.in_lane_passes[s.vtype[me]] += 1

    # Llegada a la parada: queda detenido el tiempo de descenso y ascenso.
    if p.any_stop:
        for i in range(n):
            if not (s.stop_state[i] == STOP_AHEAD and s.x[i] >= s.stop_pos[i] - EPS):
                continue
            t = s.vtype[i]
            if s.stop_kind[i] == BOTTLENECK_STOP:
                if bn_skip[i]:  # pasó el punto fuera de los carriles de la detención: no se detiene
                    s.stop_state[i] = NO_STOP
                    continue
                ticks = normal_tick(rngs.bottleneck_time[t], p.bn_mean[t], p.bn_std[t], p.bn_fixed[t])
                s.bn_stops[t] += 1
                s.bn_ticks[t] += ticks
            else:
                ticks = normal_tick(rngs.stop[t], p.stop_mean[t], p.stop_std[t], p.stop_fixed[t])
                s.stops[t] += 1
                s.stop_time_ticks[t] += ticks
            if ticks == 0:
                s.stop_state[i] = NO_STOP
                continue
            s.stop_state[i] = AT_STOP
            s.dwell[i] = ticks
            s.stopped[i] = True
            s.react[i] = -1

    # Temporizadores de cambio de carril.
    for i in range(n):
        if s.cooldown[i] > 0:
            s.cooldown[i] -= 1
        if s.lc_target[i] >= 0:
            s.lc_timer[i] -= 1
            if s.lc_timer[i] <= 0:
                s.lane[i] = s.lc_target[i]
                s.lc_target[i] = -1
                s.cooldown[i] = p.cooldown_ticks_t[s.vtype[i]]

    # Cruce de la línea de alto. Tiempo de recorrido solo de quien recorrió todo el tramo (no los de la condición
    # inicial).
    cap_exit = s.exit_ring.shape[1]
    any_gone = False
    gone = np.zeros(n, np.bool_)
    for i in range(n):
        if not s.crossed[i] and s.x[i] > p.L:
            t = s.vtype[i]
            s.cum_veh[t] += 1
            s.cum_pax[t] += s.pax[i]
            if s.entry_tick[i] >= 0:
                s.timed_veh[t] += 1
                s.travel_ticks[t] += tick + 1 - s.entry_tick[i]
            s.crossed[i] = True
            ln = s.lane[i]
            s.lane_cross[ln] += 1
            s.lane_crossed[ln] += 1
            if p.any_exit and math.isfinite(p.exit_rate[ln]):
                size = s.vlen[i] + p.gap_stop_t[t]
                s.exit_ring[ln, (s.exit_head[ln] + s.exit_count[ln]) % cap_exit] = size
                s.exit_count[ln] += 1
                s.exit_q[ln] += size
        # Sale del sistema cuando su parte trasera rebasa la línea.
        gone[i] = s.x[i] - s.vlen[i] > p.L
        any_gone = any_gone or gone[i]
    if any_gone:
        compact(s, gone)


@jit
def emit(p, s, adv):
    """Gramos que emite en el paso cada vehículo que aún no cruza y cuyo tipo emite, con su velocidad (adv) y su
    aceleración respecto al paso anterior, acotada a [−decel, accel] de su tipo (un frenado de emergencia no dispara
    el término a²). También lo que emitiría recorriendo lo mismo a flujo libre (a velocidad constante: su máxima
    dentro del límite del carril) y dónde lo emite."""
    n_pol = p.emis_on.shape[1]
    n_bins = s.emis_pos.shape[2]
    grams = np.zeros((p.n_types, n_pol))
    free = np.zeros((p.n_types, n_pol))
    for i in range(s.ints[N]):
        t = s.vtype[i]
        if s.crossed[i] or not p.emits_t[t]:
            continue
        step = adv[i]
        v = step / DT
        a = min(max((step - s.v_last[i]) / (DT * DT), -p.emis_dmax[t]), p.emis_amax[t])
        row = 2 * t + (0 if a >= EMISSION_DECEL else 1)
        vf = min(s.vmax[i], p.lane_vmax[s.lane[i]])  # m/paso
        v_free = vf / DT
        frac = step / vf
        ln = s.lane[i]
        b = min(max(int(s.x[i] / EMIS_BIN), 0), n_bins - 1)
        for k in range(n_pol):
            on = 1.0 if p.emis_on[t, k] else 0.0
            c = p.coef_sets[row, k]
            e = c[0] + c[1] * v + c[2] * v * v + c[3] * a + c[4] * a * a + c[5] * v * a
            g = max(e, 0.0) * DT * on
            cf = p.coef_sets[2 * t, k]  # a velocidad constante: el juego de a ≥ −0.5, sin los términos de a
            f = max(cf[0] + cf[1] * v_free + cf[2] * v_free * v_free, 0.0) * DT * frac * on
            grams[t, k] += g
            free[t, k] += f
            s.emis_pos[k, ln, b] += g
    s.emis_g[:] += grams
    s.emis_interval[:] += grams
    s.emis_free_g[:] += free


# ------------------------------------------------------------------ cambio de carril


@jit
def can_brake_behind(p, s, i, j, gap):
    """Con frenado gradual, `i` puede quedar `gap` m detrás de `j` sin frenar más que su `decel` (con un cambio de
    carril, sea quien cambia o el seguidor del carril destino). Sin frenado gradual, siempre."""
    dec = p.dec_t[s.vtype[i]]
    if not math.isfinite(dec):
        return True
    stopped = s.stopped[j]
    room = gap - (p.gap_stop_t[s.vtype[i]] if stopped else p.gap_run_t[s.vtype[i]])
    safe = safe_speed_1(room, 0.0 if stopped else s.v_last[j], dec, p.dec_t[s.vtype[j]])
    # También el límite duro: en un paso no avanza más que `room` (medido desde donde está j ahora).
    return s.v_last[i] - dec <= min(safe, room) + EPS


@jit
def neighbors(occ, bounds, lane, x):
    """(líder, seguidor) que tendría en `lane` un vehículo con el frente en x; -1 si no hay."""
    lo, hi = bounds[lane], bounds[lane + 1]
    k = lo + np.searchsorted(occ.x[lo:hi], x)
    return (occ.veh[k] if k < hi else -1), (occ.veh[k - 1] if k > lo else -1)


@jit
def pass_lane(p, s, occ, bounds, used, k, gap, stuck_speed):
    """Carril adyacente para rebasar al líder lento de la entrada k; -1 si ninguno conviene."""
    i, xi, li = occ.veh[k], occ.x[k], occ.lane[k]
    vi = s.vtype[i]
    over = p.overtake[vi]
    look = p.lookahead_t[vi]
    best, best_score = -1, min(gap, look) + p.min_adv_t[vi]
    for tl in (li + 1, li - 1):  # prefiere rebasar por la izquierda
        if tl < 0 or tl >= p.n_lanes or used[tl] or not p.allowed[vi, tl]:
            continue
        v_there = min(s.vmax[i], p.lane_vmax[tl])
        if v_there <= stuck_speed:  # ese carril no mejora la velocidad actual
            continue
        j, f = neighbors(occ, bounds, tl, xi)
        score = look
        if j >= 0:  # líder en el carril destino
            front = s.x[j] - s.vlen[j] - xi
            if front < p.gap_stop_t[vi] or not can_brake_behind(p, s, i, j, front):
                continue
            v_j = s.v_last[j] + EPS if over else min(s.vmax[j], p.lane_vmax[tl])
            if v_j < v_there or s.stopped[j]:
                score = min(front, look)
        if f >= 0:  # seguidor en el carril destino: se comprime, pero sin traslape
            back = xi - s.vlen[i] - s.x[f]
            if back < p.gap_stop_t[s.vtype[f]] or not can_brake_behind(p, s, f, i, back):
                continue
        if score >= best_score:
            best, best_score = tl, score
    return best


@jit
def rise_lane(p, s, occ, bounds, used, k):
    """Carril adyacente de mayor límite al que la entrada k puede subir; -1 si no hay lugar."""
    i, xi, li = occ.veh[k], occ.x[k], occ.lane[k]
    vi = s.vtype[i]
    best, best_v = -1, min(s.vmax[i], p.lane_vmax[li]) + EPS
    for tl in (li + 1, li - 1):
        if tl < 0 or tl >= p.n_lanes or used[tl] or not p.allowed[vi, tl]:
            continue
        v_there = min(s.vmax[i], p.lane_vmax[tl])
        if v_there <= best_v:
            continue
        j, f = neighbors(occ, bounds, tl, xi)
        if j >= 0:
            front = s.x[j] - s.vlen[j] - xi
            if front < p.gap_run_t[vi] or not can_brake_behind(p, s, i, j, front):
                continue
            # Un líder cercano detenido o no más rápido que él le quita la ventaja.
            if front < p.lookahead_t[vi] and (s.stopped[j] or s.v_last[j] <= s.v_last[i] + EPS):
                continue
        if f >= 0:
            back = xi - s.vlen[i] - s.x[f]
            if back < p.gap_run_t[s.vtype[f]] or not can_brake_behind(p, s, f, i, back):
                continue
        best, best_v = tl, v_there
    return best


@jit
def lane_changes(p, s, rngs):
    """Decisiones de cambio de carril de quienes circulan (no detenidos, sin maniobra ni enfriamiento), en orden
    aleatorio y con a lo más una entrada a cada carril por ronda.

    Condición para cualquier cambio: un líder dentro de su `lookahead` cuya velocidad real en el último paso esté al
    menos `leader_slowdown` (p. ej. 25 %) por debajo del límite del carril (en un carril sin límite, de la velocidad
    máxima del propio vehículo). Sin líder, no cambia.

    * Rebase: con un líder lento dentro de su `lookahead`, busca un carril adyacente donde avance más rápido y tenga
      `min_advantage` m más de espacio libre. Los tipos con `overtake` juzgan la lentitud por la velocidad real del
      último paso (líder en maniobra, frenado por su cola) y aceptan un carril de menor límite si ahí avanzan más que
      detrás del líder.
    * Subida: si no puede rebasar, pasa a un carril adyacente de mayor límite si hay lugar sin comprimir a nadie
      (`gap_run` adelante y atrás) y ahí no lo frena un líder dentro de su `lookahead`.
    """
    n = s.ints[N]
    cand = np.zeros(n, np.bool_)
    for i in range(n):
        cand[i] = (p.can_change[s.vtype[i]] and s.lc_target[i] < 0 and s.cooldown[i] == 0
                   and not s.stopped[i] and not s.crossed[i])  # fmt: skip
    if not cand.any():
        return
    occ = occupancy(p, s)
    m = occ.veh.size
    gap = np.empty(m)
    stuck_speed = np.empty(m)
    passing = np.zeros(m, np.bool_)
    rising = np.zeros(m, np.bool_)
    for k in range(m):
        i, ld = occ.veh[k], occ.leader[k]
        t = s.vtype[i]
        gap[k] = s.x[ld] - s.vlen[ld] - occ.x[k]
        # Velocidades efectivas en el carril de cada entrada (máxima del vehículo, sin rebasar el límite).
        lane_v = p.lane_vmax[occ.lane[k]]
        v_self = s.vmax[i]
        eff_self = min(v_self, lane_v)
        eff_lead = min(s.vmax[ld], lane_v)
        lead_stopped = s.stopped[ld]
        slow = occ.has_leader[k] and (eff_lead < eff_self or lead_stopped)
        stuck_speed[k] = 0.0 if lead_stopped else eff_lead  # a la que lo obliga su líder
        if p.overtake[t]:
            # Rebase agresivo: el líder es lento si en el último paso avanzó menos de lo que podía él.
            slow = slow or (occ.has_leader[k] and s.v_last[ld] < s.vcap_last[i] - EPS)
            stuck_speed[k] = s.v_last[ld]
        # Condición común: líder cercano que va al menos `leader_slowdown` por debajo del límite del carril.
        lane_max = lane_v if math.isfinite(lane_v) else v_self
        v_ahead = 0.0 if lead_stopped else s.v_last[ld]
        blocked = (
            cand[i]
            and occ.has_leader[k]
            and gap[k] < p.lookahead_t[t]
            and v_ahead < (1.0 - p.leader_slowdown) * lane_max - EPS
        )  # fmt: skip
        passing[k] = blocked and slow
        if p.any_rise and blocked and p.can_change[t]:
            # Candidato a subir: un carril adyacente permitido donde su velocidad máxima (acotada por el límite) es
            # mayor que en el propio.
            for d in (1, -1):
                adj = occ.lane[k] + d
                if 0 <= adj < p.n_lanes and p.allowed[t, adj] and min(v_self, p.lane_vmax[adj]) > eff_self + EPS:
                    rising[k] = True
    idx = np.flatnonzero(passing | rising)
    if idx.size == 0:
        return
    bounds = np.searchsorted(occ.lane, np.arange(p.n_lanes + 1))
    used = np.zeros(p.n_lanes, np.bool_)
    for k in rngs.lane_change.permutation(idx):
        best = pass_lane(p, s, occ, bounds, used, k, gap[k], stuck_speed[k]) if passing[k] else -1
        if best < 0 and rising[k]:
            best = rise_lane(p, s, occ, bounds, used, k)
        if best >= 0:
            i = occ.veh[k]
            used[best] = True
            s.lc_target[i] = best
            s.lc_timer[i] = rngs.lane_change.integers(p.lc_min, p.lc_max, endpoint=True)
            s.ints[LANE_CHANGES] += 1
            s.lane_changes_t[s.vtype[i]] += 1


# ------------------------------------------------------------------ muestreo


@jit
def sample(p, s):
    """Una muestra de las series del Recorder; reinicia los acumuladores del intervalo."""
    n = s.ints[N]
    pax_on = np.zeros(p.n_types)
    footprint = np.zeros(p.n_types)
    lane_sat = np.zeros(p.n_lanes)
    if n:
        for i in range(n):
            if not s.crossed[i]:
                pax_on[s.vtype[i]] += s.pax[i]
        occ = occupancy(p, s)
        for k in range(occ.veh.size):
            i, ld = occ.veh[k], occ.leader[k]
            t = s.vtype[i]
            gap = s.x[ld] - s.vlen[ld] - occ.x[k] if occ.has_leader[k] else INF
            # Detenido ocupa su largo + gap comprimido; en marcha, a lo más largo + gap en marcha. Espacio de los que
            # llevan pasajeros: los de mercancía no cuentan en pax/m.
            cap = p.gap_stop_t[t] if s.stopped[i] else p.gap_run_t[t]
            if not s.crossed[i] and s.pax[i] > 0:
                footprint[t] += s.vlen[i] + min(gap, cap)
        lane_sat = lane_saturation(p, s, occ)
    c = s.ints[SAMPLES]
    if c < s.r_cum_pax.shape[0]:
        s.r_cum_pax[c] = s.cum_pax
        s.r_pax_m[c] = s.pax_m
        s.r_pax_on[c] = pax_on
        s.r_footprint[c] = footprint
        s.r_lane_sat[c] = lane_sat
        s.r_lane_dist[c] = s.lane_dist
        s.r_lane_time[c] = s.lane_time
        s.r_lane_cross[c] = s.lane_cross
        s.r_exit_q[c] = s.exit_q
        s.r_entry_q[c] = s.q_count
        s.r_emis[c] = s.emis_interval
        s.ints[SAMPLES] = c + 1
    s.emis_interval[:] = 0.0
    s.lane_cross[:] = 0.0
    s.pax_m[:] = 0.0
    s.lane_dist[:] = 0.0
    s.lane_time[:] = 0.0


# ------------------------------------------------------------------ ciclo


@jit
def has_room(p, s, arr):
    """Hay lugar para el paso actual: un vehículo más por carril en el tramo, las llegadas del paso en cualquier cola
    de entrada y todos los del tramo en cualquier cola de salida."""
    n = s.ints[N]
    if n + p.n_lanes > s.x.size:
        return False
    tick = s.ints[TICK]
    k = s.ints[ARRIVAL_POS]
    arriving = 0
    while k + arriving < arr.tick.size and arr.tick[k + arriving] == tick:
        arriving += 1
    if (s.q_count + arriving > s.q_vt.shape[1]).any():
        return False
    return not (p.any_exit and (s.exit_count + n > s.exit_ring.shape[1]).any())


# ------------------------------------------------------------------ desde Python


@api
def advance(p, s, rngs, arr, until):
    """Corre pasos hasta el paso `until`. Regresa antes (NEED_ARRIVALS o NEED_ROOM) si necesita el siguiente bloque
    de llegadas o más lugar en los arreglos; DONE al llegar."""
    while s.ints[TICK] < until:
        tick = s.ints[TICK]
        if tick - tick % ARRIVAL_CHUNK != arr.start:
            return NEED_ARRIVALS
        if not has_room(p, s, arr):
            return NEED_ROOM
        # Un paso.
        arrivals(p, s, rngs, arr)
        spawn(p, s, rngs)
        if p.any_exit:
            drain_exit(p, s)
        if s.ints[N]:
            move(p, s, rngs, line_phase(p, 0, tick), tick)
        s.ints[TICK] = tick + 1
        if p.n_lanes > 1 and s.ints[N] and (tick + 1) % p.lc_every == 0:
            lane_changes(p, s, rngs)
        if (tick + 1) % p.sample_ticks == 0:
            sample(p, s)
    return DONE


@api
def py_safe_speed_1(space, v_lead, dec, dec_lead):
    return safe_speed_1(space, v_lead, dec, dec_lead)


@api
def py_run_gap(p, k, v_step):
    return run_gap(p, k, v_step)


@api
def py_bump_speed_1(p, k, lane, front, rear):
    return bump_speed_1(p, k, lane, front, rear)


@api
def py_line_phase(p, j, tick):
    return line_phase(p, j, tick)


@api
def py_add_vehicle(p, s, vt, pax, lane, length, x, bottleneck_at, speed_kmh, v0):
    add_vehicle(p, s, vt, pax, lane, length, x, bottleneck_at, speed_kmh, v0)


@api
def py_occupancy(p, s):
    return occupancy(p, s)


@api
def py_rows(s, occ):
    return rows(s, occ)


@api
def py_side_slot(p, s, occ, beside, rid, row_last):
    return side_slot(p, s, occ, beside, rid, row_last)


@api
def py_lane_saturation(p, s, occ):
    return lane_saturation(p, s, occ)


@api
def py_lane_changes(p, s, rngs):
    lane_changes(p, s, rngs)
