"""Simulation step compiled with numba: the step loop of `Simulation` (engine.py) over its state in arrays.

The model is the one engine.py describes; this module computes it, vehicle by vehicle, in simple loops that numba
compiles (`@njit(cache=True)`: the first run compiles and stores the result in `__pycache__`). With
`NUMBA_DISABLE_JIT=1` the same code runs as pure Python, for debugging.

All the state lives in three `NamedTuple`s of arrays built by `Simulation`:

  * `Params`: read-only tables (per type, per lane, stop lines and their phase schedule).
  * `State`: the vehicles (compact structure of arrays in [0, n)), the entry and exit queues (rings per
    lane), the accumulators and the `Recorder` series. The scalars (n, step, …) go in `State.ints`.
  * `Rngs`: the numpy generators of each random process and type (MAX_TYPES per tuple, the surplus unused).
    numba uses them with the same algorithms as numpy, so the draws are identical to those of the numpy engine.

The kernel does not receive the tuples but a reference to them (`as_ref`): numba passes a tuple by value, field by
field, and with ~90 arrays that makes compiling take minutes. With the reference they are used the same way (`s.x[i]`, `p.L`).

Poisson arrivals and what is drawn on arrival (passengers, length, speed, halt) are computed by Python in
chunks of ARRIVAL_CHUNK steps (`Arrivals`); `advance` returns to ask for the next chunk, or for more room when an
array fills up. Sums are done in the same order as in numpy (a `bincount` sums in index order), so the
results are bit-for-bit identical to those of the previous engine with the same seed.
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

from traffic.config import DT, EMISSION_DECEL, GREEN, RED, YELLOW
from traffic.emissions import EMIS_BIN

# The step functions are compiled without a Python wrapper (they are not called from Python); those engine.py uses
# are at the end (`py_*` and `advance`) and numba compiles each one the first time it is used.
jit = njit(cache=True, no_cpython_wrapper=True)
api = njit(cache=True)

EPS = 1e-6
INF = np.inf
NAN = np.nan  # «no value» in floating point arguments (Python's None)
SIDE_EPS = 0.01  # m the front of a vehicle stopping alongside stays behind (keeps the order)
ARRIVAL_CHUNK = 1024  # steps per arrival drawing chunk
# Stop states: no pending stop, before the stop, stopped at it.
NO_STOP, STOP_AHEAD, AT_STOP = 0, 1, 2
# Kind of stop: the type's own (e.g. the bus) or a [bottleneck] halt.
TYPE_STOP, BOTTLENECK_STOP = 0, 1
# How the reaction time is drawn: uniform, fixed or truncated normal.
REACT_UNIFORM, REACT_FIXED, REACT_NORMAL = 0, 1, 2
# Kind of each stop line (phase schedule): always green, fixed cycle or precomputed schedule (pedestrians).
LINE_OPEN, LINE_CYCLE, LINE_SCHEDULE = 0, 1, 2
# Why `advance` returns: it reached the requested step, needs the next arrival chunk or more room.
DONE, NEED_ARRIVALS, NEED_ROOM = 0, 1, 2
# Indices of State.ints.
N, TICK, LANE_CHANGES, SAMPLES, ARRIVAL_POS = range(5)


class Params(NamedTuple):
    """Read-only tables of a replica."""

    L: float
    n_lanes: int
    n_types: int
    n_ticks: int
    lc_every: int  # steps between lane change rounds
    sample_ticks: int
    key_stride: float  # separates the lanes in the sort key
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
    react_lo: int  # steps (uniform)
    react_hi: int
    react_mean: float  # s (fixed or normal)
    react_std: float
    react_min: float
    react_max: float
    lc_min: int  # steps a lane change lasts
    lc_max: int
    # Per vehicle type.
    v_step: np.ndarray
    length_t: np.ndarray
    gap_run_t: np.ndarray
    gap_stop_t: np.ndarray
    has_headway_t: np.ndarray
    headway_t: np.ndarray
    can_change: np.ndarray
    stop_x: np.ndarray
    has_stop: np.ndarray
    stop_mean: np.ndarray  # s, duration of the type's own stop
    stop_std: np.ndarray
    stop_fixed: np.ndarray  # steps if stop_std ≤ 0
    bn_mean: np.ndarray  # s, duration of the [bottleneck] halt
    bn_std: np.ndarray
    bn_fixed: np.ndarray
    free_type: np.ndarray
    abreast: np.ndarray
    pass_in_lane: np.ndarray
    allowed: np.ndarray  # (types, lanes)
    entry_lanes: np.ndarray  # (types, lanes): entry lanes of each type, the first entry_count
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
    emis_on: np.ndarray  # (types, pollutants)
    coef_sets: np.ndarray  # (2·types, pollutants, 6): row 2·type for a ≥ −0.5, 2·type + 1 for a < −0.5
    emis_amax: np.ndarray
    emis_dmax: np.ndarray
    # Per lane.
    lane_vmax: np.ndarray
    bn_lane: np.ndarray
    exit_rate: np.ndarray
    exit_limited: np.ndarray
    exit_add: np.ndarray
    exit_storage: np.ndarray
    # Bumps.
    bump_x: np.ndarray
    bump_lane: np.ndarray  # (bumps, lanes)
    # Intermediate stop lines (cfg.stop_lines): position, types that do not obey them, lanes where they apply.
    inner_x: np.ndarray
    inner_free: np.ndarray  # (lines, types)
    inner_lane: np.ndarray  # (lines, lanes)
    # Phases: row 0 = the traffic light at the end of the segment, row 1 + j = intermediate line j.
    line_kind: np.ndarray  # LINE_OPEN | LINE_CYCLE | LINE_SCHEDULE
    line_red: np.ndarray  # steps
    line_green: np.ndarray
    line_yellow: np.ndarray
    line_start_red: np.ndarray
    line_offset: np.ndarray  # steps
    line_sched: np.ndarray  # (lines + 1, n_ticks + 1) phase of each step of the LINE_SCHEDULE lines


class State(NamedTuple):
    """Mutable state of a replica (see `Simulation`)."""

    ints: np.ndarray  # N, TICK, LANE_CHANGES, SAMPLES, ARRIVAL_POS
    # Vehicles in the segment, [0, n).
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
    # Entry queue of each lane (ring): type, pax, length, halt, speed km/h, arrival step.
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
    # Exit queue of each lane (ring with what each vehicle takes).
    exit_ring: np.ndarray
    exit_head: np.ndarray
    exit_count: np.ndarray
    exit_q: np.ndarray
    exit_credit: np.ndarray
    exit_closed: np.ndarray
    exit_blocked_ticks: np.ndarray
    # Accumulators.
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
    # Recorder series (metrics.py).
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
    """Generators of the step's random processes: a tuple of MAX_TYPES per process (one per type)."""

    react: tuple  # reaction times
    stop: tuple  # duration of the stop
    bottleneck_time: tuple  # duration of the [bottleneck] halt
    queue_react: tuple  # reaction of a vehicle starting in the stopped entry queue
    entry: tuple  # entry lane tie-break
    lane_change: np.random.Generator  # decision order and duration of lane changes


class Arrivals(NamedTuple):
    """Arrivals of a chunk of ARRIVAL_CHUNK steps starting at `start`, in step and type order."""

    start: int
    tick: np.ndarray
    vt: np.ndarray
    pax: np.ndarray
    length: np.ndarray
    at: np.ndarray  # position of its [bottleneck] halt; inf = it does not stop
    speed: np.ndarray  # km/h


@structref.register
class Ref(types.StructRef):
    """Reference to the fields of a NamedTuple (see `as_ref`)."""

    def preprocess_fields(self, fields):
        return tuple((name, types.unliteral(typ)) for name, typ in fields)


class RefProxy(structref.StructRefProxy):
    """What Python sees of a `Ref`: it is only passed back to the kernel."""


structref.define_boxing(Ref, RefProxy)


@intrinsic
def _as_ref(typingctx, tup):
    """A new `Ref` with the fields of the named tuple `tup` (the arrays are shared, not copied)."""
    ref_type = Ref(list(zip(tup.fields, tup.types)))

    def codegen(context, builder, signature, args):
        # Like numba.experimental.structref.new, filling each field with the tuple's.
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
    """Reference to a named tuple to pass it to the kernel; with NUMBA_DISABLE_JIT=1, the tuple itself."""
    return tup if numba.config.DISABLE_JIT else _new_ref(tup)


class Occupancy(NamedTuple):
    """Lane-vehicle entries sorted by (lane, x). A vehicle changing lanes appears twice."""

    order: np.ndarray  # sorted position -> entry index (primary [0,n), secondary [n,m))
    veh: np.ndarray  # vehicle index of each sorted entry
    lane: np.ndarray  # lane of each sorted entry
    x: np.ndarray  # front position of each sorted entry
    has_leader: np.ndarray  # the next entry is in the same lane
    leader: np.ndarray  # vehicle index of the leader (valid if has_leader)
    changing: np.ndarray  # vehicles with a lane change in progress (order of the secondary entries)


# ------------------------------------------------------------------ formulas


@jit
def gipps(space, lead, dec, finite):
    """Maximum speed (m/step) at which it can still stop braking at `dec` (m/step²), with one reaction step, without
    exceeding `space` m plus what the one ahead covers while stopping (`lead` = v_lead²·dec/max(dec_lead, dec), or 0 if
    it does not count): v + v²/(2·dec) ≤ space + v_lead²/(2·dec_lead) (Gipps' rule). INF if braking is not `finite`.

    The one ahead is assumed to brake at least as hard as it does (dec_lead ≥ dec): that way, following a leader at
    constant speed, the equilibrium is space = v_lead, the same that the hard space limit imposes
    (parallel update), and it closes in without harsh braking. With a leader that brakes less (e.g. a car behind a
    bike), the rule would let it get closer and the hard limit would brake it abruptly."""
    if not finite:
        return INF
    return math.sqrt(dec * dec + 2 * dec * max(space, 0.0) + lead) - dec


@jit
def safe_speed_1(space, v_lead, dec, dec_lead):
    """`gipps` with the braking of the one ahead: INF if `dec` is infinite (instantaneous braking, no anticipation); a
    leader with instantaneous braking stops dead."""
    if not math.isfinite(dec):
        return INF
    lead = v_lead * v_lead * dec / max(dec_lead, dec) if math.isfinite(dec_lead) else 0.0
    return -dec + math.sqrt(dec * dec + 2 * dec * max(space, 0.0) + lead)


@jit
def run_gap(p, k, v_step):
    """Gap (m) a vehicle of type k keeps behind a moving leader when going at `v_step` m/step: gap_run or, with
    time_headway, gap_stop plus what it covers in that time."""
    if not p.has_headway_t[k]:
        return p.gap_run_t[k]
    return p.gap_stop_t[k] + p.headway_t[k] * v_step / DT


@jit
def bump_speed_1(p, k, lane, front, rear):
    """Maximum speed (m/step) of a vehicle of type k in `lane` because of the bumps: the type's if it is on one; before
    it, with gradual braking, the one that still lets it reach that speed braking at `decel`; INF if it does not apply
    (no bump in the lane, already past them or the type goes over without braking). With several, the lowest."""
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
    """Phase (RED, GREEN or YELLOW) of line j at step `tick`: 0 = the traffic light at the end of the segment, 1 + i =
    intermediate line i. A fixed cycle goes green → yellow → red, delayed by `offset` steps."""
    kind = p.line_kind[j]
    if kind == LINE_OPEN:
        return GREEN
    if kind == LINE_SCHEDULE:
        return p.line_sched[j, min(tick, p.n_ticks)]
    red, green, yellow = p.line_red[j], p.line_green[j], p.line_yellow[j]
    cycle = red + green + yellow
    pos = (tick - p.line_offset[j]) % cycle
    if p.line_start_red[j]:
        pos = (pos - red) % cycle  # the cycle starts at red
    if pos < green:
        return GREEN
    return YELLOW if pos < green + yellow else RED


# ------------------------------------------------------------------ draws (as in distributions.py)


@jit
def reaction_ticks(g, p, size):
    """Reaction time in steps (at least one), like distributions.reaction_ticks: uniform in [reaction_min,
    reaction_max] or normal(reaction_mean, reaction_std) truncated to that range by resampling."""
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
    return np.maximum(ticks, 1)  # with 0 steps it would never start


@jit
def normal_tick(g, mean, std, fixed):
    """A duration in steps, like distributions.normal_ticks(…, 1): normal(mean, std) truncated to ≥ 0 by resampling,
    quantized to DT; `fixed` if std ≤ 0."""
    if std <= 0:
        return fixed
    while True:
        draw = g.normal(mean, std, 10)  # 2·1 + 8, like normal_ticks
        for v in draw:
            if v >= 0:
                return int(np.rint(v / DT))


# ------------------------------------------------------------------ order in the lanes


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
    # The sort key is numpy's (lane · separation + x), so the order is the same.
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
    """Side-by-side rows in the order of `occ`: (beside, row id, index of the last one of its row).

    `beside` marks a vehicle that overlaps with its leader, i.e. it is next to it in the same row. A row is a run
    of consecutive entries joined this way; without `abreast`, each vehicle is its own row."""
    m = occ.veh.size
    beside = np.zeros(m, np.bool_)
    for k in range(m):
        ld = occ.leader[k]
        beside[k] = occ.has_leader[k] and s.x[ld] - s.vlen[ld] - occ.x[k] < 0
    rid = np.empty(m, np.int64)
    row = -1
    for k in range(m):
        if k == 0 or not beside[k - 1]:  # entry k continues in the row of k-1 if k-1 is beside it
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
    """Entries that can stop beside their leader: their type allows `abreast` > 1, the leader is stopped, is of
    the same type (as is its whole row) and the row has room. Only a vehicle arriving while moving takes the place (or one
    already in it): a vehicle already stopped in a row, or with another one beside it, does not move ahead to another row.
    It does not apply to vehicles changing lanes or that have already crossed."""
    m = occ.veh.size
    out = np.zeros(m, np.bool_)
    if m < 2:
        return out
    # Minimum and maximum type of each row.
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
        nxt = min(k + 1, m - 1)  # entry of the leader (the next one in the order)
        ahead = row_last[nxt] - k  # vehicles in the leader's row that are ahead
        companion = k > 0 and beside[k - 1]  # someone stopped beside it (the previous entry has it as leader)
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
    """Overtaking within the lane between moving vehicles of the same type with `pass_in_lane`.

    Returns, per entry, (does not count its leader, is overtaking it). A vehicle overtakes when it has a higher maximum
    speed than its leader (both bounded by the lane's limit), as long as the leader does not already have another one
    beside it and nobody is catching up with it on one side: that way there are never more than two side by side. The
    leader does not count either for the overtaken vehicle while the faster one passing it is beside it or pulling away
    without being `gap_run` plus one step ahead yet.

    `beside` (overlaps with its leader) and `side` (can stop beside its leader) are those of rows and side_slot."""
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
        # The leader already has another one beside it or is going to stop beside its own; someone is catching up
        # with it on one side.
        lead_beside = k + 1 < m and (beside[k + 1] or side[k + 1])
        behind_beside = k > 0 and beside[k - 1]
        faster = v_me > v_lead + EPS
        if p.any_dynamics:
            # With gradual dynamics it must also already be going at least as fast as it (it does not overtake while
            # accelerating).
            faster = faster and (not p.dyn_t[t] or s.v_last[me] >= s.v_last[ld] - EPS)
        passing[k] = faster and not lead_beside and not behind_beside
        gap = s.x[ld] - s.vlen[ld] - occ.x[k]
        # While they are side by side and, afterwards, while the one passing actually pulls away (it advanced in the last
        # step at least what it can), until the space lets it advance without braking. With gradual dynamics it is
        # compared with its actual speed (it cannot accelerate at once to the desired one).
        mine = s.v_last[me] if p.any_dynamics and p.dyn_t[t] else s.vcap_last[me]
        pulling_away = s.v_last[ld] >= mine - EPS and gap < p.gap_run_t[t] + v_me
        yielding[k] = v_lead > v_me + EPS and (gap < 0 or pulling_away)
    # It does not overtake a vehicle that is overtaking: three would end up side by side.
    for k in range(m - 1):
        if passing[k + 1]:
            passing[k] = False
    return passing | yielding, passing


@jit
def lane_saturation(p, s, occ):
    """Saturation of each lane: length of its queue of stopped vehicles within the segment (length + stopped gap of each)
    divided by the segment length, capped at 1. A vehicle changing lanes counts in both its lanes; a side-by-side row
    counts once."""
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
    """Rear of the last vehicle of each lane, whether it is stopped and its index (-1 = empty lane)."""
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


# ------------------------------------------------------------------ additions and removals


@jit
def add_vehicle(p, s, vt, pax, lane, length, x, bottleneck_at, speed_kmh, v0):
    """Add a vehicle at the start of the segment; pax = 0 if it carries freight. Without `length` (NAN), it measures its
    type's mean length; without `speed_kmh`, its maximum speed is the type's mean. With `x` (front), it is a vehicle of
    the initial condition: it is already in the segment, does not count as an arrival and has no travel time.
    `bottleneck_at` is the position of its [bottleneck] halt (inf = it does not stop). `v0` is its speed on entering
    (m/step); without it, it enters at its maximum. There must be room in the arrays."""
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
    s.x[i] = x if initial else s.vlen[i]  # enters with its rear at x = 0
    s.vmax[i] = p.v_step[vt] if math.isnan(speed_kmh) else speed_kmh / 3.6 * DT
    s.stopped[i] = False
    s.crossed[i] = False
    s.entry_tick[i] = -1 if initial else s.ints[TICK]
    s.vid[i] = int(s.entered_veh.sum())
    if p.has_stop[vt]:
        s.stop_pos[i], s.stop_kind[i] = p.stop_x[vt], TYPE_STOP
    else:
        s.stop_pos[i], s.stop_kind[i] = bottleneck_at, BOTTLENECK_STOP
    ahead = s.x[i] < s.stop_pos[i] - EPS  # if it starts past its stop, it does not stop
    s.stop_state[i] = STOP_AHEAD if ahead else NO_STOP
    s.dwell[i] = 0
    s.vcap_last[i] = min(s.vmax[i], p.lane_vmax[lane])  # enters moving
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
    """Remove the vehicles marked in `gone`, keeping the order of the others."""
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


# ------------------------------------------------------------------ arrivals and entry


@jit
def entry_lane(p, s, rngs, vt):
    if p.n_lanes == 1:
        return 0
    count = p.entry_count[vt]  # without the exclusive lanes of other types
    lanes = p.entry_lanes[vt, :count]
    if count == 1:  # fixed lane: those that do not change lanes
        return lanes[0]
    # Those that change lanes: shortest entry queue; tie-break by more free space at the entry.
    shortest = s.q_count[lanes].min()
    best = lanes[s.q_count[lanes] == shortest]
    if best.size > 1:
        tail = lane_tails(p, s)[0][best]
        best = best[tail == tail.max()]
    if best.size == 1:
        return best[0]
    return best[rngs.entry[vt].integers(0, best.size)]  # the same as Generator.choice(best)


@jit
def arrivals(p, s, rngs, arr):
    """Arrivals of the current step (from chunk `arr`): they count as arrivals and wait in the entry queue."""
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
    """Speed (m/step) at which a vehicle with gradual acceleration or braking enters: 0 if it leaves the stopped entry
    queue and accelerates gradually; otherwise, at most the one that lets it brake behind the lane's last vehicle.
    NAN = at its maximum, as without dynamics."""
    if from_stop and math.isfinite(p.acc_t[vt]):
        return 0.0
    if tail < 0 or not math.isfinite(p.dec_t[vt]):
        return NAN
    v_lead = 0.0 if s.stopped[tail] else s.v_last[tail]
    safe = safe_speed_1(space, v_lead, p.dec_t[vt], p.dec_t[s.vtype[tail]])
    return min(speed_kmh / 3.6 * DT, safe)


@jit
def queue_ready(p, s, rngs, ln, vt, fits, tail_stopped):
    """The first vehicle of the stopped entry queue of lane `ln` has already reacted and can enter.

    It starts reacting when the one ahead moves (or there is already room for it) and enters when its reaction ends,
    as soon as it fits."""
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
        elif p.has_headway_t[vt]:  # enters at its desired speed: keeps the time_headway gap at it
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
                if math.isfinite(bump):  # enters at most at the speed at which it still reaches the bump
                    v0 = min(speed / 3.6 * DT if math.isnan(v0) else v0, bump)
            add_vehicle(p, s, vt, pax, ln, length, NAN, at, speed, v0)
            s.queue_react[ln] = -1
            # The next one was waiting behind a stopped vehicle: it also starts with a reaction.
            s.queue_stopped[ln] = s.queue_stopped[ln] and s.q_count[ln] > 0
        elif p.queue_reaction and tail_stopped[ln]:
            s.queue_stopped[ln] = True  # the segment's queue reaches the entry stopped


@jit
def drain_exit(p, s):
    """The vehicles each exit queue's capacity accepts in this step leave it. Without a queue, the accumulated
    fraction does not exceed one exit (exits are not saved for later)."""
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
            s.exit_q[ln] = 0.0  # no rounding leftovers
            s.exit_credit[ln] = min(s.exit_credit[ln], 1.0)


# ------------------------------------------------------------------ avance


@jit
def move(p, s, rngs, phase, tick):
    n = s.ints[N]
    dynamics = p.any_dynamics
    occ = occupancy(p, s)
    m = occ.veh.size

    # Space available to each entry with respect to its leader.
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
        # Overtaking within the lane: the leader being overtaken (or moving ahead) does not count; the next one rules.
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
            # With gradual dynamics, the gap behind a moving leader grows with the vehicle's own speed: gap_stop when almost
            # stopped, gap_run at its desired speed. That way, when the queue starts moving, gap_run is not suddenly required of
            # a vehicle that was at gap_stop.
            own = s.v_last[i] / min(s.vmax[i], p.lane_vmax[occ.lane[k]])
            gap_need = p.gap_stop_t[t] + (p.gap_run_t[t] - p.gap_stop_t[t]) * min(max(own, 0.0), 1.0)
        if p.any_headway and p.has_headway_t[t] and not ls:
            # With time_headway, behind a moving leader it keeps gap_stop plus what it covers in that time at its
            # speed: the actual one of the previous step with gradual dynamics; without it, the desired one (reached at once).
            v_own = s.v_last[i] if p.dyn_t[t] else min(s.vmax[i], p.lane_vmax[occ.lane[k]])
            gap_need = p.gap_stop_t[t] + p.headway_t[t] * v_own / DT
        sp = s.x[ld] - s.vlen[ld] - gap_need - occ.x[k]
        yielding = skip[k] and not passing[k]
        if yielding:
            # The vehicle letting another pass may end up beside the one passing it, but not overtake it again; if both
            # stop behind the same vehicle, it stays SIDE_EPS behind (as in a side-by-side row).
            front = s.x[occ.leader[k]] - SIDE_EPS - occ.x[k]
            if dynamics and p.dyn_t[t]:
                # With gradual dynamics it counts what the one passing advanced in the last step: if it barely
                # overtook it, it does not force it to brake dead.
                front = front + s.v_last[occ.leader[k]]
            sp = min(sp - SIDE_EPS, front)
        if passing[k]:
            # If the next vehicle brakes both, the overtaking one stays SIDE_EPS behind the overtaken one: they go side by
            # side instead of tied at the same position.
            sp = sp - SIDE_EPS
        if side[k]:
            # Free place beside the stopped leader: it stops with its front aligned with the leader's.
            sp = s.x[occ.leader[k]] - SIDE_EPS - occ.x[k]
        e = occ.order[k]
        stop_e[e] = sp if ls else INF
        move_e[e] = sp if has_lead[k] and not ls else INF
        if dynamics:
            # Safe speed with respect to the one ahead (the one that rules for each entry).
            v_lead = 0.0 if ls else s.v_last[ld]
            if yielding:
                v_lead = min(v_lead, s.v_last[occ.leader[k]])
            dec_e = p.dec_safe_t[t]
            vt_lead = s.vtype[ld]
            lead_term = v_lead * v_lead * dec_e / max(p.dec_t[vt_lead], dec_e) if p.dec_fin_t[vt_lead] else 0.0
            safe_e[e] = gipps(sp, lead_term, dec_e, p.dec_fin_t[t]) if has_lead[k] else INF

    # Per vehicle: minimum between its lane and, if it is changing, the target lane.
    stop_space = stop_e[:n].copy()
    move_space = move_e[:n].copy()
    vsafe = safe_e[:n].copy()
    for j in range(occ.changing.size):
        i = occ.changing[j]
        stop_space[i] = min(stop_space[i], stop_e[n + j])
        move_space[i] = min(move_space[i], move_e[n + j])
        if dynamics:
            vsafe[i] = min(vsafe[i], safe_e[n + j])

    # Line at the end of the segment. Not subject to the light: those that already crossed and the types with an
    # exclusive lane without a light.
    ruled_out = np.empty(n, np.bool_)
    safe_line = np.empty(n)
    for i in range(n):
        t = s.vtype[i]
        ruled_out[i] = s.crossed[i] or p.free_type[t]
        if dynamics:
            to_line = INF if s.crossed[i] else p.L - s.x[i]
            safe_line[i] = gipps(to_line, 0.0, p.dec_safe_t[t], p.dec_fin_t[t])
            if phase == YELLOW:
                # Yellow: a vehicle that can still stop before the line (braking at decel) stops.
                dec_v = p.dec_t[t]
                if math.isfinite(dec_v) and not ruled_out[i] and s.v_last[i] - dec_v <= safe_line[i] + EPS:
                    stop_space[i] = min(stop_space[i], to_line)
                    vsafe[i] = min(vsafe[i], safe_line[i])
            elif phase == RED and not ruled_out[i]:
                vsafe[i] = min(vsafe[i], safe_line[i])
        if phase == RED and not ruled_out[i]:
            stop_space[i] = min(stop_space[i], p.L - s.x[i])

    # Intermediate stop lines (traffic lights and pedestrian crossings): on red, a vehicle that has not passed them (its
    # front) stops at the line; on yellow, it stops if it still can (with gradual braking) or slows down near
    # it, as at the light at the end. A pedestrian crossing only applies in its bump's lanes.
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
        # Exit queue without room: a vehicle that does not fit in the free m of its lane's queue (or the target's, if it is
        # changing) does not cross the line. An empty queue accepts anyone, even if it does not fit whole.
        closed = np.zeros(n, np.bool_)
        front = np.full(p.n_lanes, -1, np.int64)  # the vehicle ahead that has not crossed, per lane
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
        # Line closed in the lane if the next one to cross does not fit.
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
            # [bottleneck] halts that do not apply now: outside their lanes or while changing lanes.
            bn_lane_ok = p.bn_lane[s.lane[i]] and s.lc_target[i] < 0
            bn_skip[i] = s.stop_kind[i] == BOTTLENECK_STOP and not bn_lane_ok
            # Stop: countdown of the vehicle at it; when it ends it starts (it was already ready).
            if s.stop_state[i] == AT_STOP:
                s.dwell[i] -= 1
                if s.dwell[i] <= 0:
                    s.stop_state[i] = NO_STOP
                    s.stopped[i] = False
                    s.react[i] = -1
            # While the stop is pending or in progress, it is a stop at its position.
            pending = s.stop_state[i] == AT_STOP or (s.stop_state[i] == STOP_AHEAD and not bn_skip[i])
            if pending:
                t = s.vtype[i]
                stop_space[i] = min(stop_space[i], s.stop_pos[i] - s.x[i])
                if dynamics:
                    vsafe[i] = min(vsafe[i], gipps(s.stop_pos[i] - s.x[i], 0.0, p.dec_safe_t[t], p.dec_fin_t[t]))

    # State machine: starting reaction of the stopped vehicles.
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
        for t in range(p.n_types):  # each type with its generator, in order of arrival
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

    # Advance: maximum speed within the lane's limit (the lower of its two lanes if it is changing),
    # reduced during the manoeuvre, at the bumps and on yellow near the stop line.
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
            # Bump: a vehicle on it, or that would reach it in this step, does not exceed its type's speed. With gradual
            # braking, it also brakes beforehand at decel: with the distance to the bump, the safe speed (Gipps, one reaction
            # step) only lets it reach it in one step at the bump's speed or less. The origin lane counts,
            # and the target lane if it is changing. Each bump separately; the lowest speed remains.
            vb = p.bump_v_t[t]
            for b in range(p.bump_x.size):
                bump_x = p.bump_x[b]
                in_lane = p.bump_lane[b, s.lane[i]] or (tgt >= 0 and p.bump_lane[b, max(tgt, 0)])
                before = in_lane and math.isfinite(vb) and s.x[i] - s.vlen[i] < bump_x  # not entirely past it yet
                if before and s.x[i] + cap > bump_x:
                    cap = min(cap, vb)
                if dynamics and before and s.x[i] < bump_x:
                    dec_vf = p.dec_safe_t[t]
                    lead_term = vb * vb * dec_vf / dec_vf  # the one ahead is the bump: it brakes as for it
                    vsafe[i] = min(vsafe[i], gipps(bump_x - s.x[i], lead_term, dec_vf, p.dec_fin_t[t]))
        # A vehicle approaching the stop line on yellow slows down; with gradual braking it decides: it stops
        # if it still can (above) or continues without braking.
        near = phase == YELLOW and not ruled_out[i] and p.L - s.x[i] <= p.yellow_approach
        if near and not (dynamics and math.isfinite(dec_v)):
            cap = cap * p.yellow_speed_factor
        for j in range(n_lines):
            if inner_near[j, i]:
                cap = cap * p.yellow_speed_factor
        vcap[i] = cap
        if dynamics:
            # Towards the desired speed by at most accel·DT per step (and down by at most decel·DT), without exceeding the
            # safe speed; the space to the one ahead is still a hard limit.
            v_prev = s.v_last[i]
            v_new = min(max(min(v_prev + p.acc_t[t], cap), v_prev - dec_v), vsafe[i])
        else:
            v_new = cap
        moving = not s.stopped[i]
        adv[i] = max(min(v_new, space[i]), 0.0) if moving else 0.0
        if moving and adv[i] < EPS and binding_stop[i]:
            s.stopped[i] = True
    if p.any_emissions:
        emit(p, s, adv)  # before updating v_last: the acceleration uses the previous step's

    # Distance of the step: passengers·m and m per type; distance and time per lane and per type of those that are
    # within the segment (including stopped ones, in their current lane). In the step in which it crosses the line only
    # the fraction of the step spent within the segment counts. Each sum per type or lane is accumulated separately and
    # then added, like a bincount.
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
        if passing[k]:  # the overtaking vehicle finished passing the one it was overtaking
            me, other = occ.veh[k], occ.leader[k]
            if s.x[me] > s.x[other] and s.x[me] - adv[me] <= s.x[other] - adv[other]:
                s.in_lane_passes[s.vtype[me]] += 1

    # Arrival at the stop: it stays stopped for the time passengers get off and on.
    if p.any_stop:
        for i in range(n):
            if not (s.stop_state[i] == STOP_AHEAD and s.x[i] >= s.stop_pos[i] - EPS):
                continue
            t = s.vtype[i]
            if s.stop_kind[i] == BOTTLENECK_STOP:
                if bn_skip[i]:  # it passed the point outside the halt's lanes: it does not stop
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

    # Lane change timers.
    for i in range(n):
        if s.cooldown[i] > 0:
            s.cooldown[i] -= 1
        if s.lc_target[i] >= 0:
            s.lc_timer[i] -= 1
            if s.lc_timer[i] <= 0:
                s.lane[i] = s.lc_target[i]
                s.lc_target[i] = -1
                s.cooldown[i] = p.cooldown_ticks_t[s.vtype[i]]

    # Crossing the stop line. Travel time only for vehicles that covered the whole segment (not those of the initial
    # condition).
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
        # It leaves the system when its rear passes the line.
        gone[i] = s.x[i] - s.vlen[i] > p.L
        any_gone = any_gone or gone[i]
    if any_gone:
        compact(s, gone)


@jit
def emit(p, s, adv):
    """Grams each vehicle that has not crossed yet and whose type emits emits in the step, with its speed (adv) and its
    acceleration with respect to the previous step, clamped to its type's [−decel, accel] (an emergency braking does not
    trigger the a² term). Also what it would emit covering the same distance at free flow (at constant speed: its
    maximum within the lane's limit) and where it emits it."""
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
        vf = min(s.vmax[i], p.lane_vmax[s.lane[i]])  # m/step
        v_free = vf / DT
        frac = step / vf
        ln = s.lane[i]
        b = min(max(int(s.x[i] / EMIS_BIN), 0), n_bins - 1)
        for k in range(n_pol):
            on = 1.0 if p.emis_on[t, k] else 0.0
            c = p.coef_sets[row, k]
            e = c[0] + c[1] * v + c[2] * v * v + c[3] * a + c[4] * a * a + c[5] * v * a
            g = max(e, 0.0) * DT * on
            cf = p.coef_sets[2 * t, k]  # at constant speed: the set for a ≥ −0.5, without the a terms
            f = max(cf[0] + cf[1] * v_free + cf[2] * v_free * v_free, 0.0) * DT * frac * on
            grams[t, k] += g
            free[t, k] += f
            s.emis_pos[k, ln, b] += g
    s.emis_g[:] += grams
    s.emis_interval[:] += grams
    s.emis_free_g[:] += free


# ------------------------------------------------------------------ lane change


@jit
def can_brake_behind(p, s, i, j, gap):
    """With gradual braking, `i` can be `gap` m behind `j` without braking harder than its `decel` (in a lane
    change, whether it is the one changing or the follower in the target lane). Without gradual braking, always."""
    dec = p.dec_t[s.vtype[i]]
    if not math.isfinite(dec):
        return True
    stopped = s.stopped[j]
    room = gap - (p.gap_stop_t[s.vtype[i]] if stopped else p.gap_run_t[s.vtype[i]])
    safe = safe_speed_1(room, 0.0 if stopped else s.v_last[j], dec, p.dec_t[s.vtype[j]])
    # Also the hard limit: in one step it advances no more than `room` (measured from where j is now).
    return s.v_last[i] - dec <= min(safe, room) + EPS


@jit
def neighbors(occ, bounds, lane, x):
    """(leader, follower) a vehicle with its front at x would have in `lane`; -1 if there is none."""
    lo, hi = bounds[lane], bounds[lane + 1]
    k = lo + np.searchsorted(occ.x[lo:hi], x)
    return (occ.veh[k] if k < hi else -1), (occ.veh[k - 1] if k > lo else -1)


@jit
def pass_lane(p, s, occ, bounds, used, k, gap, stuck_speed):
    """Adjacent lane to overtake the slow leader of entry k; -1 if none is worth it."""
    i, xi, li = occ.veh[k], occ.x[k], occ.lane[k]
    vi = s.vtype[i]
    over = p.overtake[vi]
    look = p.lookahead_t[vi]
    best, best_score = -1, min(gap, look) + p.min_adv_t[vi]
    for tl in (li + 1, li - 1):  # prefers to overtake on the left
        if tl < 0 or tl >= p.n_lanes or used[tl] or not p.allowed[vi, tl]:
            continue
        v_there = min(s.vmax[i], p.lane_vmax[tl])
        if v_there <= stuck_speed:  # that lane does not improve the current speed
            continue
        j, f = neighbors(occ, bounds, tl, xi)
        score = look
        if j >= 0:  # leader in the target lane
            front = s.x[j] - s.vlen[j] - xi
            if front < p.gap_stop_t[vi] or not can_brake_behind(p, s, i, j, front):
                continue
            v_j = s.v_last[j] + EPS if over else min(s.vmax[j], p.lane_vmax[tl])
            if v_j < v_there or s.stopped[j]:
                score = min(front, look)
        if f >= 0:  # follower in the target lane: it compresses, but without overlapping
            back = xi - s.vlen[i] - s.x[f]
            if back < p.gap_stop_t[s.vtype[f]] or not can_brake_behind(p, s, f, i, back):
                continue
        if score >= best_score:
            best, best_score = tl, score
    return best


@jit
def rise_lane(p, s, occ, bounds, used, k):
    """Adjacent lane with a higher limit entry k can move up to; -1 if there is no room."""
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
            # A close leader that is stopped or not faster than it takes away the advantage.
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
    """Lane change decisions of the moving vehicles (not stopped, without a manoeuvre or cooldown), in random
    order and with at most one entry into each lane per round.

    Condition for any change: a leader within its `lookahead` whose actual speed in the last step is at
    least `leader_slowdown` (e.g. 25 %) below the lane's limit (in a lane without a limit, below the vehicle's own
    maximum speed). Without a leader, it does not change.

    * Overtaking: with a slow leader within its `lookahead`, it looks for an adjacent lane where it advances faster and has
      `min_advantage` m more free space. Types with `overtake` judge slowness by the actual speed of the
      last step (leader in a manoeuvre, braked by its queue) and accept a lane with a lower limit if they advance more there
      than behind the leader.
    * Moving up: if it cannot overtake, it moves to an adjacent lane with a higher limit if there is room without
      compressing anyone (`gap_run` ahead and behind) and no leader within its `lookahead` slows it down there.
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
        # Effective speeds in each entry's lane (the vehicle's maximum, without exceeding the limit).
        lane_v = p.lane_vmax[occ.lane[k]]
        v_self = s.vmax[i]
        eff_self = min(v_self, lane_v)
        eff_lead = min(s.vmax[ld], lane_v)
        lead_stopped = s.stopped[ld]
        slow = occ.has_leader[k] and (eff_lead < eff_self or lead_stopped)
        stuck_speed[k] = 0.0 if lead_stopped else eff_lead  # the one its leader forces on it
        if p.overtake[t]:
            # Aggressive overtaking: the leader is slow if in the last step it advanced less than this vehicle could.
            slow = slow or (occ.has_leader[k] and s.v_last[ld] < s.vcap_last[i] - EPS)
            stuck_speed[k] = s.v_last[ld]
        # Common condition: a close leader going at least `leader_slowdown` below the lane's limit.
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
            # Candidate to move up: an allowed adjacent lane where its maximum speed (bounded by the limit) is
            # higher than in its own.
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


# ------------------------------------------------------------------ sampling


@jit
def sample(p, s):
    """A sample of the Recorder series; resets the interval's accumulators."""
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
            # A stopped vehicle takes its length + compressed gap; a moving one, at most length + moving gap. Space of those
            # carrying passengers: freight vehicles do not count in pax/m.
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


# ------------------------------------------------------------------ loop


@jit
def has_room(p, s, arr):
    """There is room for the current step: one more vehicle per lane in the segment, the step's arrivals in any entry
    queue and all those in the segment in any exit queue."""
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


# ------------------------------------------------------------------ from Python


@api
def advance(p, s, rngs, arr, until):
    """Run steps up to step `until`. Returns earlier (NEED_ARRIVALS or NEED_ROOM) if it needs the next arrival chunk
    or more room in the arrays; DONE when it gets there."""
    while s.ints[TICK] < until:
        tick = s.ints[TICK]
        if tick - tick % ARRIVAL_CHUNK != arr.start:
            return NEED_ARRIVALS
        if not has_room(p, s, arr):
            return NEED_ROOM
        # One step.
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
