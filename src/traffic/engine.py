"""Microsimulation engine: a segment of `lanes` lanes with a traffic light at x = length.

State in a compact structure of arrays (SoA): the active vehicles occupy
positions [0, n) of each array; those that leave are removed by compacting.
The initial capacity is the physical bound on the vehicles that fit in the segment, so
memory does not grow with the length of the run.

Model per step of DT seconds (parallel update with the previous positions):
  * Each vehicle advances at its maximum speed (drawn on arrival if the type's
    varies) without exceeding the lane's limit; the advance is limited by the
    space to the lane's leader and, on red, by the stop line.
  * Traffic light: cycle green → yellow → red. On yellow the stop line stays
    open, but a vehicle less than `yellow_approach` m from it advances at
    `yellow_speed_factor` of its speed; on red, it stops. Types with an
    exclusive lane in one of `free_lanes` obey neither the red nor the
    yellow (they do not change lanes, so they always use it). Besides the
    light at the end of the segment there can be others at `position` m (`Light`),
    each with its phases, its initial phase and its `free_lanes`: vehicles stop at
    their line as at the end one, but flow and queues are measured at the
    end of the segment.
  * Pedestrians (optional, `pedestrian`): a pedestrian light is green for vehicles until pedestrians are
    waiting; a bump can be a pedestrian crossing. Its phase schedule (green/yellow/red) is computed when the
    replica is created (pedestrians.py) and vehicles obey it like an intermediate light, in the bump's lanes.
  * Initial condition (optional): the lanes start with moving vehicles
    occupying a fraction of the segment, so as not to wait for it to fill up.
  * Compression: behind a stopped leader `gap_stop` (< `gap_run`) is enough.
  * Stretching: a stopped vehicle that becomes free (the light turned green or its
    leader started) waits a reaction time U[1,5] s before advancing, and also
    needs to reopen `gap_run` behind the moving leader.
  * Gradual acceleration and braking (types with `accel`/`decel`): the speed
    rises by at most `accel`·DT per step towards the desired one and falls by at most `decel`·DT
    if the desired one falls; the gap behind a moving leader goes from `gap_stop`
    (almost stopped) to `gap_run` (at its desired speed). It also does not exceed the speed at which it can
    still stop, braking at `decel` (with one reaction step), behind the one
    ahead (counting what that one needs to stop), before the line on
    red or before a stop; on yellow it stops if it still can and, if
    not, crosses without slowing down (`yellow_speed_factor` does not apply). The
    space to the one ahead is still a hard limit (emergency braking).
    When starting after the reaction it starts from 0.
  * Exit queue (optional, [exit]): a vehicle crossing the line enters the exit
    queue of its lane, which accepts at most `capacity` veh/min and measures
    `storage` m; each vehicle takes its length + its stopped gap in it. If the
    next one does not fit, that lane's line closes for it as on red
    (even if it is green): it stops at it and starts with its reaction when
    there is room; if more arrive than it accepts, the lane saturates. An empty
    queue accepts anyone.
  * Bumps (optional, one [[speed_bump]] section each): in its lanes `lanes`, at `position` m from the
    start, a vehicle on it (from the front to the rear) does not exceed its
    type's `speed_bump_kmh`; without `decel` it slows down at once, with `decel` it brakes
    beforehand to reach that speed. Types without `speed_bump_kmh` do not brake.
  * Entry queue: vehicles that do not fit wait outside the segment and enter
    moving as soon as there is room. With `queue_reaction`, if the segment's queue
    reaches the entry stopped, the entry queue is stopped too:
    each vehicle starts with its reaction time when the one ahead moves,
    as in the rest of the queue.
  * Lane change (only types that can): the manoeuvre lasts U[1,4] s, the
    vehicle occupies both lanes at reduced speed and the followers in the
    two lanes compress behind it. Vehicles change to overtake a slow leader
    or to move up to an adjacent lane with a higher limit when there is room.
    Types with `overtake` (e.g. motorcycles) judge the leader by its actual
    speed and move even to a lane with a lower limit if they advance more there; each
    type can set its thresholds (`lookahead`, `min_advantage`, cooldown).
  * Side by side (types with `abreast` > 1, e.g. bikes): a vehicle arriving behind
    a stopped vehicle of the same type stops beside it, with the fronts
    aligned, while the row has room; the next one queues behind. When
    starting they leave one after another (usual reaction and moving gap).
  * Overtaking within the lane (types with `pass_in_lane`, e.g. bikes): while moving,
    a vehicle with a higher maximum speed (bounded by the lane's limit) than
    the one ahead of the same type passes it on one side without changing lanes;
    it only respects the next vehicle. The overtaken one does not brake for the one passing it
    until it pulls `gap_run` ahead. Never more than two side by side.
  * Stop (types with `stop_position`, e.g. the bus): the vehicle stops
    with its front at that position for a normal(μ, σ) time of passengers getting
    off and on, and then starts without additional reaction time. Those
    behind it in its lane wait and compress.

The computation of each step is in kernel.py, compiled with numba. This module builds the tables and the state of
the replica, the initial condition, the pedestrian schedules, the arrivals in chunks and the summary.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from traffic import kernel
from traffic.config import DT, MAX_TYPES, PEDESTRIAN_YIELD, RED, SimConfig
from traffic.distributions import sample_lengths, sample_passengers, sample_rates, sample_speeds
from traffic.emissions import EMIS_BIN, coefficient_table
from traffic.kernel import ARRIVAL_CHUNK, EPS, INF, NAN, Occupancy
from traffic.metrics import Recorder
from traffic.pedestrians import Schedule, arrivals, bump_schedule, fixed_phases, light_schedule

PER_TYPE_STREAMS = 5  # random generators per vehicle type
MAX_PAX = 255  # bound on passengers per vehicle (pax is uint8)
INITIAL_MISSES = 30  # consecutive draws that do not fit before a lane counts as full in the initial condition
# Recorder series written by the step (State.r_<name>).
RECORDER_FIELDS = ("cum_pax", "pax_m", "pax_on", "footprint", "lane_sat", "lane_dist", "lane_time", "lane_cross",
                   "exit_q", "entry_q", "emis")  # fmt: skip


def compile_kernel(cfg: SimConfig) -> None:
    """Compile the kernel in this process, or load it from numba's cache, with a short replica of `cfg`. It has an
    initial condition so that what builds it is compiled too."""
    Simulation(replace(cfg, run=1.0, time_scale=1.0, initial_occupancy=0.5), np.random.default_rng(0)).advance(1)


class Simulation:
    _FIELDS = (
        "vtype", "pax", "lane", "lc_target", "lc_timer", "cooldown",
        "react", "x", "stopped", "crossed", "entry_tick", "vid", "stop_state", "dwell", "v_last", "vcap_last", "vlen",
        "stop_pos", "stop_kind", "vmax",
    )  # fmt: skip
    _QUEUE_FIELDS = ("q_vt", "q_pax", "q_len", "q_at", "q_speed", "q_arrived")
    NO_STOP, STOP_AHEAD, AT_STOP = kernel.NO_STOP, kernel.STOP_AHEAD, kernel.AT_STOP
    TYPE_STOP, BOTTLENECK_STOP = kernel.TYPE_STOP, kernel.BOTTLENECK_STOP

    def __init__(self, cfg: SimConfig, rng: np.random.Generator):
        self.cfg = cfg
        # One generator per random process and per type, children of the replica's seed (rng.spawn
        # consumes no draws). That way, changing a process —the stop, the rate or the passengers of a
        # type— does not shift the draws of the others: two scenarios with the same seed share
        # the random numbers where the change is not involved. Those of each type come after the
        # common ones and in the order of the types, so adding a new type does not alter the earlier ones.
        streams = rng.spawn(1 + PER_TYPE_STREAMS * cfg.n_types)
        self.rng_lane_change = streams[0]  # decision order and duration of lane changes
        per_type = streams[1:]
        self.rng_arrivals = per_type[0::PER_TYPE_STREAMS]  # Poisson arrivals
        self.rng_pax = per_type[1::PER_TYPE_STREAMS]  # passengers per vehicle
        self.rng_react = per_type[2::PER_TYPE_STREAMS]  # reaction times
        self.rng_entry = per_type[3::PER_TYPE_STREAMS]  # entry lane (tie-breaks)
        self.rng_stop = per_type[4::PER_TYPE_STREAMS]  # duration of the stop
        # Children of the passenger generator (spawn consumes none of its draws): whether it carries freight and its
        # length. Adding them does not alter the earlier draws and they do not depend on the number of types.
        self.rng_cargo, self.rng_length = zip(*(g.spawn(2) for g in self.rng_pax))
        # Bottlenecks: whether it stops and where (on arrival) and for how long (when stopping).
        self.rng_bottleneck, self.rng_bottleneck_time = zip(*(g.spawn(2) for g in self.rng_pax))
        # Child of the arrival generator: the rate of each interval for the types with a variable rate.
        self.rng_rate = [g.spawn(1)[0] for g in self.rng_arrivals]
        # Child of the passenger generator: the maximum speed of each vehicle (if it varies).
        self.rng_speed = [g.spawn(1)[0] for g in self.rng_pax]
        # Child of the reaction generator: the reaction of a vehicle starting in the stopped entry queue.
        self.rng_queue_react = [g.spawn(1)[0] for g in self.rng_react]
        specs = cfg.specs
        b = cfg.behavior
        self.L = float(cfg.length)
        self.n_lanes = cfg.lanes
        self.n_types = cfg.n_types

        # Tables per vehicle type (indexed by vtype).
        self.v_step = np.array([s.speed * DT for s in specs])  # mean speed; each vehicle's is in vmax
        self.length_t = np.array([s.length for s in specs])  # mean length; each vehicle's is in vlen
        self.gap_run_t = np.array([s.gap_run for s in specs])
        self.gap_stop_t = np.array([s.gap_stop for s in specs])
        # Car-following headway (s) of the types with time_headway: when moving, gap_stop + time_headway · v.
        self.has_headway_t = np.array([s.time_headway is not None for s in specs])
        self.headway_t = np.array([s.time_headway or 0.0 for s in specs])
        self.any_headway = bool(self.has_headway_t.any())
        self.can_change = np.array([s.can_change_lane for s in specs])
        self.stop_x = np.array([np.inf if s.stop_position is None else s.stop_position for s in specs])
        self.has_stop = np.isfinite(self.stop_x)
        # Probability and duration (mean, σ) of each type's halt: its own or that of [bottleneck].
        self.bn_prob = np.array([cfg.bottleneck_prob(k) for k in range(self.n_types)])
        self.bn_time = [cfg.bottleneck_time(k) for k in range(self.n_types)]
        self.bn_type = self.bn_prob > 0  # types that can stop at a bottleneck
        self.free_type = np.array([cfg.ignores_light(k) for k in range(self.n_types)], np.bool_)  # no traffic light
        # Intermediate stop lines (intermediate traffic lights and the bumps' pedestrian crossings): position, types that do
        # not obey them (exclusive lane in the light's free_lanes) and lanes where they apply (all if unspecified).
        self.lines = cfg.stop_lines
        self.inner_x = np.array([s.position for s in self.lines], np.float64)
        self.inner_free = np.zeros((len(self.lines), self.n_types), np.bool_)
        self.inner_lane = np.ones((len(self.lines), cfg.lanes), np.bool_)
        for j, s in enumerate(self.lines):
            self.inner_free[j] = [s.light is not None and cfg.ignores_light(k, s.light) for k in range(self.n_types)]
            if s.lanes is not None:
                self.inner_lane[j] = np.isin(np.arange(cfg.lanes), s.lanes)
        self.bn_lane = np.zeros(cfg.lanes, np.bool_)
        self.bn_lane[list(cfg.bottleneck_lanes())] = True
        self.bn_zone = cfg.bottleneck_zone()
        self.any_stop = bool(self.has_stop.any() or self.bn_type.any())
        self.abreast = np.array([s.abreast for s in specs])
        self.any_abreast = bool((self.abreast > 1).any())
        self.pass_in_lane = np.array([s.pass_in_lane for s in specs])  # overtake within the lane
        self.any_pass_in_lane = bool(self.pass_in_lane.any())
        self.allowed = np.zeros((self.n_types, cfg.lanes), np.bool_)  # lanes each type can use
        for k in range(self.n_types):
            self.allowed[k, list(cfg.allowed_lanes(k))] = True
        # Entry lanes of each type (the first entry_count of its row).
        self.entry_count = np.array([len(cfg.entry_lanes(k)) for k in range(self.n_types)], np.int64)
        self.entry_lanes = np.zeros((self.n_types, cfg.lanes), np.int64)
        for k in range(self.n_types):
            self.entry_lanes[k, : self.entry_count[k]] = cfg.entry_lanes(k)
        self.rate_tick = np.array(cfg.rates, dtype=np.float64) / 60.0 * DT
        # Variable rates: arrivals per step in each rate_interval interval, drawn at the start.
        n_intervals = -(-cfg.n_ticks // cfg.rate_ticks)
        self.rate_by_interval = {
            k: sample_rates(self.rng_rate[k], cfg.rate(k), n_intervals) / 60.0 * DT
            for k in range(self.n_types) if cfg.rate(k).variable
        }  # fmt: skip

        self.lane_vmax = np.array(cfg.lane_max_kmh) / 3.6 * DT  # limit of each lane, m/step
        self.lc_every = max(1, round(b.lane_change_interval / DT))
        # Lane change thresholds per type: the type's own or, if it does not set them, those of [behavior].
        def own(field: str) -> np.ndarray:
            return np.array([getattr(b, field) if getattr(s, field) is None else getattr(s, field) for s in specs])

        self.lookahead_t = own("lookahead")
        self.min_adv_t = own("min_advantage")
        self.cooldown_ticks_t = np.round(own("lane_change_cooldown") / DT).astype(np.int16)
        self.overtake = np.array([s.overtake for s in specs])
        # Acceleration and braking per type in m/step² (inf = instantaneous, no anticipation).
        self.acc_t = np.array([INF if s.accel is None else s.accel * DT * DT for s in specs])
        self.dec_t = np.array([INF if s.decel is None else s.decel * DT * DT for s in specs])
        self.dyn_t = np.isfinite(self.acc_t) | np.isfinite(self.dec_t)  # types with gradual dynamics
        self.dec_fin_t = np.isfinite(self.dec_t)
        self.dec_safe_t = np.where(self.dec_fin_t, self.dec_t, 1.0)  # without infinities, for kernel.gipps
        self.any_dynamics = bool(self.dyn_t.any())
        # Bump ([speed_bump]): position, lanes where it is and maximum speed of each type when on it (m/step).
        self.bump_x = np.array([b.position for b in cfg.bumps], dtype=np.float64)  # one or several bumps
        self.bump_lane = np.zeros((len(cfg.bumps), cfg.lanes), np.bool_)  # lanes of each bump
        for i, b in enumerate(cfg.bumps):
            self.bump_lane[i, list(cfg.bump_lanes(b))] = True
        self.bump_v_t = np.array([INF if s.speed_bump_kmh is None else s.speed_bump_kmh / 3.6 * DT for s in specs])
        self.any_bump = bool(self.bump_x.size and self.bump_lane.any() and np.isfinite(self.bump_v_t).any())
        # Emissions (Int Panis et al., 2006): coefficients per type and pollutant, bounds of the acceleration that
        # enters the model (m/s²) and accumulated grams: over the run, at free flow (the same trip at constant
        # speed), in the sampling interval and by position (pollutant × lane × EMIS_BIN m bin).
        self.emis_coef, self.emis_on = coefficient_table(specs)
        self.any_emissions = bool(self.emis_on.any())
        self.emits_t = self.emis_on.any(axis=1)  # the type emits some pollutant
        # Coefficients per (type, set): row 2·type is the one for a ≥ −0.5 and 2·type + 1 the one for a < −0.5.
        self._coef_sets = np.ascontiguousarray(self.emis_coef.transpose(0, 2, 1, 3)).reshape(-1, *self.emis_coef.shape[1::2])
        self.emis_amax = np.array([s.accel or 0.0 for s in specs])
        self.emis_dmax = np.array([s.decel or 0.0 for s in specs])
        n_pol = self.emis_on.shape[1]
        self.emis_g = np.zeros((self.n_types, n_pol))
        self.emis_free_g = np.zeros((self.n_types, n_pol))
        self.emis_interval = np.zeros((self.n_types, n_pol))
        self.emis_pos = np.zeros((n_pol, cfg.lanes, max(1, math.ceil(cfg.length / EMIS_BIN))))
        self.veh_m = np.zeros(self.n_types)  # m travelled within the segment, per type
        # Without different limits between lanes there is nowhere to move up to.
        self.any_rise = len(set(cfg.lane_max_kmh)) > 1
        self._key_stride = self.L + 1e4  # separates the lanes in the sort key

        self._ints = np.zeros(5, np.int64)  # n, step, lane changes, samples, position in the arrival chunk
        min_slot = min(s.shortest + s.gap_stop for s in specs)
        per_lane = math.ceil((self.L + max(s.longest for s in specs)) / min_slot) + 2
        self._alloc(cfg.lanes * per_lane)
        # Entry queue of each lane (ring): (type, pax, length, position of its [bottleneck] halt,
        # maximum speed km/h, step of arrival at the queue).
        queue_cap = 64
        self.q_vt = np.zeros((cfg.lanes, queue_cap), np.int64)
        self.q_pax = np.zeros((cfg.lanes, queue_cap), np.int64)
        self.q_len = np.zeros((cfg.lanes, queue_cap))
        self.q_at = np.zeros((cfg.lanes, queue_cap))
        self.q_speed = np.zeros((cfg.lanes, queue_cap))
        self.q_arrived = np.zeros((cfg.lanes, queue_cap), np.int64)
        self.q_head = np.zeros(cfg.lanes, np.int64)
        self.q_count = np.zeros(cfg.lanes, np.int64)
        # Stopped entry queue per lane and reaction steps its first vehicle still needs (-1 = not started).
        self.queue_stopped = np.zeros(cfg.lanes, np.bool_)
        self.queue_react = np.full(cfg.lanes, -1, np.int32)
        # Current arrival chunk (none: it is drawn at the start).
        self._arr = kernel.Arrivals(-1, *(np.zeros(0, d) for d in (np.int64,) * 3 + (np.float64,) * 3))

        self.arrived_veh = np.zeros(self.n_types)
        self.arrived_pax = np.zeros(self.n_types)
        self.arrived_cargo = np.zeros(self.n_types)  # arrivals with freight (pax = 0)
        self.entered_veh = np.zeros(self.n_types)
        self.cum_veh = np.zeros(self.n_types)
        self.cum_pax = np.zeros(self.n_types)
        self.travel_ticks = np.zeros(self.n_types)
        self.pax_m = np.zeros(self.n_types)  # passengers·m within the segment in the sampling interval
        self.lane_dist = np.zeros(cfg.lanes)  # m travelled in each lane in the sampling interval
        self.lane_time = np.zeros(cfg.lanes)  # vehicle·s present in each lane in the interval
        self.type_dist = np.zeros(self.n_types)  # m travelled within the segment per type, over the whole run
        self.type_time = np.zeros(self.n_types)  # vehicle·s within the segment per type (including stopped ones)
        self.lane_changes_t = np.zeros(self.n_types)  # lane changes per type
        self.in_lane_passes = np.zeros(self.n_types)  # overtakes within the lane, per type of the overtaking vehicle
        self.stops = np.zeros(self.n_types)  # completed stops per type
        self.stop_time_ticks = np.zeros(self.n_types)  # steps stopped at the stop, per type
        self.queue_wait_ticks = np.zeros(self.n_types)  # steps in the entry queue of those that already entered
        self.bn_stops = np.zeros(self.n_types)  # [bottleneck] halts per type
        self.bn_ticks = np.zeros(self.n_types)  # steps stopped in them, per type
        self.pax_hist = np.zeros((self.n_types, MAX_PAX + 1), np.int64)  # vehicles arrived per number of passengers
        # Exit queue per lane: vehicles it accepts per step (inf = no exit queue).
        exit_cap = np.array(cfg.lane_exit_capacity, dtype=np.float64)
        self.exit_rate = np.where(exit_cap > 0, exit_cap / 60.0 * DT, INF)
        self._exit_limited = np.isfinite(self.exit_rate)  # lanes with an exit queue
        self._exit_add = np.where(self._exit_limited, self.exit_rate, 0.0)  # exits it accepts per step
        # m of each exit queue; without an exit queue, no limit.
        self.exit_storage = np.where(exit_cap > 0, np.array(cfg.lane_exit_storage, dtype=np.float64), INF)
        self.any_exit = bool((exit_cap > 0).any())
        self.exit_q = np.zeros(cfg.lanes)  # m occupied in the exit queue (length + stopped gap of each vehicle)
        # What each vehicle takes in the exit queue, in order of arrival (ring): the first in is the first out.
        self.exit_ring = np.zeros((cfg.lanes, self.x.size))
        self.exit_head = np.zeros(cfg.lanes, np.int64)
        self.exit_count = np.zeros(cfg.lanes, np.int64)
        self.exit_credit = np.zeros(cfg.lanes)  # accumulated fraction of the next exit
        self.exit_closed = np.zeros(cfg.lanes, np.bool_)  # the lane's next vehicle does not fit in the exit
        self.exit_blocked_ticks = np.zeros(cfg.lanes)  # steps with the line closed by the exit queue
        self.lane_cross = np.zeros(cfg.lanes)  # line crossings per lane in the sampling interval
        self.lane_crossed = np.zeros(cfg.lanes)  # line crossings per lane over the whole run
        self.recorder = Recorder(cfg.n_samples, cfg.n_types, cfg.lanes)
        self.initial_veh = np.zeros(self.n_types)  # vehicles of the initial condition
        self.initial_pax = np.zeros(self.n_types)
        self.timed_veh = np.zeros(self.n_types)  # crossings with a travel time (without the initial ones)
        # Own generators (new children: they do not alter the earlier ones): the initial filling and the pedestrians. They
        # are always created, so that the index of each one does not depend on the configuration.
        fill_rng, ped_rng = rng.spawn(1)[0], rng.spawn(1)[0]
        self._init_pedestrians(ped_rng)
        self._p = kernel.as_ref(self._params())
        self._st = kernel.as_ref(self._state())
        self._rngs = kernel.as_ref(self._kernel_rngs())
        if any(v > 0 for v in cfg.lane_initial_occupancy):
            self._initial_fill(fill_rng)

    # ------------------------------------------------------------------ estado

    @property
    def n(self) -> int:
        """Vehicles in the segment: they occupy positions [0, n) of each array."""
        return int(self._ints[kernel.N])

    @property
    def tick(self) -> int:
        return int(self._ints[kernel.TICK])

    @property
    def lane_changes(self) -> int:
        return int(self._ints[kernel.LANE_CHANGES])

    @property
    def queues(self) -> list[list[tuple[int, int, float, float, float, int]]]:
        """Entry queue of each lane, from the first to the last: (type, pax, length, position of its
        [bottleneck] halt, maximum speed km/h, arrival step). It is a copy."""
        cap = self.q_vt.shape[1]
        out = []
        for ln in range(self.n_lanes):
            idx = (self.q_head[ln] + np.arange(self.q_count[ln])) % cap
            out.append([(int(self.q_vt[ln, j]), int(self.q_pax[ln, j]), float(self.q_len[ln, j]),
                         float(self.q_at[ln, j]), float(self.q_speed[ln, j]), int(self.q_arrived[ln, j]))
                        for j in idx])  # fmt: skip
        return out

    @property
    def queue_len(self) -> np.ndarray:
        """Vehicles in the entry queue of each lane."""
        return self.q_count.copy()

    def queued_types(self) -> np.ndarray:
        """Vehicles of each type in the entry queues."""
        cap = self.q_vt.shape[1]
        types = [self.q_vt[ln, (self.q_head[ln] + np.arange(self.q_count[ln])) % cap] for ln in range(self.n_lanes)]
        return np.bincount(np.concatenate(types), minlength=self.n_types).astype(np.float64)

    @property
    def exit_items(self) -> list[list[float]]:
        """What each vehicle takes in the exit queue of each lane, from the first to the last. It is a copy."""
        cap = self.exit_ring.shape[1]
        return [self.exit_ring[ln, (self.exit_head[ln] + np.arange(self.exit_count[ln])) % cap].tolist()
                for ln in range(self.n_lanes)]  # fmt: skip

    def _queue_exit(self, ln: int, size: float) -> None:
        """Put a vehicle that takes `size` m into the exit queue of lane `ln`."""
        cap = self.exit_ring.shape[1]
        self.exit_ring[ln, (self.exit_head[ln] + self.exit_count[ln]) % cap] = size
        self.exit_count[ln] += 1
        self.exit_q[ln] += size

    def _alloc(self, cap: int) -> None:
        self.vtype = np.zeros(cap, np.uint8)
        self.pax = np.zeros(cap, np.uint8)
        self.lane = np.zeros(cap, np.int8)
        self.lc_target = np.full(cap, -1, np.int8)
        self.lc_timer = np.zeros(cap, np.int16)
        self.cooldown = np.zeros(cap, np.int16)
        self.react = np.full(cap, -1, np.int16)
        self.x = np.zeros(cap, np.float64)
        self.stopped = np.zeros(cap, np.bool_)
        self.crossed = np.zeros(cap, np.bool_)
        self.entry_tick = np.zeros(cap, np.int32)
        self.vid = np.zeros(cap, np.int32)  # vehicle identifier (order of entry into the segment)
        self.stop_state = np.zeros(cap, np.int8)  # NO_STOP | STOP_AHEAD | AT_STOP
        self.dwell = np.zeros(cap, np.int32)  # steps it has left at the stop
        self.v_last = np.zeros(cap, np.float64)  # m advanced in the last step (actual speed)
        self.vcap_last = np.zeros(cap, np.float64)  # m it could advance in it without a leader or a stop
        self.vlen = np.zeros(cap, np.float64)  # length of the vehicle (m)
        self.vmax = np.zeros(cap, np.float64)  # maximum speed of the vehicle (m/step)
        self.stop_pos = np.full(cap, np.inf)  # position of its stop or halt (front); inf = none
        self.stop_kind = np.zeros(cap, np.int8)  # TYPE_STOP | BOTTLENECK_STOP

    def _make_room(self) -> None:
        """Double the arrays that are not big enough for the current step (see kernel.has_room)."""
        n = self.n
        if n + self.n_lanes > self.x.size:
            old = {f: getattr(self, f) for f in self._FIELDS}
            self._alloc(2 * old["x"].size)
            for f, arr in old.items():
                getattr(self, f)[: arr.size] = arr
        k = self._ints[kernel.ARRIVAL_POS]
        arriving = int(np.count_nonzero(self._arr.tick[k:] == self.tick))
        while (self.q_count + arriving > self.q_vt.shape[1]).any():
            self._grow_ring(self._QUEUE_FIELDS, self.q_head, self.q_count)
        while self.any_exit and (self.exit_count + n > self.exit_ring.shape[1]).any():
            self._grow_ring(("exit_ring",), self.exit_head, self.exit_count)
        self._st = kernel.as_ref(self._state())

    def _grow_ring(self, fields: tuple[str, ...], head: np.ndarray, count: np.ndarray) -> None:
        """Double the rings `fields` (lanes × capacity); each lane is left in order from position 0."""
        cap = getattr(self, fields[0]).shape[1]
        for f in fields:
            old = getattr(self, f)
            new = np.zeros((old.shape[0], 2 * cap), old.dtype)
            for ln in range(old.shape[0]):
                new[ln, : count[ln]] = old[ln, (head[ln] + np.arange(count[ln])) % cap]
            setattr(self, f, new)
        head[:] = 0

    def _params(self) -> kernel.Params:
        cfg, b = self.cfg, self.cfg.behavior
        specs = cfg.specs
        if b.reaction_std is None:
            react_mode, react_mean = kernel.REACT_UNIFORM, 0.0
        else:
            react_mean = b.reaction_mean if b.reaction_mean is not None else (b.reaction_min + b.reaction_max) / 2
            react_mode = kernel.REACT_FIXED if b.reaction_std <= 0 else kernel.REACT_NORMAL
        # Phases of the stop lines: the one at the end of the segment and the intermediate ones.
        n_lines = 1 + len(self.lines)
        kind = np.zeros(n_lines, np.int64)
        red, green, yellow, offset = (np.zeros(n_lines, np.int64) for _ in range(4))
        start_red = np.zeros(n_lines, np.bool_)
        sched = np.zeros((n_lines, cfg.n_ticks + 1), np.int8)
        lights = [cfg.exit_light] + [s.light for s in self.lines]
        for j, (light, schedule) in enumerate(zip(lights, [self._exit_sched] + self._line_sched)):
            if schedule is not None:
                kind[j] = kernel.LINE_SCHEDULE
                sched[j] = schedule.phases
            elif light is not None and light.active and not light.pedestrian:
                kind[j] = kernel.LINE_CYCLE
                red[j], green[j], yellow[j] = round(light.red / DT), round(light.green / DT), round(light.yellow / DT)
                offset[j] = round(light.offset / DT)
                start_red[j] = light.start_phase == "red"
        stop_mean = np.array([s.stop_time_mean for s in specs], np.float64)
        stop_std = np.array([s.stop_time_std for s in specs], np.float64)
        bn_mean = np.array([m for m, _ in self.bn_time], np.float64)
        bn_std = np.array([sd for _, sd in self.bn_time], np.float64)
        return kernel.Params(
            L=self.L, n_lanes=self.n_lanes, n_types=self.n_types, n_ticks=cfg.n_ticks, lc_every=self.lc_every,
            sample_ticks=cfg.sample_ticks, key_stride=self._key_stride,
            any_headway=self.any_headway, any_stop=self.any_stop, any_abreast=self.any_abreast,
            any_pass_in_lane=self.any_pass_in_lane, any_dynamics=self.any_dynamics, any_bump=self.any_bump,
            any_emissions=self.any_emissions, any_exit=self.any_exit, any_rise=self.any_rise,
            yellow_approach=float(b.yellow_approach), yellow_speed_factor=float(b.yellow_speed_factor),
            release_space=float(b.release_space), lane_change_speed_factor=float(b.lane_change_speed_factor),
            leader_slowdown=float(b.leader_slowdown), queue_reaction=bool(b.queue_reaction),
            react_mode=react_mode, react_lo=round(b.reaction_min / DT), react_hi=round(b.reaction_max / DT),
            react_mean=float(react_mean), react_std=float(b.reaction_std or 0.0),
            react_min=float(b.reaction_min), react_max=float(b.reaction_max),
            lc_min=round(b.lane_change_min / DT), lc_max=round(b.lane_change_max / DT),
            v_step=self.v_step, length_t=self.length_t, gap_run_t=self.gap_run_t, gap_stop_t=self.gap_stop_t,
            has_headway_t=self.has_headway_t, headway_t=self.headway_t, can_change=self.can_change,
            stop_x=self.stop_x, has_stop=self.has_stop, stop_mean=stop_mean, stop_std=stop_std,
            stop_fixed=np.array([round(max(m, 0.0) / DT) for m in stop_mean], np.int64),
            bn_mean=bn_mean, bn_std=bn_std, bn_fixed=np.array([round(max(m, 0.0) / DT) for m in bn_mean], np.int64),
            free_type=self.free_type, abreast=self.abreast, pass_in_lane=self.pass_in_lane, allowed=self.allowed,
            entry_lanes=self.entry_lanes, entry_count=self.entry_count, lookahead_t=self.lookahead_t,
            min_adv_t=self.min_adv_t, cooldown_ticks_t=self.cooldown_ticks_t, overtake=self.overtake,
            acc_t=self.acc_t, dec_t=self.dec_t, dyn_t=self.dyn_t, dec_fin_t=self.dec_fin_t, dec_safe_t=self.dec_safe_t,
            bump_v_t=self.bump_v_t, emits_t=self.emits_t, emis_on=self.emis_on, coef_sets=self._coef_sets,
            emis_amax=self.emis_amax, emis_dmax=self.emis_dmax,
            lane_vmax=self.lane_vmax, bn_lane=self.bn_lane, exit_rate=self.exit_rate, exit_limited=self._exit_limited,
            exit_add=self._exit_add, exit_storage=self.exit_storage, bump_x=self.bump_x, bump_lane=self.bump_lane,
            inner_x=self.inner_x, inner_free=self.inner_free, inner_lane=self.inner_lane,
            line_kind=kind, line_red=red, line_green=green, line_yellow=yellow, line_start_red=start_red,
            line_offset=offset, line_sched=sched,
        )  # fmt: skip

    def _state(self) -> kernel.State:
        """The state for the kernel: the simulation's own arrays (no copies)."""
        rec = {f"r_{name}": getattr(self.recorder, name) for name in RECORDER_FIELDS}
        return kernel.State(ints=self._ints, **{f: rec[f] if f in rec else getattr(self, f)
                                                for f in kernel.State._fields if f != "ints"})  # fmt: skip

    def _kernel_rngs(self) -> kernel.Rngs:
        """The step's generators, MAX_TYPES per process (numba compiles only once for any number of
        types); the filler ones are not used."""
        spare = [np.random.default_rng(0) for _ in range(MAX_TYPES - self.n_types)]

        def full(gens) -> tuple:
            return tuple(gens) + tuple(spare)

        return kernel.Rngs(full(self.rng_react), full(self.rng_stop), full(self.rng_bottleneck_time),
                           full(self.rng_queue_react), full(self.rng_entry), self.rng_lane_change)  # fmt: skip

    def _add(
        self, vt: int, pax: int, lane: int, length: float | None = None, x: float | None = None,
        bottleneck_at: float = INF, speed_kmh: float | None = None, v0: float | None = None,
    ) -> None:  # fmt: skip
        """Add a vehicle (see kernel.add_vehicle); without `x`, it enters at the start of the segment."""
        if self.n == self.x.size:
            self._make_room()
        kernel.py_add_vehicle(self._p, self._st, int(vt), int(pax), int(lane), NAN if length is None else float(length),
                           NAN if x is None else float(x), float(bottleneck_at),
                           NAN if speed_kmh is None else float(speed_kmh), NAN if v0 is None else float(v0))  # fmt: skip

    def _initial_fill(self, rng: np.random.Generator) -> None:
        """Initial condition: each lane starts with vehicles occupying the fraction `occupancy`
        of its length, each one measuring length + gap_stop (1 = compact queue, with no more space than
        the stopped gap). Types are chosen in proportion to their arrival rate among those that can
        use that lane; passengers, freight and length are drawn as on arrival. If a
        vehicle does not fit another one is drawn, up to INITIAL_MISSES consecutive attempts.

        The free space is first shared so that each one has its gap_run (moving) and the rest
        at random. If it is not enough, all compress equally and start stopped, like a
        jam: they start with their reaction time when space opens up."""
        cfg = self.cfg
        for lane, occupancy in enumerate(cfg.lane_initial_occupancy):
            types = [k for k in range(self.n_types) if cfg.rates[k] > 0 and lane in cfg.entry_lanes(k)]
            if occupancy <= 0 or not types:
                continue
            weights = np.array([cfg.rates[k] for k in types])
            weights = weights / weights.sum()
            chosen, used, misses = [], 0.0, 0
            while misses < INITIAL_MISSES:
                k = types[rng.choice(len(types), p=weights)]
                spec = cfg.specs[k]
                length = float(sample_lengths(rng, spec, 1)[0])
                if used + length + self.gap_stop_t[k] > occupancy * self.L + EPS:
                    misses += 1
                    continue
                misses = 0
                pax = int(sample_passengers(rng, spec, 1)[0])
                if spec.cargo_prob > 0 and rng.random() < spec.cargo_prob:
                    pax = 0
                at = rng.uniform(*self.bn_zone) if self.bn_type[k] and rng.random() < self.bn_prob[k] else INF
                speed = float(sample_speeds(rng, spec, 1)[0])  # without a deviation it consumes no draws
                chosen.append((k, pax, length, at, speed))
                used += length + self.gap_stop_t[k]
            if not chosen:
                continue
            # Space above the minimum (gap_stop behind each leader; the first has no leader).
            kinds = np.array([k for k, *_ in chosen])
            pool = self.L - sum(length for _, _, length, *_ in chosen) - self.gap_stop_t[kinds[1:]].sum()
            # up to the moving gap (gap_run, or that of time_headway at its desired speed)
            run_gap = [kernel.py_run_gap(self._p, k, min(speed / 3.6 * DT, self.lane_vmax[lane]))
                       for k, *_, speed in chosen]  # fmt: skip
            need = np.r_[0.0, np.array(run_gap[1:]) - self.gap_stop_t[kinds[1:]]]
            if pool >= need.sum():
                spare = np.diff(np.sort(rng.uniform(0.0, pool - need.sum(), len(chosen))), prepend=0.0)
                extra = need + spare
            else:  # not enough to be moving: they compress equally
                extra = need * pool / need.sum()
            # Vehicles are placed from the traffic light backwards; one less than its gap_run behind starts stopped.
            rear = self.L
            ahead = -1  # the one placed just ahead in this lane
            for idx, ((k, pax, length, at, speed), free) in enumerate(zip(chosen, extra)):
                gap = (self.gap_stop_t[k] if idx else 0.0) + free
                self._add(k, pax, lane, length, x=rear - gap, bottleneck_at=at, speed_kmh=speed)
                i = self.n - 1
                if idx and gap < run_gap[idx] - EPS:
                    self.stopped[i] = True
                    self.v_last[i] = 0.0
                elif self.dyn_t[k]:
                    # With gradual braking it starts at the speed at which it can still brake behind the one ahead
                    # (or before the line if the run starts on red), not all at once in the first step.
                    if ahead >= 0:
                        room = self.x[ahead] - self.vlen[ahead] - run_gap[idx] - self.x[i]
                        v_ahead, dec_ahead = float(self.v_last[ahead]), self.dec_t[self.vtype[ahead]]
                    else:
                        room = self.L - self.x[i] if self.exit_phase(0) == RED and not self.free_type[k] else INF
                        v_ahead, dec_ahead = 0.0, INF
                    if np.isfinite(room):
                        self.v_last[i] = min(self.v_last[i],
                                             kernel.py_safe_speed_1(room, v_ahead, self.dec_t[k], dec_ahead))  # fmt: skip
                    light = self._line_room(k, self.x[i], lane)
                    if np.isfinite(light):  # intermediate stop line on red at the start
                        self.v_last[i] = min(self.v_last[i], kernel.py_safe_speed_1(light, 0.0, self.dec_t[k], INF))
                    bump = kernel.py_bump_speed_1(self._p, k, lane, self.x[i], self.x[i] - length)
                    self.v_last[i] = min(self.v_last[i], bump)
                ahead = i
                rear = self.x[i] - length

    def _line_room(self, k: int, x: float, lane: int) -> float:
        """Distance from `x` to the closest intermediate stop line ahead (traffic light or pedestrian crossing) that is red at
        the start of the run and that type k obeys in that lane; INF if there is none."""
        return min((pos - x for pos, free, ok, ph in zip(self.inner_x, self.inner_free, self.inner_lane, self.line_phases(0))
                    if ph == RED and x <= pos and not free[k] and ok[lane]), default=INF)  # fmt: skip

    # ------------------------------------------------------------- pedestrians

    def _init_pedestrians(self, rng: np.random.Generator) -> None:
        """The replica's pedestrian schedule (see pedestrians.py): one per pedestrian light and per bump with
        pedestrians. The generators go by fixed index —the light at the end of the segment, each intermediate light and
        each bump— so removing one of the other elements does not move the draws of the rest."""
        cfg = self.cfg
        n = cfg.n_ticks
        lights_rng, bumps_rng = rng.spawn(2)
        light_gens = lights_rng.spawn(1 + len(cfg.extra_lights))
        bump_gens = bumps_rng.spawn(len(cfg.bumps))

        def light_schedule_of(light, gen) -> Schedule:
            before = cfg.previous_light(light)
            prev_red = fixed_phases(before, n + 1) == RED if before else None  # waits for the previous light's red
            counts = arrivals(*gen.spawn(2), light.pedestrian_crossing, cfg)
            return light_schedule(counts, light, prev_red, n)

        self.ped_schedules: dict[tuple[str, float], Schedule] = {}
        self._exit_sched: Schedule | None = None
        if cfg.exit_light.active and cfg.exit_light.pedestrian:
            self._exit_sched = light_schedule_of(cfg.exit_light, light_gens[0])
            self.ped_schedules[("semáforo", cfg.length)] = self._exit_sched
        for i, lt in enumerate(cfg.extra_lights):
            if lt.active and lt.pedestrian:
                self.ped_schedules[("semáforo", lt.position)] = light_schedule_of(lt, light_gens[1 + i])
        wait = round(PEDESTRIAN_YIELD / DT)
        for bump, gen in zip(cfg.bumps, bump_gens):  # each bump with pedestrians, with its own generators
            if bump.pedestrian:
                counts = arrivals(*gen.spawn(2), bump.pedestrian_crossing, cfg)
                cross = round(bump.pedestrian_time / DT)
                self.ped_schedules[("tope", bump.position)] = bump_schedule(counts, cross, wait, n)
        # Schedule of each intermediate stop line (None = fixed-cycle light: its phase comes from its cycle).
        self._line_sched = [self.ped_schedules.get((s.kind, s.position)) if s.pedestrian else None for s in self.lines]

    def exit_phase(self, tick: int) -> int:
        """Phase (RED, GREEN or YELLOW) of the traffic light at the end of the segment at step `tick`."""
        return int(kernel.py_line_phase(self._p, 0, tick))

    def line_phases(self, tick: int) -> tuple[int, ...]:
        """Phase of each intermediate stop line (`cfg.stop_lines`) at step `tick`."""
        return tuple(int(kernel.py_line_phase(self._p, 1 + j, tick)) for j in range(len(self.lines)))

    # ------------------------------------------------------------------- step

    def advance(self, until: int) -> None:
        """Run the steps up to step `until` (the kernel; here only the arrivals of each chunk and the room)."""
        while (code := kernel.advance(self._p, self._st, self._rngs, self._arr, until)) != kernel.DONE:
            if code == kernel.NEED_ARRIVALS:
                self._load_arrivals()
            else:
                self._make_room()
        self.recorder.count = int(self._ints[kernel.SAMPLES])

    def run(self) -> None:
        self.advance(self.tick + self.cfg.n_ticks)

    def step(self) -> None:
        self.advance(self.tick + 1)

    def _load_arrivals(self) -> None:
        """Draw the Poisson arrivals of the ARRIVAL_CHUNK-step chunk that starts at the current step, and what belongs to each
        arrival (passengers, freight, length, speed and halt), with each type's generators and in step
        order: the same numbers as drawing them on arrival."""
        start = self.tick
        chunk = np.column_stack([self._arrival_counts(k) for k in range(self.n_types)])
        cols: list[list[np.ndarray]] = [[], [], [], [], [], []]
        for offset in np.flatnonzero(chunk.any(axis=1)):
            counts = chunk[offset]
            for vt in np.flatnonzero(counts):
                k = int(counts[vt])
                spec = self.cfg.specs[vt]
                paxs = sample_passengers(self.rng_pax[vt], spec, k)
                if spec.cargo_prob > 0:  # those carrying freight carry no passengers
                    paxs[self.rng_cargo[vt].random(k) < spec.cargo_prob] = 0
                lengths = sample_lengths(self.rng_length[vt], spec, k)
                speeds = sample_speeds(self.rng_speed[vt], spec, k)
                at = np.full(k, INF)  # where it will stop (bottleneck); inf = it does not stop
                if self.bn_type[vt]:
                    g = self.rng_bottleneck[vt]
                    stops = g.random(k) < self.bn_prob[vt]
                    at = np.where(stops, np.maximum(g.uniform(*self.bn_zone, k), lengths + 0.01), INF)
                for col, values in zip(cols, (np.full(k, start + offset), np.full(k, vt), paxs, lengths, at, speeds)):
                    col.append(values)
        dtypes = (np.int64,) * 3 + (np.float64,) * 3
        self._arr = kernel.Arrivals(start, *(np.concatenate(c).astype(d) if c else np.zeros(0, d)
                                             for c, d in zip(cols, dtypes)))  # fmt: skip
        self._ints[kernel.ARRIVAL_POS] = 0

    def _arrival_counts(self, k: int) -> np.ndarray:
        """Arrivals of type k at each step of the chunk that starts at the current step."""
        g = self.rng_arrivals[k]
        by_interval = self.rate_by_interval.get(k)
        if by_interval is None:
            return g.poisson(self.rate_tick[k], ARRIVAL_CHUNK)
        idx = (self.tick + np.arange(ARRIVAL_CHUNK)) // self.cfg.rate_ticks
        return g.poisson(by_interval[np.minimum(idx, by_interval.size - 1)])

    # ------------------------------------------------------- state queries

    def occupancy(self) -> Occupancy:
        return kernel.py_occupancy(self._p, self._st)

    def rows(self, occ: Occupancy) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Side-by-side rows in the order of `occ` (see kernel.rows)."""
        return kernel.py_rows(self._st, occ)

    def row_positions(self) -> tuple[np.ndarray, np.ndarray]:
        """Per vehicle [0, n): (place in its side-by-side row, 0 = the one in front; size of the row)."""
        n = self.n
        rank = np.zeros(n, np.int8)
        size = np.ones(n, np.int8)
        if n < 2 or not self.any_abreast:
            return rank, size
        occ = self.occupancy()
        _, rid, row_last = self.rows(occ)
        idx = np.arange(occ.order.size)
        first = np.concatenate(([True], rid[1:] != rid[:-1])).nonzero()[0][rid]
        primary = occ.order < n  # a vehicle changing lanes appears twice; its current lane is used
        rank[occ.veh[primary]] = (row_last - idx)[primary]
        size[occ.veh[primary]] = (row_last - first + 1)[primary]
        return rank, size

    def _side_slot(self, occ: Occupancy, rows: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None) -> np.ndarray:
        """Entries that can stop beside their leader (see kernel.side_slot)."""
        return kernel.py_side_slot(self._p, self._st, occ, *(self.rows(occ) if rows is None else rows))

    def _lane_saturation(self, occ: Occupancy) -> np.ndarray:
        return kernel.py_lane_saturation(self._p, self._st, occ)

    def _lane_changes(self) -> None:
        """One round of lane change decisions (see kernel.lane_changes)."""
        kernel.py_lane_changes(self._p, self._st, self._rngs)

    # --------------------------------------------------------------- summary

    def summary(self) -> dict[str, np.ndarray]:
        n = self.n
        on_road = np.bincount(
            self.vtype[:n][~self.crossed[:n]], minlength=self.n_types
        ).astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            travel_time = self.travel_ticks * DT / self.timed_veh
            # Mean wait in the entry queue of those that arrived from demand and already entered the segment.
            queue_wait = self.queue_wait_ticks * DT / (self.entered_veh - self.initial_veh)
            pax_per_veh = self.arrived_pax / (self.arrived_veh - self.arrived_cargo)  # only the passenger ones
            # Mean speed within the segment: distance / time of its vehicles, including stopped ones.
            mean_speed = self.type_dist / self.type_time * 3.6
        return {
            "arrived_veh": self.arrived_veh.copy(),
            "arrived_pax": self.arrived_pax.copy(),
            "arrived_cargo": self.arrived_cargo.copy(),
            "entered_veh": self.entered_veh.copy(),
            "initial_veh": self.initial_veh.copy(),
            "initial_pax": self.initial_pax.copy(),
            "crossed_veh": self.cum_veh.copy(),
            "crossed_pax": self.cum_pax.copy(),
            "travel_time": travel_time,
            "mean_speed": mean_speed,
            "queue_wait": queue_wait,
            "pax_per_veh": pax_per_veh,
            "queued": self.queued_types(),
            "on_road": on_road,
            "lane_changes": np.array([float(self.lane_changes)]),
            "lane_changes_type": self.lane_changes_t.copy(),
            "in_lane_passes": self.in_lane_passes.copy(),
            "lane_crossed": self.lane_crossed.copy(),  # line crossings per lane
            "exit_blocked": self.exit_blocked_ticks / max(self.tick, 1),  # fraction of the time with the line closed
            "stop_time": self.stop_time_ticks * DT / np.where(self.stops > 0, self.stops, np.nan),
            "bottleneck_stops": self.bn_stops.copy(),
            "bottleneck_time": self.bn_ticks * DT / np.where(self.bn_stops > 0, self.bn_stops, np.nan),
            "veh_km": self.veh_m / 1000.0,  # km travelled within the segment, per type
            "emissions": self.emis_g.copy(),  # g per type and pollutant (POLLUTANTS)
            "emissions_free": self.emis_free_g.copy(),  # g of the same trip at constant speed
            "emissions_pos": self.emis_pos.copy(),  # g per pollutant, lane and EMIS_BIN m bin
            **self._pedestrian_summary(),
        }

    def _pedestrian_summary(self) -> dict[str, np.ndarray]:
        """One row per point with pedestrians (`cfg.pedestrian_spots`): pedestrians that crossed, closures of the crossing, s
        the crossing was closed and total wait of the pedestrians (s). Empty if there are no pedestrians."""
        spots = self.cfg.pedestrian_spots
        if not spots:
            return {}
        rows = [(s.crossed, s.crossings, s.closed_ticks * DT, s.wait_ticks * DT)
                for s in (self.ped_schedules[spot] for spot in spots)]  # fmt: skip
        return {"pedestrians": np.array(rows, dtype=np.float64)}
