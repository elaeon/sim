"""Crossing pedestrians: the phase schedule that vehicles see at a pedestrian crossing or a pedestrian light.

Pedestrians do not depend on vehicles: they appear on their own (Poisson arrivals at the rate of
`pedestrian_crossing`, pedestrians/min), so a replica's whole schedule is computed before running it and the
engine only looks up the phase of each step (GREEN, YELLOW or RED), just as with a fixed-cycle traffic light.

  * Pedestrian crossing at a bump (`bump_schedule`): each pedestrian waits `PEDESTRIAN_YIELD` s at the kerb
    (yellow: vehicles that can still brake stop) and crosses for `pedestrian_time` s (red). The crossing is closed
    while someone is crossing: whoever arrives during the crossing extends it.
  * Pedestrian light (`light_schedule`): green for vehicles while nobody is waiting. With pedestrians in the queue,
    once the minimum green (`green`) has passed and with the previous fixed-cycle light on red (if there is one), it
    turns yellow (`yellow`) and then red (`red`, the time it takes everyone waiting when the red starts to cross).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from traffic.config import DT, GREEN, RED, YELLOW, Light, Rate, SimConfig
from traffic.distributions import sample_rates


@dataclass(frozen=True, slots=True)
class Schedule:
    """Schedule of a point with pedestrians in one replica."""

    phases: np.ndarray  # (n_ticks + 1,) phase of each step for vehicles (GREEN, YELLOW or RED)
    crossed: float  # pedestrians that crossed
    crossings: float  # times the crossing was closed (crossings)
    closed_ticks: float  # steps with the crossing on red
    wait_ticks: float  # total wait of the pedestrians that crossed, in steps


def arrivals(rng_rate: np.random.Generator, rng_count: np.random.Generator, rate: Rate, cfg: SimConfig) -> np.ndarray:
    """Pedestrians arriving in each of the `n_ticks` steps: Poisson with the interval's rate (pedestrians/min), which
    is drawn from the truncated normal of `rate` every `rate_interval` s (like the vehicle demand)."""
    n = cfg.n_ticks
    intervals = -(-n // cfg.rate_ticks)
    per_tick = sample_rates(rng_rate, rate, intervals) / 60.0 * DT
    return rng_count.poisson(per_tick[np.arange(n) // cfg.rate_ticks])


def fixed_phases(light: Light, n: int) -> np.ndarray:
    """Phase of a fixed-cycle traffic light in each of the steps [0, n) (the same as `Light.phase`)."""
    out = np.full(n, GREEN, np.int8)
    if not light.active or light.pedestrian:
        return out
    red, green, yellow = round(light.red / DT), round(light.green / DT), round(light.yellow / DT)
    cycle = red + green + yellow
    pos = (np.arange(n) - round(light.offset / DT)) % cycle
    if light.start_phase == "red":
        pos = (pos - red) % cycle
    out[pos >= green] = YELLOW
    out[pos >= green + yellow] = RED
    return out


def bump_schedule(counts: np.ndarray, cross_ticks: int, yield_ticks: int, n: int) -> Schedule:
    """Schedule of a bump's pedestrian crossing. `counts[t]` pedestrians arrive at the kerb in step t; each waits
    `yield_ticks` and crosses for `cross_ticks`. While someone is crossing the crossing is red; the yellow is the wait
    of the first pedestrian of each closure."""
    phases = np.full(n + 1, GREEN, np.int8)
    crossed = crossings = closed = 0
    wait = 0.0
    start = red_start = end = -1  # the current closure (end = -1: none)

    def close() -> int:
        phases[start : min(red_start, n + 1)] = YELLOW
        phases[red_start : min(end, n + 1)] = RED
        return max(0, min(end, n) - red_start)

    for t in np.flatnonzero(counts).tolist():
        m = int(counts[t])
        if t < end:
            if t >= red_start:  # already crossing: joins and extends the crossing
                end = max(end, t + cross_ticks)
                waited = 0
            else:  # waits at the kerb along with the first one
                waited = red_start - t
        else:
            if end >= 0:
                closed += close()
            start, red_start = t, t + yield_ticks
            end = red_start + cross_ticks
            crossings += 1
            waited = yield_ticks
        crossed += m
        wait += m * waited
    if end >= 0:
        closed += close()
    return Schedule(phases, crossed, crossings, closed, wait)


def light_schedule(counts: np.ndarray, light: Light, prev_red: np.ndarray | None, n: int) -> Schedule:
    """Schedule of a pedestrian light. `counts[t]` pedestrians join the queue in step t. `prev_red` (n steps or
    more) marks when the previous fixed-cycle light is red; None = there is none and only the yellow is waited."""
    phases = np.full(n + 1, GREEN, np.int8)
    yellow, red, green = round(light.yellow / DT), round(light.red / DT), round(light.green / DT)
    arrival = np.flatnonzero(counts).tolist()
    j = 0  # next arrival not yet in the queue
    waiting: list[tuple[int, int]] = []  # (arrival step, pedestrians)
    last_end = -green  # the first crossing does not wait for the minimum green
    t = 0
    crossed = crossings = closed = 0
    wait = 0.0
    while t < n:
        while j < len(arrival) and arrival[j] <= t:
            waiting.append((arrival[j], int(counts[arrival[j]])))
            j += 1
        if not waiting:
            if j >= len(arrival):
                break
            t = arrival[j]
            continue
        ready = max(t, last_end + green)
        if prev_red is not None:  # waits for the previous light's red
            hits = np.flatnonzero(prev_red[ready:n])
            if hits.size == 0:
                break
            ready += int(hits[0])
        if ready >= n:
            break
        if ready > t:
            t = ready  # more pedestrians arrive meanwhile: they join the queue
            continue
        red_start = t + yellow
        while j < len(arrival) and arrival[j] <= red_start:  # those arriving during the yellow also cross
            waiting.append((arrival[j], int(counts[arrival[j]])))
            j += 1
        end = red_start + red
        phases[t : min(red_start, n + 1)] = YELLOW
        phases[red_start : min(end, n + 1)] = RED
        crossings += 1
        closed += max(0, min(end, n) - red_start)
        crossed += sum(m for _, m in waiting)
        wait += sum(m * (red_start - a) for a, m in waiting)
        waiting.clear()
        last_end = t = end
    return Schedule(phases, crossed, crossings, closed, wait)
