"""Execution of Monte Carlo replicas in parallel (CPU processes) and their aggregation."""

from __future__ import annotations

import os
import resource
import time
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass, field

import numpy as np

from traffic.config import DT, SimConfig
from traffic.engine import Simulation, compile_kernel
from traffic.metrics import EMIS_SERIES, LANE_STATS, SERIES, RunningStats, derive_series, lane_speed_histogram


@dataclass(slots=True)
class ReplicaResult:
    series: dict[str, np.ndarray]  # (samples, types) float32
    summary: dict[str, np.ndarray]
    wall: float  # wall-clock s the replica took
    max_rss_kb: int  # peak resident memory of the process that ran it
    pax_hist: np.ndarray  # (types, passengers) vehicles arrived with each number of passengers
    lane_speed_hist: np.ndarray  # (lanes, speed bins) samples with vehicles in the lane


def run_replica(cfg: SimConfig, seed: np.random.SeedSequence) -> ReplicaResult:
    """Run a full replica; returns only aggregated series, not trajectories."""
    t0 = time.perf_counter()
    sim = Simulation(cfg, np.random.default_rng(seed))
    sim.run()
    return ReplicaResult(
        series=derive_series(sim.recorder, cfg),
        summary=sim.summary(),
        wall=time.perf_counter() - t0,
        max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        pax_hist=sim.pax_hist,
        lane_speed_hist=lane_speed_histogram(sim.recorder, cfg),
    )


@dataclass(slots=True)
class Aggregate:
    """Statistics across replicas. Memory O(samples), independent of the number of replicas."""

    cfg: SimConfig
    series: dict[str, RunningStats]
    summary: dict[str, RunningStats] = field(default_factory=dict)
    replica_wall: RunningStats = field(default_factory=lambda: RunningStats(()))
    replicas: int = 0
    workers: int = 1
    wall: float = 0.0
    max_rss_kb: int = 0
    pax_hist: np.ndarray | None = None  # sum over replicas of the passenger histograms
    lane_speed_hist: np.ndarray | None = None  # sum over replicas of the per-lane speed histograms

    @classmethod
    def empty(cls, cfg: SimConfig) -> Aggregate:
        series = {name: RunningStats((cfg.n_samples, cfg.n_types)) for name in (*SERIES, *EMIS_SERIES)}
        for name in LANE_STATS:
            series[name] = RunningStats((cfg.n_samples, cfg.lanes))
        return cls(cfg=cfg, series=series)

    def push(self, res: ReplicaResult) -> None:
        for name, stats in self.series.items():
            stats.push(res.series[name])
        for name, value in res.summary.items():
            if name not in self.summary:
                self.summary[name] = RunningStats(value.shape)
            self.summary[name].push(value)
        self.replica_wall.push(res.wall)
        self.max_rss_kb = max(self.max_rss_kb, res.max_rss_kb)
        self.pax_hist = res.pax_hist.copy() if self.pax_hist is None else self.pax_hist + res.pax_hist
        hist = res.lane_speed_hist
        self.lane_speed_hist = hist.copy() if self.lane_speed_hist is None else self.lane_speed_hist + hist
        self.replicas += 1

    @property
    def times(self) -> np.ndarray:
        """Time (simulated s) of each sample."""
        step = self.cfg.sample_ticks * DT
        return step * np.arange(1, self.cfg.n_samples + 1)


def run_parallel(
    cfg: SimConfig,
    replicas: int,
    workers: int,
    seed: int,
    progress: Callable[[int, int], None] | None = None,
) -> Aggregate:
    """Run `replicas` independent replicas in `workers` processes (0 = as many as CPUs).

    Seeds are derived with SeedSequence.spawn and results are aggregated
    in replica order, so the result does not depend on `workers`. At most
    2×workers results are in memory at once.
    """
    report = (lambda _, done, total: progress(done, total)) if progress else None
    return run_many([cfg], replicas, workers, seed, progress=report)[0]


def run_many(
    cfgs: list[SimConfig],
    replicas: int,
    workers: int,
    seed: int,
    progress: Callable[[int, int, int], None] | None = None,
) -> list[Aggregate]:
    """Run `replicas` replicas of each configuration in a single pool of `workers` processes (0 = as many as CPUs):
    processes do not wait for one configuration to finish before starting the next.

    All configurations use the same seeds (common random numbers, like `run_parallel` with each
    one) and each aggregate receives its replicas in order, so the result is the same as running them
    separately and does not depend on `workers`. `progress(configuration, replicas done, replicas)` is called in order.
    At most 2×workers results are in memory at once."""
    # New seeds for each configuration: a SeedSequence keeps count of the children it derives (the replica
    # derives its own), so reusing the same object in the same process would change the next one's numbers.
    seeds = [np.random.SeedSequence(seed).spawn(replicas) for _ in cfgs]
    aggs = [Aggregate.empty(cfg) for cfg in cfgs]
    tasks = [(i, r) for i in range(len(cfgs)) for r in range(replicas)]
    if workers <= 0:
        workers = os.process_cpu_count() or 1
    workers = max(1, min(workers, len(tasks)))
    t0 = time.perf_counter()
    last = t0

    def push(k: int, res: ReplicaResult) -> None:
        nonlocal last
        i, r = tasks[k]
        aggs[i].push(res)
        if progress:
            progress(i, r + 1, replicas)
        if r + 1 == replicas:  # wall-clock time of each configuration: since the previous one finished
            now = time.perf_counter()
            aggs[i].wall, last = now - last, now

    if workers == 1:
        for k, (i, r) in enumerate(tasks):
            push(k, run_replica(cfgs[i], seeds[i][r]))
    else:
        compile_kernel(cfgs[0])  # once here: the processes load it from the cache instead of each compiling it
        with ProcessPoolExecutor(max_workers=workers) as pool:
            window = 2 * workers
            running: dict[Future, int] = {}
            ready: dict[int, ReplicaResult] = {}
            next_submit = next_push = 0
            while next_push < len(tasks):
                while next_submit < len(tasks) and len(running) + len(ready) < window:
                    i, r = tasks[next_submit]
                    running[pool.submit(run_replica, cfgs[i], seeds[i][r])] = next_submit
                    next_submit += 1
                done, _ = wait(running, return_when=FIRST_COMPLETED)
                for fut in done:
                    ready[running.pop(fut)] = fut.result()
                while next_push in ready:
                    push(next_push, ready.pop(next_push))
                    next_push += 1

    for agg in aggs:
        agg.workers = workers
    return aggs
