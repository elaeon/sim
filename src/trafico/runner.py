"""Ejecución de réplicas Monte Carlo en paralelo (procesos de CPU) y su agregación."""

from __future__ import annotations

import resource
import time
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass, field

import numpy as np

from trafico.config import DT, SimConfig
from trafico.engine import Simulation
from trafico.metrics import LANE_SERIES, SERIES, RunningStats, derive_series


@dataclass(slots=True)
class ReplicaResult:
    series: dict[str, np.ndarray]  # (muestras, tipos) float32
    summary: dict[str, np.ndarray]
    wall: float  # s reales que tardó la réplica
    max_rss_kb: int  # memoria residente máxima del proceso que la corrió
    pax_hist: np.ndarray  # (tipos, pasajeros) vehículos llegados con cada número de pasajeros


def run_replica(cfg: SimConfig, seed: np.random.SeedSequence) -> ReplicaResult:
    """Corre una réplica completa; solo devuelve series agregadas, no trayectorias."""
    t0 = time.perf_counter()
    sim = Simulation(cfg, np.random.default_rng(seed))
    sim.run()
    return ReplicaResult(
        series=derive_series(sim.recorder, cfg),
        summary=sim.summary(),
        wall=time.perf_counter() - t0,
        max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        pax_hist=sim.pax_hist,
    )


@dataclass(slots=True)
class Aggregate:
    """Estadísticos entre réplicas. Memoria O(muestras), independiente del número de réplicas."""

    cfg: SimConfig
    series: dict[str, RunningStats]
    summary: dict[str, RunningStats] = field(default_factory=dict)
    replica_wall: RunningStats = field(default_factory=lambda: RunningStats(()))
    replicas: int = 0
    workers: int = 1
    wall: float = 0.0
    max_rss_kb: int = 0
    pax_hist: np.ndarray | None = None  # suma entre réplicas de los histogramas de pasajeros

    @classmethod
    def empty(cls, cfg: SimConfig) -> Aggregate:
        series = {name: RunningStats((cfg.n_samples, cfg.n_types)) for name in SERIES}
        for name in LANE_SERIES:
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
        self.replicas += 1

    @property
    def times(self) -> np.ndarray:
        """Instante (s simulados) de cada muestra."""
        step = self.cfg.sample_ticks * DT
        return step * np.arange(1, self.cfg.n_samples + 1)


def run_parallel(
    cfg: SimConfig,
    replicas: int,
    workers: int,
    seed: int,
    progress: Callable[[int, int], None] | None = None,
) -> Aggregate:
    """Corre `replicas` réplicas independientes en `workers` procesos.

    Las semillas se derivan con SeedSequence.spawn y los resultados se agregan
    en orden de réplica, así que el resultado no depende de `workers`. Como
    mucho hay 2×workers resultados en memoria a la vez.
    """
    seeds = np.random.SeedSequence(seed).spawn(replicas)
    agg = Aggregate.empty(cfg)
    agg.workers = workers
    t0 = time.perf_counter()

    if workers <= 1:
        for i, s in enumerate(seeds):
            agg.push(run_replica(cfg, s))
            if progress:
                progress(i + 1, replicas)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            window = 2 * workers
            running: dict[Future, int] = {}
            ready: dict[int, ReplicaResult] = {}
            next_submit = next_push = 0
            while next_push < replicas:
                while next_submit < replicas and len(running) + len(ready) < window:
                    running[pool.submit(run_replica, cfg, seeds[next_submit])] = next_submit
                    next_submit += 1
                done, _ = wait(running, return_when=FIRST_COMPLETED)
                for fut in done:
                    ready[running.pop(fut)] = fut.result()
                while next_push in ready:
                    agg.push(ready.pop(next_push))
                    next_push += 1
                    if progress:
                        progress(next_push, replicas)

    agg.wall = time.perf_counter() - t0
    return agg
