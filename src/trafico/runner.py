"""Ejecución de réplicas Monte Carlo en paralelo (procesos de CPU) y su agregación."""

from __future__ import annotations

import os
import resource
import time
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass, field

import numpy as np

from trafico.config import DT, SimConfig
from trafico.engine import Simulation
from trafico.metrics import EMIS_SERIES, LANE_STATS, SERIES, RunningStats, derive_series, lane_speed_histogram


@dataclass(slots=True)
class ReplicaResult:
    series: dict[str, np.ndarray]  # (muestras, tipos) float32
    summary: dict[str, np.ndarray]
    wall: float  # s reales que tardó la réplica
    max_rss_kb: int  # memoria residente máxima del proceso que la corrió
    pax_hist: np.ndarray  # (tipos, pasajeros) vehículos llegados con cada número de pasajeros
    lane_speed_hist: np.ndarray  # (carriles, intervalos de velocidad) muestras con vehículos en el carril


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
        lane_speed_hist=lane_speed_histogram(sim.recorder, cfg),
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
    lane_speed_hist: np.ndarray | None = None  # suma entre réplicas de los histogramas de velocidad por carril

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
    """Corre `replicas` réplicas independientes en `workers` procesos (0 = tantos como CPU).

    Las semillas se derivan con SeedSequence.spawn y los resultados se agregan
    en orden de réplica, así que el resultado no depende de `workers`. Como
    mucho hay 2×workers resultados en memoria a la vez.
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
    """Corre `replicas` réplicas de cada configuración en un solo grupo de `workers` procesos (0 = tantos como CPU):
    los procesos no esperan a que termine una configuración para empezar la siguiente.

    Todas las configuraciones usan las mismas semillas (números aleatorios comunes, como `run_parallel` con cada
    una) y cada agregado recibe sus réplicas en orden, así que el resultado es el mismo que corriéndolas por
    separado y no depende de `workers`. `progress(configuración, réplicas listas, réplicas)` se llama en orden.
    Como mucho hay 2×workers resultados en memoria a la vez."""
    # Semillas nuevas para cada configuración: una SeedSequence lleva la cuenta de los hijos que deriva (la réplica
    # deriva los suyos), así que reusar el mismo objeto en el mismo proceso cambiaría los números de la siguiente.
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
        if r + 1 == replicas:  # tiempo real de cada configuración: desde que terminó la anterior
            now = time.perf_counter()
            aggs[i].wall, last = now - last, now

    if workers == 1:
        for k, (i, r) in enumerate(tasks):
            push(k, run_replica(cfgs[i], seeds[i][r]))
    else:
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
