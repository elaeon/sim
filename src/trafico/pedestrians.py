"""Peatones que cruzan: el calendario de fases que ven los vehículos en un paso peatonal o en un semáforo peatonal.

Los peatones no dependen de los vehículos: aparecen por su cuenta (llegadas Poisson con la tasa de
`pedestrian_crossing`, peatones/min), así que todo el calendario de una réplica se calcula antes de correrla y el
motor solo consulta la fase de cada paso (GREEN, YELLOW o RED), igual que con un semáforo de ciclo fijo.

  * Paso peatonal en un tope (`bump_schedule`): cada peatón espera `PEDESTRIAN_YIELD` s en la orilla (amarillo: los
    vehículos que aún pueden frenar se detienen) y cruza durante `pedestrian_time` s (rojo). El paso está cerrado
    mientras cruza alguien: quien llega durante el cruce lo extiende.
  * Semáforo peatonal (`light_schedule`): verde para los vehículos mientras nadie espere. Con peatones en la cola,
    pasado el verde mínimo (`green`) y con el semáforo de ciclo fijo anterior en rojo (si lo hay), pasa a amarillo
    (`yellow`) y a rojo (`red`, lo que tardan en cruzar todos los que esperan al empezar el rojo).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trafico.config import DT, GREEN, RED, YELLOW, Light, Rate, SimConfig
from trafico.distributions import sample_rates


@dataclass(frozen=True, slots=True)
class Schedule:
    """Calendario de un punto con peatones en una réplica."""

    phases: np.ndarray  # (n_ticks + 1,) fase de cada paso para los vehículos (GREEN, YELLOW o RED)
    crossed: float  # peatones que cruzaron
    crossings: float  # veces que se cerró el paso (cruces)
    closed_ticks: float  # pasos con el paso en rojo
    wait_ticks: float  # espera total de los peatones que cruzaron, en pasos


def arrivals(rng_rate: np.random.Generator, rng_count: np.random.Generator, rate: Rate, cfg: SimConfig) -> np.ndarray:
    """Peatones que llegan en cada uno de los `n_ticks` pasos: Poisson con la tasa (peatones/min) del intervalo, que
    se sortea de la normal truncada de `rate` cada `rate_interval` s (como la demanda de vehículos)."""
    n = cfg.n_ticks
    intervals = -(-n // cfg.rate_ticks)
    per_tick = sample_rates(rng_rate, rate, intervals) / 60.0 * DT
    return rng_count.poisson(per_tick[np.arange(n) // cfg.rate_ticks])


def fixed_phases(light: Light, n: int) -> np.ndarray:
    """Fase de un semáforo de ciclo fijo en cada uno de los pasos [0, n) (la misma que `Light.phase`)."""
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
    """Calendario del paso peatonal de un tope. `counts[t]` peatones llegan a la orilla en el paso t; cada uno espera
    `yield_ticks` y cruza `cross_ticks`. Mientras cruza alguien el paso está en rojo; el amarillo es la espera del
    primero de cada cierre."""
    phases = np.full(n + 1, GREEN, np.int8)
    crossed = crossings = closed = 0
    wait = 0.0
    start = red_start = end = -1  # el cierre en curso (end = -1: ninguno)

    def close() -> int:
        phases[start : min(red_start, n + 1)] = YELLOW
        phases[red_start : min(end, n + 1)] = RED
        return max(0, min(end, n) - red_start)

    for t in np.flatnonzero(counts).tolist():
        m = int(counts[t])
        if t < end:
            if t >= red_start:  # ya están cruzando: se une y alarga el cruce
                end = max(end, t + cross_ticks)
                waited = 0
            else:  # espera en la orilla junto al primero
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
    """Calendario de un semáforo peatonal. `counts[t]` peatones llegan a la cola en el paso t. `prev_red` (n pasos o
    más) marca cuándo está en rojo el semáforo de ciclo fijo anterior; None = no hay y solo espera el amarillo."""
    phases = np.full(n + 1, GREEN, np.int8)
    yellow, red, green = round(light.yellow / DT), round(light.red / DT), round(light.green / DT)
    arrival = np.flatnonzero(counts).tolist()
    j = 0  # siguiente llegada que aún no está en la cola
    waiting: list[tuple[int, int]] = []  # (paso de llegada, peatones)
    last_end = -green  # el primer cruce no espera verde mínimo
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
        if prev_red is not None:  # espera al rojo del anterior
            hits = np.flatnonzero(prev_red[ready:n])
            if hits.size == 0:
                break
            ready += int(hits[0])
        if ready >= n:
            break
        if ready > t:
            t = ready  # llegan más peatones mientras tanto: se suman a la cola
            continue
        red_start = t + yellow
        while j < len(arrival) and arrival[j] <= red_start:  # los que llegan durante el amarillo también cruzan
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
