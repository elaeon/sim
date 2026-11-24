"""Calibration: measures how a vehicle type from the configuration behaves in standard traffic engineering
situations and compares it with published reference values (`traffic-calibrate`).

Three measurements, each on a single-lane segment with only that type (no bumps, stops, bottlenecks or
pedestrians), with the configuration's speed, lengths, gaps, acceleration, braking and behaviour ([behavior]):

  * Discharge of a queue on green: a queue stopped at a long red starts when the light turns green; the
    headway between successive crossings of the line (front of the vehicle) is measured by position in the queue.
    This gives the saturation headway (mean of positions `SATURATION_FROM` onwards), the saturation flow
    (3600 / headway) and the start-up lost time (HCM: Σ over the first 4 positions of headway − saturation).
  * Continuous flow (no traffic light) with increasing demand: flow at the exit and mean density in the middle
    third of the segment. Capacity is the highest flow; from it, the moving headway (3600 / capacity).
  * Jam density: 1000 / (mean length + gap_stop), vehicles per lane km in a stopped queue.

The references are for cars (HCM 6th ed. and traffic flow literature); for other types the values are reported
without comparing them. They are not local data: counts from the street are worth more than these ranges.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace

import numpy as np

from traffic.config import DT, SimConfig
from traffic.engine import Simulation
from traffic.settings import ConfigError

QUEUE = 15  # queue positions that are measured
SATURATION_FROM = 5  # the saturation headway is the mean from the 5th position (index 4) onwards
LOST_POSITIONS = 4  # HCM: the start-up lost time is counted over the first 4 vehicles
RED_S = 120.0  # s of red to build the queue
FLOW_RATES = (10, 20, 30, 45, 60, 90, 120)  # veh/min of demand in continuous flow
FLOW_LENGTH = 600.0  # m of segment in continuous flow
FLOW_SECONDS = 900.0  # simulated s per demand (the first quarter does not count)


@dataclass(frozen=True, slots=True)
class Reference:
    low: float
    high: float
    source: str


# Ranges for cars. See the module docstring and the documentation (docs/config.html#calibracion).
REFERENCES = {
    "flujo_saturacion_veh_h": Reference(
        1700, 1950, "HCM 6th ed.: base saturation flow of 1,900 veh/h/lane (1,750 in cities of fewer than "
                    "250 thousand inhabitants); urban counts usually come out somewhat lower"),
    "intervalo_saturacion_s": Reference(1.85, 2.1, "the inverse of the saturation flow (≈ 1.9–2.1 s between cars)"),
    "tiempo_perdido_s": Reference(1.0, 3.0, "HCM: 2 s per phase by default"),
    "capacidad_marcha_veh_h": Reference(
        1700, 2400, "capacity per lane in continuous flow: ≈ 1,800–2,000 on urban streets, up to 2,400 on "
                    "motorways (HCM)"),
    "intervalo_marcha_s": Reference(1.5, 2.1, "headway between cars at capacity (≈ 1.5–2 s in the literature)"),
    "densidad_embotellamiento_veh_km": Reference(
        115, 160, "120–160 veh/km/lane in stopped queues (≈ 6.5–8 m between fronts)"),
}  # fmt: skip


def _single(base: SimConfig, k: int, **kw) -> SimConfig:
    """A single lane with only type k, without what is not part of the type (stops, bottlenecks, bumps, pedestrians,
    exit queue, initial occupancy). The lane's limit is that of the type's lane in the configuration."""
    spec = replace(base.specs[k], bottleneck_prob=0.0, stop_position=None, lane=None, exclusive=False)
    limit = base.lane_max_kmh[base.specs[k].lane or 0]
    return SimConfig(specs=(spec,), lanes=1, lane_speed_limit=limit if np.isfinite(limit) else None,
                     behavior=base.behavior, time_scale=10.0, **kw)  # fmt: skip


def discharge_config(base: SimConfig, k: int) -> SimConfig:
    spec = base.specs[k]
    spacing = spec.length + 3 * spec.length_std + spec.gap_stop
    length = max(150.0, (QUEUE + 10) * spacing)
    green = QUEUE * 6.0 + 30.0
    return _single(base, k, length=length, rates=(60.0,), red=RED_S, green=green, yellow=0.0, start_phase="red",
                   run=(RED_S + green) / 10.0)  # fmt: skip


def discharge_replica(cfg: SimConfig, seed: np.random.SeedSequence) -> np.ndarray | None:
    """Headways (s) between the line crossings of the first QUEUE vehicles of the queue, from the green; None if
    the queue did not reach QUEUE stopped vehicles."""
    sim = Simulation(cfg, np.random.default_rng(seed))
    green_at = round(cfg.red / DT)
    while sim.tick < green_at:
        sim.step()
    if np.count_nonzero(sim.stopped[: sim.n]) < QUEUE:
        return None
    times: list[float] = []
    before = int(sim.cum_veh.sum())
    while sim.tick < cfg.n_ticks and len(times) < QUEUE:
        sim.step()
        now = int(sim.cum_veh.sum())
        times += [(sim.tick - green_at) * DT] * (now - before)
        before = now
    if len(times) < QUEUE:
        return None
    return np.diff(np.concatenate(([0.0], times[:QUEUE])))


def flow_config(base: SimConfig, k: int, rate: float, seconds: float) -> SimConfig:
    return _single(base, k, length=FLOW_LENGTH, rates=(float(rate),), traffic_light=False, run=seconds / 10.0)


def flow_replica(cfg: SimConfig, seed: np.random.SeedSequence) -> tuple[float, float, int]:
    """(flow at the exit in veh/h, mean density in the middle third in veh/km, vehicles in the entry queue at the
    end), without the first quarter of the run."""
    sim = Simulation(cfg, np.random.default_rng(seed))
    warm = cfg.n_ticks // 4
    lo, hi = cfg.length / 3, 2 * cfg.length / 3
    count = 0
    crossed0 = 0
    for tick in range(cfg.n_ticks):
        sim.step()
        if tick == warm:
            crossed0 = int(sim.cum_veh.sum())
        if tick >= warm:
            x = sim.x[: sim.n]
            count += int(np.count_nonzero((x >= lo) & (x < hi)))
    steps = cfg.n_ticks - warm
    seconds = steps * DT
    flow = (int(sim.cum_veh.sum()) - crossed0) / seconds * 3600.0
    density = count / steps / ((hi - lo) / 1000.0)
    return flow, density, int(sim.queue_len.sum())


def _run(tasks: list[tuple], workers: int) -> list:
    if workers <= 1:
        return [fn(cfg, seed) for fn, cfg, seed in tasks]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_call, tasks))


def _call(task: tuple):
    fn, cfg, seed = task
    return fn(cfg, seed)


def measure(
    base: SimConfig, vehicle: str, queues: int = 30, seed: int = 0, workers: int = 0,
    rates: tuple[float, ...] | None = None, flow_seconds: float | None = None,
) -> dict:  # fmt: skip
    """The three measurements for the type `vehicle` (its key, e.g. "car"): `queues` queues with different seeds and a
    run of `flow_seconds` simulated s per demand of `rates` veh/min (FLOW_RATES and FLOW_SECONDS by default)."""
    rates = tuple(FLOW_RATES if rates is None else rates)
    flow_seconds = FLOW_SECONDS if flow_seconds is None else flow_seconds
    keys = [s.key for s in base.specs]
    if vehicle not in keys:
        raise ConfigError(f"there is no vehicle type {vehicle!r} (keys: {', '.join(keys)})")
    if queues < 2:
        raise ConfigError("at least 2 queues are needed")
    k = keys.index(vehicle)
    spec = base.specs[k]
    if spec.speed_kmh <= 0:
        raise ConfigError(f"the vehicle type {vehicle!r} has no speed")
    seeds = np.random.SeedSequence(seed).spawn(queues + len(rates))
    dcfg = discharge_config(base, k)
    tasks = [(discharge_replica, dcfg, seeds[i]) for i in range(queues)]
    tasks += [(flow_replica, flow_config(base, k, rate, flow_seconds), seeds[queues + j]) for j, rate in enumerate(rates)]
    workers = min(workers or os.process_cpu_count() or 1, len(tasks))
    results = _run(tasks, workers)
    heads = [h for h in results[:queues] if h is not None]
    if len(heads) < 2:
        raise ConfigError(f"no queue of {QUEUE} stopped {spec.name} vehicles was formed: check their speed and length")
    H = np.array(heads)
    by_pos = H.mean(axis=0)
    sat = float(by_pos[SATURATION_FROM - 1:].mean())
    lost = float((by_pos[:LOST_POSITIONS] - sat).sum())
    flows = results[queues:]
    q = np.array([f for f, _, _ in flows])
    kd = np.array([d for _, d, _ in flows])
    cap = float(q.max())
    jam = 1000.0 / (spec.length + spec.gap_stop)
    return {
        "tipo": vehicle, "nombre": spec.name, "colas": len(heads),
        "intervalo_por_posicion_s": by_pos, "intervalo_por_posicion_sd_s": H.std(axis=0, ddof=1),
        "intervalo_saturacion_s": sat, "flujo_saturacion_veh_h": 3600.0 / sat, "tiempo_perdido_s": lost,
        "demanda_veh_min": list(rates), "flujo_veh_h": q, "densidad_veh_km": kd,
        "velocidad_kmh": np.divide(q, kd, out=np.full(q.size, np.nan), where=kd > 0),
        "cola_entrada": [c for _, _, c in flows],
        "capacidad_marcha_veh_h": cap, "intervalo_marcha_s": 3600.0 / cap if cap > 0 else None,
        "densidad_embotellamiento_veh_km": jam,
        "s_simulados": len(heads) * dcfg.sim_seconds + len(rates) * flow_seconds,
        "parametros": {"speed_kmh": spec.speed_kmh, "limite_kmh": dcfg.lane_max_kmh[0], "largo_m": spec.length,
                       "gap_run_m": spec.gap_run, "gap_stop_m": spec.gap_stop, "accel": spec.accel,
                       "decel": spec.decel, "reaccion_s": _reaction_text(base)},
    }  # fmt: skip


def _reaction_text(cfg: SimConfig) -> str:
    b = cfg.behavior
    if b.reaction_std is None:
        return f"U[{b.reaction_min:g}, {b.reaction_max:g}]"
    mean = b.reaction_mean if b.reaction_mean is not None else (b.reaction_min + b.reaction_max) / 2
    return f"N({mean:g}, {b.reaction_std:g}) in [{b.reaction_min:g}, {b.reaction_max:g}]"


def compare(m: dict) -> list[dict]:
    """Each measurement against its reference: value, range, source and whether it is "dentro" (inside), "abajo" (below)
    or "arriba" (above) (None if the type is not a car: the references are for cars)."""
    rows = []
    for key, ref in REFERENCES.items():
        value = m[key]
        state = None
        if m["tipo"] == "car" and value is not None:
            state = "abajo" if value < ref.low else "arriba" if value > ref.high else "dentro"
        rows.append({"medida": key, "valor": value, "referencia": [ref.low, ref.high], "fuente": ref.source,
                     "estado": state})  # fmt: skip
    return rows


LABELS = {
    "flujo_saturacion_veh_h": ("Saturation flow on green", "veh/h/lane", "{:,.0f}"),
    "intervalo_saturacion_s": ("Saturation headway", "s", "{:.2f}"),
    "tiempo_perdido_s": ("Start-up lost time", "s", "{:.1f}"),
    "capacidad_marcha_veh_h": ("Capacity in continuous flow", "veh/h/lane", "{:,.0f}"),
    "intervalo_marcha_s": ("Headway between vehicles at capacity", "s", "{:.2f}"),
    "densidad_embotellamiento_veh_km": ("Jam density", "veh/km/lane", "{:,.0f}"),
}


def _value(v: float | None, unit_: str, none: str) -> str:
    return none if v is None else f"{v:g} {unit_}"


def summary_text(m: dict, rows: list[dict]) -> str:
    p = m["parametros"]
    lines = [
        f"Calibration of {m['nombre']} ({m['tipo']}): speed {p['speed_kmh']:g} km/h (lane at {p['limite_kmh']:g} km/h), "
        f"length {p['largo_m']:g} m, gap_run {p['gap_run_m']:g} m, gap_stop {p['gap_stop_m']:g} m, "
        f"accel {_value(p['accel'], 'm/s²', 'instantaneous')}, decel {_value(p['decel'], 'm/s²', 'dead stop')}, "
        f"reaction {p['reaccion_s']} s",
        "",
        f"{'measurement':<40}{'simulator':>14}{'reference':>20}  ",
    ]  # fmt: skip
    for r in rows:
        label, unit_, fmt = LABELS[r["medida"]]
        value = "—" if r["valor"] is None else fmt.format(r["valor"])
        ref = f"{fmt.format(r['referencia'][0])}–{fmt.format(r['referencia'][1])}"
        mark = {"dentro": "ok", "abajo": "LOW", "arriba": "HIGH", None: ""}[r["estado"]]
        lines.append(f"{label + ' (' + unit_ + ')':<40}{value:>14}{ref:>20}  {mark}")
    lines += ["", "Headway per queue position (s, mean of " + f"{m['colas']} queues): "
              + " ".join(f"{v:.2f}" for v in m["intervalo_por_posicion_s"])]  # fmt: skip
    lines.append("Continuous flow, demand → flow, density, speed: " + " · ".join(
        f"{d} veh/min → {q:,.0f} veh/h, {k:,.0f} veh/km, {v:,.0f} km/h"
        for d, q, k, v in zip(m["demanda_veh_min"], m["flujo_veh_h"], m["densidad_veh_km"], m["velocidad_kmh"])))  # fmt: skip
    if m["tipo"] != "car":
        lines.append("The references are for cars: for this type the values are only reported.")
    lines += ["", "Sources of the references:"] + [f"  {LABELS[r['medida']][0]}: {r['fuente']}" for r in rows]
    return "\n".join(lines)


PLOT_NAME = "calibration.png"
TABLE_CSV = "calibration.csv"
QUEUE_CSV = "queue_headways.csv"
FLOW_CSV = "continuous_flow.csv"
SUMMARY_NAME = "summary.txt"


def results_metrics(m: dict, rows: list[dict]) -> dict:
    """`results.json` metrics: each measurement with its reference and the full measurements."""
    return {"tipo": m["tipo"], "medidas": rows, **{k: v for k, v in m.items() if k not in ("tipo",)}}


def write_outputs(folder, m: dict, rows: list[dict]) -> str:
    """Tables and summary; returns the summary as text."""
    import csv

    with open(folder / TABLE_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["measurement", "value", "reference_min", "reference_max", "status", "source"])
        for r in rows:
            w.writerow([r["medida"], "" if r["valor"] is None else f"{r['valor']:.6g}", r["referencia"][0],
                        r["referencia"][1], r["estado"] or "", r["fuente"]])  # fmt: skip
    with open(folder / QUEUE_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["position", "headway_s_mean", "headway_s_sd"])
        for i, (mean, sd) in enumerate(zip(m["intervalo_por_posicion_s"], m["intervalo_por_posicion_sd_s"]), 1):
            w.writerow([i, f"{mean:.4f}", f"{sd:.4f}"])
    with open(folder / FLOW_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["demand_veh_min", "flow_veh_h", "density_veh_km", "speed_kmh", "final_entry_queue"])
        for row in zip(m["demanda_veh_min"], m["flujo_veh_h"], m["densidad_veh_km"], m["velocidad_kmh"],
                       m["cola_entrada"]):  # fmt: skip
            w.writerow([row[0], f"{row[1]:.1f}", f"{row[2]:.2f}", f"{row[3]:.2f}", row[4]])
    text = summary_text(m, rows)
    (folder / SUMMARY_NAME).write_text(text + "\n", encoding="utf-8")
    return text
