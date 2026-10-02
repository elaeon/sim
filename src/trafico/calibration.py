"""Calibración: mide cómo se comporta un tipo de vehículo de la configuración en situaciones estándar de ingeniería de
tránsito y lo compara con valores de referencia publicados (`trafico-calibrar`).

Tres mediciones, cada una en un tramo de un carril con solo ese tipo (sin topes, paradas, cuellos de botella ni
peatones), con su velocidad, largos, gaps, aceleración, frenado y la conducta ([behavior]) de la configuración:

  * Descarga de una cola en verde: una cola detenida ante un rojo largo arranca al ponerse en verde; se mide el
    intervalo entre cruces sucesivos de la línea (frente del vehículo) por posición en la cola. De ahí salen el
    intervalo de saturación (media de las posiciones `SATURATION_FROM` en adelante), el flujo de saturación
    (3600 / intervalo) y el tiempo perdido al arrancar (HCM: Σ de las primeras 4 posiciones de intervalo − saturación).
  * Flujo continuo (sin semáforo) con demanda creciente: flujo a la salida y densidad media en el tercio central del
    tramo. La capacidad es el mayor flujo; de ahí el intervalo en marcha (3600 / capacidad).
  * Densidad de embotellamiento: 1000 / (largo medio + gap_stop), vehículos por km de carril en una cola detenida.

Las referencias son de autos (HCM 6.ª ed. y literatura de flujo vehicular); para otros tipos se informan los
valores sin compararlos. No son datos locales: si hay aforos de la calle, valen más que estos rangos.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace

import numpy as np

from trafico.config import DT, SimConfig
from trafico.engine import Simulation
from trafico.settings import ConfigError

QUEUE = 15  # posiciones de la cola que se miden
SATURATION_FROM = 5  # el intervalo de saturación es la media desde la 5.ª posición (índice 4) en adelante
LOST_POSITIONS = 4  # HCM: el tiempo perdido al arrancar se cuenta en los primeros 4 vehículos
RED_S = 120.0  # s de rojo para formar la cola
FLOW_RATES = (10, 20, 30, 45, 60, 90, 120)  # veh/min de demanda en flujo continuo
FLOW_LENGTH = 600.0  # m del tramo en flujo continuo
FLOW_SECONDS = 900.0  # s simulados por demanda (el primer cuarto no cuenta)


@dataclass(frozen=True, slots=True)
class Reference:
    low: float
    high: float
    source: str


# Rangos para autos. Ver el docstring del módulo y la documentación (docs/configuracion.html#calibracion).
REFERENCES = {
    "flujo_saturacion_veh_h": Reference(
        1700, 1950, "HCM 6.ª ed.: flujo de saturación base de 1,900 veh/h/carril (1,750 en ciudades de menos de "
                    "250 mil habitantes); los aforos urbanos suelen quedar algo por debajo"),
    "intervalo_saturacion_s": Reference(1.85, 2.1, "el inverso del flujo de saturación (≈ 1.9–2.1 s entre autos)"),
    "tiempo_perdido_s": Reference(1.0, 3.0, "HCM: 2 s por fase por omisión"),
    "capacidad_marcha_veh_h": Reference(
        1700, 2400, "capacidad por carril en flujo continuo: ≈ 1,800–2,000 en calles urbanas, hasta 2,400 en "
                    "autopistas (HCM)"),
    "intervalo_marcha_s": Reference(1.5, 2.1, "intervalo entre autos a capacidad (≈ 1.5–2 s en la literatura)"),
    "densidad_embotellamiento_veh_km": Reference(
        115, 160, "120–160 veh/km/carril en colas detenidas (≈ 6.5–8 m entre frentes)"),
}  # fmt: skip


def _single(base: SimConfig, k: int, **kw) -> SimConfig:
    """Un carril con solo el tipo k, sin lo que no es del tipo (paradas, cuellos de botella, topes, peatones, cola de
    salida, ocupación inicial). El límite del carril es el del carril del tipo en la configuración."""
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
    """Intervalos (s) entre los cruces de la línea de los primeros QUEUE vehículos de la cola, desde el verde; None si
    la cola no llegó a QUEUE vehículos detenidos."""
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
    """(flujo a la salida en veh/h, densidad media en el tercio central en veh/km, vehículos en la cola de entrada al
    final), sin el primer cuarto de la corrida."""
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
    return flow, density, sum(len(q) for q in sim.queues)


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
    """Las tres mediciones para el tipo `vehicle` (su clave, p. ej. "car"): `queues` colas con distinta semilla y una
    corrida de `flow_seconds` s simulados por demanda de `rates` veh/min (por omisión, FLOW_RATES y FLOW_SECONDS)."""
    rates = tuple(FLOW_RATES if rates is None else rates)
    flow_seconds = FLOW_SECONDS if flow_seconds is None else flow_seconds
    keys = [s.key for s in base.specs]
    if vehicle not in keys:
        raise ConfigError(f"no existe el tipo {vehicle!r} (claves: {', '.join(keys)})")
    if queues < 2:
        raise ConfigError("se necesitan al menos 2 colas")
    k = keys.index(vehicle)
    spec = base.specs[k]
    if spec.speed_kmh <= 0:
        raise ConfigError(f"el tipo {vehicle!r} no tiene velocidad")
    seeds = np.random.SeedSequence(seed).spawn(queues + len(rates))
    dcfg = discharge_config(base, k)
    tasks = [(discharge_replica, dcfg, seeds[i]) for i in range(queues)]
    tasks += [(flow_replica, flow_config(base, k, rate, flow_seconds), seeds[queues + j]) for j, rate in enumerate(rates)]
    workers = min(workers or os.process_cpu_count() or 1, len(tasks))
    results = _run(tasks, workers)
    heads = [h for h in results[:queues] if h is not None]
    if len(heads) < 2:
        raise ConfigError(f"no se formó una cola de {QUEUE} {spec.name}s detenidos: revisa su velocidad y su largo")
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
    return f"N({mean:g}, {b.reaction_std:g}) en [{b.reaction_min:g}, {b.reaction_max:g}]"


def compare(m: dict) -> list[dict]:
    """Cada medida contra su referencia: valor, rango, fuente y si queda "dentro", "abajo" o "arriba" (None si el tipo no
    es un auto: las referencias son de autos)."""
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
    "flujo_saturacion_veh_h": ("Flujo de saturación en verde", "veh/h/carril", "{:,.0f}"),
    "intervalo_saturacion_s": ("Intervalo de saturación", "s", "{:.2f}"),
    "tiempo_perdido_s": ("Tiempo perdido al arrancar", "s", "{:.1f}"),
    "capacidad_marcha_veh_h": ("Capacidad en flujo continuo", "veh/h/carril", "{:,.0f}"),
    "intervalo_marcha_s": ("Intervalo entre vehículos a capacidad", "s", "{:.2f}"),
    "densidad_embotellamiento_veh_km": ("Densidad de embotellamiento", "veh/km/carril", "{:,.0f}"),
}


def _value(v: float | None, unit_: str, none: str) -> str:
    return none if v is None else f"{v:g} {unit_}"


def summary_text(m: dict, rows: list[dict]) -> str:
    p = m["parametros"]
    lines = [
        f"Calibración de {m['nombre']} ({m['tipo']}): velocidad {p['speed_kmh']:g} km/h (carril a {p['limite_kmh']:g} km/h), "
        f"largo {p['largo_m']:g} m, gap_run {p['gap_run_m']:g} m, gap_stop {p['gap_stop_m']:g} m, "
        f"accel {_value(p['accel'], 'm/s²', 'instantánea')}, decel {_value(p['decel'], 'm/s²', 'en seco')}, "
        f"reacción {p['reaccion_s']} s",
        "",
        f"{'medida':<40}{'simulador':>14}{'referencia':>20}  ",
    ]  # fmt: skip
    for r in rows:
        label, unit_, fmt = LABELS[r["medida"]]
        value = "—" if r["valor"] is None else fmt.format(r["valor"])
        ref = f"{fmt.format(r['referencia'][0])}–{fmt.format(r['referencia'][1])}"
        mark = {"dentro": "ok", "abajo": "BAJO", "arriba": "ALTO", None: ""}[r["estado"]]
        lines.append(f"{label + ' (' + unit_ + ')':<40}{value:>14}{ref:>20}  {mark}")
    lines += ["", "Intervalo por posición en la cola (s, media de " + f"{m['colas']} colas): "
              + " ".join(f"{v:.2f}" for v in m["intervalo_por_posicion_s"])]  # fmt: skip
    lines.append("Flujo continuo, demanda → flujo, densidad, velocidad: " + " · ".join(
        f"{d} veh/min → {q:,.0f} veh/h, {k:,.0f} veh/km, {v:,.0f} km/h"
        for d, q, k, v in zip(m["demanda_veh_min"], m["flujo_veh_h"], m["densidad_veh_km"], m["velocidad_kmh"])))  # fmt: skip
    if m["tipo"] != "car":
        lines.append("Las referencias son de autos: para este tipo solo se informan los valores.")
    lines += ["", "Fuentes de las referencias:"] + [f"  {LABELS[r['medida']][0]}: {r['fuente']}" for r in rows]
    return "\n".join(lines)


PLOT_NAME = "calibracion.png"
TABLE_CSV = "calibracion.csv"
QUEUE_CSV = "intervalos_cola.csv"
FLOW_CSV = "flujo_continuo.csv"
SUMMARY_NAME = "resumen.txt"


def results_metrics(m: dict, rows: list[dict]) -> dict:
    """Métricas de `resultados.json`: cada medida con su referencia y las mediciones completas."""
    return {"tipo": m["tipo"], "medidas": rows, **{k: v for k, v in m.items() if k not in ("tipo",)}}


def write_outputs(folder, m: dict, rows: list[dict]) -> str:
    """Tablas y resumen; devuelve el resumen en texto."""
    import csv

    with open(folder / TABLE_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["medida", "valor", "referencia_min", "referencia_max", "estado", "fuente"])
        for r in rows:
            w.writerow([r["medida"], "" if r["valor"] is None else f"{r['valor']:.6g}", r["referencia"][0],
                        r["referencia"][1], r["estado"] or "", r["fuente"]])  # fmt: skip
    with open(folder / QUEUE_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["posicion", "intervalo_s_media", "intervalo_s_sd"])
        for i, (mean, sd) in enumerate(zip(m["intervalo_por_posicion_s"], m["intervalo_por_posicion_sd_s"]), 1):
            w.writerow([i, f"{mean:.4f}", f"{sd:.4f}"])
    with open(folder / FLOW_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["demanda_veh_min", "flujo_veh_h", "densidad_veh_km", "velocidad_kmh", "cola_entrada_final"])
        for row in zip(m["demanda_veh_min"], m["flujo_veh_h"], m["densidad_veh_km"], m["velocidad_kmh"],
                       m["cola_entrada"]):  # fmt: skip
            w.writerow([row[0], f"{row[1]:.1f}", f"{row[2]:.2f}", f"{row[3]:.2f}", row[4]])
    text = summary_text(m, rows)
    (folder / SUMMARY_NAME).write_text(text + "\n", encoding="utf-8")
    return text
