"""Comparación de variantes: el escenario de una configuración con distintos largos de tramo y
repartos del semáforo (`uv run trafico-variantes`).

Cada combinación de largo y semáforo es un escenario que corre `replicas` réplicas con las mismas
semillas que `trafico` (SeedSequence.spawn), así que las variantes comparten los números
aleatorios. De cada réplica se guarda la velocidad media de cada carril en toda la corrida y los
vehículos en la cola de entrada de cada tipo en cada muestra.
"""

from __future__ import annotations

import csv
import json
import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from trafico.config import DT, SimConfig
from trafico.engine import Simulation
from trafico.settings import ConfigError, RunOptions, validate_config

PLOT_NAME = "variantes_semaforo.png"
DATA_NAME = "datos.npz"
META_NAME = "variantes.json"
BASE_CONFIG_NAME = "config_base.toml"  # no se llama config.toml: la carpeta no es una corrida de `trafico`
SPEED_CSV = "velocidad_por_carril.csv"
QUEUE_CSV = "cola_entrada.csv"
SUMMARY_NAME = "resumen.txt"


@dataclass(frozen=True, slots=True)
class Variants:
    """Qué se compara y cómo."""

    lengths: tuple[float, ...]  # m de cada tramo
    lights: tuple[tuple[float, float], ...]  # (rojo, verde) en s de cada semáforo
    lanes: tuple[int, ...] | None  # carriles de la configuración base que se conservan (en orden); None = todos
    without: tuple[str, ...]  # claves de los tipos que no participan
    queue_types: tuple[str, ...]  # claves de los tipos que se cuentan en la cola de entrada
    replicas: int
    run: float  # s de proceso (× time_scale = s simulados)


def light_name(red: float, green: float) -> str:
    """Nombre de un reparto del semáforo: «rojo > verde», «rojo = verde» o «verde > rojo»."""
    if abs(red - green) < DT / 2:
        return "rojo = verde"
    return "rojo > verde" if red > green else "verde > rojo"


def reduce_config(base: SimConfig, lanes: tuple[int, ...] | None, without: tuple[str, ...]) -> SimConfig:
    """La configuración base con solo los carriles `lanes` (renumerados desde 0 en ese orden, con sus
    límites, congestión, ocupación inicial y carriles de cuello de botella) y sin los tipos `without`.

    Los tipos que no participan (en `without` o con tasa 0) no pesan en la validación: pierden su
    carril fijo, su exclusividad, su parada y sus detenciones. Un tipo que participa con carril fijo
    fuera de `lanes` es un error."""
    keys = [s.key for s in base.specs]
    for key in without:
        if key not in keys:
            raise ConfigError(f"--sin: no existe el tipo {key!r} (claves: {', '.join(keys)})")
    old = tuple(range(base.lanes)) if lanes is None else lanes
    for ln in old:
        if not 0 <= ln < base.lanes:
            raise ConfigError(f"--carriles: cada carril debe estar entre 0 y {base.lanes - 1}")
    if len(set(old)) != len(old):
        raise ConfigError("--carriles: hay carriles repetidos")
    new_of = {ln: i for i, ln in enumerate(old)}

    specs, rates = [], []
    for spec, rate in zip(base.specs, base.rates):
        if spec.key in without:
            rate = 0.0
        if rate == 0:
            spec = replace(spec, lane=None, exclusive=False, stop_position=None, bottleneck_prob=0.0)
        elif spec.lane is not None:
            if spec.lane not in new_of:
                raise ConfigError(f"[vehicles.{spec.key}] lane = {spec.lane} no está en --carriles "
                                  f"({', '.join(map(str, old))}); agrega ese carril o saca el tipo con --sin")  # fmt: skip
            spec = replace(spec, lane=new_of[spec.lane])
        specs.append(spec)
        rates.append(rate)
    if not any(rates):
        raise ConfigError("no queda ningún tipo con tasa > 0 (revisa --sin y [demand])")

    bn_lanes = base.bottleneck.stop_lanes
    if bn_lanes is not None:
        bn_lanes = tuple(new_of[ln] for ln in bn_lanes if ln in new_of)  # () = en ninguno
    return replace(
        base, lanes=len(old), specs=tuple(specs), rates=tuple(rates),
        lane_speed_limit=None if base.lane_speed_limit is None else tuple(base.lane_max_kmh[i] for i in old),
        behavior=replace(base.behavior, congestion_factor=tuple(base.lane_congestion[i] for i in old)),
        initial_occupancy=tuple(base.lane_initial_occupancy[i] for i in old),
        bottleneck=replace(base.bottleneck, stop_lanes=bn_lanes),
    )  # fmt: skip


def scenario(reduced: SimConfig, v: Variants, length: float, red: float, green: float) -> SimConfig:
    """Un escenario validado: la configuración reducida con ese largo, semáforo y duración."""
    cfg = replace(reduced, length=length, red=red, green=green, run=v.run)
    try:
        validate_config(cfg, RunOptions(replicas=v.replicas))
    except ConfigError as exc:
        raise ConfigError(f"tramo de {length:g} m, semáforo {red:g}/{green:g}: {exc}") from None
    return cfg


def run_variant_replica(cfg: SimConfig, seed: np.random.SeedSequence) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Una réplica: (velocidad media de cada carril en km/h, incluidos los detenidos; vehículos de
    cada tipo en la cola de entrada en cada muestra (tipos × muestras); vehículos que cruzan por tipo)."""
    sim = Simulation(cfg, np.random.default_rng(seed))
    queue = np.zeros((cfg.n_types, cfg.n_samples), np.float32)
    k = 0
    for _ in range(cfg.n_ticks):
        sim.step()
        if sim.tick % cfg.sample_ticks == 0 and k < cfg.n_samples:
            for q in sim.queues:
                for vt, *_ in q:
                    queue[vt, k] += 1
            k += 1
    rec = sim.recorder
    with np.errstate(invalid="ignore", divide="ignore"):
        speed = rec.lane_dist[: rec.count].sum(axis=0) / rec.lane_time[: rec.count].sum(axis=0) * 3.6
    return speed, queue, sim.cum_veh.copy()


def run_variants(
    base: SimConfig, v: Variants, seed: int, workers: int = 0,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, np.ndarray]:  # fmt: skip
    """Corre todos los escenarios. Devuelve arreglos indexados por (largo, semáforo, réplica, …)."""
    reduced = reduce_config(base, v.lanes, v.without)
    configs = [[scenario(reduced, v, L, r, g) for r, g in v.lights] for L in v.lengths]  # valida todo antes
    seeds = np.random.SeedSequence(seed).spawn(v.replicas)
    shape = (len(v.lengths), len(v.lights), v.replicas)
    cfg0 = configs[0][0]
    speed = np.full(shape + (reduced.lanes,), np.nan)
    queue = np.zeros(shape + (reduced.n_types, cfg0.n_samples), np.float32)
    crossed = np.zeros(shape + (reduced.n_types,))
    tasks = [(li, vi, r) for li in range(shape[0]) for vi in range(shape[1]) for r in range(shape[2])]
    workers = min(workers or os.process_cpu_count() or 1, len(tasks))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_variant_replica, configs[li][vi], seeds[r]): (li, vi, r) for li, vi, r in tasks}
        for n, fut in enumerate(as_completed(futures), 1):
            idx = futures[fut]
            speed[idx], queue[idx], crossed[idx] = fut.result()
            if progress:
                progress(n, len(tasks))
    t = (np.arange(cfg0.n_samples) + 1) * cfg0.sample_ticks * DT
    return {"speed": speed, "queue": queue, "crossed": crossed, "t": t}


def metadata(base: SimConfig, v: Variants, seed: int, argv: list[str]) -> dict:
    """Lo necesario para describir, graficar y repetir la comparación."""
    reduced = reduce_config(base, v.lanes, v.without)
    keys = [s.key for s in reduced.specs]
    limits = reduced.lane_max_kmh
    old = list(range(base.lanes)) if v.lanes is None else list(v.lanes)
    return {
        "comando": "uv run trafico-variantes " + " ".join(argv),
        "semilla": seed,
        "largos_m": list(v.lengths),
        "semaforos": [{"rojo_s": r, "verde_s": g, "nombre": light_name(r, g)} for r, g in v.lights],
        "carriles_base": old,
        "carriles": [
            f"carril {i} ({limits[i]:g} km/h)" if np.isfinite(limits[i]) else f"carril {i}" for i in range(reduced.lanes)
        ],
        "sin": list(v.without),
        "tipos": keys,
        "nombres": [s.name for s in reduced.specs],
        "tasas": list(reduced.rates),
        "cola_tipos": [keys.index(k) for k in v.queue_types],
        "replicas": v.replicas,
        "run_s": v.run,
        "time_scale": base.time_scale,
        "s_simulados": v.run * base.time_scale,
        "amarillo_s": base.yellow,
    }


def write_outputs(folder: Path, meta: dict, data: dict[str, np.ndarray]) -> str:
    """Tablas, datos crudos y parámetros; devuelve el resumen en texto."""
    np.savez(folder / DATA_NAME, **data)
    (folder / META_NAME).write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    speed, crossed, t = data["speed"], data["crossed"], data["t"]
    queue = data["queue"][:, :, :, meta["cola_tipos"]].sum(axis=3)  # (largo, semáforo, réplica, muestra)
    queue_label = ", ".join(meta["nombres"][k] for k in meta["cola_tipos"])
    lights = meta["semaforos"]
    with open(folder / SPEED_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["tramo_m", "semaforo", "rojo_s", "verde_s", "carril", "velocidad_kmh_media", "velocidad_kmh_sd",
                    "cola_final_media", "cola_max_media", "cruzan_media"])  # fmt: skip
        for li, L in enumerate(meta["largos_m"]):
            for vi, light in enumerate(lights):
                q = queue[li, vi]
                cross = crossed[li, vi][:, meta["cola_tipos"]].sum(axis=1)
                for lane in range(speed.shape[3]):
                    s = speed[li, vi, :, lane]
                    w.writerow([f"{L:g}", light["nombre"], f"{light['rojo_s']:g}", f"{light['verde_s']:g}", lane,
                                f"{np.nanmean(s):.2f}", f"{np.nanstd(s, ddof=1) if s.size > 1 else 0:.2f}",
                                f"{q[:, -1].mean():.1f}", f"{q.max(axis=1).mean():.1f}", f"{cross.mean():.1f}"])  # fmt: skip
    with open(folder / QUEUE_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        cols = [(li, vi) for li in range(len(meta["largos_m"])) for vi in range(len(lights))]
        w.writerow(["t_sim_s"] + [f"{meta['largos_m'][li]:g}m_{lights[vi]['rojo_s']:g}-{lights[vi]['verde_s']:g}_media"
                                  for li, vi in cols])  # fmt: skip
        means = queue.mean(axis=2)
        for i, ti in enumerate(t):
            w.writerow([f"{ti:g}"] + [f"{means[li, vi, i]:.2f}" for li, vi in cols])
    return summary_text(meta, speed, queue, queue_label)


def summary_text(meta: dict, speed: np.ndarray, queue: np.ndarray, queue_label: str) -> str:
    lanes = meta["carriles"]
    head = ["tramo", "semáforo"] + lanes + [f"cola final ({queue_label})"]
    rows = []
    for li, L in enumerate(meta["largos_m"]):
        for vi, light in enumerate(meta["semaforos"]):
            name = f"{light['nombre']} ({light['rojo_s']:g}/{light['verde_s']:g})"
            speeds = [f"{np.nanmean(speed[li, vi, :, k]):.1f} km/h" for k in range(len(lanes))]
            rows.append([f"{L:g} m", name] + speeds + [f"{queue[li, vi, :, -1].mean():.1f}"])
    widths = [max(len(r[i]) for r in [head] + rows) for i in range(len(head))]
    lines = ["  ".join(c.rjust(w) if i > 1 else c.ljust(w) for i, (c, w) in enumerate(zip(r, widths)))
             for r in [head] + rows]  # fmt: skip
    return "\n".join(["Velocidad media por carril (toda la corrida, incluye detenidos) y vehículos en la cola de entrada"
                      " al final (media de las réplicas):", "", *lines])  # fmt: skip


def load(folder: Path) -> tuple[dict, dict[str, np.ndarray]]:
    """Parámetros y datos de una comparación ya corrida."""
    meta_path, data_path = folder / META_NAME, folder / DATA_NAME
    if not (meta_path.is_file() and data_path.is_file()):
        raise ConfigError(f"{folder} no es una carpeta de trafico-variantes (faltan {META_NAME} o {DATA_NAME})")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    with np.load(data_path) as d:
        data = {k: d[k] for k in d.files}
    return meta, data
