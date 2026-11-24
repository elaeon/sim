"""Variant comparison: a configuration's scenario with different segment lengths and
traffic light splits (`uv run traffic-variants`).

Each combination of length and light is a scenario that runs `replicas` replicas with the same
seeds as `traffic` (SeedSequence.spawn), so the variants share the random numbers. From each
replica it keeps the mean speed of each lane over the whole run and the vehicles of each type
in the entry queue at each sample.
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

from traffic.config import DT, Rate, SimConfig
from traffic.engine import Simulation
from traffic.settings import ConfigError, RunOptions, validate_config

PLOT_NAME = "light_variants.png"
DATA_NAME = "data.npz"
META_NAME = "variants.json"
BASE_CONFIG_NAME = "base_config.toml"  # not called config.toml: the folder is not a `traffic` run
SPEED_CSV = "speed_per_lane.csv"
QUEUE_CSV = "entry_queue.csv"
SUMMARY_NAME = "summary.txt"


@dataclass(frozen=True, slots=True)
class Variants:
    """What is compared and how."""

    lengths: tuple[float, ...]  # m of each segment
    lights: tuple[tuple[float, float], ...]  # (red, green) in s of each traffic light
    lanes: tuple[int, ...] | None  # lanes of the base configuration that are kept (in order); None = all
    without: tuple[str, ...]  # keys of the types that do not take part
    queue_types: tuple[str, ...]  # keys of the types counted in the entry queue
    replicas: int
    run: float  # process s (× time_scale = simulated s)


def light_name(red: float, green: float) -> str:
    """Name of a traffic light split: «red > green», «red = green» or «green > red»."""
    if abs(red - green) < DT / 2:
        return "red = green"
    return "red > green" if red > green else "green > red"


def reduce_config(base: SimConfig, lanes: tuple[int, ...] | None, without: tuple[str, ...]) -> SimConfig:
    """The base configuration with only the lanes `lanes` (renumbered from 0 in that order, with their
    limits, initial occupancy, exit queue, lanes without a traffic light, with a bump and with a bottleneck) and
    without the types `without`.

    Types that do not take part (in `without` or with rate 0) do not count in the validation: they lose their
    fixed lane, their exclusivity, their stop and their halts. A participating type with a fixed lane
    outside `lanes` is an error."""
    keys = [s.key for s in base.specs]
    for key in without:
        if key not in keys:
            raise ConfigError(f"--without: there is no vehicle type {key!r} (keys: {', '.join(keys)})")
    old = tuple(range(base.lanes)) if lanes is None else lanes
    for ln in old:
        if not 0 <= ln < base.lanes:
            raise ConfigError(f"--lanes: each lane must be between 0 and {base.lanes - 1}")
    if len(set(old)) != len(old):
        raise ConfigError("--lanes: there are repeated lanes")
    new_of = {ln: i for i, ln in enumerate(old)}

    specs, rates = [], []
    for k, spec in enumerate(base.specs):
        rate = Rate.fixed(0.0) if spec.key in without else base.rate(k)
        if rate.expected == 0:
            spec = replace(spec, lane=None, exclusive=False, stop_position=None, bottleneck_prob=0.0)
        elif spec.lane is not None:
            if spec.lane not in new_of:
                raise ConfigError(f"[vehicles.{spec.key}] lane = {spec.lane} is not in --lanes "
                                  f"({', '.join(map(str, old))}); add that lane or remove the type with --without")  # fmt: skip
            spec = replace(spec, lane=new_of[spec.lane])
        specs.append(spec)
        rates.append(rate)
    if not any(rate.expected > 0 for rate in rates):
        raise ConfigError("no type with a rate > 0 is left (check --without and [demand])")

    bn_lanes = base.bottleneck.stop_lanes
    bumps = []
    for bump in base.speed_bumps:  # each bump keeps its lanes; if none is left, there is no bump
        if bump.lanes is None:
            bumps.append(bump)
            continue
        kept = tuple(new_of[ln] for ln in bump.lanes if ln in new_of)
        if kept:
            bumps.append(replace(bump, lanes=kept))
    if bn_lanes is not None:
        bn_lanes = tuple(new_of[ln] for ln in bn_lanes if ln in new_of)  # () = in none
    return replace(
        base, lanes=len(old), specs=tuple(specs), rate_dists=tuple(rates),
        lane_speed_limit=None if base.lane_speed_limit is None else tuple(base.lane_max_kmh[i] for i in old),
        initial_occupancy=tuple(base.lane_initial_occupancy[i] for i in old),
        exit_capacity=tuple(base.lane_exit_capacity[i] for i in old),
        exit_storage=tuple(base.lane_exit_storage[i] for i in old),
        # Without a traffic light only the lanes that are still exclusive to a participating type remain.
        free_lanes=tuple(new_of[ln] for ln in base.free_lanes
                         if ln in new_of and any(s.exclusive and s.lane == new_of[ln] for s in specs)),  # fmt: skip
        extra_lights=tuple(
            replace(lt, free_lanes=tuple(new_of[ln] for ln in lt.free_lanes
                                         if ln in new_of and any(s.exclusive and s.lane == new_of[ln] for s in specs)))
            for lt in base.extra_lights
        ),  # fmt: skip
        bottleneck=replace(base.bottleneck, stop_lanes=bn_lanes),
        speed_bumps=tuple(bumps),
    )  # fmt: skip


def scenario(reduced: SimConfig, v: Variants, length: float, red: float, green: float) -> SimConfig:
    """A validated scenario: the reduced configuration with that length, light and duration. The `red`/`green` split
    is that of the light at the end of the segment (fixed-cycle, even if it is a pedestrian light in the
    configuration); the intermediate ones keep their phases and position."""
    cfg = replace(reduced, length=length, red=red, green=green, traffic_light=True, light_pedestrian=False, run=v.run)
    try:
        validate_config(cfg, RunOptions(replicas=v.replicas))
    except ConfigError as exc:
        raise ConfigError(f"{length:g} m segment, light {red:g}/{green:g}: {exc}") from None
    return cfg


def run_variant_replica(cfg: SimConfig, seed: np.random.SeedSequence) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One replica: (mean speed of each lane in km/h, including stopped vehicles; vehicles of
    each type in the entry queue at each sample (types × samples); vehicles crossing per type)."""
    sim = Simulation(cfg, np.random.default_rng(seed))
    queue = np.zeros((cfg.n_types, cfg.n_samples), np.float32)
    for k in range(cfg.n_samples):
        sim.advance((k + 1) * cfg.sample_ticks)
        queue[:, k] = sim.queued_types()
    sim.advance(cfg.n_ticks)
    rec = sim.recorder
    with np.errstate(invalid="ignore", divide="ignore"):
        speed = rec.lane_dist[: rec.count].sum(axis=0) / rec.lane_time[: rec.count].sum(axis=0) * 3.6
    return speed, queue, sim.cum_veh.copy()


def run_variants(
    base: SimConfig, v: Variants, seed: int, workers: int = 0,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, np.ndarray]:  # fmt: skip
    """Run every scenario. Returns arrays indexed by (length, light, replica, …)."""
    reduced = reduce_config(base, v.lanes, v.without)
    configs = [[scenario(reduced, v, L, r, g) for r, g in v.lights] for L in v.lengths]  # validates everything first
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
    """What is needed to describe, plot and repeat the comparison."""
    reduced = reduce_config(base, v.lanes, v.without)
    keys = [s.key for s in reduced.specs]
    limits = reduced.lane_max_kmh
    old = list(range(base.lanes)) if v.lanes is None else list(v.lanes)
    return {
        "comando": "uv run traffic-variants " + " ".join(argv),
        "semilla": seed,
        "largos_m": list(v.lengths),
        "semaforos": [{"rojo_s": r, "verde_s": g, "nombre": light_name(r, g)} for r, g in v.lights],
        "carriles_base": old,
        "carriles": [
            f"lane {i} ({limits[i]:g} km/h)" if np.isfinite(limits[i]) else f"lane {i}" for i in range(reduced.lanes)
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
    """Tables, raw data and parameters; returns the summary as text."""
    np.savez(folder / DATA_NAME, **data)
    (folder / META_NAME).write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    speed, crossed, t = data["speed"], data["crossed"], data["t"]
    queue = data["queue"][:, :, :, meta["cola_tipos"]].sum(axis=3)  # (length, light, replica, sample)
    queue_label = ", ".join(meta["nombres"][k] for k in meta["cola_tipos"])
    lights = meta["semaforos"]
    with open(folder / SPEED_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["segment_m", "light", "red_s", "green_s", "lane", "speed_kmh_mean", "speed_kmh_sd",
                    "final_queue_mean", "max_queue_mean", "crossing_mean"])  # fmt: skip
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
        w.writerow(["t_sim_s"] + [f"{meta['largos_m'][li]:g}m_{lights[vi]['rojo_s']:g}-{lights[vi]['verde_s']:g}_mean"
                                  for li, vi in cols])  # fmt: skip
        means = queue.mean(axis=2)
        for i, ti in enumerate(t):
            w.writerow([f"{ti:g}"] + [f"{means[li, vi, i]:.2f}" for li, vi in cols])
    return summary_text(meta, speed, queue, queue_label)


def summary_text(meta: dict, speed: np.ndarray, queue: np.ndarray, queue_label: str) -> str:
    lanes = meta["carriles"]
    head = ["segment", "traffic light"] + lanes + [f"final queue ({queue_label})"]
    rows = []
    for li, L in enumerate(meta["largos_m"]):
        for vi, light in enumerate(meta["semaforos"]):
            name = f"{light['nombre']} ({light['rojo_s']:g}/{light['verde_s']:g})"
            speeds = [f"{np.nanmean(speed[li, vi, :, k]):.1f} km/h" for k in range(len(lanes))]
            rows.append([f"{L:g} m", name] + speeds + [f"{queue[li, vi, :, -1].mean():.1f}"])
    widths = [max(len(r[i]) for r in [head] + rows) for i in range(len(head))]
    lines = ["  ".join(c.rjust(w) if i > 1 else c.ljust(w) for i, (c, w) in enumerate(zip(r, widths)))
             for r in [head] + rows]  # fmt: skip
    return "\n".join(["Mean speed per lane (whole run, including stopped vehicles) and vehicles in the entry queue"
                      " at the end (mean of the replicas):", "", *lines])  # fmt: skip


def results_metrics(meta: dict, data: dict[str, np.ndarray]) -> dict:
    """`results.json` metrics: per segment and light split, the mean speed of each lane and the entry queue
    (final and maximum, mean of the replicas) of the types counted in it."""
    speed, crossed = data["speed"], data["crossed"]
    queue = data["queue"][:, :, :, meta["cola_tipos"]].sum(axis=3)  # (length, light, replica, sample)
    scenarios = []
    for li, length in enumerate(meta["largos_m"]):
        for vi, light in enumerate(meta["semaforos"]):
            q = queue[li, vi]
            scenarios.append({
                "largo_m": length, "semaforo": light,
                "velocidad_kmh_por_carril": [float(np.nanmean(speed[li, vi, :, k])) for k in range(speed.shape[3])],
                "cola_final": float(q[:, -1].mean()), "cola_maxima": float(q.max(axis=1).mean()),
                "cruzan": float(crossed[li, vi][:, meta["cola_tipos"]].sum(axis=1).mean()),
            })  # fmt: skip
    return {"carriles": meta["carriles"], "cola_tipos": [meta["tipos"][k] for k in meta["cola_tipos"]],
            "escenarios": scenarios, "parametros": meta}  # fmt: skip


def load(folder: Path) -> tuple[dict, dict[str, np.ndarray]]:
    """Parameters and data of a comparison that has already run."""
    from traffic.results import existing

    meta_path, data_path = existing(folder, META_NAME), existing(folder, DATA_NAME)
    if not (meta_path.is_file() and data_path.is_file()):
        raise ConfigError(f"{folder} is not a traffic-variants folder ({META_NAME} or {DATA_NAME} is missing)")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    with np.load(data_path) as d:
        data = {k: d[k] for k in d.files}
    return meta, data
