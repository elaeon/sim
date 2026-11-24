"""Comparison of emissions between scenarios of the same street: with and without a bump (or with bumps at different
positions) and, optionally, with and without a traffic light (`uv run traffic-emissions`).

All scenarios run the same replicas with the same seed as `traffic` (SeedSequence.spawn): the arrivals,
passengers and drawn speeds are the same, so the differences come from the bump or the traffic light. For
each scenario it keeps the means across replicas of the flow leaving the segment, the vehicles, times and
speeds per type, and the emissions per type, per km and along the segment.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from traffic.config import POLLUTANT_LABELS, POLLUTANTS, SimConfig, SpeedBump, as_positions
from traffic.emissions import EMIS_BIN, emitted, unit
from traffic.metrics import LANE_EXIT_FLOW
from traffic.runner import run_many
from traffic.settings import ConfigError, RunOptions, validate_config
from traffic.variants import reduce_config

PLOT_NAME = "emission_comparison.png"
DATA_NAME = "data.npz"
META_NAME = "scenarios.json"
BASE_CONFIG_NAME = "base_config.toml"  # not called config.toml: the folder is not a `traffic` run
TABLE_CSV = "emission_comparison.csv"
POSITION_CSV = "emissions_by_position.csv"
SUMMARY_NAME = "summary.txt"


@dataclass(frozen=True, slots=True)
class Scenarios:
    """What is compared and how."""

    # Bumps of each scenario: one position (m), several (tuple) or None / empty = no bump.
    bumps: tuple[float | tuple[float, ...] | None, ...]
    bump_lanes: tuple[int, ...] | None  # lanes of every bump (already renumbered); None = those of each bump
    lights: tuple[bool | None, ...]  # traffic light enabled in each variant; None = as in the configuration
    length: float | None  # m of the segment; None = [road] length
    lanes: tuple[int, ...] | None  # lanes of the base configuration that are kept; None = all
    without: tuple[str, ...]  # keys of the types that do not take part
    replicas: int
    run: float  # process s (× time_scale = simulated s)


def bump_positions(bump: float | tuple[float, ...] | None) -> tuple[float, ...]:
    """Positions (m) of a scenario's bumps, from the entry to the exit (empty = no bump)."""
    return as_positions(bump)


def scenario_label(bump: float | tuple[float, ...] | None, light: bool | None) -> str:
    """Name of the scenario, e.g. «bump at 100 m», «bumps at 35 and 70 m» or, when traffic lights are compared,
    «no bump · no traffic light»."""
    at = bump_positions(bump)
    name = "no bump" if not at else f"{'bumps' if len(at) > 1 else 'bump'} at {' and '.join(f'{p:g}' for p in at)} m"
    if light is not None:
        name += " · with traffic light" if light else " · without traffic light"
    return name


def scenario_bumps(base: SimConfig, positions: tuple[float, ...], lanes: tuple[int, ...] | None) -> tuple[SpeedBump, ...]:
    """A scenario's bumps at those positions: the configuration's bump that is there (with its lanes and pedestrians)
    or, if there is none, a new one like the configuration's first (or without pedestrians, if there is none). `lanes`
    replaces the lanes of all of them; without positions there is no bump or pedestrian crossing."""
    configured = {b.position: b for b in base.bumps}
    template = base.bumps[0] if base.bumps else SpeedBump(0.0)
    bumps = (configured.get(p) or replace(template, position=p) for p in positions)
    return tuple(replace(b, lanes=lanes) if lanes is not None else b for b in bumps)


def build_configs(base: SimConfig, s: Scenarios) -> list[tuple[str, SimConfig]]:
    """(name, validated configuration) of each scenario: each traffic light setting for each bump setting."""
    reduced = reduce_config(base, s.lanes, s.without)
    if s.length is not None:
        reduced = replace(reduced, length=s.length)
    out = []
    for light in s.lights:
        for bump in s.bumps:
            cfg = replace(reduced if light is None else reduced.with_lights(light), run=s.run,
                          speed_bumps=scenario_bumps(reduced, bump_positions(bump), s.bump_lanes))  # fmt: skip
            name = scenario_label(bump, cfg.any_light if len(s.lights) > 1 else None)
            try:
                validate_config(cfg, RunOptions(replicas=s.replicas))
            except ConfigError as exc:
                raise ConfigError(f"scenario «{name}»: {exc}") from None
            out.append((name, cfg))
    if not emitted(out[0][1])[0]:
        raise ConfigError("no participating type emits: [vehicles.<key>.emissions], accel and decel are needed")
    return out


def run_scenarios(
    configs: list[tuple[str, SimConfig]], replicas: int, seed: int, workers: int = 0,
    progress: Callable[[str, int, int], None] | None = None,
) -> dict[str, np.ndarray]:  # fmt: skip
    """Run each scenario with the same seeds. Returns arrays indexed by scenario (mean across
    replicas; «_sd», its standard deviation)."""
    keys = ("crossed_veh", "travel_time", "mean_speed", "veh_km", "emissions", "emissions_free", "emissions_pos")
    out: dict[str, list] = {k: [] for k in (*keys, "crossed_veh_sd", "emissions_sd", "flow")}
    names = [name for name, _ in configs]
    # A single process pool for all replicas of all scenarios (the same seeds in each one).
    aggs = run_many([cfg for _, cfg in configs], replicas, workers, seed,
                    progress=(lambda i, d, n: progress(names[i], d, n)) if progress else None)  # fmt: skip
    for agg in aggs:
        for k in keys:
            out[k].append(agg.summary[k].mean)
        out["crossed_veh_sd"].append(agg.summary["crossed_veh"].std)
        out["emissions_sd"].append(agg.summary["emissions"].std)
        out["flow"].append(np.nansum(agg.series[LANE_EXIT_FLOW].mean, axis=1))  # veh/min leaving, all lanes
    t = aggs[-1].times
    data = {k: np.array(v) for k, v in out.items()}
    data["t"] = np.asarray(t)
    return data


def metadata(base: SimConfig, s: Scenarios, configs: list[tuple[str, SimConfig]], seed: int, argv: list[str]) -> dict:
    """What is needed to describe, plot and repeat the comparison."""
    cfg = configs[0][1]
    pols, types = emitted(cfg)
    active = [k for k, rate in enumerate(cfg.rates) if rate > 0]
    limits = cfg.lane_max_kmh
    return {
        "comando": "uv run traffic-emissions " + " ".join(argv),
        "semilla": seed,
        "escenarios": [name for name, _ in configs],
        "topes_m": [[b.position for b in c.bumps] for _, c in configs],
        "semaforo": [c.any_light for _, c in configs],
        "carriles_tope": sorted({ln for _, c in configs for b in c.bumps for ln in c.bump_lanes(b)}) or None,
        # Pedestrians at the bumps ([[speed_bump]] pedestrian): mean rate (pedestrians/min) and s per crossing of the first
        # one with pedestrians; None = no pedestrians.
        "peatones_tope": next(({"peatones_min": b.pedestrian_crossing.expected, "s_cruce": b.pedestrian_time}
                               for _, c in configs for b in c.crossings), None),  # fmt: skip
        "largo_m": cfg.length,
        "carriles_base": list(range(base.lanes)) if s.lanes is None else list(s.lanes),
        "carriles": [f"lane {i} ({limits[i]:g} km/h)" if np.isfinite(limits[i]) else f"lane {i}"
                     for i in range(cfg.lanes)],  # fmt: skip
        "sin": list(s.without),
        "tipos": [sp.key for sp in cfg.specs],
        "nombres": [sp.name for sp in cfg.specs],
        "tasas": list(cfg.rates),
        "activos": active,
        "emisores": types,
        "contaminantes": [POLLUTANTS[p] for p in pols],
        "velocidad_tope_kmh": {cfg.specs[k].name: cfg.specs[k].speed_bump_kmh for k in active},
        "paradas_m": {cfg.specs[k].name: cfg.specs[k].stop_position for k in active
                      if cfg.specs[k].stop_position is not None},  # fmt: skip
        "semaforo_config": base.lights_label,
        "replicas": s.replicas,
        "run_s": s.run,
        "time_scale": base.time_scale,
        "s_simulados": s.run * base.time_scale,
        "intervalo_posicion_m": EMIS_BIN,
    }


def per_km(data: dict, meta: dict) -> np.ndarray:
    """(scenario, type, pollutant) in each pollutant's unit per km travelled (NaN if it does not emit it)."""
    km = data["veh_km"][:, :, None]
    scale = np.array([unit(pol)[1] for pol in POLLUTANTS])
    with np.errstate(invalid="ignore", divide="ignore"):
        out = data["emissions"] * scale / km
    emits = data["emissions"].sum(axis=0) > 0  # (type, pollutant)
    return np.where(emits[None], out, np.nan)


def write_outputs(folder: Path, meta: dict, data: dict[str, np.ndarray]) -> str:
    """Tables, raw data and parameters; returns the summary as text."""
    np.savez(folder / DATA_NAME, **data)
    (folder / META_NAME).write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    names = meta["nombres"]
    minutes = meta["s_simulados"] / 60
    km = per_km(data, meta)
    pidx = [POLLUTANTS.index(p) for p in meta["contaminantes"]]
    with open(folder / TABLE_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        header = ["scenario", "type", "veh_min", "travel_time_s", "mean_speed_kmh"]
        for p in pidx:
            name = f"{POLLUTANTS[p]}_{unit(POLLUTANTS[p])[0]}_km"
            header += [name, f"{POLLUTANTS[p]}_change_pct", f"{POLLUTANTS[p]}_excess_free_flow_pct"]
        w.writerow(header)
        for si, scen in enumerate(meta["escenarios"]):
            for k in meta["activos"]:
                row = [scen, names[k], f"{data['crossed_veh'][si, k] / minutes:.3f}",
                       f"{data['travel_time'][si, k]:.2f}", f"{data['mean_speed'][si, k]:.2f}"]  # fmt: skip
                for p in pidx:
                    v, ref = km[si, k, p], km[0, k, p]
                    free = data["emissions_free"][si, k, p]
                    excess = 100 * (data["emissions"][si, k, p] / free - 1) if free > 0 else np.nan
                    row += [_num(v), _num(100 * (v / ref - 1) if np.isfinite(v) and ref > 0 else np.nan), _num(excess)]
                w.writerow(row)
    pos = data["emissions_pos"].sum(axis=2) / EMIS_BIN / (meta["s_simulados"] / 3600.0)  # (scenario, pollutant, x)
    with open(folder / POSITION_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["x_start_m", "x_end_m"] + [f"{POLLUTANTS[p]}_g_m_h_{i}" for p in pidx
                                                for i in range(len(meta["escenarios"]))])  # fmt: skip
        for b in range(pos.shape[2]):
            w.writerow([f"{b * EMIS_BIN:g}", f"{min((b + 1) * EMIS_BIN, meta['largo_m']):g}"]
                       + [f"{pos[i, p, b]:.6g}" for p in pidx for i in range(len(meta["escenarios"]))])  # fmt: skip
    return summary_text(meta, data)


def _num(v: float) -> str:
    return "" if not np.isfinite(v) else f"{v:.4g}"


def summary_text(meta: dict, data: dict[str, np.ndarray]) -> str:
    """Table per scenario: vehicles per minute and travel time of each type, and emissions per km of each
    emitting type, with the change from the first scenario."""
    names = meta["nombres"]
    minutes = meta["s_simulados"] / 60
    km = per_km(data, meta)
    rows: list[tuple[str, list[str]]] = []
    for k in meta["activos"]:
        rows.append((f"{names[k]}: veh/min", [f"{data['crossed_veh'][i, k] / minutes:,.2f}" for i in range(len(km))]))
        rows.append((f"{names[k]}: travel time (s)", [f"{data['travel_time'][i, k]:,.1f}" for i in range(len(km))]))
    for p in (POLLUTANTS.index(pol) for pol in meta["contaminantes"]):
        pol = POLLUTANTS[p]
        for k in meta["emisores"]:
            ref = km[0, k, p]
            if not np.isfinite(ref):
                continue
            cells = []
            for i in range(len(km)):
                v = km[i, k, p]
                change = "" if i == 0 or ref <= 0 else f" ({100 * (v / ref - 1):+.0f} %)"
                cells.append(f"{v:,.1f}{change}")
            rows.append((f"{names[k]}: {POLLUTANT_LABELS[pol]} ({unit(pol)[0]}/km)", cells))
    head = [""] + meta["escenarios"]
    table = [head] + [[label] + cells for label, cells in rows]
    widths = [max(len(r[i]) for r in table) for i in range(len(head))]
    lines = ["  ".join(c.ljust(w) if i == 0 else c.rjust(w) for i, (c, w) in enumerate(zip(r, widths))) for r in table]
    return "\n".join([
        f"Emissions per scenario (mean of {meta['replicas']} replicas, {meta['s_simulados']:,.0f} simulated s; in "
        f"brackets, the change compared with «{meta['escenarios'][0]}»):", "", *lines,
    ])  # fmt: skip


def results_metrics(meta: dict, data: dict[str, np.ndarray]) -> dict:
    """`results.json` metrics: per scenario and type, vehicles per minute, travel time, mean speed and,
    per emitted pollutant, the amount emitted per km, its change from the first scenario and its excess over free flow."""
    minutes = meta["s_simulados"] / 60
    km = per_km(data, meta)
    scenarios = []
    for si, name in enumerate(meta["escenarios"]):
        types = {}
        for k in meta["activos"]:
            row = {"veh_min": data["crossed_veh"][si, k] / minutes, "tiempo_recorrido_s": data["travel_time"][si, k],
                   "velocidad_media_kmh": data["mean_speed"][si, k]}  # fmt: skip
            for pol in meta["contaminantes"]:
                p = POLLUTANTS.index(pol)
                v, ref, free = km[si, k, p], km[0, k, p], data["emissions_free"][si, k, p]
                if not np.isfinite(v):
                    continue
                row[pol] = {
                    "unidad": f"{unit(pol)[0]}/km", "por_km": v,
                    "cambio_pct": 100 * (v / ref - 1) if ref > 0 else None,
                    "exceso_flujo_libre_pct": 100 * (data["emissions"][si, k, p] / free - 1) if free > 0 else None,
                }  # fmt: skip
            types[meta["tipos"][k]] = row
        scenarios.append({"nombre": name, "topes_m": meta["topes_m"][si], "semaforo": meta["semaforo"][si],
                          "tipos": types})  # fmt: skip
    return {"referencia": meta["escenarios"][0], "contaminantes": meta["contaminantes"], "escenarios": scenarios,
            "parametros": meta}  # fmt: skip


def load(folder: Path) -> tuple[dict, dict[str, np.ndarray]]:
    """Parameters and data of a comparison that has already run."""
    from traffic.results import existing

    meta_path, data_path = existing(folder, META_NAME), existing(folder, DATA_NAME)
    if not (meta_path.is_file() and data_path.is_file()):
        raise ConfigError(f"{folder} is not a traffic-emissions folder ({META_NAME} or {DATA_NAME} is missing)")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    with np.load(data_path) as d:
        data = {k: d[k] for k in d.files}
    return meta, data
