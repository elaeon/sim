"""Comparación de emisiones entre escenarios de una misma calle: con y sin tope (o con topes en distintas
posiciones) y, opcionalmente, con y sin semáforo (`uv run trafico-emisiones`).

Todos los escenarios corren las mismas réplicas con la misma semilla que `trafico` (SeedSequence.spawn): las
llegadas, pasajeros y velocidades sorteadas son las mismas, así que las diferencias vienen del tope o del
semáforo. De cada escenario se guardan las medias entre réplicas del flujo que sale del tramo, los vehículos,
tiempos y velocidades por tipo, y las emisiones por tipo, por km y a lo largo del tramo.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from trafico.config import POLLUTANT_LABELS, POLLUTANTS, SimConfig, SpeedBump, as_positions
from trafico.emissions import EMIS_BIN, emitted, unit
from trafico.metrics import LANE_EXIT_FLOW
from trafico.runner import run_parallel
from trafico.settings import ConfigError, RunOptions, validate_config
from trafico.variants import reduce_config

PLOT_NAME = "comparacion_emisiones.png"
DATA_NAME = "datos.npz"
META_NAME = "escenarios.json"
BASE_CONFIG_NAME = "config_base.toml"  # no se llama config.toml: la carpeta no es una corrida de `trafico`
TABLE_CSV = "comparacion_emisiones.csv"
POSITION_CSV = "emisiones_posicion.csv"
SUMMARY_NAME = "resumen.txt"


@dataclass(frozen=True, slots=True)
class Scenarios:
    """Qué se compara y cómo."""

    # Topes de cada escenario: una posición (m), varias (tupla) o None / vacía = sin tope.
    bumps: tuple[float | tuple[float, ...] | None, ...]
    bump_lanes: tuple[int, ...] | None  # carriles de todos los topes (ya renumerados); None = los de cada tope
    lights: tuple[bool | None, ...]  # semáforo activado en cada variante; None = como en la configuración
    length: float | None  # m del tramo; None = [road] length
    lanes: tuple[int, ...] | None  # carriles de la configuración base que se conservan; None = todos
    without: tuple[str, ...]  # claves de los tipos que no participan
    replicas: int
    run: float  # s de proceso (× time_scale = s simulados)


def bump_positions(bump: float | tuple[float, ...] | None) -> tuple[float, ...]:
    """Posiciones (m) de los topes de un escenario, de la entrada a la salida (vacía = sin tope)."""
    return as_positions(bump)


def scenario_label(bump: float | tuple[float, ...] | None, light: bool | None) -> str:
    """Nombre del escenario, p. ej. «tope a 100 m», «topes a 35 y 70 m» o, si se comparan semáforos, «sin tope ·
    sin semáforo»."""
    at = bump_positions(bump)
    name = "sin tope" if not at else f"{'topes' if len(at) > 1 else 'tope'} a {' y '.join(f'{p:g}' for p in at)} m"
    if light is not None:
        name += " · con semáforo" if light else " · sin semáforo"
    return name


def scenario_bumps(base: SimConfig, positions: tuple[float, ...], lanes: tuple[int, ...] | None) -> tuple[SpeedBump, ...]:
    """Topes de un escenario en esas posiciones: el de la configuración que está ahí (con sus carriles y peatones) o,
    si no hay, uno nuevo como el primero de la configuración (o sin peatones, si no hay ninguno). `lanes` reemplaza
    los carriles de todos; sin posiciones no hay tope ni paso peatonal."""
    configured = {b.position: b for b in base.bumps}
    template = base.bumps[0] if base.bumps else SpeedBump(0.0)
    bumps = (configured.get(p) or replace(template, position=p) for p in positions)
    return tuple(replace(b, lanes=lanes) if lanes is not None else b for b in bumps)


def build_configs(base: SimConfig, s: Scenarios) -> list[tuple[str, SimConfig]]:
    """(nombre, configuración validada) de cada escenario: cada semáforo por cada tope."""
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
                raise ConfigError(f"escenario «{name}»: {exc}") from None
            out.append((name, cfg))
    if not emitted(out[0][1])[0]:
        raise ConfigError("ningún tipo que participa emite: hacen falta [vehicles.<clave>.emissions], accel y decel")
    return out


def run_scenarios(
    configs: list[tuple[str, SimConfig]], replicas: int, seed: int, workers: int = 0,
    progress: Callable[[str, int, int], None] | None = None,
) -> dict[str, np.ndarray]:  # fmt: skip
    """Corre cada escenario con las mismas semillas. Devuelve arreglos indexados por escenario (media entre
    réplicas; «_sd», su desviación estándar)."""
    keys = ("crossed_veh", "travel_time", "mean_speed", "veh_km", "emissions", "emissions_free", "emissions_pos")
    out: dict[str, list] = {k: [] for k in (*keys, "crossed_veh_sd", "emissions_sd", "flow")}
    t = None
    for name, cfg in configs:
        agg = run_parallel(cfg, replicas, workers, seed,
                           progress=(lambda d, n, name=name: progress(name, d, n)) if progress else None)  # fmt: skip
        for k in keys:
            out[k].append(agg.summary[k].mean)
        out["crossed_veh_sd"].append(agg.summary["crossed_veh"].std)
        out["emissions_sd"].append(agg.summary["emissions"].std)
        out["flow"].append(np.nansum(agg.series[LANE_EXIT_FLOW].mean, axis=1))  # veh/min que salen, todos los carriles
        t = agg.times
    data = {k: np.array(v) for k, v in out.items()}
    data["t"] = np.asarray(t)
    return data


def metadata(base: SimConfig, s: Scenarios, configs: list[tuple[str, SimConfig]], seed: int, argv: list[str]) -> dict:
    """Lo necesario para describir, graficar y repetir la comparación."""
    cfg = configs[0][1]
    pols, types = emitted(cfg)
    active = [k for k, rate in enumerate(cfg.rates) if rate > 0]
    limits = cfg.lane_max_kmh
    return {
        "comando": "uv run trafico-emisiones " + " ".join(argv),
        "semilla": seed,
        "escenarios": [name for name, _ in configs],
        "topes_m": [[b.position for b in c.bumps] for _, c in configs],
        "semaforo": [c.any_light for _, c in configs],
        "carriles_tope": sorted({ln for _, c in configs for b in c.bumps for ln in c.bump_lanes(b)}) or None,
        # Peatones en los topes ([[speed_bump]] pedestrian): tasa media (peatones/min) y s por cruce del primero con
        # peatones; None = sin peatones.
        "peatones_tope": next(({"peatones_min": b.pedestrian_crossing.expected, "s_cruce": b.pedestrian_time}
                               for _, c in configs for b in c.crossings), None),  # fmt: skip
        "largo_m": cfg.length,
        "carriles_base": list(range(base.lanes)) if s.lanes is None else list(s.lanes),
        "carriles": [f"carril {i} ({limits[i]:g} km/h)" if np.isfinite(limits[i]) else f"carril {i}"
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
    """(escenario, tipo, contaminante) en la unidad de cada contaminante por km recorrido (NaN si no emite)."""
    km = data["veh_km"][:, :, None]
    scale = np.array([unit(pol)[1] for pol in POLLUTANTS])
    with np.errstate(invalid="ignore", divide="ignore"):
        out = data["emissions"] * scale / km
    emits = data["emissions"].sum(axis=0) > 0  # (tipo, contaminante)
    return np.where(emits[None], out, np.nan)


def write_outputs(folder: Path, meta: dict, data: dict[str, np.ndarray]) -> str:
    """Tablas, datos crudos y parámetros; devuelve el resumen en texto."""
    np.savez(folder / DATA_NAME, **data)
    (folder / META_NAME).write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    names = meta["nombres"]
    minutes = meta["s_simulados"] / 60
    km = per_km(data, meta)
    pidx = [POLLUTANTS.index(p) for p in meta["contaminantes"]]
    with open(folder / TABLE_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        header = ["escenario", "tipo", "veh_min", "t_recorrido_s", "velocidad_media_kmh"]
        for p in pidx:
            name = f"{POLLUTANTS[p]}_{unit(POLLUTANTS[p])[0]}_km"
            header += [name, f"{POLLUTANTS[p]}_cambio_pct", f"{POLLUTANTS[p]}_exceso_flujo_libre_pct"]
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
    pos = data["emissions_pos"].sum(axis=2) / EMIS_BIN / (meta["s_simulados"] / 3600.0)  # (escenario, contaminante, x)
    with open(folder / POSITION_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["x_inicio_m", "x_fin_m"] + [f"{POLLUTANTS[p]}_g_m_h_{i}" for p in pidx
                                                for i in range(len(meta["escenarios"]))])  # fmt: skip
        for b in range(pos.shape[2]):
            w.writerow([f"{b * EMIS_BIN:g}", f"{min((b + 1) * EMIS_BIN, meta['largo_m']):g}"]
                       + [f"{pos[i, p, b]:.6g}" for p in pidx for i in range(len(meta["escenarios"]))])  # fmt: skip
    return summary_text(meta, data)


def _num(v: float) -> str:
    return "" if not np.isfinite(v) else f"{v:.4g}"


def summary_text(meta: dict, data: dict[str, np.ndarray]) -> str:
    """Tabla por escenario: vehículos por minuto y tiempo de recorrido de cada tipo, y emisiones por km de cada
    tipo que emite, con el cambio frente al primer escenario."""
    names = meta["nombres"]
    minutes = meta["s_simulados"] / 60
    km = per_km(data, meta)
    rows: list[tuple[str, list[str]]] = []
    for k in meta["activos"]:
        rows.append((f"{names[k]}: veh/min", [f"{data['crossed_veh'][i, k] / minutes:,.2f}" for i in range(len(km))]))
        rows.append((f"{names[k]}: recorrido (s)", [f"{data['travel_time'][i, k]:,.1f}" for i in range(len(km))]))
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
        f"Emisiones por escenario (media de {meta['replicas']} réplicas, {meta['s_simulados']:,.0f} s simulados; entre "
        f"paréntesis, el cambio frente a «{meta['escenarios'][0]}»):", "", *lines,
    ])  # fmt: skip


def results_metrics(meta: dict, data: dict[str, np.ndarray]) -> dict:
    """Métricas de `resultados.json`: por escenario y tipo, vehículos por minuto, tiempo de recorrido, velocidad media y,
    por contaminante emitido, lo emitido por km, su cambio frente al primer escenario y su exceso frente a flujo libre."""
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
    """Parámetros y datos de una comparación ya corrida."""
    meta_path, data_path = folder / META_NAME, folder / DATA_NAME
    if not (meta_path.is_file() and data_path.is_file()):
        raise ConfigError(f"{folder} no es una carpeta de trafico-emisiones (faltan {META_NAME} o {DATA_NAME})")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    with np.load(data_path) as d:
        data = {k: d[k] for k in d.files}
    return meta, data
