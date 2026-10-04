"""Línea de comandos: `uv run trafico [nombre | carpeta]`."""

from __future__ import annotations

import argparse
import contextlib
import functools
import json
import os
import secrets
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from trafico.config import (
    DT, FUEL_CO2_G_PER_L, FUEL_LABELS, MAX_TYPES, PEDESTRIAN_YIELD, POLLUTANT_LABELS, POLLUTANTS, as_positions,
)
from trafico.emissions import EMIS_BIN, acceleration_table, emissions_label, emitted, unit
from trafico.fuel import constant_speed_table, cost, liters
from trafico.metrics import (
    EMIS_SERIES, ENTRY_QUEUE, EXIT_QUEUE, LANE_EXIT_FLOW, LANE_SATURATION, LANE_SPEED, SERIES,
)
from trafico.results import document, read_results, write_results
from trafico.runner import Aggregate, run_parallel
from trafico.settings import (
    CONFIG_NAME,
    ConfigError,
    config_copy_text,
    default_config_path,
    load_settings,
    make_run_dir,
    resolve_output_dir,
    resolve_target,
    safe_name,
)

PLOT_NAME = "movilidad_pasajeros.png"
PAX_PLOT_NAME = "distribucion_pasajeros.png"
CSV_NAME = "series.csv"
EMIS_PLOT_NAME = "emisiones_posicion.png"
EMIS_CSV_NAME = "emisiones_posicion.csv"
SUMMARY_NAME = "resumen.txt"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="trafico",
        description=(
            "Simula el flujo vehicular en un tramo de vía con semáforo al final y grafica la "
            f"capacidad de movilidad de pasajeros. Todos los parámetros se leen de {CONFIG_NAME}; "
            "los resultados y una copia de la configuración se guardan en "
            "<output.dir>/<fecha-hora>_<nombre>/."
        ),
    )
    p.add_argument(
        "target",
        nargs="?",
        default=None,
        metavar="nombre | carpeta",
        help=(
            f"una carpeta existente (p. ej. resultados/<ID>_<nombre>) de la que se lee su {CONFIG_NAME}, "
            f"o el nombre de la corrida, que se agrega a la carpeta de resultados y usa {default_config_path()}"
        ),
    )
    _add_json(p)
    return p


def _progress(done: int, total: int) -> None:
    width = 30
    filled = round(width * done / total)
    sys.stderr.write(f"\r  réplicas [{'█' * filled}{'·' * (width - filled)}] {done}/{total}")
    if done == total:
        sys.stderr.write("\n")
    sys.stderr.flush()


def _fmt(value: float, digits: int = 1) -> str:
    return "—" if not np.isfinite(value) else f"{value:,.{digits}f}"


def summary_rows(agg: Aggregate) -> tuple[list[tuple[str, np.ndarray, int, str]], dict]:
    """Filas de la tabla del resumen, (etiqueta, valor por tipo, decimales, clave de resultados.json), y las medias entre
    réplicas de cada estadístico."""
    cfg = agg.cfg
    s = {k: v.mean for k, v in agg.summary.items()}
    sd = {k: v.std for k, v in agg.summary.items()}
    minutes = cfg.sim_seconds / 60.0
    footprint = np.array([sp.length + sp.gap_run for sp in cfg.specs])
    rows = [
        ("Llegadas (veh/min)", s["arrived_veh"] / minutes, 1, "llegadas_veh_min"),
        ("Vehículos que cruzan", s["crossed_veh"], 1, "vehiculos_cruzan"),
        ("Pasajeros que cruzan", s["crossed_pax"], 1, "pasajeros_cruzan"),
        ("  ± σ entre réplicas", sd["crossed_pax"], 1, "pasajeros_cruzan_sd"),
        ("Pasajeros/min cruzando", s["crossed_pax"] / minutes, 1, "pasajeros_min"),
        ("Pasajeros por vehículo", s["pax_per_veh"], 2, "pasajeros_por_vehiculo"),
        ("Pax/m de carril en marcha", s["pax_per_veh"] / footprint, 2, "pasajeros_por_m_carril"),
        ("T. recorrido medio (s)", s["travel_time"], 1, "tiempo_recorrido_s"),
        ("T. medio en cola de entrada (s)", s["queue_wait"], 1, "tiempo_cola_entrada_s"),
        ("T. a flujo libre (s)", np.array([cfg.free_flow_time(k) + sp.expected_stop_time
                                           for k, sp in enumerate(cfg.specs)]), 1, "tiempo_flujo_libre_s"),
        ("Velocidad media (km/h)", s["mean_speed"], 1, "velocidad_media_kmh"),
        ("En el tramo al final", s["on_road"], 1, "en_tramo_final"),
        ("En cola de entrada al final", s["queued"], 1, "en_cola_final"),
        *([("En el tramo al inicio", s["initial_veh"], 1, "en_tramo_inicio")] if any(cfg.lane_initial_occupancy) else []),
        ("Cambios de carril/veh", np.divide(s["lane_changes_type"], s["entered_veh"], out=np.full(cfg.n_types, np.nan),
                                            where=s["entered_veh"] > 0), 2, "cambios_carril_por_veh"),  # fmt: skip
    ]
    active = [k for k, rate in enumerate(cfg.rates) if rate > 0]  # tipos que participan
    if any(cfg.specs[k].stop_position is not None for k in active):
        rows.insert(9, ("T. medio en la parada (s)", s["stop_time"], 1, "tiempo_parada_s"))
    if cfg.bottleneck_active:
        rows.append(("Detenciones (bottleneck)", s["bottleneck_stops"], 1, "detenciones_bottleneck"))
        rows.append(("T. medio detenido (s)", s["bottleneck_time"], 1, "tiempo_detenido_s"))
    passers = np.array([sp.pass_in_lane for sp in cfg.specs])
    if passers[active].any():
        # Rebases dentro del carril por vehículo que entró; «—» para los tipos que no rebasan así.
        per_veh = np.divide(s["in_lane_passes"], s["entered_veh"], out=np.full(cfg.n_types, np.nan),
                            where=passers & (s["entered_veh"] > 0))  # fmt: skip
        rows.append(("Rebases en el carril/veh", per_veh, 2, "rebases_carril_por_veh"))
    if any(cfg.specs[k].cargo_prob > 0 for k in active):
        # Los de mercancía cuentan como vehículos, pero no en las filas de pasajeros.
        cargo = np.divide(100.0 * s["arrived_cargo"], s["arrived_veh"], out=np.full(cfg.n_types, np.nan),
                          where=s["arrived_veh"] > 0)  # fmt: skip
        rows.insert(1, ("Con mercancía (%)", cargo, 1, "mercancia_pct"))
    rows += _emission_rows(cfg, s)
    rows += _fuel_rows(cfg, s)
    return rows, {"s": s, "sd": sd}


def format_summary(agg: Aggregate) -> str:
    cfg = agg.cfg
    rows, stats = summary_rows(agg)
    s = stats["s"]
    active = [k for k, rate in enumerate(cfg.rates) if rate > 0]  # tipos que participan
    w0 = max(len(r[0]) for r in rows) + 2
    widths = {k: max(10, len(cfg.specs[k].name) + 2) for k in active}
    speed = cfg.sim_seconds / agg.replica_wall.mean
    lines = ["", " " * w0 + "".join(f"{cfg.specs[k].name:>{widths[k]}}" for k in active)]
    for label, values, digits, _ in rows:
        lines.append(f"{label:<{w0}}" + "".join(f"{_fmt(values[k], digits):>{widths[k]}}" for k in active))
    minutes_run = cfg.sim_seconds / 60.0
    lines += ["", f"Cruzan {cfg.line_name} por carril (veh/min): " + " · ".join(
        f"{lane}: {_fmt(v / minutes_run)}" for lane, v in enumerate(s["lane_crossed"]))]  # fmt: skip
    if any(c > 0 for c in cfg.lane_exit_capacity):
        lines.append("Línea cerrada por la cola de salida (% del tiempo, el siguiente no cabe): " + " · ".join(
            f"{lane}: {_fmt(100 * v)}" for lane, v in enumerate(s["exit_blocked"])
            if cfg.lane_exit_capacity[lane] > 0))  # fmt: skip
    lines += _pedestrian_lines(cfg, s)
    lines += _acceleration_lines(cfg)
    lines += _cruise_lines(cfg)
    lines += [
        f"Cambios de carril por réplica: {_fmt(s['lane_changes'][0])}",
        f"Tiempo de proceso nominal: {cfg.run:g} s ≙ {cfg.sim_seconds:g} s simulados "
        f"({cfg.n_ticks:,} pasos de {DT} s)",
        f"Tiempo real: {agg.wall:.2f} s para {agg.replicas} réplicas con {agg.workers} proceso(s) · "
        f"{float(agg.replica_wall.mean):.2f} s por réplica · {float(speed):,.0f} s simulados por s real",
        f"Memoria máxima (RSS) por proceso: {agg.max_rss_kb / 1024:.1f} MiB",
    ]
    return "\n".join(lines)


def _pedestrian_lines(cfg, s: dict) -> list[str]:
    """Peatones de cada semáforo peatonal y de cada tope con paso peatonal: cuántos cruzaron, cuántas veces se cerró el
    paso, la espera media de quien cruzó y el tiempo que estuvo cerrado a los vehículos (en rojo)."""
    spots = cfg.pedestrian_spots
    if not spots or "pedestrians" not in s:
        return []
    lines = ["", "Peatones (media por réplica):"]
    for (kind, pos), (crossed, closings, closed, wait) in zip(spots, s["pedestrians"]):
        name = "semáforo peatonal" if kind == "semáforo" else "paso peatonal del tope"
        mean_wait = wait / crossed if crossed > 0 else np.nan
        lines.append(f"  {name} a {pos:g} m: {_fmt(crossed)} cruzaron en {_fmt(closings)} cierres · espera media "
                     f"{_fmt(mean_wait)} s · en rojo para los vehículos {_fmt(closed)} s "
                     f"({_fmt(100 * closed / cfg.sim_seconds)} % del tiempo)")  # fmt: skip
    return lines


def _emission_rows(cfg, s: dict) -> list[tuple[str, np.ndarray, int, str]]:
    """Por contaminante emitido: por km recorrido, por recorrido completo del tramo y el exceso frente al mismo
    recorrido a velocidad constante (paradas, arranques, tope, cola). «—» en los tipos que no lo emiten."""
    pols, types = emitted(cfg)
    rows = []
    km = s["veh_km"]
    for p in pols:
        pol = POLLUTANTS[p]
        name, scale = unit(pol)
        emits = np.array([k in types and cfg.specs[k].emission_coefs(pol) is not None for k in range(cfg.n_types)])
        grams = s["emissions"][:, p]
        free = s["emissions_free"][:, p]
        per_km = np.divide(grams * scale, km, out=np.full(cfg.n_types, np.nan), where=emits & (km > 0))
        excess = np.divide(100.0 * grams, free, out=np.full(cfg.n_types, np.nan), where=emits & (free > 0)) - 100.0
        label = POLLUTANT_LABELS[pol]
        rows += [
            (f"{label} ({name}/km)", per_km, 1 if scale == 1 else 0, f"{pol}_{name}_km"),
            (f"{label} ({name} por recorrido del tramo)", per_km * cfg.length / 1000.0, 2 if scale == 1 else 1,
             f"{pol}_{name}_recorrido"),
            (f"{label} exceso vs flujo libre (%)", excess, 1, f"{pol}_exceso_flujo_libre_pct"),
        ]
    return rows


def _burners(cfg) -> list[int]:
    """Tipos que participan y queman combustible (tienen `fuel` y emiten CO2)."""
    return [k for k, sp in enumerate(cfg.specs) if cfg.rates[k] > 0 and sp.burns]


def _fuel_rows(cfg, s: dict) -> list[tuple[str, np.ndarray, int, str]]:
    """Consumo (del CO2 emitido) y su costo: por km, por recorrido completo del tramo, lo que se gasta de más frente
    al mismo recorrido a velocidad constante y el total de la corrida. «—» en los tipos sin combustible o sin
    precio."""
    burners = _burners(cfg)
    if not burners:
        return []
    co2 = POLLUTANTS.index("co2")
    nan = np.full(cfg.n_types, np.nan)
    total_l, extra_l, l_km = nan.copy(), nan.copy(), nan.copy()
    total_cost, trip_cost, extra_cost = nan.copy(), nan.copy(), nan.copy()
    km = s["veh_km"]
    for k in burners:
        spec = cfg.specs[k]
        total_l[k] = liters(spec, s["emissions"][k, co2])
        if km[k] > 0:
            l_km[k] = total_l[k] / km[k]
            extra_l[k] = liters(spec, s["emissions"][k, co2] - s["emissions_free"][k, co2]) / km[k]
        total_cost[k] = cost(cfg, spec, total_l[k])
        trip_cost[k] = cost(cfg, spec, l_km[k] * cfg.length / 1000.0)
        extra_cost[k] = cost(cfg, spec, extra_l[k] * cfg.length / 1000.0)
    trip = cfg.length / 1000.0
    cur = cfg.currency
    return [
        ("Combustible (L/100 km)", l_km * 100.0, 2, "combustible_l_100km"),
        ("Combustible por recorrido (mL)", l_km * trip * 1000.0, 1, "combustible_recorrido_ml"),
        ("  de más vs flujo libre (mL)", extra_l * trip * 1000.0, 1, "combustible_extra_recorrido_ml"),
        (f"Costo por recorrido ({cur})", trip_cost, 3, "costo_recorrido"),
        (f"  de más vs flujo libre ({cur})", extra_cost, 3, "costo_extra_recorrido"),
        ("Combustible en la corrida (L)", total_l, 2, "combustible_corrida_l"),
        (f"Costo en la corrida ({cur})", total_cost, 2, "costo_corrida"),
    ]


def _cruise_lines(cfg) -> list[str]:
    """Tabla de referencia: rendimiento de cada tipo a velocidad constante con el modelo (del CO2) y con la curva
    física (con `mass_kg`), en L/100 km, km/L y costo por km."""
    burners = _burners(cfg)
    if not burners:
        return []
    lines = ["", "Rendimiento a velocidad constante (del CO2 del modelo de emisiones; «física»: ralentí + potencia "
             "para vencer rodadura y aire, de referencia):"]  # fmt: skip
    for k in burners:
        spec = cfg.specs[k]
        price = cfg.price(spec.fuel)
        extra = f", {price:g} {cfg.currency}/L" if price is not None else ", sin precio"
        mass = f", {spec.mass_kg:g} kg" if spec.mass_kg is not None else ""
        lines.append(f"  {spec.name} ({FUEL_LABELS[spec.fuel]}{extra}{mass}):")

        def fmt(l100: float) -> str:
            money = f", {l100 / 100 * price:.2f} {cfg.currency}/km" if price is not None else ""
            return f"{l100:.1f} L/100 km ({100 / l100:.1f} km/L{money})"

        for row in constant_speed_table(spec):
            physical = f" · física {fmt(row.physical_l_100km)}" if row.physical_l_100km is not None else ""
            lines.append(f"    {row.kmh:g} km/h: {fmt(row.l_100km)}{physical}")
    return lines


def _acceleration_lines(cfg) -> list[str]:
    """Tabla de referencia: lo que emite cada tipo al acelerar de 0 a 20, 20 a 40, 40 a 60 y 60 a 80 km/h a su
    accel, y, entre paréntesis, a velocidad constante en el mismo tiempo y distancia."""
    _, types = emitted(cfg)
    if not types:
        return []
    lines = ["", "Emisiones al acelerar (modelo de Int Panis et al., 2006, a la accel de cada tipo; entre "
             "paréntesis, a velocidad constante en el mismo tiempo y distancia):"]  # fmt: skip
    over = False
    for k in types:
        spec = cfg.specs[k]
        lines.append(f"  {spec.name} ({spec.accel:g} m/s², velocidad máxima {spec.fastest_kmh:g} km/h):")
        for row in acceleration_table(spec):
            parts = []
            for pol, grams in row.grams.items():
                name, scale = unit(pol)
                parts.append(f"{POLLUTANT_LABELS[pol]} {grams * scale:,.2f} ({row.cruise[pol] * scale:,.2f}) {name}")
            mark = " *" if row.v2 > spec.fastest_kmh else ""
            over |= bool(mark)
            lines.append(f"    {row.v1:g}→{row.v2:g} km/h{mark} en {row.seconds:.1f} s y {row.meters:.0f} m: "
                         + " · ".join(parts))  # fmt: skip
    if over:
        lines.append("  * pasa la velocidad máxima del tipo: fuera del rango en que circula (y en que se ajustó el "
                     "modelo); tómese con reserva")  # fmt: skip
    return lines


def write_emissions_csv(agg: Aggregate, path: Path) -> None:
    """g/(m·h) de cada contaminante emitido, por carril, en intervalos de EMIS_BIN m a lo largo del tramo (media
    entre réplicas)."""
    cfg = agg.cfg
    pols, _ = emitted(cfg)
    pos = agg.summary["emissions_pos"].mean / EMIS_BIN / (cfg.sim_seconds / 3600.0)
    edges = np.arange(pos.shape[2] + 1) * EMIS_BIN
    cols = [edges[:-1], np.minimum(edges[1:], cfg.length)]
    header = ["x_inicio_m", "x_fin_m"]
    for p in pols:
        for lane in range(cfg.lanes):
            cols.append(pos[p, lane])
            header.append(f"{POLLUTANTS[p]}_g_m_h_carril{lane}")
    np.savetxt(path, np.column_stack(cols), delimiter=",", header=",".join(header), comments="", fmt="%.6g")


def write_csv(agg: Aggregate, path: Path) -> None:
    cfg = agg.cfg
    t = agg.times
    cols = [t, t / cfg.time_scale]
    header = ["t_sim_s", "t_proceso_s"]
    for key in SERIES:
        mean, std = agg.series[key].mean, agg.series[key].std
        for k, sp in enumerate(cfg.specs):
            cols += [mean[:, k], std[:, k]]
            header += [f"{key}_{sp.name}_media", f"{key}_{sp.name}_sd"]
    pols, types = emitted(cfg)
    for p in pols:  # g/min de cada contaminante emitido, por tipo que lo emite
        stats = agg.series[EMIS_SERIES[p]]
        for k in types:
            if cfg.specs[k].emission_coefs(POLLUTANTS[p]) is not None:
                cols += [stats.mean[:, k], stats.std[:, k]]
                header += [f"{EMIS_SERIES[p]}_g_min_{cfg.specs[k].name}_media", f"{EMIS_SERIES[p]}_g_min_{cfg.specs[k].name}_sd"]
    if "co2" in [POLLUTANTS[p] for p in pols]:  # L/min de combustible, del CO2
        stats = agg.series[EMIS_SERIES[POLLUTANTS.index("co2")]]
        for k in _burners(cfg):
            g_per_l = FUEL_CO2_G_PER_L[cfg.specs[k].fuel]
            cols += [stats.mean[:, k] / g_per_l, stats.std[:, k] / g_per_l]
            header += [f"comb_l_min_{cfg.specs[k].name}_media", f"comb_l_min_{cfg.specs[k].name}_sd"]
    lane_series = [
        (LANE_SATURATION, "saturacion"), (LANE_SPEED, "velocidad_kmh"), (LANE_EXIT_FLOW, "cruzan_veh_min"),
        (ENTRY_QUEUE, "cola_entrada_veh"),
    ]  # fmt: skip
    if any(c > 0 for c in cfg.lane_exit_capacity):
        lane_series.append((EXIT_QUEUE, "cola_salida_m"))
    for key, label in lane_series:
        stats = agg.series[key]
        for lane in range(cfg.lanes):
            cols += [stats.mean[:, lane], stats.std[:, lane]]
            header += [f"{label}_carril{lane}_media", f"{label}_carril{lane}_sd"]
    np.savetxt(path, np.column_stack(cols), delimiter=",", header=",".join(header), comments="", fmt="%.6g")


def _rate_text(rate) -> str:
    """Una tasa variable o fija, p. ej. «1.5 ± 1 en [0.5, 3], media 1.6»."""
    if not rate.variable:
        return f"{rate.expected:g}"
    return f"{rate.mean:g} ± {rate.std:g} en [{rate.min:g}, {rate.max:g}], media {rate.expected:.3g}"


def _lane_lines(cfg) -> list[str]:
    """Límite de velocidad, carriles sin semáforo, paradas y demás ajustes por carril o por tipo."""
    lines = []
    if cfg.lane_speed_limit is not None:
        limits = " · ".join(f"{i}: {v:g}" for i, v in enumerate(cfg.lane_max_kmh))
        lines.append(f"Límite de velocidad por carril (km/h, 0 = derecho): {limits}")
    bumps = cfg.bumps
    if bumps:
        slow = [f"{sp.name} ≤ {sp.speed_bump_kmh:g}" for k, sp in enumerate(cfg.specs)
                if sp.speed_bump_kmh is not None and cfg.rates[k] > 0]  # fmt: skip

        def where(bump) -> str:
            lanes = cfg.bump_lanes(bump)
            return "todos los carriles" if len(lanes) == cfg.lanes else (
                ("carril " if len(lanes) == 1 else "carriles ") + ", ".join(map(str, lanes)))

        at = " · ".join(f"{b.position:g} m ({where(b)})" for b in bumps)
        lines.append(f"{'Topes' if len(bumps) > 1 else 'Tope'} a {at}, km/h al pasarlo: "
                     + (" · ".join(slow) if slow else "ningún tipo frena (sin speed_bump_kmh)"))  # fmt: skip
    for bump in cfg.crossings:
        lines.append(f"Paso peatonal en el tope a {bump.position:g} m (peatones/min "
                     f"{_rate_text(bump.pedestrian_crossing)}; espera {PEDESTRIAN_YIELD:g} s y cruza en "
                     f"{bump.pedestrian_time:g} s): los vehículos se detienen del todo mientras cruza alguien")  # fmt: skip
    for light in cfg.lights:
        if light.pedestrian:
            prev = cfg.previous_light(light)
            when = (f"con el semáforo de {prev.position:g} m en rojo" if prev else "sin esperar a otro semáforo")
            lines.append(f"Semáforo peatonal a {light.position:g} m (peatones/min {_rate_text(light.pedestrian_crossing)}): "
                         f"verde para los vehículos salvo con peatones en espera, {when}; amarillo {light.yellow:g} s, "
                         f"rojo {light.red:g} s, verde mínimo {light.green:g} s entre cruces")  # fmt: skip
        free = [f"{sp.name} (carril {sp.lane})" for k, sp in enumerate(cfg.specs)
                if cfg.ignores_light(k, light) and cfg.rates[k] > 0]  # fmt: skip
        if free:
            at = f" a {light.position:g} m" if cfg.inner_lights else ""
            lines.append(f"Sin semáforo{at}, siguen en rojo y en amarillo (carril exclusivo en free_lanes): "
                         + ", ".join(free))  # fmt: skip
    stops = [
        f"{sp.name} a {sp.stop_position:g} m, {sp.stop_time_mean:g} ± {sp.stop_time_std:g} s"
        for k, sp in enumerate(cfg.specs) if sp.stop_position is not None and cfg.rates[k] > 0
    ]  # fmt: skip
    if stops:
        where = "antes del semáforo" if cfg.has_light else "en el tramo"
        lines.append(f"Parada {where} (descenso y ascenso): " + " · ".join(stops))
    variable = [(sp.name, cfg.rate(k)) for k, sp in enumerate(cfg.specs) if cfg.rate(k).variable]
    if variable:
        lines.append(f"Demanda variable (veh/min, nueva tasa cada {cfg.rate_interval:g} s): " + " · ".join(
            f"{name} {r.mean:g} ± {r.std:g} en [{r.min:g}, {r.max:g}], media {r.expected:.3g}" for name, r in variable))
    speeds = [sp for k, sp in enumerate(cfg.specs) if sp.speed_std > 0 and cfg.rates[k] > 0]
    if speeds:
        lines.append("Velocidad máxima variable (km/h, sorteada por vehículo): " + " · ".join(
            f"{sp.name} {sp.speed_kmh:g} ± {sp.speed_std:g} en [{sp.slowest_kmh:g}, {sp.fastest_kmh:g}]" for sp in speeds))
    reserved = sorted(cfg.reserved_lanes)
    if reserved:
        owners = {ln: [s.name for s in cfg.specs if s.lane == ln] for ln in reserved}
        lines.append("Carriles exclusivos: " + " · ".join(f"{ln}: {', '.join(owners[ln])}" for ln in reserved))
    if cfg.bottleneck_active:
        lo, hi = cfg.bottleneck_zone()
        per_type = " · ".join(
            f"{sp.name} {100 * cfg.bottleneck_prob(k):g} % {'%g ± %g s' % cfg.bottleneck_time(k)}"
            for k, sp in enumerate(cfg.specs) if cfg.rates[k] > 0 and cfg.bottleneck_prob(k) > 0
        )  # fmt: skip
        lines.append(f"Cuello de botella (se detienen en carril(es) {', '.join(map(str, cfg.bottleneck_lanes()))} "
                     f"entre {lo:g} y {hi:g} m): {per_type}")  # fmt: skip
    occupancy = cfg.lane_initial_occupancy
    if any(occupancy):
        lines.append("Condición inicial, ocupación por carril (0 = derecho): "
                     + " · ".join(f"{i}: {100 * v:g} %" for i, v in enumerate(occupancy)))  # fmt: skip
    gradual = [sp for k, sp in enumerate(cfg.specs) if cfg.rates[k] > 0 and (sp.accel or sp.decel)]
    if gradual:
        def fmt(v):
            return "—" if v is None else f"{v:g}"
        lines.append("Aceleración / frenado graduales (m/s², — = instantáneo): " + " · ".join(
            f"{sp.name} {fmt(sp.accel)} / {fmt(sp.decel)}" for sp in gradual))  # fmt: skip
    if any(c > 0 for c in cfg.lane_exit_capacity):
        lines.append("Cola de salida por carril (acepta veh/min, mide m): " + " · ".join(
            f"{lane}: " + (f"{c:g}/min, {st:g} m" if c > 0 else "sin límite")
            for lane, (c, st) in enumerate(zip(cfg.lane_exit_capacity, cfg.lane_exit_storage))))  # fmt: skip
    if cfg.behavior.queue_reaction:
        lines.append("Cola de entrada detenida: cada vehículo arranca con su tiempo de reacción ([behavior] queue_reaction)")
    emis = emissions_label(cfg)
    if emis:
        lines.append(emis)
    fuel = _fuel_label(cfg)
    if fuel:
        lines.append(fuel)
    return lines


def _fuel_label(cfg) -> str | None:
    """Línea de la cabecera: combustible y precio de cada tipo que quema combustible, y los que tienen `fuel`
    pero no CO2."""
    burners = _burners(cfg)
    if not burners:
        return None

    def price(fuel: str) -> str:
        p = cfg.price(fuel)
        return f"{p:g} {cfg.currency}/L" if p is not None else "sin precio"

    parts = [f"{cfg.specs[k].name} {FUEL_LABELS[cfg.specs[k].fuel]} ({price(cfg.specs[k].fuel)})" for k in burners]
    missing = [sp.name for k, sp in enumerate(cfg.specs) if cfg.rates[k] > 0 and sp.fuel and not sp.burns]
    return "Combustible (del CO2 emitido): " + " · ".join(parts) + (
        f" · sin consumo (sin CO2 o sin accel/decel): {', '.join(missing)}" if missing else "")


def _json_aware(command):
    """Añade `--json` a un comando: lo que imprime pasa a stderr y en stdout queda solo el `resultados.json` de la
    carpeta que devuelve (sin `--json` no cambia nada). El comando recibe siempre la lista de argumentos."""

    @functools.wraps(command)
    def wrapper(argv: list[str] | None = None) -> Path:
        argv = list(sys.argv[1:] if argv is None else argv)
        if "--json" not in argv:
            return command(argv)
        argv = [a for a in argv if a != "--json"]
        with contextlib.redirect_stdout(sys.stderr):
            folder = command(argv)
        print(json.dumps(read_results(folder), ensure_ascii=False, indent=2))
        return folder

    return wrapper


def _add_json(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", action="store_true",
                        help="imprime en stdout el resultados.json de la corrida (el resto de la salida va a stderr)")  # fmt: skip


def run_metrics(agg: Aggregate) -> dict:
    """Métricas de `resultados.json` de una corrida de `trafico`: por tipo los mismos estadísticos de la tabla del
    resumen (media entre réplicas), por carril, peatones y la ejecución."""
    cfg = agg.cfg
    rows, stats = summary_rows(agg)
    s = stats["s"]
    minutes = cfg.sim_seconds / 60.0
    active = [k for k, rate in enumerate(cfg.rates) if rate > 0]
    types = {cfg.specs[k].key: {"nombre": cfg.specs[k].name, **{key: values[k] for _, values, _, key in rows}}
             for k in active}  # fmt: skip
    pedestrians = [
        {"tipo": "semaforo" if kind == "semáforo" else "tope", "posicion_m": pos, "cruzaron": crossed,
         "cierres": closings, "espera_media_s": wait / crossed if crossed > 0 else None, "rojo_s": closed,
         "rojo_pct": 100 * closed / cfg.sim_seconds}
        for (kind, pos), (crossed, closings, closed, wait) in zip(cfg.pedestrian_spots, s.get("pedestrians", []))
    ]  # fmt: skip
    return {
        "tipos": types,
        "carriles": {"cruzan_veh_min": [v / minutes for v in s["lane_crossed"]],
                     "linea_cerrada_pct": [100 * v if cfg.lane_exit_capacity[i] > 0 else None
                                           for i, v in enumerate(s["exit_blocked"])]},  # fmt: skip
        "peatones": pedestrians,
        "cambios_carril_por_replica": s["lane_changes"][0],
        "ejecucion": {"tiempo_real_s": agg.wall, "procesos": agg.workers, "s_por_replica": float(agg.replica_wall.mean),
                      "s_simulados_por_s_real": cfg.sim_seconds / float(agg.replica_wall.mean),
                      "rss_max_mib": agg.max_rss_kb / 1024.0},  # fmt: skip
    }


@_json_aware
def run(argv: list[str] | None = None) -> Path:
    """Ejecuta una corrida completa y devuelve su carpeta de resultados."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        target = resolve_target(args.target)
        settings = load_settings(target.config)
    except ConfigError as exc:
        parser.error(str(exc))
    cfg, opts = settings.sim, settings.run
    # Prioridad del nombre: el del comando, [output] name, y si no el de la carpeta leída.
    name = safe_name(target.name or opts.name or target.fallback_name)
    seed = opts.seed if opts.seed is not None else secrets.randbelow(2**32)
    workers = min(opts.workers or os.process_cpu_count() or 1, opts.replicas)

    now = datetime.now()
    run_dir = make_run_dir(resolve_output_dir(opts.output_dir), name, now)
    copy = config_copy_text(settings, seed, name, run_dir, now)
    (run_dir / CONFIG_NAME).write_text(copy, encoding="utf-8")

    header = "\n".join(
        [
            f"Corrida {run_dir.name} · configuración {target.config}",
            f"Tramo {cfg.length:g} m · {cfg.lanes} carril(es) · "
            + (f"semáforos {cfg.lights_label}" if cfg.inner_lights else
               f"semáforo {cfg.light_label}" if cfg.has_light else "sin semáforo"),
            f"run {cfg.run:g} s de proceso × {cfg.time_scale:g} = {cfg.sim_seconds:g} s simulados · "
            f"{opts.replicas} réplicas en {workers} proceso(s) · semilla {seed}",
            *_lane_lines(cfg),
            *settings.notices,
        ]
    )
    print(header)
    agg = run_parallel(cfg, opts.replicas, workers, seed, progress=_progress if opts.progress else None)
    summary = format_summary(agg)
    print(summary)
    (run_dir / SUMMARY_NAME).write_text(header + "\n" + summary + "\n", encoding="utf-8")

    if opts.series_csv:
        write_csv(agg, run_dir / CSV_NAME)

    from trafico.plotting import plot_mobility, plot_passenger_distribution

    footer = f"corrida {run_dir.name} · semilla {seed}"
    if opts.passengers_plot and any(rate > 0 and sp.carries_passengers for sp, rate in zip(cfg.specs, cfg.rates)):
        plot_passenger_distribution(agg, run_dir / PAX_PLOT_NAME, footer=footer)
    if opts.mobility_plot:
        plot_mobility(agg, run_dir / PLOT_NAME, footer=footer)
    if emitted(cfg)[0]:
        if opts.emisiones_csv:
            write_emissions_csv(agg, run_dir / EMIS_CSV_NAME)
        if opts.emissions_plot:
            from trafico.plotting import plot_emissions_by_position

            plot_emissions_by_position(agg, run_dir / EMIS_PLOT_NAME, footer=footer)
    if opts.animation:
        from trafico.movement import visualize

        print()
        for path in visualize(cfg, seed, settings.animation, run_dir, run_dir.name):
            print(f"  {path.name}")
    write_results(run_dir, document(
        "corrida", run_dir, command=("uv run trafico " + " ".join(argv)).strip(), seed=seed, replicas=opts.replicas,
        sim_seconds=cfg.sim_seconds, metrics=run_metrics(agg)))  # fmt: skip
    print(f"\nResultados en {run_dir}")
    return run_dir


def main(argv: list[str] | None = None) -> None:
    run(argv)


# ------------------------------------------------------------------ trafico-ver


def build_viewer_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="trafico-ver",
        description=(
            "Visualiza el movimiento de los vehículos de una corrida: vuelve a simular una réplica "
            "(idéntica, con la semilla de la copia de la configuración) y guarda en la carpeta un "
            "diagrama espacio-tiempo por carril (PNG) y un video visto desde arriba (WebM por defecto; MP4 o GIF "
            "con [animation] format). "
            "La ventana de tiempo y el ritmo se leen de [animation]; las opciones los reemplazan."
        ),
    )
    p.add_argument(
        "carpeta", nargs="?", default=None,
        help="carpeta de una corrida (resultados/<ID>_<nombre>); por defecto, la más reciente",
    )  # fmt: skip
    p.add_argument("--inicio", type=float, help="s simulados donde empieza la ventana ([animation] start)")
    p.add_argument("--duracion", type=float, help="s simulados de la ventana ([animation] duration; por defecto 2 ciclos)")
    p.add_argument("--velocidad", type=float, help="s simulados por s de video ([animation] speed)")
    p.add_argument("--replica", type=int, help="réplica a visualizar, 1 = la primera ([animation] replica)")
    p.add_argument("--formato", choices=("webm", "mp4", "gif"), help="formato del video ([animation] format)")
    return p


def latest_run_dir(base: Path) -> Path | None:
    """Carpeta de corrida más reciente (los nombres empiezan con la fecha y hora)."""
    runs = sorted(p for p in base.glob("*") if (p / CONFIG_NAME).is_file()) if base.is_dir() else []
    return runs[-1] if runs else None


def view(argv: list[str] | None = None) -> list[Path]:
    import dataclasses

    from trafico.movement import visualize
    from trafico.settings import validate_animation

    parser = build_viewer_parser()
    args = parser.parse_args(argv)
    try:
        if args.carpeta is None:
            root = load_settings(default_config_path()).run.output_dir
            folder = latest_run_dir(resolve_output_dir(root))
            if folder is None:
                raise ConfigError(f"no hay corridas en {resolve_output_dir(root)}; corre primero `uv run trafico`")
        else:
            folder = Path(args.carpeta).expanduser()
            if not (folder / CONFIG_NAME).is_file():
                raise ConfigError(f"{folder} no es una carpeta de corrida (no tiene {CONFIG_NAME})")
        settings = load_settings(folder / CONFIG_NAME)
        if settings.run.seed is None:
            raise ConfigError(f"{folder / CONFIG_NAME} no tiene [execution] seed: no se puede repetir la réplica")
        changes = {
            name: value
            for name, value in (("start", args.inicio), ("duration", args.duracion),
                                ("speed", args.velocidad), ("replica", args.replica),
                                ("format", args.formato))
            if value is not None
        }  # fmt: skip
        anim = dataclasses.replace(settings.animation, **changes)
        validate_animation(anim, settings.sim, settings.run.replicas)
    except ConfigError as exc:
        parser.error(str(exc))
    print(f"Corrida {folder.resolve().name}")
    paths = visualize(settings.sim, settings.run.seed, anim, folder, folder.resolve().name)
    for path in paths:
        print(f"  {path}")
    return paths


def view_main(argv: list[str] | None = None) -> None:
    view(argv)


# ------------------------------------------------------------ trafico-variantes

VARIANTS_NAME = "variantes_semaforo"


def build_variants_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="trafico-variantes",
        description=(
            f"Compara variantes de un escenario: corre la configuración ({CONFIG_NAME}) con cada combinación de "
            "largo de tramo y reparto del semáforo, y grafica la velocidad media de cada carril y los vehículos en "
            "la cola de entrada a lo largo del tiempo. Guarda la gráfica, tablas CSV, un resumen, los datos crudos "
            "y los parámetros en <output.dir>/<fecha-hora>_<nombre>/."
        ),
    )
    p.add_argument(
        "target", nargs="?", default=None, metavar="nombre | carpeta",
        help=f"como en `trafico`: una carpeta de la que se lee su {CONFIG_NAME}, o el nombre de la comparación "
             f"(por defecto, {VARIANTS_NAME}) con {default_config_path()}",
    )  # fmt: skip
    p.add_argument("--largos", type=float, nargs="+", metavar="M",
                   help="largos del tramo en m (por defecto, [road] length)")  # fmt: skip
    p.add_argument("--semaforos", nargs="+", metavar="ROJO/VERDE",
                   help="repartos del semáforo en s, p. ej. 40/20 30/30 20/40 (por defecto, esos tres "
                        "repartos del ciclo rojo + verde de la configuración)")  # fmt: skip
    p.add_argument("--carriles", type=int, nargs="+", metavar="N",
                   help="carriles de la configuración que se conservan, renumerados desde 0 en ese orden, con "
                        "sus límites, congestión, ocupación inicial y cuellos de botella (por defecto, todos)")  # fmt: skip
    p.add_argument("--sin", nargs="+", default=[], metavar="CLAVE",
                   help="tipos de vehículo que no participan, p. ej. bike bus")  # fmt: skip
    p.add_argument("--cola", nargs="+", metavar="CLAVE",
                   help="tipos que se cuentan en la cola de entrada (por defecto, car; si no participa, todos)")  # fmt: skip
    p.add_argument("--replicas", type=int, help="réplicas por escenario (por defecto, [execution] replicas)")
    p.add_argument("--run", type=float, help="s de proceso de cada réplica (por defecto, [execution] run)")
    p.add_argument("--redibujar", metavar="CARPETA",
                   help="no simula: vuelve a dibujar la gráfica de una comparación ya corrida (sin sobrescribir)")  # fmt: skip
    _add_json(p)
    return p


def _lights(values: list[str] | None, red: float, green: float) -> tuple[tuple[float, float], ...]:
    if values is None:  # tres repartos del mismo ciclo: rojo 2/3, 1/2 y 1/3
        cycle = red + green
        return tuple((round(round(cycle * f / DT) * DT, 6), round(round(cycle * (1 - f) / DT) * DT, 6))
                     for f in (2 / 3, 1 / 2, 1 / 3))  # fmt: skip
    out = []
    for text in values:
        parts = text.split("/")
        try:
            red_s, green_s = (float(v) for v in parts)
        except ValueError:
            raise ConfigError(f"--semaforos: {text!r} debe ser ROJO/VERDE en s, p. ej. 40/20") from None
        out.append((red_s, green_s))
    return tuple(out)


@_json_aware
def variants(argv: list[str] | None = None) -> Path:
    """Corre una comparación de variantes (o la redibuja) y devuelve su carpeta."""
    from trafico.movement import _free_path
    from trafico.plotting import plot_variants
    from trafico.settings import _set_key
    from trafico.variants import (
        BASE_CONFIG_NAME, PLOT_NAME as VARIANTS_PLOT, SUMMARY_NAME as VARIANTS_SUMMARY, Variants, load, metadata,
        run_variants, write_outputs,
    )  # fmt: skip
    from trafico.variants import results_metrics as variants_metrics

    argv = sys.argv[1:] if argv is None else argv
    parser = build_variants_parser()
    args = parser.parse_args(argv)
    try:
        if args.redibujar:
            folder = Path(args.redibujar).expanduser()
            meta, data = load(folder)
            path = _free_path(folder / VARIANTS_PLOT)
            plot_variants(path, meta, data)
            print(f"Gráfica en {path}")
            return folder
        target = resolve_target(args.target)
        settings = load_settings(target.config)
        base, opts = settings.sim, settings.run
        keys = [s.key for s in base.specs]
        without = tuple(args.sin)
        playing = [k for k, rate in zip(keys, base.rates) if rate > 0 and k not in without]
        queue_types = tuple(args.cola or (["car"] if "car" in playing else playing))
        for key in queue_types:
            if key not in keys:
                raise ConfigError(f"--cola: no existe el tipo {key!r} (claves: {', '.join(keys)})")
        v = Variants(
            lengths=tuple(args.largos or (base.length,)),
            lights=_lights(args.semaforos, base.red, base.green),
            lanes=tuple(args.carriles) if args.carriles else None,
            without=without, queue_types=queue_types,
            replicas=args.replicas if args.replicas is not None else opts.replicas,
            run=args.run if args.run is not None else base.run,
        )  # fmt: skip
        if v.replicas < 1:
            raise ConfigError("--replicas debe ser al menos 1")
        if len(v.lights) > MAX_TYPES:
            raise ConfigError(f"--semaforos: se admiten hasta {MAX_TYPES} repartos (colores de la gráfica)")
        seed = opts.seed if opts.seed is not None else secrets.randbelow(2**32)
        meta = metadata(base, v, seed, argv)  # valida carriles y tipos antes de crear la carpeta
    except ConfigError as exc:
        parser.error(str(exc))

    n_scen = len(v.lengths) * len(v.lights)
    print(f"Comparación de variantes · configuración {target.config}")
    print(f"{len(v.lengths)} largo(s) × {len(v.lights)} semáforo(s) = {n_scen} escenarios × {v.replicas} réplicas · "
          f"run {v.run:g} s × {base.time_scale:g} = {v.run * base.time_scale:g} s simulados · semilla {seed}")  # fmt: skip
    try:
        data = run_variants(base, v, seed, opts.workers, progress=_progress if opts.progress else None)
    except ConfigError as exc:
        parser.error(str(exc))

    now = datetime.now()
    folder = make_run_dir(resolve_output_dir(opts.output_dir), safe_name(target.name or VARIANTS_NAME), now)
    text = settings.text if opts.seed is not None else _set_key(settings.text, "execution", "seed", seed, "semilla usada")
    (folder / BASE_CONFIG_NAME).write_text(
        f"# Configuración base de la comparación {folder.name}\n# {meta['comando']}\n\n{text}", encoding="utf-8"
    )
    summary = write_outputs(folder, meta, data)
    print("\n" + summary)
    (folder / VARIANTS_SUMMARY).write_text(f"{meta['comando']}\n\n{summary}\n", encoding="utf-8")
    plot_variants(folder / VARIANTS_PLOT, meta, data)
    write_results(folder, document("variantes", folder, command=meta["comando"], seed=seed, replicas=v.replicas,
                                   sim_seconds=meta["s_simulados"], metrics=variants_metrics(meta, data)))  # fmt: skip
    print(f"\nResultados en {folder}")
    return folder


def variants_main(argv: list[str] | None = None) -> None:
    variants(argv)


EMISSIONS_NAME = "emisiones_topes"
SPACING_NAME = "separacion_topes"
LIGHT_SPACING_NAME = "separacion_semaforos"


def build_emissions_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="trafico-emisiones",
        description=(
            f"Compara las emisiones de una calle en varios escenarios: corre la configuración ({CONFIG_NAME}) sin "
            "tope y con topes en las posiciones dadas y, si se pide, con y sin semáforo, con la misma semilla. "
            "Grafica el flujo, los vehículos por minuto, las emisiones por km de cada tipo y a lo largo de la calle. "
            "Guarda la gráfica, tablas CSV, un resumen, los datos crudos y los parámetros en "
            "<output.dir>/<fecha-hora>_<nombre>/. Las emisiones requieren [vehicles.<clave>.emissions], accel y decel."
        ),
    )
    p.add_argument(
        "target", nargs="?", default=None, metavar="nombre | carpeta",
        help=f"como en `trafico`: una carpeta de la que se lee su {CONFIG_NAME}, o el nombre de la comparación "
             f"(por defecto, {EMISSIONS_NAME}) con {default_config_path()}",
    )  # fmt: skip
    p.add_argument("--topes", nargs="+", metavar="M|M,M|sin",
                   help="topes de cada escenario: la posición (m desde la entrada), varias separadas por comas "
                        "(35,70: dos topes en el mismo escenario) o «sin» (por defecto, sin tope y con los de "
                        "[[speed_bump]]; si no hay, uno a la mitad del tramo)")  # fmt: skip
    p.add_argument("--carriles-tope", type=int, nargs="+", metavar="N",
                   help="carriles con tope, en la numeración después de --carriles (por defecto, los de "
                        "[[speed_bump]] lanes de cada tope; sin ella, todos)")  # fmt: skip
    p.add_argument("--separacion", nargs="*", type=float, default=None, metavar="D",
                   help="en vez de la comparación normal, evalúa cuánto deben separarse dos topes para diluir la huella de "
                        "emisiones: corre sin tope, con un tope y con dos topes (o --cadena N) separados D m (por omisión "
                        "10 15 20 25 30 40 50 60 80 100) y grafica aditividad, media en el tramo, gradiente y costo de cada tope")  # fmt: skip
    p.add_argument("--primer-tope", type=float, metavar="M",
                   help="con --separacion, posición del primer tope en m (por omisión, el primer [[speed_bump]] o un cuarto "
                        "del tramo)")  # fmt: skip
    p.add_argument("--cadena", type=int, metavar="N",
                   help="con --separacion, topes de cada cadena (por defecto 2, una pareja): N topes separados D m "
                        "(P1, P1 + D, …); calcula también el costo de cada tope añadido")  # fmt: skip
    p.add_argument("--tolerancia", type=float, default=0.05, metavar="T",
                   help="con --separacion, cuánto puede apartarse la pareja de dos topes aislados para llamarla aditiva "
                        "(por defecto 0.05)")  # fmt: skip
    p.add_argument("--umbral", type=float, default=0.10, metavar="U",
                   help="con --separacion, fracción del pico de un tope solo bajo la cual su exceso cuenta como diluido "
                        "(por defecto 0.10)")  # fmt: skip
    p.add_argument("--separacion-semaforos", nargs="*", type=float, default=None, metavar="D",
                   help="como --separacion, pero con semáforos de ciclo fijo: corre sin semáforo, con uno y con dos (o "
                        "--cadena N) separados D m (por omisión 25 50 75 100 150 200 250 300), sin los semáforos ni los "
                        "topes de la configuración; el tramo se amplía si hace falta (salvo con --largo)")  # fmt: skip
    p.add_argument("--primer-semaforo", type=float, metavar="M",
                   help="con --separacion-semaforos, posición del primer semáforo en m (por omisión 100: deja sitio a "
                        "su cola)")  # fmt: skip
    p.add_argument("--ciclo", metavar="ROJO/VERDE[/AMARILLO]",
                   help="con --separacion-semaforos, fases de todos los semáforos en s, p. ej. 30/30/3 (por omisión, las "
                        "del primer semáforo de ciclo fijo de la configuración o, si no hay, 30/30/3)")  # fmt: skip
    p.add_argument("--desfase", metavar="rojo|verde|igual[,…]", default=None,
                   help="con --separacion-semaforos: «rojo» (por omisión) pone cada semáforo en rojo justo cuando llega "
                        "la cabeza del pelotón que salió del anterior al ponerse en verde, a cualquier separación; "
                        "«verde», en verde (onda verde); «igual», todos en la misma fase. Con --cadena, uno por tramo "
                        "separados por comas, p. ej. verde,rojo")  # fmt: skip
    p.add_argument("--separaciones-fijas", type=float, nargs="+", metavar="M",
                   help="con --separacion-semaforos y --cadena N (N ≥ 3): separaciones fijas en m de los tramos después "
                        "del primero (N − 2 valores); el primero es el que se barre")  # fmt: skip
    p.add_argument("--semaforo", choices=("config", "si", "no", "ambos"), default="config",
                   help="semáforo en todos los escenarios: como en la configuración, activado, desactivado o ambos "
                        "(cada tope con y sin semáforo)")  # fmt: skip
    p.add_argument("--largo", type=float, metavar="M", help="largo del tramo en m (por defecto, [road] length)")
    p.add_argument("--carriles", type=int, nargs="+", metavar="N",
                   help="carriles de la configuración que se conservan, renumerados desde 0 en ese orden (por "
                        "defecto, todos)")  # fmt: skip
    p.add_argument("--sin", nargs="+", default=[], metavar="CLAVE",
                   help="tipos de vehículo que no participan, p. ej. bike motorbike")  # fmt: skip
    p.add_argument("--replicas", type=int, help="réplicas por escenario (por defecto, [execution] replicas)")
    p.add_argument("--run", type=float, help="s de proceso de cada réplica (por defecto, [execution] run)")
    p.add_argument("--redibujar", metavar="CARPETA",
                   help="no simula: vuelve a dibujar la gráfica de una comparación ya corrida (sin sobrescribir)")  # fmt: skip
    _add_json(p)
    return p


def _bumps(values: list[str] | None, config_positions, length: float) -> tuple[tuple[float, ...], ...]:
    """Topes de cada escenario: «sin» (ninguno), una posición en m o varias separadas por comas (p. ej. 35,70).
    Por defecto, sin tope y con los de [[speed_bump]] (o uno a la mitad del tramo si no hay)."""
    if values is None:
        configured = as_positions(config_positions)
        return ((), configured or (length / 2,))
    out: list[tuple[float, ...]] = []
    for text in values:
        if text.lower() == "sin":
            out.append(())
            continue
        try:
            out.append(tuple(sorted(float(part) for part in text.split(","))))
        except ValueError:
            raise ConfigError(f"--topes: {text!r} debe ser «sin», una posición en m o varias separadas por comas "
                              "(p. ej. 35,70)") from None
    return tuple(out)


@_json_aware
def emissions(argv: list[str] | None = None) -> Path:
    """Corre una comparación de emisiones entre escenarios (o la redibuja) y devuelve su carpeta."""
    from trafico.emission_scenarios import (
        BASE_CONFIG_NAME, PLOT_NAME as EMIS_PLOT, SUMMARY_NAME as EMIS_SUMMARY, Scenarios, build_configs, load,
        metadata, run_scenarios, write_outputs,
    )  # fmt: skip
    from trafico.emission_scenarios import results_metrics as emission_metrics
    from trafico.movement import _free_path
    from trafico.plotting import plot_emission_comparison
    from trafico.settings import _set_key
    from trafico.variants import reduce_config

    argv = sys.argv[1:] if argv is None else argv
    parser = build_emissions_parser()
    args = parser.parse_args(argv)
    try:
        if args.redibujar:
            folder = Path(args.redibujar).expanduser()
            meta, data = load(folder)
            if meta.get("modo") == "separacion":
                from trafico.bump_spacing import analyze, plot_name
                from trafico.plotting import plot_bump_spacing

                path = _free_path(folder / plot_name(meta))
                plot_bump_spacing(path, meta, data, analyze(meta, data))
                print(f"Gráfica en {path}")
                return folder
            path = _free_path(folder / EMIS_PLOT)
            plot_emission_comparison(path, meta, data)
            print(f"Gráfica en {path}")
            return folder
        target = resolve_target(args.target)
        settings = load_settings(target.config)
        base, opts = settings.sim, settings.run
        lanes = tuple(args.carriles) if args.carriles else None
        reduced = reduce_config(base, lanes, tuple(args.sin))
        length = args.largo if args.largo is not None else reduced.length
        lights_sweep = args.separacion_semaforos is not None
        if args.separacion is not None and lights_sweep:
            parser.error("--separacion y --separacion-semaforos no se combinan: corre uno y luego el otro")
        if args.cadena is not None and args.separacion is None and not lights_sweep:
            parser.error("--cadena requiere --separacion o --separacion-semaforos")
        given = [f for f, v in (("--primer-semaforo", args.primer_semaforo), ("--ciclo", args.ciclo),
                                ("--desfase", args.desfase), ("--separaciones-fijas", args.separaciones_fijas))
                 if v is not None]  # fmt: skip
        if given and not lights_sweep:
            parser.error(f"{', '.join(given)} requiere --separacion-semaforos")
        if args.separacion is not None or lights_sweep:
            return _spacing("semaforo" if lights_sweep else "tope", args, argv, parser, target, settings, reduced, lanes,
                            length)  # fmt: skip
        lights = {"config": (None,), "si": (True,), "no": (False,), "ambos": (True, False)}[args.semaforo]
        s = Scenarios(
            bumps=_bumps(args.topes, tuple(b.position for b in reduced.bumps), length),
            bump_lanes=tuple(args.carriles_tope) if args.carriles_tope else None,
            lights=lights, length=args.largo, lanes=lanes, without=tuple(args.sin),
            replicas=args.replicas if args.replicas is not None else opts.replicas,
            run=args.run if args.run is not None else base.run,
        )  # fmt: skip
        if s.replicas < 1:
            raise ConfigError("--replicas debe ser al menos 1")
        if len(s.bumps) * len(s.lights) > MAX_TYPES:
            raise ConfigError(f"se admiten hasta {MAX_TYPES} escenarios (colores de la gráfica)")
        configs = build_configs(base, s)  # valida todos los escenarios antes de correr
        seed = opts.seed if opts.seed is not None else secrets.randbelow(2**32)
        meta = metadata(base, s, configs, seed, argv)
    except ConfigError as exc:
        parser.error(str(exc))

    print(f"Comparación de emisiones · configuración {target.config}")
    print(f"{len(configs)} escenarios × {s.replicas} réplicas · run {s.run:g} s × {base.time_scale:g} = "
          f"{s.run * base.time_scale:g} s simulados · semilla {seed}")  # fmt: skip
    if not any(v for v in meta["velocidad_tope_kmh"].values()) and any(s.bumps):
        print("Aviso: ningún tipo que participa tiene speed_bump_kmh: el tope no frena a nadie")

    def progress(name: str, done: int, total: int) -> None:
        if done == 1:
            sys.stderr.write(f"  {name}\n")
        _progress(done, total)

    data = run_scenarios(configs, s.replicas, seed, opts.workers, progress=progress if opts.progress else None)
    now = datetime.now()
    folder = make_run_dir(resolve_output_dir(opts.output_dir), safe_name(target.name or EMISSIONS_NAME), now)
    text = settings.text if opts.seed is not None else _set_key(settings.text, "execution", "seed", seed, "semilla usada")
    (folder / BASE_CONFIG_NAME).write_text(
        f"# Configuración base de la comparación {folder.name}\n# {meta['comando']}\n\n{text}", encoding="utf-8"
    )
    summary = write_outputs(folder, meta, data)
    print("\n" + summary)
    (folder / EMIS_SUMMARY).write_text(f"{meta['comando']}\n\n{summary}\n", encoding="utf-8")
    plot_emission_comparison(folder / EMIS_PLOT, meta, data)
    write_results(folder, document("emisiones", folder, command=meta["comando"], seed=seed, replicas=s.replicas,
                                   sim_seconds=meta["s_simulados"], metrics=emission_metrics(meta, data)))  # fmt: skip
    print(f"\nResultados en {folder}")
    return folder


def _cycle(text: str):
    """`--ciclo ROJO/VERDE[/AMARILLO]` en s, como semáforo de plantilla (la posición no se usa)."""
    from trafico.config import Light

    try:
        values = [float(v) for v in text.split("/")]
    except ValueError:
        values = []
    if len(values) not in (2, 3):
        raise ConfigError(f"--ciclo: {text!r} debe ser ROJO/VERDE o ROJO/VERDE/AMARILLO en s, p. ej. 30/30/3")
    return Light(0.0, red=values[0], green=values[1], yellow=values[2] if len(values) == 3 else 0.0)


def _spacing(kind, args, argv, parser, target, settings, reduced, lanes, length) -> Path:
    """`trafico-emisiones --separacion` (kind «tope») o `--separacion-semaforos` (kind «semaforo»): barrido de la
    distancia entre dos topes o dos semáforos (ver bump_spacing.py)."""
    from trafico.bump_spacing import (
        DEFAULT_DISTANCES, LIGHT, LIGHT_CYCLE, LIGHT_DISTANCES, LIGHT_FIRST, LIGHT_ROOM, NOUNS,
        SUMMARY_NAME as SPACING_SUMMARY, Spacing, analyze, build_configs, metadata, plot_name, write_outputs,
    )  # fmt: skip
    from trafico.bump_spacing import results_metrics as spacing_metrics
    from trafico.emission_scenarios import BASE_CONFIG_NAME, Scenarios
    from trafico.emission_scenarios import metadata as base_metadata
    from trafico.emission_scenarios import run_scenarios
    from trafico.movement import _free_path
    from trafico.plotting import plot_bump_spacing
    from trafico.settings import _set_key

    base, opts = settings.sim, settings.run
    light = kind == LIGHT
    flag = "--separacion-semaforos" if light else "--separacion"
    nn = NOUNS[kind]
    notes = []
    try:
        if args.semaforo != "config" and light:
            raise ConfigError("--separacion-semaforos no admite --semaforo: los semáforos son los del barrido")
        if args.semaforo == "ambos":
            raise ConfigError("--separacion no admite --semaforo ambos (usa config, si o no)")
        fixed: tuple[float, ...] = ()
        if args.topes is not None:
            if not light:
                raise ConfigError("--separacion no se combina con --topes (los topes son el primero y el que se aleja)")
            scenarios = _bumps(args.topes, (), length)
            if len(scenarios) != 1:
                raise ConfigError("con --separacion-semaforos, --topes da los topes fijos de todos los escenarios: un solo "
                                  "valor, p. ej. 35,100 (o «sin»)")  # fmt: skip
            fixed = scenarios[0]
        distances = tuple(args.separacion_semaforos if light else args.separacion) or (
            LIGHT_DISTANCES if light else DEFAULT_DISTANCES)
        if any(d <= 0 for d in distances) or len(set(distances)) != len(distances):
            raise ConfigError(f"{flag}: cada distancia debe ser mayor que 0 m y no repetirse")
        if args.cadena is not None and args.cadena < 2:
            raise ConfigError(f"--cadena debe ser de al menos 2 {nn.many}")
        if not 0 < args.tolerancia < 1 or not 0 < args.umbral < 1:
            raise ConfigError("--tolerancia y --umbral deben estar entre 0 y 1")
        count = args.cadena or 2
        street = args.largo
        if light:
            first = args.primer_semaforo if args.primer_semaforo is not None else LIGHT_FIRST
            configured = next((lt for lt in reduced.lights if not lt.pedestrian), None)
            cycle = _cycle(args.ciclo) if args.ciclo is not None else configured or LIGHT_CYCLE
            fixed_gaps = tuple(args.separaciones_fijas or ())
            if any(g <= 0 for g in fixed_gaps):
                raise ConfigError("--separaciones-fijas: cada separación debe ser mayor que 0 m")
            span = max(distances) + sum(fixed_gaps) if fixed_gaps else (count - 1) * max(distances)
            needed = first + span + LIGHT_ROOM
            if street is None and needed > length:
                street = float(np.ceil(needed / EMIS_BIN) * EMIS_BIN)
                notes.append(f"el tramo se amplía de {length:g} a {street:g} m para que quepa la separación mayor y la "
                             f"aceleración tras el último semáforo (usa --largo para fijarlo)")  # fmt: skip
        else:
            first = args.primer_tope if args.primer_tope is not None else (
                reduced.bumps[0].position if reduced.bumps else length / 4)
            cycle, fixed_gaps = None, ()
        spacing = Spacing(
            first=first, distances=tuple(sorted(distances)),
            light=None if light else {"config": None, "si": True, "no": False}[args.semaforo],
            bump_lanes=tuple(args.carriles_tope) if args.carriles_tope else None, length=street,
            tolerance=args.tolerancia, threshold=args.umbral, count=count,
            replicas=args.replicas if args.replicas is not None else opts.replicas,
            run=args.run if args.run is not None else base.run,
            kind=kind, cycle=cycle, offset_mode=(args.desfase or "rojo").replace(" ", ""), bumps=fixed,
            fixed_gaps=fixed_gaps,
        )  # fmt: skip
        if spacing.replicas < 1:
            raise ConfigError("--replicas debe ser al menos 1")
        configs, skipped = build_configs(reduced, spacing)
        kept = [d for d in spacing.distances if d not in skipped]
        seed = opts.seed if opts.seed is not None else secrets.randbelow(2**32)
        scen = Scenarios(bumps=((),), bump_lanes=spacing.bump_lanes, lights=(None,), length=street, lanes=lanes,
                         without=tuple(args.sin), replicas=spacing.replicas, run=spacing.run)  # fmt: skip
        meta = metadata(base_metadata(base, scen, configs, seed, argv), spacing, kept, skipped,
                        configs[0][1] if light else None)  # fmt: skip
    except ConfigError as exc:
        parser.error(str(exc))

    print(f"Separación entre {nn.many} · configuración {target.config}")
    print(f"{len(configs)} escenarios × {spacing.replicas} réplicas · run {spacing.run:g} s × {base.time_scale:g} = "
          f"{spacing.run * base.time_scale:g} s simulados · semilla {seed}")  # fmt: skip
    for note in notes:
        print(f"Aviso: {note}")
    if skipped:
        print("Aviso: no caben en el tramo y se omiten las separaciones " + ", ".join(f"{d:g}" for d in skipped) + " m")

    def progress(name: str, done: int, total: int) -> None:
        if done == 1:
            sys.stderr.write(f"  {name}\n")
        _progress(done, total)

    data = run_scenarios(configs, spacing.replicas, seed, opts.workers, progress=progress if opts.progress else None)
    default_name = LIGHT_SPACING_NAME if light else SPACING_NAME
    folder = make_run_dir(resolve_output_dir(opts.output_dir), safe_name(target.name or default_name), datetime.now())
    text = settings.text if opts.seed is not None else _set_key(settings.text, "execution", "seed", seed, "semilla usada")
    (folder / BASE_CONFIG_NAME).write_text(
        f"# Configuración base de la comparación {folder.name}\n# {meta['comando']}\n\n{text}", encoding="utf-8"
    )
    summary = write_outputs(folder, meta, data)
    print("\n" + summary)
    (folder / SPACING_SUMMARY).write_text(f"{meta['comando']}\n\n{summary}\n", encoding="utf-8")
    plot_bump_spacing(_free_path(folder / plot_name(meta)), meta, data, analyze(meta, data))
    write_results(folder, document("separacion", folder, command=meta["comando"], seed=seed,
                                   replicas=spacing.replicas, sim_seconds=meta["s_simulados"],
                                   metrics=spacing_metrics(meta, data)))  # fmt: skip
    print(f"\nResultados en {folder}")
    return folder


def emissions_main(argv: list[str] | None = None) -> None:
    emissions(argv)


# ------------------------------------------------------------------ trafico-calibrar

CALIBRATION_NAME = "calibracion"


def build_calibration_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="trafico-calibrar",
        description=(
            "Mide un tipo de vehículo de la configuración en situaciones estándar (descarga de una cola en verde, "
            "flujo continuo con demanda creciente, cola detenida) y lo compara con valores de referencia publicados "
            "(flujo de saturación, tiempo perdido al arrancar, capacidad, densidad de embotellamiento). Guarda la "
            "gráfica, tablas, el resumen y resultados.json en <output.dir>/<fecha-hora>_<nombre>/."
        ),
    )
    p.add_argument(
        "target", nargs="?", default=None, metavar="nombre | carpeta",
        help=f"como en `trafico`: una carpeta de la que se lee su {CONFIG_NAME}, o el nombre de la calibración "
             f"(por defecto, {CALIBRATION_NAME}) con {default_config_path()}",
    )  # fmt: skip
    p.add_argument("--tipo", default="car", metavar="CLAVE",
                   help="tipo de vehículo que se mide (por defecto car; las referencias son de autos)")  # fmt: skip
    p.add_argument("--colas", type=int, default=30, metavar="N",
                   help="colas que se descargan, cada una con su semilla (por defecto 30)")  # fmt: skip
    _add_json(p)
    return p


@_json_aware
def calibrate(argv: list[str] | None = None) -> Path:
    """Corre una calibración y devuelve su carpeta."""
    from trafico.calibration import FLOW_RATES, PLOT_NAME as CAL_PLOT, compare, measure, results_metrics, write_outputs
    from trafico.plotting import plot_calibration
    from trafico.settings import _set_key

    argv = sys.argv[1:] if argv is None else argv
    parser = build_calibration_parser()
    args = parser.parse_args(argv)
    command = ("uv run trafico-calibrar " + " ".join(argv)).strip()
    try:
        target = resolve_target(args.target)
        settings = load_settings(target.config)
        base, opts = settings.sim, settings.run
        seed = opts.seed if opts.seed is not None else secrets.randbelow(2**32)
        print(f"Calibración de {args.tipo} · configuración {target.config} · semilla {seed}")
        print(f"  {args.colas} colas que se descargan en verde y {len(FLOW_RATES)} demandas en flujo continuo…")
        m = measure(base, args.tipo, queues=args.colas, seed=seed, workers=opts.workers)
    except ConfigError as exc:
        parser.error(str(exc))
    rows = compare(m)
    folder = make_run_dir(resolve_output_dir(opts.output_dir), safe_name(target.name or CALIBRATION_NAME), datetime.now())
    text = settings.text if opts.seed is not None else _set_key(settings.text, "execution", "seed", seed, "semilla usada")
    (folder / "config_base.toml").write_text(
        f"# Configuración de la calibración {folder.name}\n# {command}\n\n{text}", encoding="utf-8"
    )
    print("\n" + write_outputs(folder, m, rows))
    plot_calibration(folder / CAL_PLOT, m, rows)
    write_results(folder, document("calibracion", folder, command=command, seed=seed, replicas=m["colas"],
                                   sim_seconds=m["s_simulados"], metrics=results_metrics(m, rows)))  # fmt: skip
    print(f"\nResultados en {folder}")
    return folder


def calibrate_main(argv: list[str] | None = None) -> None:
    calibrate(argv)
