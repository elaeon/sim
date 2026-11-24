"""Command line: `uv run traffic [name | folder]`."""

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

from traffic.config import (
    DT, FUEL_CO2_G_PER_L, FUEL_LABELS, MAX_TYPES, PEDESTRIAN_YIELD, POLLUTANT_LABELS, POLLUTANTS, as_positions,
)
from traffic.emissions import EMIS_BIN, acceleration_table, emissions_label, emitted, unit
from traffic.fuel import constant_speed_table, cost, liters
from traffic.metrics import (
    EMIS_SERIES, ENTRY_QUEUE, EXIT_QUEUE, LANE_EXIT_FLOW, LANE_SATURATION, LANE_SPEED, SERIES,
)
from traffic.results import document, read_results, write_results
from traffic.runner import Aggregate, run_parallel
from traffic.settings import (
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

PLOT_NAME = "passenger_mobility.png"
PAX_PLOT_NAME = "passenger_distribution.png"
CSV_NAME = "series.csv"
EMIS_PLOT_NAME = "emissions_by_position.png"
EMIS_CSV_NAME = "emissions_by_position.csv"
SUMMARY_NAME = "summary.txt"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="traffic",
        description=(
            "Simulate the vehicle flow on a road segment with a traffic light at the end and plot the "
            f"passenger mobility capacity. Every parameter is read from {CONFIG_NAME}; "
            "the results and a copy of the configuration are saved in "
            "<output.dir>/<date-time>_<name>/."
        ),
    )
    p.add_argument(
        "target",
        nargs="?",
        default=None,
        metavar="name | folder",
        help=(
            f"an existing folder (e.g. results/<ID>_<name>) whose {CONFIG_NAME} is read, "
            f"or the name of the run, added to the results folder, using {default_config_path()}"
        ),
    )
    _add_json(p)
    return p


def _progress(done: int, total: int) -> None:
    width = 30
    filled = round(width * done / total)
    sys.stderr.write(f"\r  replicas [{'█' * filled}{'·' * (width - filled)}] {done}/{total}")
    if done == total:
        sys.stderr.write("\n")
    sys.stderr.flush()


def _fmt(value: float, digits: int = 1) -> str:
    return "—" if not np.isfinite(value) else f"{value:,.{digits}f}"


def summary_rows(agg: Aggregate) -> tuple[list[tuple[str, np.ndarray, int, str]], dict]:
    """Rows of the summary table, (label, value per type, decimals, results.json key), and the means across
    replicas of each statistic."""
    cfg = agg.cfg
    s = {k: v.mean for k, v in agg.summary.items()}
    sd = {k: v.std for k, v in agg.summary.items()}
    minutes = cfg.sim_seconds / 60.0
    footprint = np.array([sp.length + sp.gap_run for sp in cfg.specs])
    rows = [
        ("Arrivals (veh/min)", s["arrived_veh"] / minutes, 1, "llegadas_veh_min"),
        ("Vehicles crossing", s["crossed_veh"], 1, "vehiculos_cruzan"),
        ("Passengers crossing", s["crossed_pax"], 1, "pasajeros_cruzan"),
        ("  ± σ across replicas", sd["crossed_pax"], 1, "pasajeros_cruzan_sd"),
        ("Passengers/min crossing", s["crossed_pax"] / minutes, 1, "pasajeros_min"),
        ("Passengers per vehicle", s["pax_per_veh"], 2, "pasajeros_por_vehiculo"),
        ("Pax/m of lane, moving", s["pax_per_veh"] / footprint, 2, "pasajeros_por_m_carril"),
        ("Mean travel time (s)", s["travel_time"], 1, "tiempo_recorrido_s"),
        ("Mean wait in entry queue (s)", s["queue_wait"], 1, "tiempo_cola_entrada_s"),
        ("Free-flow time (s)", np.array([cfg.free_flow_time(k) + sp.expected_stop_time
                                           for k, sp in enumerate(cfg.specs)]), 1, "tiempo_flujo_libre_s"),
        ("Mean speed (km/h)", s["mean_speed"], 1, "velocidad_media_kmh"),
        ("On the segment at the end", s["on_road"], 1, "en_tramo_final"),
        ("In the entry queue at the end", s["queued"], 1, "en_cola_final"),
        *([("On the segment at the start", s["initial_veh"], 1, "en_tramo_inicio")] if any(cfg.lane_initial_occupancy) else []),
        ("Lane changes/veh", np.divide(s["lane_changes_type"], s["entered_veh"], out=np.full(cfg.n_types, np.nan),
                                            where=s["entered_veh"] > 0), 2, "cambios_carril_por_veh"),  # fmt: skip
    ]
    active = [k for k, rate in enumerate(cfg.rates) if rate > 0]  # participating types
    if any(cfg.specs[k].stop_position is not None for k in active):
        rows.insert(9, ("Mean time at the stop (s)", s["stop_time"], 1, "tiempo_parada_s"))
    if cfg.bottleneck_active:
        rows.append(("Halts (bottleneck)", s["bottleneck_stops"], 1, "detenciones_bottleneck"))
        rows.append(("Mean time halted (s)", s["bottleneck_time"], 1, "tiempo_detenido_s"))
    passers = np.array([sp.pass_in_lane for sp in cfg.specs])
    if passers[active].any():
        # Overtakes within the lane per vehicle that entered; «—» for the types that do not overtake that way.
        per_veh = np.divide(s["in_lane_passes"], s["entered_veh"], out=np.full(cfg.n_types, np.nan),
                            where=passers & (s["entered_veh"] > 0))  # fmt: skip
        rows.append(("In-lane overtakes/veh", per_veh, 2, "rebases_carril_por_veh"))
    if any(cfg.specs[k].cargo_prob > 0 for k in active):
        # Freight types count as vehicles, but not in the passenger rows.
        cargo = np.divide(100.0 * s["arrived_cargo"], s["arrived_veh"], out=np.full(cfg.n_types, np.nan),
                          where=s["arrived_veh"] > 0)  # fmt: skip
        rows.insert(1, ("Carrying goods (%)", cargo, 1, "mercancia_pct"))
    rows += _emission_rows(cfg, s)
    rows += _fuel_rows(cfg, s)
    return rows, {"s": s, "sd": sd}


def format_summary(agg: Aggregate) -> str:
    cfg = agg.cfg
    rows, stats = summary_rows(agg)
    s = stats["s"]
    active = [k for k, rate in enumerate(cfg.rates) if rate > 0]  # participating types
    w0 = max(len(r[0]) for r in rows) + 2
    widths = {k: max(10, len(cfg.specs[k].name) + 2) for k in active}
    speed = cfg.sim_seconds / agg.replica_wall.mean
    lines = ["", " " * w0 + "".join(f"{cfg.specs[k].name:>{widths[k]}}" for k in active)]
    for label, values, digits, _ in rows:
        lines.append(f"{label:<{w0}}" + "".join(f"{_fmt(values[k], digits):>{widths[k]}}" for k in active))
    minutes_run = cfg.sim_seconds / 60.0
    lines += ["", f"Crossing {cfg.line_name} per lane (veh/min): " + " · ".join(
        f"{lane}: {_fmt(v / minutes_run)}" for lane, v in enumerate(s["lane_crossed"]))]  # fmt: skip
    if any(c > 0 for c in cfg.lane_exit_capacity):
        lines.append("Line closed by the exit queue (% of the time, the next one does not fit): " + " · ".join(
            f"{lane}: {_fmt(100 * v)}" for lane, v in enumerate(s["exit_blocked"])
            if cfg.lane_exit_capacity[lane] > 0))  # fmt: skip
    lines += _pedestrian_lines(cfg, s)
    lines += _acceleration_lines(cfg)
    lines += _cruise_lines(cfg)
    lines += [
        f"Lane changes per replica: {_fmt(s['lane_changes'][0])}",
        f"Nominal process time: {cfg.run:g} s ≙ {cfg.sim_seconds:g} simulated s "
        f"({cfg.n_ticks:,} steps of {DT} s)",
        f"Wall time: {agg.wall:.2f} s for {agg.replicas} replicas with {agg.workers} process(es) · "
        f"{float(agg.replica_wall.mean):.2f} s per replica · {float(speed):,.0f} simulated s per real s",
        f"Peak memory (RSS) per process: {agg.max_rss_kb / 1024:.1f} MiB",
    ]
    return "\n".join(lines)


def _pedestrian_lines(cfg, s: dict) -> list[str]:
    """Pedestrians at each pedestrian light and each bump with a pedestrian crossing: how many crossed, how many times
    the crossing closed, the mean wait of those who crossed and the time it was closed to vehicles (red)."""
    spots = cfg.pedestrian_spots
    if not spots or "pedestrians" not in s:
        return []
    lines = ["", "Pedestrians (mean per replica):"]
    for (kind, pos), (crossed, closings, closed, wait) in zip(spots, s["pedestrians"]):
        name = "pedestrian light" if kind == "semáforo" else "pedestrian crossing at the bump"
        mean_wait = wait / crossed if crossed > 0 else np.nan
        lines.append(f"  {name} at {pos:g} m: {_fmt(crossed)} crossed in {_fmt(closings)} closures · mean wait "
                     f"{_fmt(mean_wait)} s · red for vehicles {_fmt(closed)} s "
                     f"({_fmt(100 * closed / cfg.sim_seconds)} % of the time)")  # fmt: skip
    return lines


def _emission_rows(cfg, s: dict) -> list[tuple[str, np.ndarray, int, str]]:
    """Per emitted pollutant: per km travelled, per full trip along the segment and the excess over the same
    trip at constant speed (stops, starts, bump, queue). «—» for the types that do not emit it."""
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
            (f"{label} ({name} per trip along the segment)", per_km * cfg.length / 1000.0, 2 if scale == 1 else 1,
             f"{pol}_{name}_recorrido"),
            (f"{label} excess vs free flow (%)", excess, 1, f"{pol}_exceso_flujo_libre_pct"),
        ]
    return rows


def _burners(cfg) -> list[int]:
    """Participating types that burn fuel (they have `fuel` and emit CO2)."""
    return [k for k, sp in enumerate(cfg.specs) if cfg.rates[k] > 0 and sp.burns]


def _fuel_rows(cfg, s: dict) -> list[tuple[str, np.ndarray, int, str]]:
    """Fuel (from the CO2 emitted) and its cost: per km, per full trip along the segment, the extra spent over
    the same trip at constant speed and the total of the run. «—» for the types without fuel or without a
    price."""
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
        ("Fuel (L/100 km)", l_km * 100.0, 2, "combustible_l_100km"),
        ("Fuel per trip (mL)", l_km * trip * 1000.0, 1, "combustible_recorrido_ml"),
        ("  extra vs free flow (mL)", extra_l * trip * 1000.0, 1, "combustible_extra_recorrido_ml"),
        (f"Cost per trip ({cur})", trip_cost, 3, "costo_recorrido"),
        (f"  extra vs free flow ({cur})", extra_cost, 3, "costo_extra_recorrido"),
        ("Fuel in the run (L)", total_l, 2, "combustible_corrida_l"),
        (f"Cost in the run ({cur})", total_cost, 2, "costo_corrida"),
    ]


def _cruise_lines(cfg) -> list[str]:
    """Reference table: fuel economy of each type at constant speed with the model (from CO2) and with the physical
    curve (with `mass_kg`), in L/100 km, km/L and cost per km."""
    burners = _burners(cfg)
    if not burners:
        return []
    lines = ["", "Efficiency at constant speed (from the CO2 of the emission model; «physical»: idle + power to "
             "overcome rolling and air resistance, for reference):"]  # fmt: skip
    for k in burners:
        spec = cfg.specs[k]
        price = cfg.price(spec.fuel)
        extra = f", {price:g} {cfg.currency}/L" if price is not None else ", no price"
        mass = f", {spec.mass_kg:g} kg" if spec.mass_kg is not None else ""
        lines.append(f"  {spec.name} ({FUEL_LABELS[spec.fuel]}{extra}{mass}):")

        def fmt(l100: float) -> str:
            money = f", {l100 / 100 * price:.2f} {cfg.currency}/km" if price is not None else ""
            return f"{l100:.1f} L/100 km ({100 / l100:.1f} km/L{money})"

        for row in constant_speed_table(spec):
            physical = f" · physical {fmt(row.physical_l_100km)}" if row.physical_l_100km is not None else ""
            lines.append(f"    {row.kmh:g} km/h: {fmt(row.l_100km)}{physical}")
    return lines


def _acceleration_lines(cfg) -> list[str]:
    """Reference table: what each type emits when accelerating from 0 to 20, 20 to 40, 40 to 60 and 60 to 80 km/h at
    its accel and, in parentheses, at constant speed over the same time and distance."""
    _, types = emitted(cfg)
    if not types:
        return []
    lines = ["", "Emissions when accelerating (Int Panis et al., 2006 model, at the accel of each type; in "
             "brackets, at constant speed over the same time and distance):"]  # fmt: skip
    over = False
    for k in types:
        spec = cfg.specs[k]
        lines.append(f"  {spec.name} ({spec.accel:g} m/s², maximum speed {spec.fastest_kmh:g} km/h):")
        for row in acceleration_table(spec):
            parts = []
            for pol, grams in row.grams.items():
                name, scale = unit(pol)
                parts.append(f"{POLLUTANT_LABELS[pol]} {grams * scale:,.2f} ({row.cruise[pol] * scale:,.2f}) {name}")
            mark = " *" if row.v2 > spec.fastest_kmh else ""
            over |= bool(mark)
            lines.append(f"    {row.v1:g}→{row.v2:g} km/h{mark} in {row.seconds:.1f} s and {row.meters:.0f} m: "
                         + " · ".join(parts))  # fmt: skip
    if over:
        lines.append("  * goes past the maximum speed of the type: outside the range in which it drives (and in which the "
                     "model was fitted); take it with caution")  # fmt: skip
    return lines


def write_emissions_csv(agg: Aggregate, path: Path) -> None:
    """g/(m·h) of each emitted pollutant, per lane, in EMIS_BIN m bins along the segment (mean
    across replicas)."""
    cfg = agg.cfg
    pols, _ = emitted(cfg)
    pos = agg.summary["emissions_pos"].mean / EMIS_BIN / (cfg.sim_seconds / 3600.0)
    edges = np.arange(pos.shape[2] + 1) * EMIS_BIN
    cols = [edges[:-1], np.minimum(edges[1:], cfg.length)]
    header = ["x_start_m", "x_end_m"]
    for p in pols:
        for lane in range(cfg.lanes):
            cols.append(pos[p, lane])
            header.append(f"{POLLUTANTS[p]}_g_m_h_lane{lane}")
    np.savetxt(path, np.column_stack(cols), delimiter=",", header=",".join(header), comments="", fmt="%.6g")


def write_csv(agg: Aggregate, path: Path) -> None:
    cfg = agg.cfg
    t = agg.times
    cols = [t, t / cfg.time_scale]
    header = ["t_sim_s", "t_process_s"]
    for key in SERIES:
        mean, std = agg.series[key].mean, agg.series[key].std
        for k, sp in enumerate(cfg.specs):
            cols += [mean[:, k], std[:, k]]
            header += [f"{key}_{sp.name}_mean", f"{key}_{sp.name}_sd"]
    pols, types = emitted(cfg)
    for p in pols:  # g/min of each emitted pollutant, per type that emits it
        stats = agg.series[EMIS_SERIES[p]]
        for k in types:
            if cfg.specs[k].emission_coefs(POLLUTANTS[p]) is not None:
                cols += [stats.mean[:, k], stats.std[:, k]]
                header += [f"{EMIS_SERIES[p]}_g_min_{cfg.specs[k].name}_mean", f"{EMIS_SERIES[p]}_g_min_{cfg.specs[k].name}_sd"]
    if "co2" in [POLLUTANTS[p] for p in pols]:  # L/min of fuel, from CO2
        stats = agg.series[EMIS_SERIES[POLLUTANTS.index("co2")]]
        for k in _burners(cfg):
            g_per_l = FUEL_CO2_G_PER_L[cfg.specs[k].fuel]
            cols += [stats.mean[:, k] / g_per_l, stats.std[:, k] / g_per_l]
            header += [f"fuel_l_min_{cfg.specs[k].name}_mean", f"fuel_l_min_{cfg.specs[k].name}_sd"]
    lane_series = [
        (LANE_SATURATION, "saturation"), (LANE_SPEED, "speed_kmh"), (LANE_EXIT_FLOW, "crossing_veh_min"),
        (ENTRY_QUEUE, "entry_queue_veh"),
    ]  # fmt: skip
    if any(c > 0 for c in cfg.lane_exit_capacity):
        lane_series.append((EXIT_QUEUE, "exit_queue_m"))
    for key, label in lane_series:
        stats = agg.series[key]
        for lane in range(cfg.lanes):
            cols += [stats.mean[:, lane], stats.std[:, lane]]
            header += [f"{label}_lane{lane}_mean", f"{label}_lane{lane}_sd"]
    np.savetxt(path, np.column_stack(cols), delimiter=",", header=",".join(header), comments="", fmt="%.6g")


def _rate_text(rate) -> str:
    """A variable or fixed rate, e.g. «1.5 ± 1 in [0.5, 3], mean 1.6»."""
    if not rate.variable:
        return f"{rate.expected:g}"
    return f"{rate.mean:g} ± {rate.std:g} in [{rate.min:g}, {rate.max:g}], mean {rate.expected:.3g}"


def _lane_lines(cfg) -> list[str]:
    """Speed limit, lanes without a traffic light, stops and other per-lane or per-type settings."""
    lines = []
    if cfg.lane_speed_limit is not None:
        limits = " · ".join(f"{i}: {v:g}" for i, v in enumerate(cfg.lane_max_kmh))
        lines.append(f"Speed limit per lane (km/h, 0 = right): {limits}")
    bumps = cfg.bumps
    if bumps:
        slow = [f"{sp.name} ≤ {sp.speed_bump_kmh:g}" for k, sp in enumerate(cfg.specs)
                if sp.speed_bump_kmh is not None and cfg.rates[k] > 0]  # fmt: skip

        def where(bump) -> str:
            lanes = cfg.bump_lanes(bump)
            return "all lanes" if len(lanes) == cfg.lanes else (
                ("lane " if len(lanes) == 1 else "lanes ") + ", ".join(map(str, lanes)))

        at = " · ".join(f"{b.position:g} m ({where(b)})" for b in bumps)
        lines.append(f"{'Speed bumps' if len(bumps) > 1 else 'Speed bump'} at {at}, km/h over it: "
                     + (" · ".join(slow) if slow else "no type slows down (no speed_bump_kmh)"))  # fmt: skip
    for bump in cfg.crossings:
        lines.append(f"Pedestrian crossing at the bump at {bump.position:g} m (pedestrians/min "
                     f"{_rate_text(bump.pedestrian_crossing)}; waits {PEDESTRIAN_YIELD:g} s and crosses in "
                     f"{bump.pedestrian_time:g} s): vehicles stop completely while someone is crossing")  # fmt: skip
    for light in cfg.lights:
        if light.pedestrian:
            prev = cfg.previous_light(light)
            when = (f"with the light at {prev.position:g} m on red" if prev else "without waiting for another light")
            lines.append(f"Pedestrian light at {light.position:g} m (pedestrians/min {_rate_text(light.pedestrian_crossing)}): "
                         f"green for vehicles unless pedestrians are waiting, {when}; yellow {light.yellow:g} s, "
                         f"red {light.red:g} s, minimum green {light.green:g} s between crossings")  # fmt: skip
        free = [f"{sp.name} (lane {sp.lane})" for k, sp in enumerate(cfg.specs)
                if cfg.ignores_light(k, light) and cfg.rates[k] > 0]  # fmt: skip
        if free:
            at = f" at {light.position:g} m" if cfg.inner_lights else ""
            lines.append(f"No traffic light{at}, they keep going on red and yellow (exclusive lane in free_lanes): "
                         + ", ".join(free))  # fmt: skip
    stops = [
        f"{sp.name} at {sp.stop_position:g} m, {sp.stop_time_mean:g} ± {sp.stop_time_std:g} s"
        for k, sp in enumerate(cfg.specs) if sp.stop_position is not None and cfg.rates[k] > 0
    ]  # fmt: skip
    if stops:
        where = "before the traffic light" if cfg.has_light else "on the segment"
        lines.append(f"Stop {where} (passengers getting off and on): " + " · ".join(stops))
    variable = [(sp.name, cfg.rate(k)) for k, sp in enumerate(cfg.specs) if cfg.rate(k).variable]
    if variable:
        lines.append(f"Variable demand (veh/min, new rate every {cfg.rate_interval:g} s): " + " · ".join(
            f"{name} {r.mean:g} ± {r.std:g} in [{r.min:g}, {r.max:g}], mean {r.expected:.3g}" for name, r in variable))
    speeds = [sp for k, sp in enumerate(cfg.specs) if sp.speed_std > 0 and cfg.rates[k] > 0]
    if speeds:
        lines.append("Variable maximum speed (km/h, drawn per vehicle): " + " · ".join(
            f"{sp.name} {sp.speed_kmh:g} ± {sp.speed_std:g} in [{sp.slowest_kmh:g}, {sp.fastest_kmh:g}]" for sp in speeds))
    reserved = sorted(cfg.reserved_lanes)
    if reserved:
        owners = {ln: [s.name for s in cfg.specs if s.lane == ln] for ln in reserved}
        lines.append("Exclusive lanes: " + " · ".join(f"{ln}: {', '.join(owners[ln])}" for ln in reserved))
    if cfg.bottleneck_active:
        lo, hi = cfg.bottleneck_zone()
        per_type = " · ".join(
            f"{sp.name} {100 * cfg.bottleneck_prob(k):g} % {'%g ± %g s' % cfg.bottleneck_time(k)}"
            for k, sp in enumerate(cfg.specs) if cfg.rates[k] > 0 and cfg.bottleneck_prob(k) > 0
        )  # fmt: skip
        lines.append(f"Bottleneck (halts in lane(s) {', '.join(map(str, cfg.bottleneck_lanes()))} "
                     f"between {lo:g} and {hi:g} m): {per_type}")  # fmt: skip
    occupancy = cfg.lane_initial_occupancy
    if any(occupancy):
        lines.append("Initial condition, occupancy per lane (0 = right): "
                     + " · ".join(f"{i}: {100 * v:g} %" for i, v in enumerate(occupancy)))  # fmt: skip
    gradual = [sp for k, sp in enumerate(cfg.specs) if cfg.rates[k] > 0 and (sp.accel or sp.decel)]
    if gradual:
        def fmt(v):
            return "—" if v is None else f"{v:g}"
        lines.append("Gradual acceleration / braking (m/s², — = instantaneous): " + " · ".join(
            f"{sp.name} {fmt(sp.accel)} / {fmt(sp.decel)}" for sp in gradual))  # fmt: skip
    if any(c > 0 for c in cfg.lane_exit_capacity):
        lines.append("Exit queue per lane (accepts veh/min, length m): " + " · ".join(
            f"{lane}: " + (f"{c:g}/min, {st:g} m" if c > 0 else "no limit")
            for lane, (c, st) in enumerate(zip(cfg.lane_exit_capacity, cfg.lane_exit_storage))))  # fmt: skip
    if cfg.behavior.queue_reaction:
        lines.append("Stopped entry queue: each vehicle starts with its reaction time ([behavior] queue_reaction)")
    emis = emissions_label(cfg)
    if emis:
        lines.append(emis)
    fuel = _fuel_label(cfg)
    if fuel:
        lines.append(fuel)
    return lines


def _fuel_label(cfg) -> str | None:
    """Header line: fuel and price of each type that burns fuel, and those that have `fuel`
    but no CO2."""
    burners = _burners(cfg)
    if not burners:
        return None

    def price(fuel: str) -> str:
        p = cfg.price(fuel)
        return f"{p:g} {cfg.currency}/L" if p is not None else "no price"

    parts = [f"{cfg.specs[k].name} {FUEL_LABELS[cfg.specs[k].fuel]} ({price(cfg.specs[k].fuel)})" for k in burners]
    missing = [sp.name for k, sp in enumerate(cfg.specs) if cfg.rates[k] > 0 and sp.fuel and not sp.burns]
    return "Fuel (from the CO2 emitted): " + " · ".join(parts) + (
        f" · no consumption (no CO2 or no accel/decel): {', '.join(missing)}" if missing else "")


def _json_aware(command):
    """Add `--json` to a command: what it prints goes to stderr and stdout gets only the `results.json` of the
    folder it returns (without `--json` nothing changes). The command always receives the argument list."""

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
                        help="print the run's resultados.json to stdout (the rest of the output goes to stderr)")  # fmt: skip


def run_metrics(agg: Aggregate) -> dict:
    """`results.json` metrics of a `traffic` run: per type the same statistics as the summary table (mean across
    replicas), per lane, pedestrians and the execution."""
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
    """Run a full simulation and return its results folder."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        target = resolve_target(args.target)
        settings = load_settings(target.config)
    except ConfigError as exc:
        parser.error(str(exc))
    cfg, opts = settings.sim, settings.run
    # Name priority: the command's, [output] name, and otherwise that of the folder read.
    name = safe_name(target.name or opts.name or target.fallback_name)
    seed = opts.seed if opts.seed is not None else secrets.randbelow(2**32)
    workers = min(opts.workers or os.process_cpu_count() or 1, opts.replicas)

    now = datetime.now()
    run_dir = make_run_dir(resolve_output_dir(opts.output_dir), name, now)
    copy = config_copy_text(settings, seed, name, run_dir, now)
    (run_dir / CONFIG_NAME).write_text(copy, encoding="utf-8")

    header = "\n".join(
        [
            f"Run {run_dir.name} · configuration {target.config}",
            f"Segment {cfg.length:g} m · {cfg.lanes} lane(s) · "
            + (f"traffic lights {cfg.lights_label}" if cfg.inner_lights else
               f"traffic light {cfg.light_label}" if cfg.has_light else "no traffic light"),
            f"run {cfg.run:g} process s × {cfg.time_scale:g} = {cfg.sim_seconds:g} simulated s · "
            f"{opts.replicas} replicas in {workers} process(es) · seed {seed}",
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

    from traffic.plotting import plot_mobility, plot_passenger_distribution

    footer = f"run {run_dir.name} · seed {seed}"
    if opts.passengers_plot and any(rate > 0 and sp.carries_passengers for sp, rate in zip(cfg.specs, cfg.rates)):
        plot_passenger_distribution(agg, run_dir / PAX_PLOT_NAME, footer=footer)
    if opts.mobility_plot:
        plot_mobility(agg, run_dir / PLOT_NAME, footer=footer)
    if emitted(cfg)[0]:
        if opts.emisiones_csv:
            write_emissions_csv(agg, run_dir / EMIS_CSV_NAME)
        if opts.emissions_plot:
            from traffic.plotting import plot_emissions_by_position

            plot_emissions_by_position(agg, run_dir / EMIS_PLOT_NAME, footer=footer)
    if opts.animation:
        from traffic.movement import visualize

        print()
        for path in visualize(cfg, seed, settings.animation, run_dir, run_dir.name):
            print(f"  {path.name}")
    write_results(run_dir, document(
        "corrida", run_dir, command=("uv run traffic " + " ".join(argv)).strip(), seed=seed, replicas=opts.replicas,
        sim_seconds=cfg.sim_seconds, metrics=run_metrics(agg)))  # fmt: skip
    print(f"\nResults in {run_dir}")
    return run_dir


def main(argv: list[str] | None = None) -> None:
    run(argv)


# ------------------------------------------------------------------ traffic-view


def build_viewer_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="traffic-view",
        description=(
            "Show how the vehicles of a run move: simulate one replica again "
            "(identical, with the seed of the configuration copy) and save in the folder a "
            "space-time diagram per lane (PNG) and a top-down video (WebM by default; MP4 or GIF "
            "with [animation] format). "
            "The time window and the pace are read from [animation]; the options override them."
        ),
    )
    p.add_argument(
        "carpeta", nargs="?", default=None, metavar="folder",
        help="folder of a run (results/<ID>_<name>); by default, the most recent one",
    )  # fmt: skip
    p.add_argument("--start", dest="inicio", type=float, metavar="S",
                   help="simulated s where the window starts ([animation] start)")  # fmt: skip
    p.add_argument("--duration", dest="duracion", type=float, metavar="S",
                   help="simulated s of the window ([animation] duration; by default 2 cycles)")  # fmt: skip
    p.add_argument("--speed", dest="velocidad", type=float, metavar="X",
                   help="simulated s per s of video ([animation] speed)")  # fmt: skip
    p.add_argument("--replica", type=int, metavar="N", help="replica to show, 1 = the first one ([animation] replica)")
    p.add_argument("--format", dest="formato", choices=("webm", "mp4", "gif"), help="video format ([animation] format)")
    return p


def latest_run_dir(base: Path) -> Path | None:
    """Most recent run folder (names start with the date and time)."""
    runs = sorted(p for p in base.glob("*") if (p / CONFIG_NAME).is_file()) if base.is_dir() else []
    return runs[-1] if runs else None


def view(argv: list[str] | None = None) -> list[Path]:
    import dataclasses

    from traffic.movement import visualize
    from traffic.settings import validate_animation

    parser = build_viewer_parser()
    args = parser.parse_args(argv)
    try:
        if args.carpeta is None:
            root = load_settings(default_config_path()).run.output_dir
            folder = latest_run_dir(resolve_output_dir(root))
            if folder is None:
                raise ConfigError(f"there are no runs in {resolve_output_dir(root)}; run `uv run traffic` first")
        else:
            folder = Path(args.carpeta).expanduser()
            if not (folder / CONFIG_NAME).is_file():
                raise ConfigError(f"{folder} is not a run folder (it has no {CONFIG_NAME})")
        settings = load_settings(folder / CONFIG_NAME)
        if settings.run.seed is None:
            raise ConfigError(f"{folder / CONFIG_NAME} has no [execution] seed: the replica cannot be repeated")
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
    print(f"Run {folder.resolve().name}")
    paths = visualize(settings.sim, settings.run.seed, anim, folder, folder.resolve().name)
    for path in paths:
        print(f"  {path}")
    return paths


def view_main(argv: list[str] | None = None) -> None:
    view(argv)


# ------------------------------------------------------------ traffic-variants

VARIANTS_NAME = "light_variants"


def build_variants_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="traffic-variants",
        description=(
            f"Compare variants of a scenario: run the configuration ({CONFIG_NAME}) with each combination of "
            "segment length and traffic light split, and plot the mean speed of each lane and the vehicles in "
            "the entry queue over time. Save the plot, CSV tables, a summary, the raw data "
            "and the parameters in <output.dir>/<date-time>_<name>/."
        ),
    )
    p.add_argument(
        "target", nargs="?", default=None, metavar="name | folder",
        help=f"as in `traffic`: a folder whose {CONFIG_NAME} is read, or the name of the comparison "
             f"(default: {VARIANTS_NAME}) using {default_config_path()}",
    )  # fmt: skip
    p.add_argument("--lengths", dest="largos", type=float, nargs="+", metavar="M",
                   help="segment lengths in m (default: [road] length)")  # fmt: skip
    p.add_argument("--lights", dest="semaforos", nargs="+", metavar="RED/GREEN",
                   help="traffic light splits in s, e.g. 40/20 30/30 20/40 (default: those three "
                        "splits of the configuration's red + green cycle)")  # fmt: skip
    p.add_argument("--lanes", dest="carriles", type=int, nargs="+", metavar="N",
                   help="lanes of the configuration that are kept, renumbered from 0 in that order, with "
                        "their limits, congestion, initial occupancy and bottlenecks (default: all)")  # fmt: skip
    p.add_argument("--without", dest="sin", nargs="+", default=[], metavar="KEY",
                   help="vehicle types that do not take part, e.g. bike bus")  # fmt: skip
    p.add_argument("--queue", dest="cola", nargs="+", metavar="KEY",
                   help="types counted in the entry queue (default: car; if it does not take part, all)")  # fmt: skip
    p.add_argument("--replicas", type=int, help="replicas per scenario (default: [execution] replicas)")
    p.add_argument("--run", type=float, help="process s of each replica (default: [execution] run)")
    p.add_argument("--redraw", dest="redibujar", metavar="FOLDER",
                   help="do not simulate: redraw the plot of a comparison already run (without overwriting)")  # fmt: skip
    _add_json(p)
    return p


def _lights(values: list[str] | None, red: float, green: float) -> tuple[tuple[float, float], ...]:
    if values is None:  # three splits of the same cycle: red 2/3, 1/2 and 1/3
        cycle = red + green
        return tuple((round(round(cycle * f / DT) * DT, 6), round(round(cycle * (1 - f) / DT) * DT, 6))
                     for f in (2 / 3, 1 / 2, 1 / 3))  # fmt: skip
    out = []
    for text in values:
        parts = text.split("/")
        try:
            red_s, green_s = (float(v) for v in parts)
        except ValueError:
            raise ConfigError(f"--lights: {text!r} must be RED/GREEN in s, e.g. 40/20") from None
        out.append((red_s, green_s))
    return tuple(out)


@_json_aware
def variants(argv: list[str] | None = None) -> Path:
    """Run a variant comparison (or redraw it) and return its folder."""
    from traffic.movement import _free_path
    from traffic.plotting import plot_variants
    from traffic.settings import _set_key
    from traffic.variants import (
        BASE_CONFIG_NAME, PLOT_NAME as VARIANTS_PLOT, SUMMARY_NAME as VARIANTS_SUMMARY, Variants, load, metadata,
        run_variants, write_outputs,
    )  # fmt: skip
    from traffic.variants import results_metrics as variants_metrics

    argv = sys.argv[1:] if argv is None else argv
    parser = build_variants_parser()
    args = parser.parse_args(argv)
    try:
        if args.redibujar:
            folder = Path(args.redibujar).expanduser()
            meta, data = load(folder)
            path = _free_path(folder / VARIANTS_PLOT)
            plot_variants(path, meta, data)
            print(f"Plot in {path}")
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
                raise ConfigError(f"--queue: there is no vehicle type {key!r} (keys: {', '.join(keys)})")
        v = Variants(
            lengths=tuple(args.largos or (base.length,)),
            lights=_lights(args.semaforos, base.red, base.green),
            lanes=tuple(args.carriles) if args.carriles else None,
            without=without, queue_types=queue_types,
            replicas=args.replicas if args.replicas is not None else opts.replicas,
            run=args.run if args.run is not None else base.run,
        )  # fmt: skip
        if v.replicas < 1:
            raise ConfigError("--replicas must be at least 1")
        if len(v.lights) > MAX_TYPES:
            raise ConfigError(f"--lights: up to {MAX_TYPES} splits are allowed (plot colours)")
        seed = opts.seed if opts.seed is not None else secrets.randbelow(2**32)
        meta = metadata(base, v, seed, argv)  # validates lanes and types before creating the folder
    except ConfigError as exc:
        parser.error(str(exc))

    n_scen = len(v.lengths) * len(v.lights)
    print(f"Variant comparison · configuration {target.config}")
    print(f"{len(v.lengths)} length(s) × {len(v.lights)} light split(s) = {n_scen} scenarios × {v.replicas} replicas · "
          f"run {v.run:g} s × {base.time_scale:g} = {v.run * base.time_scale:g} simulated s · seed {seed}")  # fmt: skip
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
    print(f"\nResults in {folder}")
    return folder


def variants_main(argv: list[str] | None = None) -> None:
    variants(argv)


EMISSIONS_NAME = "bump_emissions"
SPACING_NAME = "bump_spacing"
LIGHT_SPACING_NAME = "light_spacing"


def build_emissions_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="traffic-emissions",
        description=(
            f"Compare the emissions of a street in several scenarios: run the configuration ({CONFIG_NAME}) "
            "without a speed bump and with bumps at the given positions and, if asked, with and without a traffic "
            "light, with the same seed. Plot the flow, the vehicles per minute, the emissions per km of each type and "
            "along the street. Save the plot, CSV tables, a summary, the raw data and the parameters in "
            "<output.dir>/<date-time>_<name>/. Emissions require [vehicles.<key>.emissions], accel and decel."
        ),
    )
    p.add_argument(
        "target", nargs="?", default=None, metavar="name | folder",
        help=f"as in `traffic`: a folder whose {CONFIG_NAME} is read, or the name of the comparison "
             f"(default: {EMISSIONS_NAME}) using {default_config_path()}",
    )  # fmt: skip
    p.add_argument("--bumps", dest="topes", nargs="+", metavar="M|M,M|none",
                   help="speed bumps of each scenario: the position (m from the entry), several separated by commas "
                        "(35,70: two bumps in the same scenario) or «none» (default: no bump and those of "
                        "[[speed_bump]]; if there are none, one at the middle of the segment)")  # fmt: skip
    p.add_argument("--bump-lanes", dest="carriles_tope", type=int, nargs="+", metavar="N",
                   help="lanes with a bump, numbered after --lanes (default: the [[speed_bump]] lanes of each "
                        "bump; without them, all)")  # fmt: skip
    p.add_argument("--spacing", dest="separacion", nargs="*", type=float, default=None, metavar="D",
                   help="instead of the normal comparison, find how far apart two bumps must be to dilute their "
                        "emission footprint: run without a bump, with one bump and with two bumps (or --chain N) D m "
                        "apart (default: 10 15 20 25 30 40 50 60 80 100) and plot additivity, mean over the segment, "
                        "gradient and cost of each bump")  # fmt: skip
    p.add_argument("--first-bump", dest="primer_tope", type=float, metavar="M",
                   help="with --spacing, position of the first bump in m (default: the first [[speed_bump]] or a "
                        "quarter of the segment)")  # fmt: skip
    p.add_argument("--chain", dest="cadena", type=int, metavar="N",
                   help="with --spacing, bumps in each chain (default: 2, a pair): N bumps D m apart "
                        "(P1, P1 + D, …); also computes the cost of each added bump")  # fmt: skip
    p.add_argument("--tolerance", dest="tolerancia", type=float, default=0.05, metavar="T",
                   help="with --spacing, how far the pair may deviate from two isolated bumps to be called additive "
                        "(default: 0.05)")  # fmt: skip
    p.add_argument("--threshold", dest="umbral", type=float, default=0.10, metavar="U",
                   help="with --spacing, fraction of the peak of a single bump below which its excess counts as "
                        "diluted (default: 0.10)")  # fmt: skip
    p.add_argument("--light-spacing", dest="separacion_semaforos", nargs="*", type=float, default=None, metavar="D",
                   help="like --spacing, but with fixed-cycle traffic lights: run without a light, with one and with "
                        "two (or --chain N) D m apart (default: 25 50 75 100 150 200 250 300), without the lights or "
                        "bumps of the configuration; the segment is extended if needed (unless --length is given)")  # fmt: skip
    p.add_argument("--first-light", dest="primer_semaforo", type=float, metavar="M",
                   help="with --light-spacing, position of the first light in m (default: 100, to leave room for "
                        "its queue)")  # fmt: skip
    p.add_argument("--cycle", dest="ciclo", metavar="RED/GREEN[/YELLOW]",
                   help="with --light-spacing, phases of all the lights in s, e.g. 30/30/3 (default: those of the "
                        "first fixed-cycle light in the configuration or, if there is none, 30/30/3)")  # fmt: skip
    p.add_argument("--offset", dest="desfase", metavar="red|green|same[,…]", default=None,
                   help="with --light-spacing: «red» (default) turns each light red just as the head of the "
                        "platoon that left the previous one on green arrives, at any spacing; «green», green "
                        "(green wave); «same», all in the same phase. With --chain, one per segment separated by "
                        "commas, e.g. green,red")  # fmt: skip
    p.add_argument("--fixed-spacings", dest="separaciones_fijas", type=float, nargs="+", metavar="M",
                   help="with --light-spacing and --chain N (N ≥ 3): fixed spacings in m of the segments after "
                        "the first one (N − 2 values); the first one is the one swept")  # fmt: skip
    p.add_argument("--light", dest="semaforo", choices=tuple(LIGHT_MODES), default="config",
                   help="traffic light in every scenario: as in the configuration (config), on (yes), off (no) or "
                        "both (each bump with and without a light)")  # fmt: skip
    p.add_argument("--length", dest="largo", type=float, metavar="M",
                   help="segment length in m (default: [road] length)")  # fmt: skip
    p.add_argument("--lanes", dest="carriles", type=int, nargs="+", metavar="N",
                   help="lanes of the configuration that are kept, renumbered from 0 in that order "
                        "(default: all)")  # fmt: skip
    p.add_argument("--without", dest="sin", nargs="+", default=[], metavar="KEY",
                   help="vehicle types that do not take part, e.g. bike motorbike")  # fmt: skip
    p.add_argument("--replicas", type=int, help="replicas per scenario (default: [execution] replicas)")
    p.add_argument("--run", type=float, help="process s of each replica (default: [execution] run)")
    p.add_argument("--redraw", dest="redibujar", metavar="FOLDER",
                   help="do not simulate: redraw the plot of a comparison already run (without overwriting)")  # fmt: skip
    _add_json(p)
    return p


# --light values: traffic light in every scenario (None = as in the configuration)
LIGHT_MODES = {"config": (None,), "yes": (True,), "no": (False,), "both": (True, False)}
# --offset values on the command line -> internal offsets of bump_spacing.OFFSETS (stored in scenarios.json)
OFFSET_MODES = {"red": "rojo", "green": "verde", "same": "igual"}


def _offset_mode(text: str | None) -> str:
    """`--offset red|green|same[,…]` as the internal offset (rojo, verde, igual), one per segment separated by commas."""
    parts = (text or "red").replace(" ", "").split(",")
    unknown = [part for part in parts if part not in OFFSET_MODES]
    if unknown:
        raise ConfigError(f"--offset: {', '.join(unknown)} is not an offset; use {', '.join(OFFSET_MODES)}")
    return ",".join(OFFSET_MODES[part] for part in parts)


def _bumps(values: list[str] | None, config_positions, length: float) -> tuple[tuple[float, ...], ...]:
    """Bumps of each scenario: «none» (no bump), a position in m or several separated by commas (e.g. 35,70).
    By default, no bump and those of [[speed_bump]] (or one at the middle of the segment if there are none)."""
    if values is None:
        configured = as_positions(config_positions)
        return ((), configured or (length / 2,))
    out: list[tuple[float, ...]] = []
    for text in values:
        if text.lower() == "none":
            out.append(())
            continue
        try:
            out.append(tuple(sorted(float(part) for part in text.split(","))))
        except ValueError:
            raise ConfigError(f"--bumps: {text!r} must be «none», a position in m or several separated by commas "
                              "(e.g. 35,70)") from None
    return tuple(out)


@_json_aware
def emissions(argv: list[str] | None = None) -> Path:
    """Run an emission comparison between scenarios (or redraw it) and return its folder."""
    from traffic.emission_scenarios import (
        BASE_CONFIG_NAME, PLOT_NAME as EMIS_PLOT, SUMMARY_NAME as EMIS_SUMMARY, Scenarios, build_configs, load,
        metadata, run_scenarios, write_outputs,
    )  # fmt: skip
    from traffic.emission_scenarios import results_metrics as emission_metrics
    from traffic.movement import _free_path
    from traffic.plotting import plot_emission_comparison
    from traffic.settings import _set_key
    from traffic.variants import reduce_config

    argv = sys.argv[1:] if argv is None else argv
    parser = build_emissions_parser()
    args = parser.parse_args(argv)
    try:
        if args.redibujar:
            folder = Path(args.redibujar).expanduser()
            meta, data = load(folder)
            if meta.get("modo") == "separacion":
                from traffic.bump_spacing import analyze, plot_name
                from traffic.plotting import plot_bump_spacing

                path = _free_path(folder / plot_name(meta))
                plot_bump_spacing(path, meta, data, analyze(meta, data))
                print(f"Plot in {path}")
                return folder
            path = _free_path(folder / EMIS_PLOT)
            plot_emission_comparison(path, meta, data)
            print(f"Plot in {path}")
            return folder
        target = resolve_target(args.target)
        settings = load_settings(target.config)
        base, opts = settings.sim, settings.run
        lanes = tuple(args.carriles) if args.carriles else None
        reduced = reduce_config(base, lanes, tuple(args.sin))
        length = args.largo if args.largo is not None else reduced.length
        lights_sweep = args.separacion_semaforos is not None
        if args.separacion is not None and lights_sweep:
            parser.error("--spacing and --light-spacing cannot be combined: run one and then the other")
        if args.cadena is not None and args.separacion is None and not lights_sweep:
            parser.error("--chain requires --spacing or --light-spacing")
        given = [f for f, v in (("--first-light", args.primer_semaforo), ("--cycle", args.ciclo),
                                ("--offset", args.desfase), ("--fixed-spacings", args.separaciones_fijas))
                 if v is not None]  # fmt: skip
        if given and not lights_sweep:
            parser.error(f"{', '.join(given)} requires --light-spacing")
        if args.separacion is not None or lights_sweep:
            return _spacing("semaforo" if lights_sweep else "tope", args, argv, parser, target, settings, reduced, lanes,
                            length)  # fmt: skip
        lights = LIGHT_MODES[args.semaforo]
        s = Scenarios(
            bumps=_bumps(args.topes, tuple(b.position for b in reduced.bumps), length),
            bump_lanes=tuple(args.carriles_tope) if args.carriles_tope else None,
            lights=lights, length=args.largo, lanes=lanes, without=tuple(args.sin),
            replicas=args.replicas if args.replicas is not None else opts.replicas,
            run=args.run if args.run is not None else base.run,
        )  # fmt: skip
        if s.replicas < 1:
            raise ConfigError("--replicas must be at least 1")
        if len(s.bumps) * len(s.lights) > MAX_TYPES:
            raise ConfigError(f"up to {MAX_TYPES} scenarios are allowed (plot colours)")
        configs = build_configs(base, s)  # validates every scenario before running
        seed = opts.seed if opts.seed is not None else secrets.randbelow(2**32)
        meta = metadata(base, s, configs, seed, argv)
    except ConfigError as exc:
        parser.error(str(exc))

    print(f"Emission comparison · configuration {target.config}")
    print(f"{len(configs)} scenarios × {s.replicas} replicas · run {s.run:g} s × {base.time_scale:g} = "
          f"{s.run * base.time_scale:g} simulated s · seed {seed}")  # fmt: skip
    if not any(v for v in meta["velocidad_tope_kmh"].values()) and any(s.bumps):
        print("Warning: no participating type has speed_bump_kmh: the bump slows nobody down")

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
    print(f"\nResults in {folder}")
    return folder


def _cycle(text: str):
    """`--cycle RED/GREEN[/YELLOW]` in s, as a template traffic light (the position is not used)."""
    from traffic.config import Light

    try:
        values = [float(v) for v in text.split("/")]
    except ValueError:
        values = []
    if len(values) not in (2, 3):
        raise ConfigError(f"--cycle: {text!r} must be RED/GREEN or RED/GREEN/YELLOW in s, e.g. 30/30/3")
    return Light(0.0, red=values[0], green=values[1], yellow=values[2] if len(values) == 3 else 0.0)


def _spacing(kind, args, argv, parser, target, settings, reduced, lanes, length) -> Path:
    """`traffic-emissions --spacing` (kind «tope») or `--light-spacing` (kind «semaforo»): sweep of the
    distance between two bumps or two traffic lights (see bump_spacing.py)."""
    from traffic.bump_spacing import (
        DEFAULT_DISTANCES, LIGHT, LIGHT_CYCLE, LIGHT_DISTANCES, LIGHT_FIRST, LIGHT_ROOM, NOUNS,
        SUMMARY_NAME as SPACING_SUMMARY, Spacing, analyze, build_configs, metadata, plot_name, write_outputs,
    )  # fmt: skip
    from traffic.bump_spacing import results_metrics as spacing_metrics
    from traffic.emission_scenarios import BASE_CONFIG_NAME, Scenarios
    from traffic.emission_scenarios import metadata as base_metadata
    from traffic.emission_scenarios import run_scenarios
    from traffic.movement import _free_path
    from traffic.plotting import plot_bump_spacing
    from traffic.settings import _set_key

    base, opts = settings.sim, settings.run
    light = kind == LIGHT
    flag = "--light-spacing" if light else "--spacing"
    nn = NOUNS[kind]
    notes = []
    try:
        if args.semaforo != "config" and light:
            raise ConfigError("--light-spacing does not accept --light: the traffic lights are those of the sweep")
        if args.semaforo == "both":
            raise ConfigError("--spacing does not accept --light both (use config, yes or no)")
        fixed: tuple[float, ...] = ()
        if args.topes is not None:
            if not light:
                raise ConfigError("--spacing cannot be combined with --bumps (the bumps are the first one and the one moved away)")
            scenarios = _bumps(args.topes, (), length)
            if len(scenarios) != 1:
                raise ConfigError("with --light-spacing, --bumps gives the fixed bumps of every scenario: a single "
                                  "value, e.g. 35,100 (or «none»)")  # fmt: skip
            fixed = scenarios[0]
        distances = tuple(args.separacion_semaforos if light else args.separacion) or (
            LIGHT_DISTANCES if light else DEFAULT_DISTANCES)
        if any(d <= 0 for d in distances) or len(set(distances)) != len(distances):
            raise ConfigError(f"{flag}: each distance must be greater than 0 m and not repeated")
        if args.cadena is not None and args.cadena < 2:
            raise ConfigError(f"--chain must be at least 2 {'lights' if light else 'bumps'}")
        if not 0 < args.tolerancia < 1 or not 0 < args.umbral < 1:
            raise ConfigError("--tolerance and --threshold must be between 0 and 1")
        count = args.cadena or 2
        street = args.largo
        if light:
            first = args.primer_semaforo if args.primer_semaforo is not None else LIGHT_FIRST
            configured = next((lt for lt in reduced.lights if not lt.pedestrian), None)
            cycle = _cycle(args.ciclo) if args.ciclo is not None else configured or LIGHT_CYCLE
            fixed_gaps = tuple(args.separaciones_fijas or ())
            if any(g <= 0 for g in fixed_gaps):
                raise ConfigError("--fixed-spacings: each spacing must be greater than 0 m")
            span = max(distances) + sum(fixed_gaps) if fixed_gaps else (count - 1) * max(distances)
            needed = first + span + LIGHT_ROOM
            if street is None and needed > length:
                street = float(np.ceil(needed / EMIS_BIN) * EMIS_BIN)
                notes.append(f"the segment is extended from {length:g} to {street:g} m to fit the largest spacing and the "
                             f"acceleration after the last light (use --length to fix it)")  # fmt: skip
        else:
            first = args.primer_tope if args.primer_tope is not None else (
                reduced.bumps[0].position if reduced.bumps else length / 4)
            cycle, fixed_gaps = None, ()
        spacing = Spacing(
            first=first, distances=tuple(sorted(distances)),
            light=None if light else LIGHT_MODES[args.semaforo][0],
            bump_lanes=tuple(args.carriles_tope) if args.carriles_tope else None, length=street,
            tolerance=args.tolerancia, threshold=args.umbral, count=count,
            replicas=args.replicas if args.replicas is not None else opts.replicas,
            run=args.run if args.run is not None else base.run,
            kind=kind, cycle=cycle, offset_mode=_offset_mode(args.desfase), bumps=fixed,
            fixed_gaps=fixed_gaps,
        )  # fmt: skip
        if spacing.replicas < 1:
            raise ConfigError("--replicas must be at least 1")
        configs, skipped = build_configs(reduced, spacing)
        kept = [d for d in spacing.distances if d not in skipped]
        seed = opts.seed if opts.seed is not None else secrets.randbelow(2**32)
        scen = Scenarios(bumps=((),), bump_lanes=spacing.bump_lanes, lights=(None,), length=street, lanes=lanes,
                         without=tuple(args.sin), replicas=spacing.replicas, run=spacing.run)  # fmt: skip
        meta = metadata(base_metadata(base, scen, configs, seed, argv), spacing, kept, skipped,
                        configs[0][1] if light else None)  # fmt: skip
    except ConfigError as exc:
        parser.error(str(exc))

    print(f"Spacing between {'traffic lights' if light else 'speed bumps'} · configuration {target.config}")
    print(f"{len(configs)} scenarios × {spacing.replicas} replicas · run {spacing.run:g} s × {base.time_scale:g} = "
          f"{spacing.run * base.time_scale:g} simulated s · seed {seed}")  # fmt: skip
    for note in notes:
        print(f"Warning: {note}")
    if skipped:
        print("Warning: these spacings do not fit on the segment and are skipped: " + ", ".join(f"{d:g}" for d in skipped) + " m")

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
    print(f"\nResults in {folder}")
    return folder


def emissions_main(argv: list[str] | None = None) -> None:
    emissions(argv)


# ------------------------------------------------------------------ traffic-calibrate

CALIBRATION_NAME = "calibration"


def build_calibration_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="traffic-calibrate",
        description=(
            "Measure a vehicle type of the configuration in standard situations (discharge of a queue on green, "
            "continuous flow with increasing demand, stopped queue) and compare it with published reference values "
            "(saturation flow, start-up lost time, capacity, jam density). Save the "
            "plot, tables, the summary and resultados.json in <output.dir>/<date-time>_<name>/."
        ),
    )
    p.add_argument(
        "target", nargs="?", default=None, metavar="name | folder",
        help=f"as in `traffic`: a folder whose {CONFIG_NAME} is read, or the name of the calibration "
             f"(default: {CALIBRATION_NAME}) using {default_config_path()}",
    )  # fmt: skip
    p.add_argument("--vehicle", dest="tipo", default="car", metavar="KEY",
                   help="vehicle type to measure (default: car; the references are for cars)")  # fmt: skip
    p.add_argument("--queues", dest="colas", type=int, default=30, metavar="N",
                   help="queues to discharge, each with its own seed (default: 30)")  # fmt: skip
    _add_json(p)
    return p


@_json_aware
def calibrate(argv: list[str] | None = None) -> Path:
    """Run a calibration and return its folder."""
    from traffic.calibration import FLOW_RATES, PLOT_NAME as CAL_PLOT, compare, measure, results_metrics, write_outputs
    from traffic.emission_scenarios import BASE_CONFIG_NAME
    from traffic.plotting import plot_calibration
    from traffic.settings import _set_key

    argv = sys.argv[1:] if argv is None else argv
    parser = build_calibration_parser()
    args = parser.parse_args(argv)
    command = ("uv run traffic-calibrate " + " ".join(argv)).strip()
    try:
        target = resolve_target(args.target)
        settings = load_settings(target.config)
        base, opts = settings.sim, settings.run
        seed = opts.seed if opts.seed is not None else secrets.randbelow(2**32)
        print(f"Calibration of {args.tipo} · configuration {target.config} · seed {seed}")
        print(f"  {args.colas} queues discharging on green and {len(FLOW_RATES)} demands in continuous flow…")
        m = measure(base, args.tipo, queues=args.colas, seed=seed, workers=opts.workers)
    except ConfigError as exc:
        parser.error(str(exc))
    rows = compare(m)
    folder = make_run_dir(resolve_output_dir(opts.output_dir), safe_name(target.name or CALIBRATION_NAME), datetime.now())
    text = settings.text if opts.seed is not None else _set_key(settings.text, "execution", "seed", seed, "semilla usada")
    (folder / BASE_CONFIG_NAME).write_text(
        f"# Configuración de la calibración {folder.name}\n# {command}\n\n{text}", encoding="utf-8"
    )
    print("\n" + write_outputs(folder, m, rows))
    plot_calibration(folder / CAL_PLOT, m, rows)
    write_results(folder, document("calibracion", folder, command=command, seed=seed, replicas=m["colas"],
                                   sim_seconds=m["s_simulados"], metrics=results_metrics(m, rows)))  # fmt: skip
    print(f"\nResults in {folder}")
    return folder


def calibrate_main(argv: list[str] | None = None) -> None:
    calibrate(argv)
