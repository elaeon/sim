"""Línea de comandos: `uv run trafico [nombre | carpeta]`."""

from __future__ import annotations

import argparse
import os
import secrets
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from trafico.config import DT
from trafico.metrics import LANE_SATURATION, LANE_SPEED, SERIES
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


def format_summary(agg: Aggregate) -> str:
    cfg = agg.cfg
    s = {k: v.mean for k, v in agg.summary.items()}
    sd = {k: v.std for k, v in agg.summary.items()}
    minutes = cfg.sim_seconds / 60.0
    footprint = np.array([sp.length + sp.gap_run for sp in cfg.specs])
    rows = [
        ("Llegadas (veh/min)", s["arrived_veh"] / minutes, 1),
        ("Vehículos que cruzan", s["crossed_veh"], 1),
        ("Pasajeros que cruzan", s["crossed_pax"], 1),
        ("  ± σ entre réplicas", sd["crossed_pax"], 1),
        ("Pasajeros/min cruzando", s["crossed_pax"] / minutes, 1),
        ("Pasajeros por vehículo", s["pax_per_veh"], 2),
        ("Pax/m de carril en marcha", s["pax_per_veh"] / footprint, 2),
        ("T. recorrido medio (s)", s["travel_time"], 1),
        ("T. a flujo libre (s)", np.array([cfg.length / (cfg.free_flow_kmh(k) / 3.6) + sp.expected_stop_time
                                           for k, sp in enumerate(cfg.specs)]), 1),
        ("En el tramo al final", s["on_road"], 1),
        ("En cola de entrada al final", s["queued"], 1),
    ]
    active = [k for k, rate in enumerate(cfg.rates) if rate > 0]  # tipos que participan
    if any(cfg.specs[k].stop_position is not None for k in active):
        rows.insert(8, ("T. medio en la parada (s)", s["stop_time"], 1))
    w0 = max(len(r[0]) for r in rows) + 2
    widths = {k: max(10, len(cfg.specs[k].name) + 2) for k in active}
    speed = cfg.sim_seconds / agg.replica_wall.mean
    lines = ["", " " * w0 + "".join(f"{cfg.specs[k].name:>{widths[k]}}" for k in active)]
    for label, values, digits in rows:
        lines.append(f"{label:<{w0}}" + "".join(f"{_fmt(values[k], digits):>{widths[k]}}" for k in active))
    lines += [
        "",
        f"Cambios de carril por réplica: {_fmt(s['lane_changes'][0])}",
        f"Tiempo de proceso nominal: {cfg.run:g} s ≙ {cfg.sim_seconds:g} s simulados "
        f"({cfg.n_ticks:,} pasos de {DT} s)",
        f"Tiempo real: {agg.wall:.2f} s para {agg.replicas} réplicas con {agg.workers} proceso(s) · "
        f"{float(agg.replica_wall.mean):.2f} s por réplica · {float(speed):,.0f} s simulados por s real",
        f"Memoria máxima (RSS) por proceso: {agg.max_rss_kb / 1024:.1f} MiB",
    ]
    return "\n".join(lines)


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
    for key, label in ((LANE_SATURATION, "saturacion"), (LANE_SPEED, "velocidad_kmh")):
        stats = agg.series[key]
        for lane in range(cfg.lanes):
            cols += [stats.mean[:, lane], stats.std[:, lane]]
            header += [f"{label}_carril{lane}_media", f"{label}_carril{lane}_sd"]
    np.savetxt(path, np.column_stack(cols), delimiter=",", header=",".join(header), comments="", fmt="%.6g")


def _lane_lines(cfg) -> list[str]:
    """Límite de velocidad y factor de congestión de cada carril."""
    lines = []
    if cfg.lane_speed_limit is not None:
        limits = " · ".join(f"{i}: {v:g}" for i, v in enumerate(cfg.lane_max_kmh))
        lines.append(f"Límite de velocidad por carril (km/h, 0 = derecho): {limits}")
    stops = [
        f"{sp.name} a {sp.stop_position:g} m, {sp.stop_time_mean:g} ± {sp.stop_time_std:g} s"
        for k, sp in enumerate(cfg.specs) if sp.stop_position is not None and cfg.rates[k] > 0
    ]  # fmt: skip
    if stops:
        lines.append("Parada antes del semáforo (descenso y ascenso): " + " · ".join(stops))
    reserved = sorted(cfg.reserved_lanes)
    if reserved:
        owners = {ln: [s.name for s in cfg.specs if s.lane == ln] for ln in reserved}
        lines.append("Carriles exclusivos: " + " · ".join(f"{ln}: {', '.join(owners[ln])}" for ln in reserved))
    return lines + _congestion_lines(cfg)


def _congestion_lines(cfg) -> list[str]:
    """Factor de congestión por carril y, si la lista no coincide con los carriles, cómo se ajustó."""
    by_lane = cfg.lane_congestion
    if not any(by_lane):
        return []
    lines = ["Congestión por carril (0 = derecho): " + " · ".join(f"{i}: {v:g}" for i, v in enumerate(by_lane))]
    given = cfg.behavior.congestion_factor
    if isinstance(given, tuple) and len(given) != cfg.lanes:
        if len(given) < cfg.lanes:
            lines.append(
                f"Aviso: congestion_factor tiene {len(given)} valores para {cfg.lanes} carriles; "
                f"los carriles {len(given)}–{cfg.lanes - 1} usan el último ({given[-1]:g})"
            )
        else:
            ignored = ", ".join(f"{v:g}" for v in given[cfg.lanes :])
            lines.append(
                f"Aviso: congestion_factor tiene {len(given)} valores para {cfg.lanes} carriles; "
                f"se ignoran los sobrantes ({ignored})"
            )
    return lines


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
            f"Tramo {cfg.length:g} m · {cfg.lanes} carril(es) · semáforo rojo {cfg.red:g} s / verde {cfg.green:g} s",
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

    if opts.csv:
        write_csv(agg, run_dir / CSV_NAME)

    from trafico.plotting import plot_mobility, plot_passenger_distribution

    footer = f"corrida {run_dir.name} · semilla {seed}"
    plot_passenger_distribution(agg, run_dir / PAX_PLOT_NAME, footer=footer)
    plot_mobility(agg, run_dir / PLOT_NAME, show=opts.show, footer=footer)
    if opts.animation:
        from trafico.movement import visualize

        print()
        for path in visualize(cfg, seed, settings.animation, run_dir, run_dir.name):
            print(f"  {path.name}")
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
