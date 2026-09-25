"""Línea de comandos: `uv run trafico [nombre | carpeta]`."""

from __future__ import annotations

import argparse
import os
import secrets
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from trafico.config import DT, MAX_TYPES
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
        ("T. medio en cola de entrada (s)", s["queue_wait"], 1),
        ("T. a flujo libre (s)", np.array([cfg.length / (cfg.free_flow_kmh(k) / 3.6) + sp.expected_stop_time
                                           for k, sp in enumerate(cfg.specs)]), 1),
        ("En el tramo al final", s["on_road"], 1),
        ("En cola de entrada al final", s["queued"], 1),
        *([("En el tramo al inicio", s["initial_veh"], 1)] if any(cfg.lane_initial_occupancy) else []),
        ("Cambios de carril/veh", np.divide(s["lane_changes_type"], s["entered_veh"], out=np.full(cfg.n_types, np.nan),
                                            where=s["entered_veh"] > 0), 2),  # fmt: skip
    ]
    active = [k for k, rate in enumerate(cfg.rates) if rate > 0]  # tipos que participan
    if any(cfg.specs[k].stop_position is not None for k in active):
        rows.insert(9, ("T. medio en la parada (s)", s["stop_time"], 1))
    if cfg.bottleneck_active:
        rows.append(("Detenciones (bottleneck)", s["bottleneck_stops"], 1))
        rows.append(("T. medio detenido (s)", s["bottleneck_time"], 1))
    if any(cfg.specs[k].cargo_prob > 0 for k in active):
        # Los de mercancía cuentan como vehículos, pero no en las filas de pasajeros.
        cargo = np.divide(100.0 * s["arrived_cargo"], s["arrived_veh"], out=np.full(cfg.n_types, np.nan),
                          where=s["arrived_veh"] > 0)  # fmt: skip
        rows.insert(1, ("Con mercancía (%)", cargo, 1))
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
            f"Tramo {cfg.length:g} m · {cfg.lanes} carril(es) · semáforo {cfg.light_label}",
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
    if any(rate > 0 and sp.carries_passengers for sp, rate in zip(cfg.specs, cfg.rates)):
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


def variants(argv: list[str] | None = None) -> Path:
    """Corre una comparación de variantes (o la redibuja) y devuelve su carpeta."""
    from trafico.movement import _free_path
    from trafico.plotting import plot_variants
    from trafico.settings import _set_key
    from trafico.variants import (
        BASE_CONFIG_NAME, PLOT_NAME as VARIANTS_PLOT, SUMMARY_NAME as VARIANTS_SUMMARY, Variants, load, metadata,
        run_variants, write_outputs,
    )  # fmt: skip

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
    print(f"\nResultados en {folder}")
    return folder


def variants_main(argv: list[str] | None = None) -> None:
    variants(argv)
