"""Gráfica de la capacidad de movilidad de pasajeros por tipo de vehículo."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator

from trafico.config import POLLUTANT_LABELS, POLLUTANTS, SpeedBump
from trafico.emissions import EMIS_BIN, emitted
from trafico.emissions import unit as pollutant_unit
from trafico.metrics import EMIS_SERIES, ENTRY_QUEUE, EXIT_QUEUE, LANE_EXIT_FLOW, LANE_SATURATION, SPEED_BIN
from trafico.runner import Aggregate

# Paleta categórica validada, en orden fijo: el color sigue al tipo (su posición en la
# configuración), no a su rango, así que no cambia al desactivar otros tipos.
TYPE_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
RED_PHASE = "#d03b3b"
YELLOW_PHASE = "#e0a800"
LEGEND_COLS = 5  # entradas por fila de la leyenda
VIOLIN_WIDTH = 0.8  # ancho máximo de un violín (en carriles)
VIOLIN_SMOOTH = 1.0  # km/h, desviación del núcleo gaussiano que suaviza el histograma del violín
VIOLIN_GAP = 0.2  # alto (en paneles) del espacio antes del panel de violines, para el eje de tiempo
LEGEND_CHARS = 150  # caracteres que caben en una fila de la leyenda a lo ancho de la figura
LEGEND_ROW = 0.017  # alto de una fila de leyenda (fracción de la figura)
MAX_END_LABELS = 4  # con más tipos, las etiquetas finales convergen: la leyenda basta
# Color fijo de cada carril (0 = derecho): la misma paleta empezando por los colores que los tipos
# usan al final (rosa, verde, violeta, rojo), para no confundir un carril con los primeros tipos.
# Con más de 8 carriles, una rampa viridis.
LANE_COLORS = tuple(TYPE_COLORS[k] for k in (4, 5, 6, 7, 3, 2, 1, 0))
LANE_COLORMAP = "viridis"
BASE_HEIGHT = 13.0  # alto (pulgadas) para el que están ajustadas las posiciones de la cabecera
BASE_PANELS = 4  # paneles que caben en BASE_HEIGHT; cada panel extra agrega su parte proporcional
LANE_PANELS = 4  # saturación, cruces contra capacidad de salida, colas de entrada y salida, violines de velocidad
EXIT_DASH = (0, (4, 3))  # trazo de la cola de salida y de las capacidades de salida

PANELS = (
    ("cum_pax", "pasajeros", "{:,.0f}"),
    ("pax_flow", "pax/min", "{:,.0f}"),
)


def _titles(cfg) -> dict[str, str]:
    return {
        "cum_pax": f"Pasajeros acumulados que cruzan {cfg.line_name}",
        "pax_flow": f"Flujo de pasajeros en {cfg.line_name} ({cfg.flow_window_label})",
    }


def _lane_colors(n: int) -> list[str]:
    """Color de cada carril (0 = derecho); no depende de qué tipos participan."""
    if n <= len(LANE_COLORS):
        return list(LANE_COLORS[:n])
    from matplotlib import colormaps
    from matplotlib.colors import to_hex

    cmap = colormaps[LANE_COLORMAP]
    return [to_hex(cmap(0.9 * i / (n - 1))) for i in range(n)]  # sin el amarillo más claro


def _lane_label(lane: int, lanes: int, assigned: list[str] = ()) -> str:
    """Nombre del carril; `assigned` son los tipos con ese carril fijo en la configuración."""
    if lanes == 1:
        return "carril único"
    notes = (["derecho"] if lane == 0 else ["izquierdo"] if lane == lanes - 1 else []) + list(assigned)
    return f"carril {lane}" + (f" ({', '.join(notes)})" if notes else "")


def _lane_assignments(cfg, active: list[int]) -> list[list[str]]:
    """Tipos que participan y tienen configurado un carril fijo ([vehicles.<tipo>] lane), por carril;
    en un carril exclusivo se anteponen con "solo". Un carril sin semáforo lo indica al final."""
    out: list[list[str]] = [[] for _ in range(cfg.lanes)]
    for k in active:
        spec = cfg.specs[k]
        if spec.lane is not None and not spec.can_change_lane:
            out[spec.lane].append(spec.name)
    for lane in cfg.reserved_lanes:
        out[lane] = ["solo " + ", ".join(out[lane])] if out[lane] else ["reservado"]
    for light in cfg.lights:
        for lane in light.free_lanes:
            out[lane].append("sin semáforo" if len(cfg.lights) == 1 else f"sin semáforo a {light.position:g} m")
    return out


def phase_spans(cfg) -> list[tuple[float, float, str]]:
    """(inicio, fin, color) de las fases en rojo y en amarillo, para sombrearlas en las gráficas."""
    return [(a, b, RED_PHASE) for a, b in cfg.red_intervals()] + [
        (a, b, YELLOW_PHASE) for a, b in cfg.yellow_intervals()
    ]


def shade_phases(ax, spans) -> None:
    for a, b, color in spans:
        ax.axvspan(a, b, color=color, alpha=0.07 if color == RED_PHASE else 0.12, linewidth=0, zorder=0)


def phase_handles(cfg, per_light: bool = False) -> list:
    """Entradas de leyenda de las fases sombreadas (ninguna sin semáforo). Con varios semáforos, el sombreado es el
    del último; `per_light` (el diagrama espacio-tiempo, que marca las fases de cada semáforo en su línea) no
    nombra cuál."""
    ref = cfg.ref_light
    if ref is None:
        return []
    which = f" a {ref.position:g} m" if cfg.inner_lights and not per_light else ""
    handles = [Patch(facecolor=RED_PHASE, alpha=0.2, label=f"semáforo{which} en rojo")]
    if ref.yellow > 0:
        handles.append(Patch(facecolor=YELLOW_PHASE, alpha=0.3, label="en amarillo"))
    return handles


def _lane_panel(ax, agg: Aggregate, t: np.ndarray, red_spans, key: str, title: str, unit: str,
                fmt: str, scale: float, labels: list[str], legend_loc: str, colors: list[str]) -> None:  # fmt: skip
    """Panel con una línea (media ± 1σ entre réplicas) por carril."""
    cfg = agg.cfg
    _style_axis(ax, fmt)
    shade_phases(ax, red_spans)
    stats = agg.series[key]
    mean, std = stats.mean * scale, stats.std * scale
    for lane in range(cfg.lanes):
        m, s = mean[:, lane], std[:, lane]
        ax.fill_between(t, np.maximum(m - s, 0), m + s, color=colors[lane], alpha=0.10, linewidth=0)
        ax.plot(t, m, color=colors[lane], linewidth=2, solid_capstyle="round", solid_joinstyle="round")
    ax.set_ylim(bottom=0)
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=8)
    ax.set_ylabel(unit, color=INK_2, fontsize=9)
    handles = [Patch(facecolor=colors[k], label=labels[k]) for k in range(cfg.lanes)]
    ax.legend(
        handles=handles, loc=legend_loc, ncol=min(cfg.lanes, 4), frameon=True, facecolor=SURFACE,
        edgecolor="none", framealpha=0.9, fontsize=8.5, labelcolor=INK_2, handlelength=1.0,
        borderaxespad=0.6,
    )  # fmt: skip


def _speed_refs(cfg, active: list[int]) -> dict[float, str]:
    """Etiqueta de la velocidad máxima de cada tipo (la media, con su rango si es variable); los tipos
    con la misma velocidad comparten una."""
    by_speed: dict[float, dict[str, list[str]]] = {}
    for k in active:
        spec = cfg.specs[k]
        spread = f" ({spec.slowest_kmh:g}–{spec.fastest_kmh:g})" if spec.speed_std > 0 else ""
        by_speed.setdefault(spec.speed_kmh, {}).setdefault(spread, []).append(spec.name)
    # Un renglón por grupo: con velocidades variables la etiqueta sería demasiado ancha en una sola línea.
    return {
        v: "\n".join(f"{' · '.join(names)} {v:g}{spread}" for spread, names in groups.items())
        for v, groups in by_speed.items()
    }  # fmt: skip


def _lane_panels(axes, agg: Aggregate, t: np.ndarray, red_spans, active: list[int]) -> None:
    cfg = agg.cfg
    assigned = _lane_assignments(cfg, active)
    colors = _lane_colors(cfg.lanes)
    names = [_lane_label(k, cfg.lanes, assigned[k]) for k in range(cfg.lanes)]
    _lane_panel(
        axes[0], agg, t, red_spans, LANE_SATURATION,
        "Saturación de cada carril (cola de detenidos / largo del tramo)",
        "% del carril", "{:,.0f} %", 100.0, names, "upper left", colors,
    )  # fmt: skip
    _exit_panel(axes[1], agg, t, red_spans, names, colors)
    _queue_panels(axes[2], agg, t, red_spans, names, colors)
    _speed_violins(axes[3], agg, active, names, colors)


def _exit_labels(cfg) -> dict[float, str]:
    """Etiqueta de la capacidad de la cola de salida, una por valor (los carriles con la misma la comparten)."""
    by_cap: dict[float, list[int]] = {}
    for lane, cap in enumerate(cfg.lane_exit_capacity):
        if cap > 0:
            by_cap.setdefault(cap, []).append(lane)
    return {
        cap: f"salida {cap:g}/min · " + ("carril " if len(lanes) == 1 else "carriles ") + ", ".join(map(str, lanes))
        for cap, lanes in by_cap.items()
    }  # fmt: skip


def _exit_panel(ax, agg: Aggregate, t: np.ndarray, red_spans, names: list[str], colors: list[str]) -> None:
    """Demanda en la línea (vehículos que cruzan por carril, ventana de un ciclo) contra la capacidad de la
    cola de salida de cada carril, como línea punteada del color del carril (neutra si la comparten)."""
    cfg = agg.cfg
    limited = any(c > 0 for c in cfg.lane_exit_capacity)
    title = f"Vehículos que cruzan {cfg.line_name} por carril ({cfg.flow_window_label})"
    if limited:
        title += " contra la capacidad de su cola de salida"
    _lane_panel(ax, agg, t, red_spans, LANE_EXIT_FLOW, title, "veh/min", "{:,.0f}", 1.0, names, "upper left", colors)
    top = ax.get_ylim()[1]
    for cap, label in _exit_labels(cfg).items():
        lanes = [k for k, c in enumerate(cfg.lane_exit_capacity) if c == cap]
        color = colors[lanes[0]] if len(lanes) == 1 else MUTED
        ax.axhline(cap, color=color, linewidth=1.5, linestyle=EXIT_DASH, zorder=4)
        ax.annotate(label, (1.0, cap), xycoords=("axes fraction", "data"), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK_2, annotation_clip=False)  # fmt: skip
        top = max(top, cap * 1.15)
    ax.set_ylim(0, top)


def _storage_labels(cfg) -> dict[float, str]:
    """Etiqueta de los m que mide la cola de salida, una por valor, solo de los carriles con salida limitada."""
    by_size: dict[float, list[int]] = {}
    for lane, (cap, size) in enumerate(zip(cfg.lane_exit_capacity, cfg.lane_exit_storage)):
        if cap > 0:
            by_size.setdefault(size, []).append(lane)
    return {
        size: f"mide {size:g} m · " + ("carril " if len(lanes) == 1 else "carriles ") + ", ".join(map(str, lanes))
        for size, lanes in by_size.items()
    }  # fmt: skip


def _queue_panels(axes, agg: Aggregate, t: np.ndarray, red_spans, names: list[str], colors: list[str]) -> None:
    """Lado a lado, cada una con su escala: la cola de entrada al tramo (vehículos que llegaron y aún no caben) y
    la cola de salida después del semáforo (m ocupados por los que cruzaron y aún no salen), por carril (media
    ± 1σ entre réplicas). La de entrada puede crecer sin límite y la de salida no pasa de lo que mide. Lo que mide
    la salida va como línea punteada del color del carril (neutra si la comparten)."""
    cfg = agg.cfg
    limited = [lane for lane, c in enumerate(cfg.lane_exit_capacity) if c > 0]
    entry_ax, exit_ax = axes
    for ax, key, lanes, unit in ((entry_ax, ENTRY_QUEUE, range(cfg.lanes), "vehículos"),
                                 (exit_ax, EXIT_QUEUE, limited, "m ocupados")):  # fmt: skip
        _style_axis(ax, "{:,.0f}")
        shade_phases(ax, red_spans)
        stats = agg.series[key]
        for lane in lanes:
            m, s = stats.mean[:, lane], stats.std[:, lane]
            ax.fill_between(t, np.maximum(m - s, 0), m + s, color=colors[lane], alpha=0.10, linewidth=0)
            ax.plot(t, m, color=colors[lane], linewidth=2, solid_capstyle="round", solid_joinstyle="round")
        ax.set_ylim(0, max(ax.get_ylim()[1], 1.0))
        ax.set_ylabel(unit, color=INK_2, fontsize=9)
    entry_ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3, integer=True))  # vehículos enteros
    entry_ax.set_title("Cola de entrada al tramo por carril", loc="left", fontsize=11, color=INK, pad=8)
    # Lugar arriba para la leyenda: la cola de entrada suele crecer hacia la esquina superior derecha.
    entry_ax.set_ylim(0, entry_ax.get_ylim()[1] * 1.45)
    entry_ax.legend(
        handles=[Patch(facecolor=colors[k], label=names[k]) for k in range(cfg.lanes)], loc="upper left",
        ncol=2, frameon=True, facecolor=SURFACE, edgecolor="none", framealpha=0.9, fontsize=8, labelcolor=INK_2,
        handlelength=1.0, borderaxespad=0.6, columnspacing=1.0,
    )  # fmt: skip
    after = "después del semáforo" if cfg.has_light else "al final del tramo"
    exit_ax.set_title(f"Cola de salida {after} por carril", loc="left", fontsize=11, color=INK, pad=8)
    if not limited:
        exit_ax.text(0.5, 0.5, "sin cola de salida ([exit] capacity = 0)", transform=exit_ax.transAxes, ha="center",
                     va="center", fontsize=9, color=MUTED)  # fmt: skip
        return
    top = exit_ax.get_ylim()[1]
    for size, label in _storage_labels(cfg).items():
        lanes = [k for k in limited if cfg.lane_exit_storage[k] == size]
        color = colors[lanes[0]] if len(lanes) == 1 else MUTED
        exit_ax.axhline(size, color=color, linewidth=1.5, linestyle=EXIT_DASH, zorder=4)
        exit_ax.annotate(label, (1.0, size), xycoords=("axes fraction", "data"), xytext=(4, 0),
                         textcoords="offset points", va="center", fontsize=8, color=INK_2, annotation_clip=False)  # fmt: skip
        top = max(top, size * 1.15)
    exit_ax.set_ylim(0, top)


def _emission_panels(axes, agg: Aggregate, t: np.ndarray, red_spans, pols: list[int], types: list[int]) -> None:
    """Lado a lado, los dos primeros contaminantes emitidos (CO2 y NOx si están) por minuto y por tipo, en la
    misma ventana que el flujo (media ± 1σ entre réplicas)."""
    cfg = agg.cfg
    for ax, p in zip(axes, [*pols[:2], None]):
        if p is None:
            _style_axis(ax, "{:,.0f}")
            ax.text(0.5, 0.5, "un solo contaminante emitido", transform=ax.transAxes, ha="center", va="center",
                    fontsize=9, color=MUTED)  # fmt: skip
            continue
        pol = POLLUTANTS[p]
        name, scale = pollutant_unit(pol)
        _style_axis(ax, "{:,.0f}")
        shade_phases(ax, red_spans)
        stats = agg.series[EMIS_SERIES[p]]
        for k in types:
            if cfg.specs[k].emission_coefs(pol) is None:
                continue
            m, sd = stats.mean[:, k] * scale, stats.std[:, k] * scale
            ax.fill_between(t, np.maximum(m - sd, 0), m + sd, color=TYPE_COLORS[k], alpha=0.12, linewidth=0)
            ax.plot(t, m, color=TYPE_COLORS[k], linewidth=2, solid_capstyle="round", solid_joinstyle="round")
        ax.set_ylim(bottom=0)
        _fit_format(ax)
        ax.set_ylabel(f"{name}/min", color=INK_2, fontsize=9)
        ax.set_title(f"{POLLUTANT_LABELS[pol]} emitido por tipo ({cfg.flow_window_label})", loc="left", fontsize=11,
                     color=INK, pad=8)  # fmt: skip


def plot_emissions_by_position(agg: Aggregate, path: Path, footer: str | None = None) -> None:
    """Emisiones a lo largo del tramo: un panel por contaminante emitido, g/(m·h) (mg para los que no son CO2)
    por intervalo de EMIS_BIN m y una línea por carril, con el tope, las paradas y la línea final marcados."""
    from matplotlib.figure import Figure

    cfg = agg.cfg
    pols, _ = emitted(cfg)
    active = [k for k, rate in enumerate(cfg.rates) if rate > 0]
    colors = _lane_colors(cfg.lanes)
    assigned = _lane_assignments(cfg, active)
    names = [_lane_label(k, cfg.lanes, assigned[k]) for k in range(cfg.lanes)]
    pos = agg.summary["emissions_pos"].mean / EMIS_BIN / (cfg.sim_seconds / 3600.0)  # g/(m·h)
    x = (np.arange(pos.shape[2]) + 0.5) * EMIS_BIN
    fig = Figure(figsize=(11, 1.9 + 2.9 * len(pols)), facecolor=SURFACE)
    axes = fig.subplots(len(pols), 1, sharex=True, squeeze=False, gridspec_kw={"hspace": 0.45})[:, 0]
    marks = [(cfg.length, "semáforo" if cfg.has_light else "final del tramo", INK_2)]
    marks += [(light.position, "semáforo", INK_2) for light in cfg.inner_lights]
    bump_lanes = cfg.speed_bump_lanes()
    if bump_lanes:
        where = "" if len(bump_lanes) == cfg.lanes else (
            (" (carril " if len(bump_lanes) == 1 else " (carriles ") + ", ".join(map(str, bump_lanes)) + ")")
        marks += [(pos, "tope" + where, YELLOW_PHASE) for pos in cfg.speed_bump.positions]
    marks += [(cfg.specs[k].stop_position, f"parada {cfg.specs[k].name}", TYPE_COLORS[k])
              for k in active if cfg.specs[k].stop_position is not None]  # fmt: skip
    for ax, p in zip(axes, pols):
        pol = POLLUTANTS[p]
        name, scale = pollutant_unit(pol)
        _style_axis(ax, "{:,.0f}")
        for lane in range(cfg.lanes):
            if pos[p, lane].any():
                ax.plot(x, pos[p, lane] * scale, color=colors[lane], linewidth=2, label=names[lane],
                        solid_joinstyle="round")  # fmt: skip
        for at, label, color in marks:
            ax.axvline(at, color=color, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
            end = at >= cfg.length  # la línea final: etiqueta hacia adentro
            ax.annotate(label, (at, 1.0), xycoords=("data", "axes fraction"), xytext=(-3 if end else 3, -3),
                        textcoords="offset points", ha="right" if end else "left", va="top", fontsize=8,
                        color=INK_2)  # fmt: skip
        ax.set_ylim(bottom=0)
        _fit_format(ax)
        ax.set_xlim(0, cfg.length)
        ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.9), ncol=min(cfg.lanes, 4), frameon=True,
                  facecolor=SURFACE, edgecolor="none", framealpha=0.9, fontsize=8.5, labelcolor=INK_2,
                  handlelength=1.2)  # fmt: skip
        ax.set_ylabel(f"{name}/(m·h)", color=INK_2, fontsize=9)
        ax.set_title(f"{POLLUTANT_LABELS[pol]} emitido a lo largo del tramo, por carril", loc="left", fontsize=11,
                     color=INK, pad=8)  # fmt: skip
    axes[-1].set_xlabel("Posición del frente del vehículo (m desde la entrada del tramo)", color=INK_2, fontsize=9)
    fh = fig.get_figheight()
    fig.suptitle("Emisiones a lo largo del tramo", x=0.075, y=1 - 0.3 / fh, ha="left", fontsize=14, color=INK,
                 fontweight="bold")  # fmt: skip
    fig.text(0.075, 1 - 0.58 / fh,
             f"Modelo de Int Panis et al. (2006) · intervalos de {EMIS_BIN:g} m · media de {agg.replicas} réplicas · "
             f"{_duration(cfg.sim_seconds)} simulados", ha="left", fontsize=9, color=INK_2)  # fmt: skip
    if footer:
        fig.text(0.075, 1 - 0.8 / fh, footer, ha="left", fontsize=8, color=MUTED)
    fig.subplots_adjust(left=0.075, right=0.97, top=1 - 1.25 / fh, bottom=0.6 / fh)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)


def _speed_violins(ax, agg: Aggregate, active: list[int], names: list[str], colors: list[str]) -> None:
    """Un violín por carril con la distribución de su velocidad media: todas las muestras de todas las
    réplicas en que el carril tenía vehículos. Adentro, el rango intercuartil y la mediana."""
    cfg = agg.cfg
    _style_axis(ax, "{:,.0f}")
    hist = agg.lane_speed_hist
    edges = np.arange(hist.shape[1] + 1) * SPEED_BIN
    y = edges[:-1] + SPEED_BIN / 2
    k = np.arange(-4 * VIOLIN_SMOOTH, 4 * VIOLIN_SMOOTH + SPEED_BIN / 2, SPEED_BIN) / VIOLIN_SMOOTH
    kernel = np.exp(-0.5 * k * k)
    samples = agg.replicas * cfg.n_samples
    ticks = []
    for lane in range(cfg.lanes):
        counts = hist[lane]
        total = int(counts.sum())
        limit = f"\nlímite {cfg.lane_max_kmh[lane]:g}" if cfg.lane_speed_limit is not None else ""
        present = f"con vehículos {100 * total / samples:.0f} % del tiempo"
        if total == 0:
            ticks.append(f"{names[lane]}{limit}\n{present}")
            ax.text(lane, 0.5, "sin vehículos", transform=ax.get_xaxis_transform(), ha="center", fontsize=8,
                    color=MUTED)  # fmt: skip
            continue
        # Densidad suavizada, recortada al rango observado (como violinplot con cut = 0).
        density = np.convolve(counts, kernel, mode="same")
        seen = np.flatnonzero(counts)
        keep = slice(seen[0], seen[-1] + 1)
        half = VIOLIN_WIDTH / 2 * density[keep] / density[keep].max()
        ys = y[keep]
        ax.fill_betweenx(ys, lane - half, lane + half, color=colors[lane], alpha=0.35, linewidth=0)
        for side in (-1, 1):
            ax.plot(lane + side * half, ys, color=colors[lane], linewidth=1)
        cum = np.concatenate(([0], np.cumsum(counts)))
        q1, median, q3 = np.interp(np.array([0.25, 0.5, 0.75]) * total, cum, edges)
        ax.plot([lane, lane], [q1, q3], color=INK_2, linewidth=4, solid_capstyle="round", zorder=3)
        ax.scatter([lane], [median], s=40, color=SURFACE, edgecolors=INK, linewidths=1.5, zorder=4)
        ticks.append(f"{names[lane]}{limit}\n{present}\nmediana {median:.1f} · intercuartil {q1:.1f}–{q3:.1f}")
    ax.set_xticks(range(cfg.lanes), ticks, fontsize=8, color=INK_2)
    ax.set_xlim(-0.6, cfg.lanes - 0.4)
    ax.set_title(
        f"Velocidad media de cada carril, con los detenidos: distribución de las muestras de {cfg.sample:g} s "
        f"con vehículos (todas las réplicas)", loc="left", fontsize=11, color=INK, pad=8,
    )  # fmt: skip
    ax.set_ylabel("km/h", color=INK_2, fontsize=9)
    # Referencia: velocidad a flujo libre de los tipos que participan (una línea por velocidad).
    for v, label in _speed_refs(cfg, active).items():
        ax.axhline(v, color=BASELINE, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
        # Una etiqueta de varios renglones crece hacia arriba desde su línea, sin tapar las de abajo.
        multi = "\n" in label
        ax.annotate(
            label, (1.0, v), xycoords=("axes fraction", "data"),
            xytext=(4, -4 if multi else 0), textcoords="offset points", va="bottom" if multi else "center",
            fontsize=8, color=MUTED, annotation_clip=False,
        )  # fmt: skip
    top = max((cfg.specs[k].fastest_kmh for k in active), default=0)
    ax.set_ylim(0, top * 1.1 if top else None)


def _fit_format(ax) -> None:
    """Sin decimales si el eje llega a 10 o más; con dos, si no."""
    fmt = "{:,.0f}" if ax.get_ylim()[1] >= 10 else "{:,.2f}"
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt.format(v)))


def _style_axis(ax, fmt: str) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
        ax.spines[side].set_linewidth(1)
    ax.grid(axis="y", color=GRID, linewidth=1, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelcolor=MUTED, labelsize=9, length=0, pad=6)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt.format(v)))


def _crossed_vehicles(ax, agg: Aggregate, active: list[int]) -> None:
    """Recuadro con los vehículos de cada tipo que cruzaron el semáforo (media entre réplicas)."""
    mean = agg.summary["crossed_veh"].mean
    std = agg.summary["crossed_veh"].std
    handles = [
        Patch(facecolor=TYPE_COLORS[k], label=f"{agg.cfg.specs[k].name}: {mean[k]:,.1f} ± {std[k]:,.1f}")
        for k in active
    ]
    box = ax.legend(
        handles=handles, loc="upper left", frameon=True, facecolor=SURFACE, edgecolor="none",
        framealpha=0.9, fontsize=8.5, labelcolor=INK_2,
        handlelength=1.0, title="Vehículos que cruzaron (media ± σ)", title_fontsize=8.5,
        alignment="left", borderaxespad=0.6,
    )  # fmt: skip
    box.get_title().set_color(INK_2)


def _duration(seconds: float) -> str:
    """Duración legible: 1000 s -> '16 min 40 s', 5000 s -> '1 h 23 min', 7200 s -> '2 h'."""
    total = round(seconds)
    if total >= 3600:
        hours, minutes = divmod(round(total / 60), 60)
        return f"{hours:,} h" + (f" {minutes} min" if minutes else "")
    if total >= 60:
        minutes, secs = divmod(total, 60)
        return f"{minutes} min" + (f" {secs} s" if secs else "")
    return f"{seconds:g} s"


def _type_label(spec, rate: float) -> str:
    """Etiqueta de leyenda: nombre, tasa y, si aplica, la parte que lleva mercancía."""
    extra = ""
    if spec.cargo_prob >= 1:
        extra = ", solo mercancía"
    elif spec.cargo_prob > 0:
        extra = f", {100 * spec.cargo_prob:g} % mercancía"
    return f"{spec.name} ({_rate_label(rate)}{extra})"


def _rate_label(rate: float) -> str:
    """Tasa de llegada legible: 20 -> '20/min'; menos de 1 veh/min -> '1 cada x min' con x = 1/rate
    redondeado (0.15 -> '1 cada 7 min')."""
    if rate >= 1:
        return f"{rate:.3g}/min"
    return f"1 cada {max(1, round(1 / rate))} min"


def _row_major(handles: list, ncol: int) -> list:
    """Reordena para que la leyenda (que matplotlib llena por columnas) se lea por filas."""
    rows = math.ceil(len(handles) / ncol)
    return [handles[r * ncol + c] for c in range(ncol) for r in range(rows) if r * ncol + c < len(handles)]


def _end_labels(ax, t_end: float, ends: list[tuple[float, str]], fmt: str) -> None:
    """Valor al final de cada línea; si chocan, se separan con una línea guía fina."""
    if not ends:
        return
    lo, hi = ax.get_ylim()
    min_sep = 0.07 * (hi - lo)
    ends = sorted(ends)
    placed: list[float] = []
    for y, _ in ends:
        placed.append(y if not placed else max(y, placed[-1] + min_sep))
    x_text = t_end * 1.012
    for (y, name), y_lab in zip(ends, placed):
        if abs(y_lab - y) > 1e-9:
            ax.plot([t_end, x_text], [y, y_lab], color=BASELINE, linewidth=0.8, clip_on=False)
        ax.annotate(
            f"{name} {fmt.format(y)}",
            (x_text, y_lab),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=INK_2,
            annotation_clip=False,
        )


def plot_mobility(agg: Aggregate, path: Path | None, footer: str | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = agg.cfg
    t = agg.times
    active = [k for k, rate in enumerate(cfg.rates) if rate > 0]
    passengers = [k for k in active if cfg.specs[k].carries_passengers]  # los que solo llevan mercancía no se grafican

    emis_pols, emis_types = emitted(cfg)
    # El último panel (velocidad por carril) no es una serie de tiempo; con emisiones, una fila más (CO2 | NOx).
    n_panels = len(PANELS) + LANE_PANELS + (1 if emis_pols else 0)
    height = BASE_HEIGHT * (n_panels + VIOLIN_GAP) / BASE_PANELS

    def fy(y: float) -> float:
        """Posición vertical de la cabecera: misma distancia en pulgadas al borde superior."""
        return 1 - (1 - y) * BASE_HEIGHT / height

    fig = plt.figure(figsize=(11, height), facecolor=SURFACE)
    grid = fig.add_gridspec(n_panels + 1, 1, hspace=0.42, height_ratios=[1] * (n_panels - 1) + [VIOLIN_GAP, 1])
    # Filas de tiempo; las de colas (entrada | salida) y emisiones (CO2 | NOx) llevan dos paneles lado a lado.
    queue_row = len(PANELS) + 2
    pair_rows = {queue_row, queue_row + 1} if emis_pols else {queue_row}
    axes: list = []
    for i in range(n_panels - 1):
        if i in pair_rows:
            pair = grid[i].subgridspec(1, 2, wspace=0.16)
            axes.append(tuple(fig.add_subplot(pair[j], sharex=axes[0]) for j in range(2)))
        else:
            axes.append(fig.add_subplot(grid[i], sharex=axes[0] if axes else None))
    for row in axes[:-1]:  # solo la última fila de tiempo lleva el eje x
        for ax in row if isinstance(row, tuple) else (row,):
            ax.tick_params(labelbottom=False)
    axes.append(fig.add_subplot(grid[n_panels]))  # violines: eje x propio (carriles)
    red_spans = phase_spans(cfg)
    titles = _titles(cfg)
    for ax, (key, unit, fmt) in zip(axes, PANELS):
        _style_axis(ax, fmt)
        shade_phases(ax, red_spans)
        mean = agg.series[key].mean
        std = agg.series[key].std
        ends = []
        for k in passengers:
            m, s = mean[:, k], std[:, k]
            color = TYPE_COLORS[k]
            ax.fill_between(t, np.maximum(m - s, 0), m + s, color=color, alpha=0.12, linewidth=0)
            ax.plot(t, m, color=color, linewidth=2, solid_capstyle="round", solid_joinstyle="round")
            if key == "cum_pax" and len(passengers) <= MAX_END_LABELS and np.isfinite(m[-1]):
                ends.append((float(m[-1]), cfg.specs[k].name))
        ax.set_ylim(bottom=0)
        ax.set_title(titles[key], loc="left", fontsize=11, color=INK, pad=8)
        ax.set_ylabel(unit, color=INK_2, fontsize=9)
        if key == "cum_pax":
            _end_labels(ax, float(t[-1]), ends, fmt)
            _crossed_vehicles(ax, agg, active)
    _lane_panels([*axes[len(PANELS):queue_row + 1], axes[-1]], agg, t, red_spans, active)
    if emis_pols:
        _emission_panels(axes[queue_row + 1], agg, t, red_spans, emis_pols, emis_types)

    for bottom in axes[-2]:  # última fila de tiempo (dos paneles)
        bottom.set_xlabel("Tiempo simulado (s)", color=INK_2, fontsize=9)
        bottom.set_xlim(0, cfg.sim_seconds)
        proc = bottom.secondary_xaxis(
            -0.3, functions=(lambda s: s / cfg.time_scale, lambda p: p * cfg.time_scale)
        )
        proc.set_xlabel("Tiempo de proceso (s)", color=INK_2, fontsize=9)
        proc.tick_params(colors=MUTED, labelcolor=MUTED, labelsize=9, length=0)
        proc.spines["bottom"].set_color(BASELINE)

    # Leyenda con la tasa de llegada de cada tipo, en filas de hasta LEGEND_COLS entradas.
    handles = [
        Patch(facecolor=TYPE_COLORS[k], label=_type_label(cfg.specs[k], cfg.rates[k])) for k in active
    ] + phase_handles(cfg)
    # Columnas del mismo ancho (la etiqueta más larga): tantas como quepan en una fila, hasta LEGEND_COLS.
    widest = max(len(h.get_label()) for h in handles) + 6  # + recuadro de color y separación
    ncol = max(1, min(len(handles), LEGEND_COLS, LEGEND_CHARS // widest))
    rows = math.ceil(len(handles) / ncol)
    fig.legend(
        handles=_row_major(handles, ncol), loc="upper left", bbox_to_anchor=(0.068, fy(0.935)),
        ncol=ncol, frameon=False, fontsize=9, labelcolor=INK_2, handlelength=1.2,
    )  # fmt: skip
    fig.suptitle(
        "Capacidad de movilidad de pasajeros por tipo de vehículo",
        x=0.075, y=fy(0.985), ha="left", fontsize=14, color=INK, fontweight="bold",
    )  # fmt: skip
    fig.text(
        0.075, fy(0.958),
        f"Tramo {cfg.length:g} m, {cfg.lanes} carril(es) · {cfg.lights_short} · "
        f"{agg.replicas} réplicas (media ± 1σ) · run {cfg.run:g} s → {cfg.sim_seconds:,g} s simulados "
        f"= {_duration(cfg.sim_seconds)} de ejecución simulada",
        ha="left", fontsize=9, color=INK_2,
    )  # fmt: skip
    if footer:
        fig.text(0.075, fy(0.944), footer, ha="left", fontsize=8, color=MUTED)
    # El margen derecho deja lugar a la etiqueta más larga a la derecha de los paneles: los
    # valores finales del primer panel y las velocidades de referencia del último.
    labels = [*_speed_refs(cfg, active).values(), *_exit_labels(cfg).values(), *_storage_labels(cfg).values()]
    longest = max(len(line) for label in labels for line in label.split("\n"))
    if len(passengers) <= MAX_END_LABELS:
        longest = max(longest, *(len(f"{cfg.specs[k].name} 0,000") for k in passengers))
    right = min(0.93, 0.975 - 0.0063 * longest)
    fig.subplots_adjust(
        left=0.075, right=right, top=fy(0.887 - LEGEND_ROW * (rows - 1)),
        bottom=0.075 * BASE_HEIGHT / height,
    )  # fmt: skip

    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


MAX_PAX_BARS = 30  # con más valores posibles de pasajeros, el histograma agrupa en intervalos
PAX_BIN_WIDTHS = (1, 2, 5, 10, 20, 50, 100)


def _pax_bin_width(n_values: int) -> int:
    """Ancho de intervalo (en pasajeros) para que el histograma tenga como mucho MAX_PAX_BARS barras."""
    return next((w for w in PAX_BIN_WIDTHS if math.ceil(n_values / w) <= MAX_PAX_BARS), PAX_BIN_WIDTHS[-1])


def _pax_bins(values: np.ndarray, weights: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Agrupa `weights` (uno por valor entero consecutivo) en intervalos de `width` valores.
    Devuelve (primer valor, último valor, suma) de cada intervalo; el último puede ser más corto."""
    starts = np.arange(0, values.size, width)
    lo = values[starts]
    hi = values[np.minimum(starts + width, values.size) - 1]
    return lo, hi, np.add.reduceat(weights, starts)


def plot_passenger_distribution(agg: Aggregate, path: Path, footer: str | None = None) -> None:
    """Distribución de pasajeros por vehículo de cada tipo: histograma de todos los vehículos que
    llegaron durante la simulación, sumando las réplicas, frente a la distribución esperada según
    la configuración. Un panel por tipo, con su propio eje x; un rango ancho se agrupa en intervalos."""
    from matplotlib.figure import Figure  # sin pyplot: no depende del backend
    from matplotlib.lines import Line2D

    from trafico.distributions import passenger_pmf

    cfg = agg.cfg
    hist = agg.pax_hist
    # Solo los tipos que llevan pasajeros; los vehículos con mercancía no entran en el histograma.
    active = [k for k, rate in enumerate(cfg.rates) if rate > 0 and cfg.specs[k].carries_passengers]
    ncols = 1 if len(active) == 1 else 2
    nrows = math.ceil(len(active) / ncols)
    fig = Figure(figsize=(11, 1.9 + 3.3 * nrows), facecolor=SURFACE)
    axes = fig.subplots(nrows, ncols, squeeze=False, gridspec_kw={"hspace": 0.95, "wspace": 0.18})
    for ax, k in zip(axes.flat, active):
        spec = cfg.specs[k]
        values, probs = passenger_pmf(spec)
        counts = hist[k, values]
        total = int(counts.sum())
        width = _pax_bin_width(values.size)
        lo, hi, binned = _pax_bins(values, counts.astype(float), width)
        _, _, expected = _pax_bins(values, 100.0 * probs, width)
        exp_mean = float((values * probs).sum())
        exp_std = float(np.sqrt(((values - exp_mean) ** 2 * probs).sum()))

        _style_axis(ax, "{:,.0f} %")
        ax.grid(axis="x", visible=False)
        ax.set_title(spec.name, loc="left", fontsize=11, color=INK, pad=34)
        ax.set_xlabel(
            "pasajeros por vehículo" + (f" (intervalos de {width})" if width > 1 else ""), color=INK_2, fontsize=9
        )  # fmt: skip
        ax.set_ylabel("% de vehículos", color=INK_2, fontsize=9)
        if values.size <= 12:
            ax.set_xticks(values)
        else:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
        ax.set_xlim(spec.pax_min - 0.6 * max(1, width), spec.pax_max + 0.6 * max(1, width))
        ax.text(
            0, 1.03,
            f"esperada: media {exp_mean:.2f} ± {exp_std:.2f} · configurado [{spec.pax_min}, {spec.pax_max}], "
            f"μ {spec.pax_mean:g}, σ {spec.pax_std:g}",
            transform=ax.transAxes, fontsize=8.5, color=MUTED,
        )  # fmt: skip
        # Esperada: escalones sobre los mismos intervalos que las barras.
        edges = np.append(lo - 0.5, hi[-1] + 0.5)
        ax.stairs(expected, edges, color=INK, linewidth=1.5, baseline=None, zorder=4)
        if total == 0:
            ax.text(0, 1.11, "sin llegadas", transform=ax.transAxes, fontsize=8.5, color=INK_2)
            ax.set_ylim(0, expected.max() * 1.15)
            continue
        share = 100.0 * binned / total
        bar_w = (hi - lo + 1) * (0.8 if width == 1 else 0.9)
        ax.bar((lo + hi) / 2, share, width=bar_w, color=TYPE_COLORS[k], linewidth=0, zorder=2)
        mean = float((values * counts).sum() / total)
        std = float(np.sqrt(((values - mean) ** 2 * counts).sum() / total))
        ax.text(
            0, 1.11, f"{total:,} vehículos · observada: media {mean:.2f} ± {std:.2f} pax",
            transform=ax.transAxes, fontsize=8.5, color=INK_2,
        )  # fmt: skip
        ax.axvline(mean, color=INK_2, linewidth=1, linestyle=(0, (4, 3)), zorder=3)
        ax.annotate(
            "media observada", (mean, 1), xycoords=("data", "axes fraction"), xytext=(4, -2),
            textcoords="offset points", va="top", fontsize=8, color=INK_2,
        )  # fmt: skip
        ax.set_ylim(0, max(share.max(), expected.max()) * 1.18)
    for ax in axes.flat[len(active):]:
        ax.set_visible(False)

    fh = fig.get_figheight()
    fig.suptitle(
        "Distribución de pasajeros por vehículo", x=0.075, y=1 - 0.3 / fh,
        ha="left", fontsize=14, color=INK, fontweight="bold",
    )  # fmt: skip
    fig.text(
        0.075, 1 - 0.58 / fh,
        f"Todos los vehículos con pasajeros que llegaron en {_duration(cfg.sim_seconds)} simulados, sumando "
        f"{agg.replicas} réplicas",
        ha="left", fontsize=9, color=INK_2,
    )  # fmt: skip
    if footer:
        fig.text(0.075, 1 - 0.8 / fh, footer, ha="left", fontsize=8, color=MUTED)
    handles = [
        Patch(facecolor=MUTED, label="observada (barras, en el color de cada tipo)"),
        Line2D([], [], color=INK, linewidth=1.5, label="esperada según la configuración (normal redondeada y truncada)"),
    ]  # fmt: skip
    fig.legend(
        handles=handles, loc="upper left", bbox_to_anchor=(0.068, 1 - 0.95 / fh), ncol=2, frameon=False,
        fontsize=9, labelcolor=INK_2, handlelength=1.6,
    )  # fmt: skip
    fig.subplots_adjust(left=0.075, right=0.97, top=1 - 1.95 / fh, bottom=0.65 / fh)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)


def plot_variants(path: Path, meta: dict, data: dict) -> None:
    """Comparación de variantes (trafico-variantes): arriba, la velocidad media de cada carril por
    semáforo (barras, media ± 1σ entre réplicas); abajo, los vehículos en la cola de entrada a lo
    largo del tiempo, una línea por semáforo. Una columna por largo de tramo."""
    from matplotlib.figure import Figure  # sin pyplot: no depende del backend
    from matplotlib.lines import Line2D

    speed, t = data["speed"], data["t"]
    queue = data["queue"][:, :, :, meta["cola_tipos"]].sum(axis=3)  # (largo, semáforo, réplica, muestra)
    lengths, lights, lane_names = meta["largos_m"], meta["semaforos"], meta["carriles"]
    n_len, n_light, n_lanes = len(lengths), len(lights), len(lane_names)
    queue_label = ", ".join(meta["nombres"][k] for k in meta["cola_tipos"])
    lane_colors = _lane_colors(n_lanes)
    light_colors = TYPE_COLORS[:n_light]  # orden fijo; los tres primeros validados en todos los pares

    fig = Figure(figsize=(max(7.0, 4.7 * n_len), 9.2), facecolor=SURFACE)
    axes = fig.subplots(2, n_len, sharey="row", squeeze=False, gridspec_kw={"hspace": 0.62, "wspace": 0.3})
    x = np.arange(n_light)
    width = 0.8 / n_lanes
    top = np.nanmax(speed) if np.isfinite(speed).any() else 1.0
    for li, L in enumerate(lengths):
        ax = axes[0, li]
        _style_axis(ax, "{:,.0f}")
        for lane in range(n_lanes):
            m = np.nanmean(speed[li, :, :, lane], axis=1)
            sd = np.nanstd(speed[li, :, :, lane], axis=1, ddof=1) if speed.shape[2] > 1 else np.zeros(n_light)
            pos = x + (lane - (n_lanes - 1) / 2) * width
            ax.bar(pos, m, width=width * 0.94, color=lane_colors[lane], linewidth=0, zorder=2)
            ax.errorbar(pos, m, yerr=sd, fmt="none", ecolor=INK_2, elinewidth=1, capsize=3, zorder=3)
            for p, v, e in zip(pos, m, sd):  # etiquetas visibles: algunos tonos tienen poco contraste
                ax.annotate(f"{v:.1f}", (p, v + e), xytext=(0, 3), textcoords="offset points", ha="center",
                            va="bottom", fontsize=8, color=INK_2)  # fmt: skip
        ax.set_xticks(x, [f"{s['nombre']}\n{s['rojo_s']:g} s / {s['verde_s']:g} s" for s in lights])
        ax.tick_params(axis="x", labelcolor=INK_2)
        ax.set_title(f"Tramo de {L:g} m", loc="left", fontsize=11, color=INK, pad=8)
        ax.set_ylim(0, top * 1.2)
        ax.grid(axis="x", visible=False)
    axes[0, 0].set_ylabel("velocidad media (km/h)", color=INK_2, fontsize=9)

    q_mean = queue.mean(axis=2)
    q_sd = queue.std(axis=2, ddof=1) if queue.shape[2] > 1 else np.zeros_like(q_mean)
    q_top = float((q_mean + q_sd).max()) * 1.06 or 1.0  # mismo eje para todos los tramos, sin cortar líneas
    for li, L in enumerate(lengths):
        ax = axes[1, li]
        _style_axis(ax, "{:,.0f}")
        ax.set_ylim(0, q_top)
        ends = []
        for vi, light in enumerate(lights):
            m, sd = q_mean[li, vi], q_sd[li, vi]
            ax.fill_between(t, np.maximum(m - sd, 0), m + sd, color=light_colors[vi], alpha=0.12, linewidth=0)
            ax.plot(t, m, color=light_colors[vi], linewidth=2, solid_capstyle="round")
            ends.append((float(m[-1]), light["nombre"]))
        ax.set_xlim(0, t[-1])
        ax.set_title(f"Tramo de {L:g} m", loc="left", fontsize=11, color=INK, pad=8)
        ax.set_xlabel("tiempo simulado (s)", color=INK_2, fontsize=9)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.tick_params(axis="x", labelcolor=MUTED)
        if n_light <= 4:  # etiquetas directas al final de cada línea, separadas para no encimarse
            gap = 0.07 * q_top
            ys: list[float] = []
            for v, _ in sorted(ends):
                ys.append(max(v, ys[-1] + gap) if ys else v)
            shift = max(0.0, ys[-1] - q_top)
            for (v, name), y in zip(sorted(ends), ys):
                ax.annotate(name, (t[-1], v), xytext=(t[-1] * 1.02, y - shift), textcoords="data", fontsize=8,
                            color=INK_2, va="center", annotation_clip=False)  # fmt: skip
    axes[1, 0].set_ylabel("vehículos en la cola de entrada", color=INK_2, fontsize=9)

    fig.suptitle("Variantes del semáforo: velocidad por carril y cola de entrada", x=0.06, y=0.985,
                 ha="left", fontsize=14, color=INK, fontweight="bold")  # fmt: skip
    names = dict(zip(meta["tipos"], meta["nombres"]))
    without = f"; sin {', '.join(names[k] for k in meta['sin'])}" if meta["sin"] else ""
    fig.text(
        0.06, 0.945,
        f"{n_lanes} carril(es): {', '.join(lane_names)}{without} · amarillo {meta['amarillo_s']:g} s · "
        f"run {meta['run_s']:g} s → {meta['s_simulados']:,g} s simulados · {meta['replicas']} réplicas por escenario "
        f"(media ± 1σ) · semilla {meta['semilla']}",
        ha="left", fontsize=9, color=INK_2,
    )  # fmt: skip
    fig.text(0.06, 0.905, "Velocidad media de cada carril en toda la corrida (incluye a los detenidos)",
             ha="left", fontsize=10.5, color=INK, fontweight="bold")  # fmt: skip
    # Cada leyenda en su propia línea, debajo del título de su fila, para no encimarse con él.
    fig.legend(handles=[Patch(facecolor=lane_colors[i], label=lane_names[i]) for i in range(n_lanes)],
               loc="upper left", bbox_to_anchor=(0.055, 0.9), ncol=min(n_lanes, 6), frameon=False, fontsize=9,
               labelcolor=INK_2)  # fmt: skip
    fig.text(0.06, 0.455, f"Vehículos en la cola de entrada: {queue_label} (ya llegaron, aún no caben en el tramo)",
             ha="left", fontsize=10.5, color=INK, fontweight="bold")  # fmt: skip
    fig.legend(handles=[Line2D([], [], color=light_colors[i], linewidth=2,
                               label=f"{s['nombre']} ({s['rojo_s']:g} s / {s['verde_s']:g} s)")
                        for i, s in enumerate(lights)],
               loc="upper left", bbox_to_anchor=(0.055, 0.45), ncol=min(n_light, 4), frameon=False, fontsize=9,
               labelcolor=INK_2)  # fmt: skip
    fig.subplots_adjust(left=0.06, right=0.9, top=0.83, bottom=0.07)
    fig.savefig(path, dpi=150, facecolor=SURFACE)


SMOOTH_FLOW_MIN = 5.0  # min de la media móvil del flujo en la comparación de emisiones


def _scenario_bars(ax, values: list[float], colors: list[str], labels: list[str], unit: str, fmt: str) -> None:
    """Una barra por escenario con su valor y, sobre las demás, el cambio frente a la primera."""
    _style_axis(ax, "{:,.0f}")
    x = np.arange(len(values))
    ax.bar(x, values, width=0.65, color=colors, edgecolor=SURFACE, linewidth=2, zorder=2)
    top = max(v for v in values if np.isfinite(v)) if any(np.isfinite(values)) else 1.0
    for i, v in enumerate(values):
        if not np.isfinite(v):
            continue
        text = fmt.format(v)
        if i > 0 and values[0] > 0:
            pct = 100 * (v / values[0] - 1)
            text += "\n" + ("sin cambio" if abs(pct) < 0.5 else f"{pct:+.0f} %")
        ax.annotate(text, (i, v), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5,
                    color=INK_2, linespacing=1.1)  # fmt: skip
    ax.set_xticks(x, labels, fontsize=8, rotation=0)
    ax.set_ylim(0, top * 1.4 if top > 0 else 1)
    _fit_format(ax)
    ax.set_ylabel(unit, color=INK_2, fontsize=9)


def plot_emission_comparison(path: Path, meta: dict, data: dict) -> None:
    """Comparación de escenarios de trafico-emisiones: flujo que sale del tramo en el tiempo, vehículos por minuto
    de cada tipo, emisiones por km de cada tipo y contaminante, y emisiones a lo largo de la calle."""
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    scenarios = meta["escenarios"]
    colors = [TYPE_COLORS[i] for i in range(len(scenarios))]
    short = [f"{i + 1}" for i in range(len(scenarios))]  # rótulos cortos bajo las barras; la leyenda los nombra
    names, minutes = meta["nombres"], meta["s_simulados"] / 60
    ncols = 4
    active = meta["activos"]
    pols = [POLLUTANTS.index(p) for p in meta["contaminantes"]]
    with np.errstate(invalid="ignore", divide="ignore"):
        scale = np.array([pollutant_unit(pol)[1] for pol in POLLUTANTS])
        km = data["emissions"] * scale / data["veh_km"][:, :, None]
    emis_panels = [(k, p) for p in pols for k in meta["emisores"] if data["emissions"][:, k, p].sum() > 0]
    rows_veh = math.ceil(len(active) / ncols)
    rows_emis = math.ceil(len(emis_panels) / ncols)
    pos_pols = [p for p in pols if POLLUTANTS[p] in ("co2", "nox")][:2] or pols[:2]
    heights = [1.0] + [0.95] * rows_veh + [0.95] * rows_emis + [1.05]
    fig = Figure(figsize=(12, 2.3 + 3.0 * len(heights)), facecolor=SURFACE)
    grid = fig.add_gridspec(len(heights), ncols, height_ratios=heights, hspace=0.75, wspace=0.45)

    # Flujo que sale del tramo (todos los carriles), con media móvil.
    ax = fig.add_subplot(grid[0, :])
    _style_axis(ax, "{:,.0f}")
    t = data["t"] / 60
    dt = (data["t"][1] - data["t"][0]) if data["t"].size > 1 else 1.0
    w = max(1, round(SMOOTH_FLOW_MIN * 60 / dt))
    for i, flow in enumerate(data["flow"]):
        # Media móvil hacia atrás; al inicio, con el historial disponible.
        cum = np.concatenate(([0.0], np.cumsum(np.nan_to_num(flow))))
        idx = np.arange(1, flow.size + 1)
        lo = np.maximum(idx - w, 0)
        ax.plot(t, (cum[idx] - cum[lo]) / (idx - lo), color=colors[i], linewidth=2)
    ax.set_ylim(bottom=0)
    ax.set_xlim(0, t[-1])
    ax.set_xlabel("tiempo simulado (min)", color=INK_2, fontsize=9)
    ax.set_ylabel("veh/min", color=INK_2, fontsize=9)
    ax.set_title(f"Vehículos que salen del tramo por minuto (todos los tipos y carriles, media móvil de "
                 f"{SMOOTH_FLOW_MIN:g} min)", loc="left", fontsize=10.5, color=INK, pad=8)  # fmt: skip

    for n, k in enumerate(active):
        ax = fig.add_subplot(grid[1 + n // ncols, n % ncols])
        values = list(data["crossed_veh"][:, k] / minutes)
        _scenario_bars(ax, values, colors, short, "veh/min", "{:,.2f}" if max(values) < 10 else "{:,.1f}")
        ax.set_title(f"Vehículos por minuto: {names[k]}", loc="left", fontsize=10.5, color=INK, pad=8)

    for n, (k, p) in enumerate(emis_panels):
        ax = fig.add_subplot(grid[1 + rows_veh + n // ncols, n % ncols])
        name = pollutant_unit(POLLUTANTS[p])[0]
        _scenario_bars(ax, list(km[:, k, p]), colors, short, f"{name}/km", "{:,.0f}")
        ax.set_title(f"{POLLUTANT_LABELS[POLLUTANTS[p]]}: {names[k]}", loc="left", fontsize=10.5, color=INK, pad=8)

    pos = data["emissions_pos"].sum(axis=2) / EMIS_BIN / (meta["s_simulados"] / 3600.0)  # (escenario, pol, x)
    x = (np.arange(pos.shape[2]) + 0.5) * EMIS_BIN
    # topes_m: por escenario, una lista de posiciones (en los datos anteriores, una posición o null)
    marks = [(m, "tope") for m in sorted({p for b in meta["topes_m"] for p in SpeedBump(position=b).positions})]
    marks_bumps = {len(SpeedBump(position=b).positions) for b in meta["topes_m"]}
    marks_bumps = range(max(marks_bumps, default=0))  # cuántos topes tiene el escenario con más
    marks += [(m, f"parada {name}") for name, m in meta["paradas_m"].items()]
    span = ncols // max(1, len(pos_pols))
    for n, p in enumerate(pos_pols):
        ax = fig.add_subplot(grid[-1, n * span:(n + 1) * span])
        _style_axis(ax, "{:,.0f}")
        name, factor = pollutant_unit(POLLUTANTS[p])
        for i in range(len(scenarios)):
            ax.plot(x, pos[i, p] * factor, color=colors[i], linewidth=2)
        for at, label in marks:
            ax.axvline(at, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
            ax.annotate(label, (at, 1.0), xycoords=("data", "axes fraction"), xytext=(3, -3),
                        textcoords="offset points", ha="left", va="top", fontsize=8, color=INK_2)  # fmt: skip
        ax.set_xlim(0, meta["largo_m"])
        ax.set_ylim(bottom=0)
        _fit_format(ax)
        ax.set_xlabel("posición en la calle (m desde la entrada)", color=INK_2, fontsize=9)
        ax.set_ylabel(f"{name}/(m·h)", color=INK_2, fontsize=9)
        ax.set_title(f"{POLLUTANT_LABELS[POLLUTANTS[p]]} a lo largo de la calle (todos los carriles)", loc="left",
                     fontsize=10.5, color=INK, pad=8)  # fmt: skip

    fh = fig.get_figheight()
    fig.suptitle("Emisiones por escenario", x=0.06, y=1 - 0.3 / fh, ha="left", fontsize=14, color=INK,
                 fontweight="bold")  # fmt: skip
    rates = " · ".join(f"{names[k]} {meta['tasas'][k]:.3g} veh/min" for k in active)
    bump_speed = " · ".join(f"{n} ≤ {v:g} km/h" for n, v in meta["velocidad_tope_kmh"].items() if v is not None)
    lanes = meta["carriles_tope"]
    where = "todos los carriles" if lanes is None or len(lanes) == len(meta["carriles"]) else (
        ("carril " if len(lanes) == 1 else "carriles ") + ", ".join(map(str, lanes)))
    lines = [
        f"Tramo {meta['largo_m']:g} m · {', '.join(meta['carriles'])} · {rates}",
        f"{'Topes' if len(marks_bumps) > 1 else 'Tope'} en {where}: {bump_speed or 'ningún tipo frena (sin speed_bump_kmh)'}",
        f"{meta['s_simulados']:,.0f} s simulados · media de {meta['replicas']} réplicas con la misma semilla "
        f"({meta['semilla']}) · emisiones por km recorrido en el tramo, modelo de Int Panis et al. (2006)",
    ]
    for i, line in enumerate(lines):
        fig.text(0.06, 1 - (0.68 + 0.22 * i) / fh, line, ha="left", va="top", fontsize=9, color=INK_2)
    handles = [Line2D([], [], color=colors[i], linewidth=3, label=f"{short[i]}: {s}") for i, s in enumerate(scenarios)]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.055, 1 - 1.35 / fh), ncol=min(len(handles), 3),
               frameon=False, fontsize=9.5, labelcolor=INK_2, handlelength=1.6)  # fmt: skip
    legend_rows = math.ceil(len(handles) / 3)
    fig.subplots_adjust(left=0.06, right=0.98, top=1 - (1.9 + 0.25 * legend_rows) / fh, bottom=0.55 / fh)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
