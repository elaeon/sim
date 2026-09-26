"""Gráfica de la capacidad de movilidad de pasajeros por tipo de vehículo."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator

from trafico.metrics import LANE_SATURATION, LANE_SERIES, LANE_SPEED, SMOOTH_S
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
LEGEND_ROW = 0.017  # alto de una fila de leyenda (fracción de la figura)
MAX_END_LABELS = 4  # con más tipos, las etiquetas finales convergen: la leyenda basta
# Color fijo de cada carril (0 = derecho): la misma paleta empezando por los colores que los tipos
# usan al final (rosa, verde, violeta, rojo), para no confundir un carril con los primeros tipos.
# Con más de 8 carriles, una rampa viridis.
LANE_COLORS = tuple(TYPE_COLORS[k] for k in (4, 5, 6, 7, 3, 2, 1, 0))
LANE_COLORMAP = "viridis"
BASE_HEIGHT = 13.0  # alto (pulgadas) para el que están ajustadas las posiciones de la cabecera

PANELS = (
    ("cum_pax", "pasajeros", "{:,.0f}"),
    ("pax_flow", "pax/min", "{:,.0f}"),
    ("paxkm_h", "pax·km/h", "{:,.0f}"),
    ("pax_per_m", "pax/m", "{:,.2f}"),
)


def _titles(cfg) -> dict[str, str]:
    return {
        "cum_pax": "Pasajeros acumulados que cruzan el semáforo",
        "pax_flow": f"Flujo de pasajeros en el semáforo (ventana de un ciclo, {cfg.cycle:g} s)",
        "paxkm_h": f"Movilidad dentro del tramo (media móvil de {SMOOTH_S:g} s)",
        "pax_per_m": "Pasajeros por metro de carril ocupado (largo + gap real al líder)",
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
    en un carril exclusivo se anteponen con "solo"."""
    out: list[list[str]] = [[] for _ in range(cfg.lanes)]
    for k in active:
        spec = cfg.specs[k]
        if spec.lane is not None and not spec.can_change_lane:
            out[spec.lane].append(spec.name)
    for lane in cfg.reserved_lanes:
        out[lane] = ["solo " + ", ".join(out[lane])] if out[lane] else ["reservado"]
    return out


def phase_spans(cfg) -> list[tuple[float, float, str]]:
    """(inicio, fin, color) de las fases en rojo y en amarillo, para sombrearlas en las gráficas."""
    return [(a, b, RED_PHASE) for a, b in cfg.red_intervals()] + [
        (a, b, YELLOW_PHASE) for a, b in cfg.yellow_intervals()
    ]


def shade_phases(ax, spans) -> None:
    for a, b, color in spans:
        ax.axvspan(a, b, color=color, alpha=0.07 if color == RED_PHASE else 0.12, linewidth=0, zorder=0)


def phase_handles(cfg) -> list:
    """Entradas de leyenda de las fases sombreadas."""
    handles = [Patch(facecolor=RED_PHASE, alpha=0.2, label="semáforo en rojo")]
    if cfg.yellow > 0:
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
    """Etiqueta de cada velocidad a flujo libre; los tipos con la misma velocidad comparten una."""
    by_speed: dict[float, list[str]] = {}
    for k in active:
        by_speed.setdefault(cfg.specs[k].speed_kmh, []).append(cfg.specs[k].name)
    return {v: f"{' · '.join(names)} {v:g}" for v, names in by_speed.items()}


def _lane_panels(axes, agg: Aggregate, t: np.ndarray, red_spans, active: list[int]) -> None:
    cfg = agg.cfg
    assigned = _lane_assignments(cfg, active)
    colors = _lane_colors(cfg.lanes)
    names = [_lane_label(k, cfg.lanes, assigned[k]) for k in range(cfg.lanes)]
    by_lane = cfg.lane_congestion
    sat_labels = [f"{n} · factor {f:g}" for n, f in zip(names, by_lane)] if any(by_lane) else names
    _lane_panel(
        axes[0], agg, t, red_spans, LANE_SATURATION,
        "Saturación de cada carril (cola de detenidos / largo del tramo) y su factor de congestión",
        "% del carril", "{:,.0f} %", 100.0, sat_labels, "upper left", colors,
    )  # fmt: skip
    limits = cfg.lane_max_kmh
    speed_labels = [f"{n} · límite {v:g}" for n, v in zip(names, limits)] if cfg.lane_speed_limit is not None else names
    _lane_panel(
        axes[1], agg, t, red_spans, LANE_SPEED,
        f"Velocidad media en cada carril, con los detenidos (media móvil de {SMOOTH_S:g} s)",
        "km/h", "{:,.0f}", 1.0, speed_labels, "upper right", colors,
    )  # fmt: skip
    # Referencia: velocidad a flujo libre de los tipos que participan (una línea por velocidad).
    for v, label in _speed_refs(cfg, active).items():
        axes[1].axhline(v, color=BASELINE, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
        axes[1].annotate(
            label, (1.0, v), xycoords=("axes fraction", "data"),
            xytext=(4, 0), textcoords="offset points", va="center", fontsize=8, color=MUTED,
            annotation_clip=False,
        )  # fmt: skip
    top = max((cfg.specs[k].speed_kmh for k in active), default=0)
    axes[1].set_ylim(0, top * 1.1 if top else None)


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
        return f"{rate:g}/min"
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


def plot_mobility(agg: Aggregate, path: Path | None, show: bool = False, footer: str | None = None) -> None:
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = agg.cfg
    t = agg.times
    active = [k for k, rate in enumerate(cfg.rates) if rate > 0]
    passengers = [k for k in active if cfg.specs[k].carries_passengers]  # los que solo llevan mercancía no se grafican

    n_panels = len(PANELS) + len(LANE_SERIES)
    height = BASE_HEIGHT * n_panels / len(PANELS)

    def fy(y: float) -> float:
        """Posición vertical de la cabecera: misma distancia en pulgadas al borde superior."""
        return 1 - (1 - y) * BASE_HEIGHT / height

    fig, axes = plt.subplots(
        n_panels, 1, figsize=(11, height), sharex=True, facecolor=SURFACE,
        gridspec_kw={"hspace": 0.42},
    )  # fmt: skip
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
    _lane_panels(axes[-2:], agg, t, red_spans, active)

    bottom = axes[-1]
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
    ncol = min(len(handles), LEGEND_COLS)
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
        f"Tramo {cfg.length:g} m, {cfg.lanes} carril(es) · {cfg.light_label} · "
        f"{agg.replicas} réplicas (media ± 1σ) · run {cfg.run:g} s → {cfg.sim_seconds:,g} s simulados "
        f"= {_duration(cfg.sim_seconds)} de ejecución simulada",
        ha="left", fontsize=9, color=INK_2,
    )  # fmt: skip
    if footer:
        fig.text(0.075, fy(0.944), footer, ha="left", fontsize=8, color=MUTED)
    # El margen derecho deja lugar a la etiqueta más larga a la derecha de los paneles: los
    # valores finales del primer panel y las velocidades de referencia del último.
    longest = max(len(label) for label in _speed_refs(cfg, active).values())
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
    if show:
        plt.show()
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
    from matplotlib.figure import Figure  # sin pyplot: no depende del backend ni de `show`
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
