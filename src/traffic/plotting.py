"""Plot of the passenger mobility capacity per vehicle type."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator

from traffic.config import POLLUTANT_LABELS, POLLUTANTS, as_positions
from traffic.emissions import EMIS_BIN, emitted
from traffic.emissions import unit as pollutant_unit
from traffic.metrics import EMIS_SERIES, ENTRY_QUEUE, EXIT_QUEUE, LANE_EXIT_FLOW, LANE_SATURATION, SPEED_BIN
from traffic.runner import Aggregate

# Validated categorical palette, in a fixed order: the colour follows the type (its position in the
# configuration), not its rank, so it does not change when other types are disabled.
TYPE_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
RED_PHASE = "#d03b3b"
YELLOW_PHASE = "#e0a800"
LEGEND_COLS = 5  # entries per legend row
VIOLIN_WIDTH = 0.8  # maximum width of a violin (in lanes)
VIOLIN_SMOOTH = 1.0  # km/h, deviation of the Gaussian kernel that smooths the violin's histogram
VIOLIN_GAP = 0.2  # height (in panels) of the space before the violin panel, for the time axis
LEGEND_CHARS = 150  # characters that fit in a legend row across the figure
LEGEND_ROW = 0.017  # height of a legend row (fraction of the figure)
MAX_END_LABELS = 4  # with more types, the end labels converge: the legend is enough
# Fixed colour of each lane (0 = right): the same palette starting with the colours the types
# use last (pink, green, violet, red), so a lane is not confused with the first types.
# With more than 8 lanes, a viridis ramp.
LANE_COLORS = tuple(TYPE_COLORS[k] for k in (4, 5, 6, 7, 3, 2, 1, 0))
LANE_COLORMAP = "viridis"
BASE_HEIGHT = 13.0  # height (inches) the header positions are tuned for
BASE_PANELS = 4  # panels that fit in BASE_HEIGHT; each extra panel adds its proportional share
LANE_PANELS = 4  # saturation, crossings against exit capacity, entry and exit queues, speed violins
EXIT_DASH = (0, (4, 3))  # line style of the exit queue and of the exit capacities

PANELS = (
    ("cum_pax", "passengers", "{:,.0f}"),
    ("pax_flow", "pax/min", "{:,.0f}"),
)


def _titles(cfg) -> dict[str, str]:
    return {
        "cum_pax": f"Cumulative passengers crossing {cfg.line_name}",
        "pax_flow": f"Passenger flow at {cfg.line_name} ({cfg.flow_window_label})",
    }


def _lane_colors(n: int) -> list[str]:
    """Colour of each lane (0 = right); it does not depend on which types take part."""
    if n <= len(LANE_COLORS):
        return list(LANE_COLORS[:n])
    from matplotlib import colormaps
    from matplotlib.colors import to_hex

    cmap = colormaps[LANE_COLORMAP]
    return [to_hex(cmap(0.9 * i / (n - 1))) for i in range(n)]  # without the lightest yellow


def _lane_label(lane: int, lanes: int, assigned: list[str] = ()) -> str:
    """Name of the lane; `assigned` are the types with that fixed lane in the configuration."""
    if lanes == 1:
        return "single lane"
    notes = (["right"] if lane == 0 else ["left"] if lane == lanes - 1 else []) + list(assigned)
    return f"lane {lane}" + (f" ({', '.join(notes)})" if notes else "")


def _lane_assignments(cfg, active: list[int]) -> list[list[str]]:
    """Participating types that have a fixed lane configured ([vehicles.<type>] lane), per lane;
    in an exclusive lane they are prefixed with "only". A lane without a traffic light says so at the end."""
    out: list[list[str]] = [[] for _ in range(cfg.lanes)]
    for k in active:
        spec = cfg.specs[k]
        if spec.lane is not None and not spec.can_change_lane:
            out[spec.lane].append(spec.name)
    for lane in cfg.reserved_lanes:
        out[lane] = ["only " + ", ".join(out[lane])] if out[lane] else ["reserved"]
    for light in cfg.lights:
        for lane in light.free_lanes:
            out[lane].append("no traffic light" if len(cfg.lights) == 1 else f"no traffic light at {light.position:g} m")
    return out


def phase_spans(cfg) -> list[tuple[float, float, str]]:
    """(start, end, colour) of the red and yellow phases, to shade them in the plots."""
    return [(a, b, RED_PHASE) for a, b in cfg.red_intervals()] + [
        (a, b, YELLOW_PHASE) for a, b in cfg.yellow_intervals()
    ]


def shade_phases(ax, spans) -> None:
    for a, b, color in spans:
        ax.axvspan(a, b, color=color, alpha=0.07 if color == RED_PHASE else 0.12, linewidth=0, zorder=0)


def phase_handles(cfg, per_light: bool = False) -> list:
    """Legend entries of the shaded phases (none without a traffic light). With several lights, the shading is that of
    the last fixed-cycle one; `per_light` (the space-time diagram, which marks the phases of each light and each
    pedestrian crossing on its line) does not name which."""
    if per_light:
        lights = cfg.lights
        crossings = bool(cfg.crossings)
        if not lights and not crossings:
            return []
        what = "traffic light" if lights and not crossings else "pedestrian crossing" if crossings and not lights else "stop"
        handles = [Patch(facecolor=RED_PHASE, alpha=0.2, label=f"{what} on red")]
        if crossings or any(lt.yellow > 0 for lt in lights):
            handles.append(Patch(facecolor=YELLOW_PHASE, alpha=0.3, label="on yellow"))
        return handles
    ref = cfg.ref_light
    if ref is None:
        return []
    which = f" at {ref.position:g} m" if cfg.inner_lights and not per_light else ""
    handles = [Patch(facecolor=RED_PHASE, alpha=0.2, label=f"traffic light{which} on red")]
    if ref.yellow > 0:
        handles.append(Patch(facecolor=YELLOW_PHASE, alpha=0.3, label="on yellow"))
    return handles


def _lane_panel(ax, agg: Aggregate, t: np.ndarray, red_spans, key: str, title: str, unit: str,
                fmt: str, scale: float, labels: list[str], legend_loc: str, colors: list[str]) -> None:  # fmt: skip
    """Panel with one line (mean ± 1σ across replicas) per lane."""
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
    """Label of each type's maximum speed (the mean, with its range if it varies); types
    with the same speed share one."""
    by_speed: dict[float, dict[str, list[str]]] = {}
    for k in active:
        spec = cfg.specs[k]
        spread = f" ({spec.slowest_kmh:g}–{spec.fastest_kmh:g})" if spec.speed_std > 0 else ""
        by_speed.setdefault(spec.speed_kmh, {}).setdefault(spread, []).append(spec.name)
    # One row per group: with variable speeds the label would be too wide on a single line.
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
        "Saturation of each lane (queue of stopped vehicles / segment length)",
        "% of the lane", "{:,.0f} %", 100.0, names, "upper left", colors,
    )  # fmt: skip
    _exit_panel(axes[1], agg, t, red_spans, names, colors)
    _queue_panels(axes[2], agg, t, red_spans, names, colors)
    _speed_violins(axes[3], agg, active, names, colors)


def _exit_labels(cfg) -> dict[float, str]:
    """Label of the exit queue capacity, one per value (lanes with the same one share it)."""
    by_cap: dict[float, list[int]] = {}
    for lane, cap in enumerate(cfg.lane_exit_capacity):
        if cap > 0:
            by_cap.setdefault(cap, []).append(lane)
    return {
        cap: f"exit {cap:g}/min · " + ("lane " if len(lanes) == 1 else "lanes ") + ", ".join(map(str, lanes))
        for cap, lanes in by_cap.items()
    }  # fmt: skip


def _exit_panel(ax, agg: Aggregate, t: np.ndarray, red_spans, names: list[str], colors: list[str]) -> None:
    """Demand at the line (vehicles crossing per lane, one-cycle window) against the exit queue capacity
    of each lane, as a dotted line in the lane's colour (neutral if they share it)."""
    cfg = agg.cfg
    limited = any(c > 0 for c in cfg.lane_exit_capacity)
    title = f"Vehicles crossing {cfg.line_name} per lane ({cfg.flow_window_label})"
    if limited:
        title += " against the capacity of their exit queue"
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
    """Label of the m the exit queue measures, one per value, only for the lanes with a limited exit."""
    by_size: dict[float, list[int]] = {}
    for lane, (cap, size) in enumerate(zip(cfg.lane_exit_capacity, cfg.lane_exit_storage)):
        if cap > 0:
            by_size.setdefault(size, []).append(lane)
    return {
        size: f"length {size:g} m · " + ("lane " if len(lanes) == 1 else "lanes ") + ", ".join(map(str, lanes))
        for size, lanes in by_size.items()
    }  # fmt: skip


def _queue_panels(axes, agg: Aggregate, t: np.ndarray, red_spans, names: list[str], colors: list[str]) -> None:
    """Side by side, each with its own scale: the entry queue to the segment (vehicles that arrived and do not fit yet) and
    the exit queue after the traffic light (m occupied by those that crossed and have not left yet), per lane (mean
    ± 1σ across replicas). The entry queue can grow without limit and the exit queue cannot exceed its length. The
    exit's length is a dotted line in the lane's colour (neutral if they share it)."""
    cfg = agg.cfg
    limited = [lane for lane, c in enumerate(cfg.lane_exit_capacity) if c > 0]
    entry_ax, exit_ax = axes
    for ax, key, lanes, unit in ((entry_ax, ENTRY_QUEUE, range(cfg.lanes), "vehicles"),
                                 (exit_ax, EXIT_QUEUE, limited, "occupied m")):  # fmt: skip
        _style_axis(ax, "{:,.0f}")
        shade_phases(ax, red_spans)
        stats = agg.series[key]
        for lane in lanes:
            m, s = stats.mean[:, lane], stats.std[:, lane]
            ax.fill_between(t, np.maximum(m - s, 0), m + s, color=colors[lane], alpha=0.10, linewidth=0)
            ax.plot(t, m, color=colors[lane], linewidth=2, solid_capstyle="round", solid_joinstyle="round")
        ax.set_ylim(0, max(ax.get_ylim()[1], 1.0))
        ax.set_ylabel(unit, color=INK_2, fontsize=9)
    entry_ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3, integer=True))  # whole vehicles
    entry_ax.set_title("Entry queue to the segment per lane", loc="left", fontsize=11, color=INK, pad=8)
    # Room at the top for the legend: the entry queue tends to grow towards the upper right corner.
    entry_ax.set_ylim(0, entry_ax.get_ylim()[1] * 1.45)
    entry_ax.legend(
        handles=[Patch(facecolor=colors[k], label=names[k]) for k in range(cfg.lanes)], loc="upper left",
        ncol=2, frameon=True, facecolor=SURFACE, edgecolor="none", framealpha=0.9, fontsize=8, labelcolor=INK_2,
        handlelength=1.0, borderaxespad=0.6, columnspacing=1.0,
    )  # fmt: skip
    after = "after the traffic light" if cfg.has_light else "at the end of the segment"
    exit_ax.set_title(f"Exit queue {after} per lane", loc="left", fontsize=11, color=INK, pad=8)
    if not limited:
        exit_ax.text(0.5, 0.5, "no exit queue ([exit] capacity = 0)", transform=exit_ax.transAxes, ha="center",
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
    """Side by side, the first two emitted pollutants (CO2 and NOx if present) per minute and per type, over the
    same window as the flow (mean ± 1σ across replicas)."""
    cfg = agg.cfg
    for ax, p in zip(axes, [*pols[:2], None]):
        if p is None:
            _style_axis(ax, "{:,.0f}")
            ax.text(0.5, 0.5, "only one pollutant emitted", transform=ax.transAxes, ha="center", va="center",
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
        ax.set_title(f"{POLLUTANT_LABELS[pol]} emitted per type ({cfg.flow_window_label})", loc="left", fontsize=11,
                     color=INK, pad=8)  # fmt: skip


def plot_emissions_by_position(agg: Aggregate, path: Path, footer: str | None = None) -> None:
    """Emissions along the segment: one panel per emitted pollutant, g/(m·h) (mg for those that are not CO2)
    per EMIS_BIN m bin and one line per lane, with the bump, the stops and the end line marked."""
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
    def light_name(light) -> str:
        return "pedestrian light" if light.pedestrian else "traffic light"

    marks = [(cfg.length, light_name(cfg.exit_light) if cfg.has_light else "end of the segment", INK_2)]
    marks += [(light.position, light_name(light), INK_2) for light in cfg.inner_lights]
    for bump in cfg.bumps:
        bump_lanes = cfg.bump_lanes(bump)
        where = "" if len(bump_lanes) == cfg.lanes else (
            (" (lane " if len(bump_lanes) == 1 else " (lanes ") + ", ".join(map(str, bump_lanes)) + ")")
        marks.append((bump.position, "bump" + (" · pedestrians" if bump.pedestrian else "") + where, YELLOW_PHASE))
    marks += [(cfg.specs[k].stop_position, f"stop {cfg.specs[k].name}", TYPE_COLORS[k])
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
            end = at >= cfg.length  # the end line: label towards the inside
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
        ax.set_title(f"{POLLUTANT_LABELS[pol]} emitted along the segment, per lane", loc="left", fontsize=11,
                     color=INK, pad=8)  # fmt: skip
    axes[-1].set_xlabel("Position of the vehicle front (m from the entry of the segment)", color=INK_2, fontsize=9)
    fh = fig.get_figheight()
    fig.suptitle("Emissions along the segment", x=0.075, y=1 - 0.3 / fh, ha="left", fontsize=14, color=INK,
                 fontweight="bold")  # fmt: skip
    fig.text(0.075, 1 - 0.58 / fh,
             f"Int Panis et al. (2006) model · {EMIS_BIN:g} m intervals · mean of {agg.replicas} replicas · "
             f"{_duration(cfg.sim_seconds)} simulated", ha="left", fontsize=9, color=INK_2)  # fmt: skip
    if footer:
        fig.text(0.075, 1 - 0.8 / fh, footer, ha="left", fontsize=8, color=MUTED)
    fig.subplots_adjust(left=0.075, right=0.97, top=1 - 1.25 / fh, bottom=0.6 / fh)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)


def _speed_violins(ax, agg: Aggregate, active: list[int], names: list[str], colors: list[str]) -> None:
    """One violin per lane with the distribution of its mean speed: every sample of every
    replica in which the lane had vehicles. Inside, the interquartile range and the median."""
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
        limit = f"\nlimit {cfg.lane_max_kmh[lane]:g}" if cfg.lane_speed_limit is not None else ""
        present = f"with vehicles {100 * total / samples:.0f} % of the time"
        if total == 0:
            ticks.append(f"{names[lane]}{limit}\n{present}")
            ax.text(lane, 0.5, "no vehicles", transform=ax.get_xaxis_transform(), ha="center", fontsize=8,
                    color=MUTED)  # fmt: skip
            continue
        # Smoothed density, clipped to the observed range (like violinplot with cut = 0).
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
        ticks.append(f"{names[lane]}{limit}\n{present}\nmedian {median:.1f} · interquartile {q1:.1f}–{q3:.1f}")
    ax.set_xticks(range(cfg.lanes), ticks, fontsize=8, color=INK_2)
    ax.set_xlim(-0.6, cfg.lanes - 0.4)
    ax.set_title(
        f"Mean speed of each lane, including stopped vehicles: distribution of the {cfg.sample:g} s samples "
        f"with vehicles (all replicas)", loc="left", fontsize=11, color=INK, pad=8,
    )  # fmt: skip
    ax.set_ylabel("km/h", color=INK_2, fontsize=9)
    # Reference: free-flow speed of the participating types (one line per speed).
    for v, label in _speed_refs(cfg, active).items():
        ax.axhline(v, color=BASELINE, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
        # A multi-row label grows upwards from its line, without covering the ones below.
        multi = "\n" in label
        ax.annotate(
            label, (1.0, v), xycoords=("axes fraction", "data"),
            xytext=(4, -4 if multi else 0), textcoords="offset points", va="bottom" if multi else "center",
            fontsize=8, color=MUTED, annotation_clip=False,
        )  # fmt: skip
    top = max((cfg.specs[k].fastest_kmh for k in active), default=0)
    ax.set_ylim(0, top * 1.1 if top else None)


def _fit_format(ax) -> None:
    """No decimals if the axis reaches 10 or more; two otherwise."""
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
    """Box with the vehicles of each type that crossed the traffic light (mean across replicas)."""
    mean = agg.summary["crossed_veh"].mean
    std = agg.summary["crossed_veh"].std
    handles = [
        Patch(facecolor=TYPE_COLORS[k], label=f"{agg.cfg.specs[k].name}: {mean[k]:,.1f} ± {std[k]:,.1f}")
        for k in active
    ]
    box = ax.legend(
        handles=handles, loc="upper left", frameon=True, facecolor=SURFACE, edgecolor="none",
        framealpha=0.9, fontsize=8.5, labelcolor=INK_2,
        handlelength=1.0, title="Vehicles that crossed (mean ± σ)", title_fontsize=8.5,
        alignment="left", borderaxespad=0.6,
    )  # fmt: skip
    box.get_title().set_color(INK_2)


def _duration(seconds: float) -> str:
    """Readable duration: 1000 s -> '16 min 40 s', 5000 s -> '1 h 23 min', 7200 s -> '2 h'."""
    total = round(seconds)
    if total >= 3600:
        hours, minutes = divmod(round(total / 60), 60)
        return f"{hours:,} h" + (f" {minutes} min" if minutes else "")
    if total >= 60:
        minutes, secs = divmod(total, 60)
        return f"{minutes} min" + (f" {secs} s" if secs else "")
    return f"{seconds:g} s"


def _type_label(spec, rate: float) -> str:
    """Legend label: name, rate and, if it applies, the share that carries freight."""
    extra = ""
    if spec.cargo_prob >= 1:
        extra = ", goods only"
    elif spec.cargo_prob > 0:
        extra = f", {100 * spec.cargo_prob:g} % goods"
    return f"{spec.name} ({_rate_label(rate)}{extra})"


def _rate_label(rate: float) -> str:
    """Readable arrival rate: 20 -> '20/min'; under 1 veh/min -> '1 every x min' with x = 1/rate
    rounded (0.15 -> '1 every 7 min')."""
    if rate >= 1:
        return f"{rate:.3g}/min"
    return f"1 every {max(1, round(1 / rate))} min"


def _row_major(handles: list, ncol: int) -> list:
    """Reorder so the legend (which matplotlib fills by columns) reads by rows."""
    rows = math.ceil(len(handles) / ncol)
    return [handles[r * ncol + c] for c in range(ncol) for r in range(rows) if r * ncol + c < len(handles)]


def _end_labels(ax, t_end: float, ends: list[tuple[float, str]], fmt: str) -> None:
    """Value at the end of each line; if they collide, they are separated with a thin leader line."""
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
    passengers = [k for k in active if cfg.specs[k].carries_passengers]  # those that only carry freight are not plotted

    emis_pols, emis_types = emitted(cfg)
    # The last panel (speed per lane) is not a time series; with emissions, one more row (CO2 | NOx).
    n_panels = len(PANELS) + LANE_PANELS + (1 if emis_pols else 0)
    height = BASE_HEIGHT * (n_panels + VIOLIN_GAP) / BASE_PANELS

    def fy(y: float) -> float:
        """Vertical position of the header: the same distance in inches from the top edge."""
        return 1 - (1 - y) * BASE_HEIGHT / height

    fig = plt.figure(figsize=(11, height), facecolor=SURFACE)
    grid = fig.add_gridspec(n_panels + 1, 1, hspace=0.42, height_ratios=[1] * (n_panels - 1) + [VIOLIN_GAP, 1])
    # Time rows; the queue rows (entry | exit) and emission rows (CO2 | NOx) have two panels side by side.
    queue_row = len(PANELS) + 2
    pair_rows = {queue_row, queue_row + 1} if emis_pols else {queue_row}
    axes: list = []
    for i in range(n_panels - 1):
        if i in pair_rows:
            pair = grid[i].subgridspec(1, 2, wspace=0.16)
            axes.append(tuple(fig.add_subplot(pair[j], sharex=axes[0]) for j in range(2)))
        else:
            axes.append(fig.add_subplot(grid[i], sharex=axes[0] if axes else None))
    for row in axes[:-1]:  # only the last time row has the x axis
        for ax in row if isinstance(row, tuple) else (row,):
            ax.tick_params(labelbottom=False)
    axes.append(fig.add_subplot(grid[n_panels]))  # violins: own x axis (lanes)
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

    for bottom in axes[-2]:  # last time row (two panels)
        bottom.set_xlabel("Simulated time (s)", color=INK_2, fontsize=9)
        bottom.set_xlim(0, cfg.sim_seconds)
        proc = bottom.secondary_xaxis(
            -0.3, functions=(lambda s: s / cfg.time_scale, lambda p: p * cfg.time_scale)
        )
        proc.set_xlabel("Process time (s)", color=INK_2, fontsize=9)
        proc.tick_params(colors=MUTED, labelcolor=MUTED, labelsize=9, length=0)
        proc.spines["bottom"].set_color(BASELINE)

    # Legend with each type's arrival rate, in rows of up to LEGEND_COLS entries.
    handles = [
        Patch(facecolor=TYPE_COLORS[k], label=_type_label(cfg.specs[k], cfg.rates[k])) for k in active
    ] + phase_handles(cfg)
    # Columns of the same width (the longest label): as many as fit in a row, up to LEGEND_COLS.
    widest = max(len(h.get_label()) for h in handles) + 6  # + colour box and spacing
    ncol = max(1, min(len(handles), LEGEND_COLS, LEGEND_CHARS // widest))
    rows = math.ceil(len(handles) / ncol)
    fig.legend(
        handles=_row_major(handles, ncol), loc="upper left", bbox_to_anchor=(0.068, fy(0.935)),
        ncol=ncol, frameon=False, fontsize=9, labelcolor=INK_2, handlelength=1.2,
    )  # fmt: skip
    fig.suptitle(
        "Passenger mobility capacity per vehicle type",
        x=0.075, y=fy(0.985), ha="left", fontsize=14, color=INK, fontweight="bold",
    )  # fmt: skip
    fig.text(
        0.075, fy(0.958),
        f"Segment {cfg.length:g} m, {cfg.lanes} lane(s) · {cfg.lights_short} · "
        f"{agg.replicas} replicas (mean ± 1σ) · run {cfg.run:g} s → {cfg.sim_seconds:,g} simulated s "
        f"= {_duration(cfg.sim_seconds)} of simulated time",
        ha="left", fontsize=9, color=INK_2,
    )  # fmt: skip
    if footer:
        fig.text(0.075, fy(0.944), footer, ha="left", fontsize=8, color=MUTED)
    # The right margin leaves room for the longest label to the right of the panels: the
    # end values of the first panel and the reference speeds of the last one.
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


MAX_PAX_BARS = 30  # with more possible passenger values, the histogram groups them into bins
PAX_BIN_WIDTHS = (1, 2, 5, 10, 20, 50, 100)


def _pax_bin_width(n_values: int) -> int:
    """Bin width (in passengers) so the histogram has at most MAX_PAX_BARS bars."""
    return next((w for w in PAX_BIN_WIDTHS if math.ceil(n_values / w) <= MAX_PAX_BARS), PAX_BIN_WIDTHS[-1])


def _pax_bins(values: np.ndarray, weights: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Group `weights` (one per consecutive integer value) into bins of `width` values.
    Returns (first value, last value, sum) of each bin; the last one may be shorter."""
    starts = np.arange(0, values.size, width)
    lo = values[starts]
    hi = values[np.minimum(starts + width, values.size) - 1]
    return lo, hi, np.add.reduceat(weights, starts)


def plot_passenger_distribution(agg: Aggregate, path: Path, footer: str | None = None) -> None:
    """Distribution of passengers per vehicle of each type: histogram of all the vehicles that
    arrived during the simulation, summing the replicas, against the distribution expected from
    the configuration. One panel per type, with its own x axis; a wide range is grouped into bins."""
    from matplotlib.figure import Figure  # no pyplot: it does not depend on the backend
    from matplotlib.lines import Line2D

    from traffic.distributions import passenger_pmf

    cfg = agg.cfg
    hist = agg.pax_hist
    # Only the types that carry passengers; vehicles with freight are not in the histogram.
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
            "passengers per vehicle" + (f" (bins of {width})" if width > 1 else ""), color=INK_2, fontsize=9
        )  # fmt: skip
        ax.set_ylabel("% of vehicles", color=INK_2, fontsize=9)
        if values.size <= 12:
            ax.set_xticks(values)
        else:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
        ax.set_xlim(spec.pax_min - 0.6 * max(1, width), spec.pax_max + 0.6 * max(1, width))
        ax.text(
            0, 1.03,
            f"expected: mean {exp_mean:.2f} ± {exp_std:.2f} · configured [{spec.pax_min}, {spec.pax_max}], "
            f"μ {spec.pax_mean:g}, σ {spec.pax_std:g}",
            transform=ax.transAxes, fontsize=8.5, color=MUTED,
        )  # fmt: skip
        # Expected: steps over the same bins as the bars.
        edges = np.append(lo - 0.5, hi[-1] + 0.5)
        ax.stairs(expected, edges, color=INK, linewidth=1.5, baseline=None, zorder=4)
        if total == 0:
            ax.text(0, 1.11, "no arrivals", transform=ax.transAxes, fontsize=8.5, color=INK_2)
            ax.set_ylim(0, expected.max() * 1.15)
            continue
        share = 100.0 * binned / total
        bar_w = (hi - lo + 1) * (0.8 if width == 1 else 0.9)
        ax.bar((lo + hi) / 2, share, width=bar_w, color=TYPE_COLORS[k], linewidth=0, zorder=2)
        mean = float((values * counts).sum() / total)
        std = float(np.sqrt(((values - mean) ** 2 * counts).sum() / total))
        ax.text(
            0, 1.11, f"{total:,} vehicles · observed: mean {mean:.2f} ± {std:.2f} pax",
            transform=ax.transAxes, fontsize=8.5, color=INK_2,
        )  # fmt: skip
        ax.axvline(mean, color=INK_2, linewidth=1, linestyle=(0, (4, 3)), zorder=3)
        ax.annotate(
            "observed mean", (mean, 1), xycoords=("data", "axes fraction"), xytext=(4, -2),
            textcoords="offset points", va="top", fontsize=8, color=INK_2,
        )  # fmt: skip
        ax.set_ylim(0, max(share.max(), expected.max()) * 1.18)
    for ax in axes.flat[len(active):]:
        ax.set_visible(False)

    fh = fig.get_figheight()
    fig.suptitle(
        "Passengers per vehicle", x=0.075, y=1 - 0.3 / fh,
        ha="left", fontsize=14, color=INK, fontweight="bold",
    )  # fmt: skip
    fig.text(
        0.075, 1 - 0.58 / fh,
        f"All vehicles with passengers that arrived in {_duration(cfg.sim_seconds)} simulated, adding up "
        f"{agg.replicas} replicas",
        ha="left", fontsize=9, color=INK_2,
    )  # fmt: skip
    if footer:
        fig.text(0.075, 1 - 0.8 / fh, footer, ha="left", fontsize=8, color=MUTED)
    handles = [
        Patch(facecolor=MUTED, label="observed (bars, in the colour of each type)"),
        Line2D([], [], color=INK, linewidth=1.5, label="expected from the configuration (rounded and truncated normal)"),
    ]  # fmt: skip
    fig.legend(
        handles=handles, loc="upper left", bbox_to_anchor=(0.068, 1 - 0.95 / fh), ncol=2, frameon=False,
        fontsize=9, labelcolor=INK_2, handlelength=1.6,
    )  # fmt: skip
    fig.subplots_adjust(left=0.075, right=0.97, top=1 - 1.95 / fh, bottom=0.65 / fh)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)


def plot_variants(path: Path, meta: dict, data: dict) -> None:
    """Variant comparison (traffic-variants): on top, the mean speed of each lane per traffic light
    (bars, mean ± 1σ across replicas); below, the vehicles in the entry queue over
    time, one line per light. One column per segment length."""
    from matplotlib.figure import Figure  # no pyplot: it does not depend on the backend
    from matplotlib.lines import Line2D

    speed, t = data["speed"], data["t"]
    queue = data["queue"][:, :, :, meta["cola_tipos"]].sum(axis=3)  # (length, light, replica, sample)
    lengths, lights, lane_names = meta["largos_m"], meta["semaforos"], meta["carriles"]
    n_len, n_light, n_lanes = len(lengths), len(lights), len(lane_names)
    queue_label = ", ".join(meta["nombres"][k] for k in meta["cola_tipos"])
    lane_colors = _lane_colors(n_lanes)
    light_colors = TYPE_COLORS[:n_light]  # fixed order; the first three validated in every pair

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
            for p, v, e in zip(pos, m, sd):  # visible labels: some hues have little contrast
                ax.annotate(f"{v:.1f}", (p, v + e), xytext=(0, 3), textcoords="offset points", ha="center",
                            va="bottom", fontsize=8, color=INK_2)  # fmt: skip
        ax.set_xticks(x, [f"{s['nombre']}\n{s['rojo_s']:g} s / {s['verde_s']:g} s" for s in lights])
        ax.tick_params(axis="x", labelcolor=INK_2)
        ax.set_title(f"{L:g} m segment", loc="left", fontsize=11, color=INK, pad=8)
        ax.set_ylim(0, top * 1.2)
        ax.grid(axis="x", visible=False)
    axes[0, 0].set_ylabel("mean speed (km/h)", color=INK_2, fontsize=9)

    q_mean = queue.mean(axis=2)
    q_sd = queue.std(axis=2, ddof=1) if queue.shape[2] > 1 else np.zeros_like(q_mean)
    q_top = float((q_mean + q_sd).max()) * 1.06 or 1.0  # the same axis for all segments, without cutting lines
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
        ax.set_title(f"{L:g} m segment", loc="left", fontsize=11, color=INK, pad=8)
        ax.set_xlabel("simulated time (s)", color=INK_2, fontsize=9)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.tick_params(axis="x", labelcolor=MUTED)
        if n_light <= 4:  # direct labels at the end of each line, separated so they do not overlap
            gap = 0.07 * q_top
            ys: list[float] = []
            for v, _ in sorted(ends):
                ys.append(max(v, ys[-1] + gap) if ys else v)
            shift = max(0.0, ys[-1] - q_top)
            for (v, name), y in zip(sorted(ends), ys):
                ax.annotate(name, (t[-1], v), xytext=(t[-1] * 1.02, y - shift), textcoords="data", fontsize=8,
                            color=INK_2, va="center", annotation_clip=False)  # fmt: skip
    axes[1, 0].set_ylabel("vehicles in the entry queue", color=INK_2, fontsize=9)

    fig.suptitle("Traffic light variants: speed per lane and entry queue", x=0.06, y=0.985,
                 ha="left", fontsize=14, color=INK, fontweight="bold")  # fmt: skip
    names = dict(zip(meta["tipos"], meta["nombres"]))
    without = f"; without {', '.join(names[k] for k in meta['sin'])}" if meta["sin"] else ""
    fig.text(
        0.06, 0.945,
        f"{n_lanes} lane(s): {', '.join(lane_names)}{without} · yellow {meta['amarillo_s']:g} s · "
        f"run {meta['run_s']:g} s → {meta['s_simulados']:,g} simulated s · {meta['replicas']} replicas per scenario "
        f"(mean ± 1σ) · seed {meta['semilla']}",
        ha="left", fontsize=9, color=INK_2,
    )  # fmt: skip
    fig.text(0.06, 0.905, "Mean speed of each lane over the whole run (including stopped vehicles)",
             ha="left", fontsize=10.5, color=INK, fontweight="bold")  # fmt: skip
    # Each legend on its own line, under its row's title, so they do not overlap.
    fig.legend(handles=[Patch(facecolor=lane_colors[i], label=lane_names[i]) for i in range(n_lanes)],
               loc="upper left", bbox_to_anchor=(0.055, 0.9), ncol=min(n_lanes, 6), frameon=False, fontsize=9,
               labelcolor=INK_2)  # fmt: skip
    fig.text(0.06, 0.455, f"Vehicles in the entry queue: {queue_label} (already arrived, they do not fit on the segment yet)",
             ha="left", fontsize=10.5, color=INK, fontweight="bold")  # fmt: skip
    fig.legend(handles=[Line2D([], [], color=light_colors[i], linewidth=2,
                               label=f"{s['nombre']} ({s['rojo_s']:g} s / {s['verde_s']:g} s)")
                        for i, s in enumerate(lights)],
               loc="upper left", bbox_to_anchor=(0.055, 0.45), ncol=min(n_light, 4), frameon=False, fontsize=9,
               labelcolor=INK_2)  # fmt: skip
    fig.subplots_adjust(left=0.06, right=0.9, top=0.83, bottom=0.07)
    fig.savefig(path, dpi=150, facecolor=SURFACE)


SMOOTH_FLOW_MIN = 5.0  # min of the flow's moving average in the emission comparison


def _scenario_bars(ax, values: list[float], colors: list[str], labels: list[str], unit: str, fmt: str) -> None:
    """One bar per scenario with its value and, over the others, the change from the first."""
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
            text += "\n" + ("no change" if abs(pct) < 0.5 else f"{pct:+.0f} %")
        ax.annotate(text, (i, v), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5,
                    color=INK_2, linespacing=1.1)  # fmt: skip
    ax.set_xticks(x, labels, fontsize=8, rotation=0)
    ax.set_ylim(0, top * 1.4 if top > 0 else 1)
    _fit_format(ax)
    ax.set_ylabel(unit, color=INK_2, fontsize=9)


def plot_emission_comparison(path: Path, meta: dict, data: dict) -> None:
    """traffic-emissions scenario comparison: flow leaving the segment over time, vehicles per minute
    of each type, emissions per km of each type and pollutant, and emissions along the street."""
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    scenarios = meta["escenarios"]
    colors = [TYPE_COLORS[i] for i in range(len(scenarios))]
    short = [f"{i + 1}" for i in range(len(scenarios))]  # short labels under the bars; the legend names them
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

    # Flow leaving the segment (all lanes), with a moving average.
    ax = fig.add_subplot(grid[0, :])
    _style_axis(ax, "{:,.0f}")
    t = data["t"] / 60
    dt = (data["t"][1] - data["t"][0]) if data["t"].size > 1 else 1.0
    w = max(1, round(SMOOTH_FLOW_MIN * 60 / dt))
    for i, flow in enumerate(data["flow"]):
        # Trailing moving average; at the start, with the available history.
        cum = np.concatenate(([0.0], np.cumsum(np.nan_to_num(flow))))
        idx = np.arange(1, flow.size + 1)
        lo = np.maximum(idx - w, 0)
        ax.plot(t, (cum[idx] - cum[lo]) / (idx - lo), color=colors[i], linewidth=2)
    ax.set_ylim(bottom=0)
    ax.set_xlim(0, t[-1])
    ax.set_xlabel("simulated time (min)", color=INK_2, fontsize=9)
    ax.set_ylabel("veh/min", color=INK_2, fontsize=9)
    ax.set_title(f"Vehicles leaving the segment per minute (all types and lanes, moving average over "
                 f"{SMOOTH_FLOW_MIN:g} min)", loc="left", fontsize=10.5, color=INK, pad=8)  # fmt: skip

    for n, k in enumerate(active):
        ax = fig.add_subplot(grid[1 + n // ncols, n % ncols])
        values = list(data["crossed_veh"][:, k] / minutes)
        _scenario_bars(ax, values, colors, short, "veh/min", "{:,.2f}" if max(values) < 10 else "{:,.1f}")
        ax.set_title(f"Vehicles per minute: {names[k]}", loc="left", fontsize=10.5, color=INK, pad=8)

    for n, (k, p) in enumerate(emis_panels):
        ax = fig.add_subplot(grid[1 + rows_veh + n // ncols, n % ncols])
        name = pollutant_unit(POLLUTANTS[p])[0]
        _scenario_bars(ax, list(km[:, k, p]), colors, short, f"{name}/km", "{:,.0f}")
        ax.set_title(f"{POLLUTANT_LABELS[POLLUTANTS[p]]}: {names[k]}", loc="left", fontsize=10.5, color=INK, pad=8)

    pos = data["emissions_pos"].sum(axis=2) / EMIS_BIN / (meta["s_simulados"] / 3600.0)  # (scenario, pol, x)
    x = (np.arange(pos.shape[2]) + 0.5) * EMIS_BIN
    # topes_m: per scenario, a list of positions (in older data, one position or null)
    marks = [(m, "bump") for m in sorted({p for b in meta["topes_m"] for p in as_positions(b)})]
    marks_bumps = {len(as_positions(b)) for b in meta["topes_m"]}
    marks_bumps = range(max(marks_bumps, default=0))  # how many bumps the scenario with the most has
    marks += [(m, f"stop {name}") for name, m in meta["paradas_m"].items()]
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
        ax.set_xlabel("position on the street (m from the entry)", color=INK_2, fontsize=9)
        ax.set_ylabel(f"{name}/(m·h)", color=INK_2, fontsize=9)
        ax.set_title(f"{POLLUTANT_LABELS[POLLUTANTS[p]]} along the street (all lanes)", loc="left",
                     fontsize=10.5, color=INK, pad=8)  # fmt: skip

    fh = fig.get_figheight()
    fig.suptitle("Emissions per scenario", x=0.06, y=1 - 0.3 / fh, ha="left", fontsize=14, color=INK,
                 fontweight="bold")  # fmt: skip
    rates = " · ".join(f"{names[k]} {meta['tasas'][k]:.3g} veh/min" for k in active)
    bump_speed = " · ".join(f"{n} ≤ {v:g} km/h" for n, v in meta["velocidad_tope_kmh"].items() if v is not None)
    lanes = meta["carriles_tope"]
    where = "all lanes" if lanes is None or len(lanes) == len(meta["carriles"]) else (
        ("lane " if len(lanes) == 1 else "lanes ") + ", ".join(map(str, lanes)))
    ped = meta.get("peatones_tope")  # missing in older data
    walkers = (f" · with pedestrians ({ped['peatones_min']:.3g}/min, crossing in {ped['s_cruce']:g} s: vehicles stop "
               "completely)") if ped else ""  # fmt: skip
    lines = [
        f"Segment {meta['largo_m']:g} m · {', '.join(meta['carriles'])} · {rates}",
        f"{'Speed bumps' if len(marks_bumps) > 1 else 'Speed bump'} in {where}: "
        f"{bump_speed or 'no type slows down (no speed_bump_kmh)'}{walkers}",
        f"{meta['s_simulados']:,.0f} simulated s · mean of {meta['replicas']} replicas with the same seed "
        f"({meta['semilla']}) · emissions per km travelled on the segment, Int Panis et al. (2006) model",
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


def plot_bump_spacing(path: Path, meta: dict, data: dict, result: dict) -> None:
    """Results of `traffic-emissions --spacing` (or `--light-spacing`): how far apart two bumps or traffic lights
    (or those of a chain) must be for the first one's emission footprint to dilute. Five metrics against the
    spacing (additivity, mean over the stretch, excess between them, gradient and cost of each added one, total and per
    metre), one line per pollutant, and the profile along the street for some spacings."""
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    from traffic.bump_spacing import (
        LIGHT, NOUNS, chain_span, cycle_text, equivalence_text, first_position, kind_of, profiles, worst_case,
    )  # fmt: skip

    light = kind_of(meta) == LIGHT
    nn = NOUNS[kind_of(meta)]

    dist = np.array(result["distancias_m"])
    pols = list(result["pollutants"])
    colors = {pol: TYPE_COLORS[i] for i, pol in enumerate(pols)}
    height = 12.6
    k = 9.4 / height  # the margins and the header were computed for a height of 9.4 in: they are kept in inches
    n = int(meta.get("topes_por_cadena", 2))
    fig = Figure(figsize=(13.5, height), facecolor=SURFACE)
    grid = fig.add_gridspec(3, 3, hspace=0.5, wspace=0.28, left=0.06, right=0.98, top=1 - 0.22 * k, bottom=0.07 * k,
                            height_ratios=(1, 1, 1.15))  # fmt: skip
    rec, huella = result["recomendada_m"], result["huella_m"]
    blocked = np.asarray(result.get("cola_entrada_d", np.zeros(len(dist), bool)), bool)

    def panel(ax, key: str, title: str, ylabel: str, fmt: str, scale: float = 1.0, marks=(), ref=None, band=None):
        _style_axis(ax, fmt)
        if band is not None:
            ax.axhspan(*band, color=GRID, alpha=0.6, linewidth=0, zorder=0)
        if ref is not None:
            ax.axhline(ref, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
        for pol in pols:
            y = np.array(result["pollutants"][pol][key]) * scale
            ax.plot(dist, y, color=colors[pol], linewidth=2, marker="o", markersize=4, markeredgewidth=0)
            if blocked.any():  # the queue reaches the entry: hollow point, not comparable
                ax.plot(dist[blocked], y[blocked], linestyle="none", marker="o", markersize=7, markerfacecolor=SURFACE,
                        markeredgecolor=colors[pol], markeredgewidth=1.5, zorder=3)  # fmt: skip
        xmax = dist.max() * 1.02
        for at, label, color in marks:
            if at is not None and dist.min() * 0.9 <= at <= xmax:
                ax.axvline(at, color=color, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
                left = at > 0.7 * xmax  # near the right edge, the label goes to the left of the line
                ax.annotate(label, (at, 1.0), xycoords=("data", "axes fraction"), xytext=(-4 if left else 4, -3),
                            textcoords="offset points", ha="right" if left else "left", va="top", fontsize=8,
                            color=INK_2)  # fmt: skip
        ax.set_xlim(dist.min() * 0.9, xmax)
        ax.set_xlabel(f"spacing between the {nn.many} (m)", color=INK_2, fontsize=9)
        ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
        ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=8)

    tol, thr = meta["tolerancia"], meta["umbral"]
    r_add = [r["d_aditiva"] for r in result["pollutants"].values() if r["d_aditiva"] is not None]
    add_mark = [(max(r_add) if r_add else None, "stable" if light else "additive", INK_2)]
    before = (result.get("huella_antes_m") or 0.0) if light else 0.0  # with traffic lights, the queue and braking before it
    # The second leaves the first one's footprint when its own footprint before (queue) does not reach the first one's wake.
    flat_mark = [(None if huella is None else huella + before, "end of the footprint", INK_2)]
    ax = fig.add_subplot(grid[0, 0])
    green = light and not worst_case(meta)
    hidden = n - meta.get("desfase", "rojo").split(",").count("verde") if green else n
    if green:  # green wave: how many isolated lights the chain is worth, against «each green wave hides one»
        for pol in pols:
            r = result["pollutants"][pol]
            r["equivalente"] = n * np.array(r["aditividad"])
        panel(ax, "equivalente", "The chain in isolated lights" if n > 2 else "The pair in isolated lights",
              "extra emission ÷ a single light", "{:.2f}", ref=float(hidden))  # fmt: skip
        ax.annotate(f"{hidden} = each green wave hides one light", (dist.min(), hidden), xytext=(2, 4),
                    textcoords="offset points", fontsize=8, color=INK_2)  # fmt: skip
    elif light:  # with the red offset the chain does not tend to isolated lights: it is compared with the widest spacing
        title = (f"The pair against the most separated one (d = {dist[-1]:g} m)" if n == 2
                 else f"The chain against the most separated one (d = {dist[-1]:g} m)")  # fmt: skip
        panel(ax, "estabilidad", title, f"extra emission ÷ that of d = {dist[-1]:g} m", "{:.2f}", marks=add_mark,
              ref=1.0, band=(1 - tol, 1 + tol))  # fmt: skip
    else:
        title = (f"The two {nn.many} against two isolated {nn.many}" if n == 2
                 else f"The {n} {nn.many} against {n} isolated {nn.many}")  # fmt: skip
        panel(ax, "aditividad", title, f"extra emission ÷ ({n} × {nn.a})", "{:.2f}", marks=add_mark, ref=1.0,
              band=(1 - tol, 1 + tol))  # fmt: skip
    ax = fig.add_subplot(grid[0, 1])
    panel(ax, "media_tramo", f"Mean over the segment of the {nn.many}", f"× the street without {nn.many}", "{:.1f}", marks=add_mark,
          ref=1.0)  # fmt: skip
    ax = fig.add_subplot(grid[0, 2])
    panel(ax, "exceso_entre", f"Mean excess between the {nn.many}", f"% of the peak of {nn.a} alone", "{:.0f}", scale=100,
          marks=flat_mark, ref=100 * thr)  # fmt: skip
    ax = fig.add_subplot(grid[1, 0])
    for pol in pols:  # each pollutant in its unit: normalized by the maximum slope of a single bump
        r = result["pollutants"][pol]
        r["gradiente_rel"] = np.array(r["gradiente_tramo"]) / r["pendiente_tope_solo"] * 100
    panel(ax, "gradiente_rel", f"Mean gradient over the segment of the {nn.many}", f"% of the maximum slope of {nn.a} alone",
          "{:.0f}")  # fmt: skip
    ax = fig.add_subplot(grid[1, 1])
    panel(ax, "costo_tope", f"Cost of each added {nn.one}", f"× {nn.a} isolated (1 = emits like a single one)", "{:.2f}",
          marks=add_mark, ref=1.0)  # fmt: skip
    ax = fig.add_subplot(grid[1, 2])
    panel(ax, "costo_por_m", "Cost per metre of spacing", f"% of {nn.a} isolated per m", "{:.2f}")

    # Profile of the first pollutant (CO2 if emitted) for some spacings, with the single bump and the baseline.
    ax = fig.add_subplot(grid[2, :])
    _style_axis(ax, "{:,.0f}")
    pi = pols.index("co2") if "co2" in pols else 0
    prof = profiles(meta, data)
    x = result["x_m"]
    picks = sorted({int(np.argmin(np.abs(dist - target))) for target in np.quantile(dist, [0.15, 0.4, 0.65, 0.9])})
    first = first_position(meta)
    if light and huella is not None:  # the footprint of the single light, on both sides of its stop line
        ax.axvspan(first - before, first + huella, color=GRID, alpha=0.6, linewidth=0, zorder=0)
    ax.plot(x, prof[0, pi], color=MUTED, linewidth=1.5, label=nn.none)
    ax.plot(x, prof[1, pi], color=INK_2, linewidth=1.5, linestyle=(0, (4, 3)), label=nn.a)
    handles = [Line2D([], [], color=MUTED, linewidth=2, label=nn.none),
               Line2D([], [], color=INK_2, linewidth=2, linestyle=(0, (4, 3)), label=nn.a)]  # fmt: skip
    if light and huella is not None:
        handles.append(Patch(facecolor=GRID, alpha=0.6, label="footprint of one traffic light"))
    ax.axvline(first, color=BASELINE, linewidth=1, zorder=1)
    ramp = ("#b9aee6", "#8d7fd1", "#6656bd", "#3d2f94")  # single-hue ramp: the spacing is a magnitude
    for i, j in enumerate(picks):
        color = ramp[i + len(ramp) - len(picks)]
        ax.plot(x, prof[2 + j, pi], color=color, linewidth=1.8)
        for at in _chain_marks(meta, first, dist[j]):
            ax.axvline(at, color=color, linewidth=1, linestyle=(0, (1, 2)), zorder=1)
        handles.append(Line2D([], [], color=color, linewidth=2, label=f"d = {dist[j]:g} m"))
    lo = max(0.0, first - 25 - before)
    hi = min(meta["largo_m"], first + chain_span(meta, float(dist.max())) + (huella or 50) + 15)
    ax.set_xlim(lo, hi)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("position on the street (m from the entry)", color=INK_2, fontsize=9)
    ax.set_ylabel(f"{POLLUTANT_LABELS[pols[pi]]}, g/(m·h) (all lanes)", color=INK_2, fontsize=9)
    ax.set_title(f"{POLLUTANT_LABELS[pols[pi]]} profile along the street by spacing", loc="left",
                 fontsize=10.5, color=INK, pad=8)  # fmt: skip
    if light:  # the profiles reach the end of the segment: the legend goes above the panel, to the right of the title
        ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=8.5,
                  labelcolor=INK_2, ncol=4)  # fmt: skip
    else:
        ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=8.5, labelcolor=INK_2, ncol=3)

    legend = [Line2D([], [], color=colors[pol], linewidth=3, label=POLLUTANT_LABELS[pol]) for pol in pols]
    fig.legend(handles=legend, loc="upper left", bbox_to_anchor=(0.055, 1 - 0.138 * k), ncol=len(pols), frameon=False,
               fontsize=9.5, labelcolor=INK_2, handlelength=1.6)  # fmt: skip
    flag = "--light-spacing" if light else "--spacing"
    verdict = (f"Minimum spacing between {nn.many} ≈ {rec:g} m" if rec is not None
               else f"No spacing in the list dilutes the footprint: widen {flag}")
    if light and not worst_case(meta):  # with a green wave the question is how many isolated lights the chain is worth
        verdict = equivalence_text(meta, result)
    fig.suptitle(verdict, x=0.06, y=1 - 0.025 * k, ha="left", fontsize=15, color=INK, fontweight="bold")
    rates = " · ".join(f"{meta['nombres'][k]} {meta['tasas'][k]:.3g} veh/min" for k in meta["activos"])
    speed = " · ".join(f"{n} ≤ {v:g} km/h" for n, v in meta["velocidad_tope_kmh"].items() if v is not None)
    ped = meta.get("peatones_tope")
    wake = f"{before:.0f} m before + {huella:.0f} m after" if light and huella is not None else f"{huella or 0:.0f} m"
    if green:
        head = (f"Footprint of one traffic light ≈ {wake} (excess > {100 * thr:g} % of the peak) · ≡ isolated: the extra "
                f"emission of the chain counted in single lights (n × R)" if huella is not None else "No measurable footprint")
    elif light:
        far = result["pollutants"][pols[0]]["aditividad"][-1]
        head = (f"Footprint of one traffic light ≈ {wake} (excess > {100 * thr:g} % of the peak) · stable: within ±{100 * tol:g} % "
                f"of the most separated one, which emits {far:.2f} × {'two' if n == 2 else n} isolated ({POLLUTANT_LABELS[pols[0]]})"
                if huella is not None else "No measurable footprint")  # fmt: skip
    else:
        head = (f"Footprint of {nn.a} ≈ {wake} (until its excess falls below {100 * thr:g} % of the peak) · additive if "
                + ("the pair emits" if n == 2 else f"the chain of {n} {nn.many} emits")
                + f" the same as {'two' if n == 2 else n} isolated {nn.many} ±{100 * tol:g} %" if huella is not None
                else "No measurable footprint")  # fmt: skip
    lines = [
        head,
        f"{f'First {nn.one} at' if n == 2 else f'Chain of {n} {nn.many} from'} {first:g} m · segment {meta['largo_m']:g} m · {rates}",
    ]  # fmt: skip
    if light:
        fixed = meta.get("topes_fijos_m") or []
        lines.append(f"Fixed-cycle traffic lights: {cycle_text(meta)} · "
                     + (f"fixed bumps at {', '.join(f'{p:g}' for p in fixed)} m" if fixed else "no bumps")
                     + (" · ○ the queue reaches the entry (not comparable)" if blocked.any() else ""))  # fmt: skip
    else:
        lines.append(f"Speed over the bump: {speed or 'no speed_bump_kmh'}"
                     + (f" · with pedestrians ({ped['peatones_min']:.3g}/min, {ped['s_cruce']:g} s per crossing)" if ped else ""))  # fmt: skip
    lines.append(f"{meta['s_simulados']:,.0f} simulated s · mean of {meta['replicas']} replicas with the same seed "
                 f"({meta['semilla']}) · Int Panis et al. (2006) emission model")  # fmt: skip
    for i, line in enumerate(lines):
        fig.text(0.06, 1 - (0.055 + 0.022 * i) * k, line, ha="left", va="top", fontsize=9, color=INK_2)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)


def _chain_marks(meta: dict, first: float, d: float) -> list[float]:
    """Positions marked on a chain's profile: the last element or, with fixed spacings, all of them after the
    first."""
    fixed = meta.get("separaciones_fijas_m") or []
    if not fixed:
        n = int(meta.get("topes_por_cadena", 2))
        return [first + (n - 1) * d]
    return (first + np.cumsum([d, *fixed])).tolist()


def plot_calibration(path: Path, m: dict, rows: list[dict]) -> None:
    """Results of `traffic-calibrate`: the headway between crossings per queue position against the reference saturation
    headway, and flow against density in continuous flow against the reference capacity and jam
    density. The headline counts how many measurements are outside their range."""
    from matplotlib.figure import Figure

    from traffic.calibration import LABELS, REFERENCES

    fig = Figure(figsize=(13.5, 6.6), facecolor=SURFACE)
    grid = fig.add_gridspec(1, 2, wspace=0.22, left=0.06, right=0.98, top=0.70, bottom=0.11)
    color = TYPE_COLORS[0]

    def band(ax, ref, axis: str, label: str) -> None:
        span = ax.axhspan if axis == "y" else ax.axvspan
        span(ref.low, ref.high, color=GRID, alpha=0.7, linewidth=0, zorder=0)
        if axis == "y":  # label on the left, below the band
            ax.annotate(label, (0.0, ref.low), xycoords=("axes fraction", "data"), xytext=(4, -3),
                        textcoords="offset points", ha="left", va="top", fontsize=8.5, color=INK_2)  # fmt: skip
        else:
            ax.annotate(label, (ref.low, 1.0), xycoords=("data", "axes fraction"), xytext=(4, -3),
                        textcoords="offset points", ha="left", va="top", fontsize=8.5, color=INK_2)  # fmt: skip

    # Headway per queue position.
    ax = fig.add_subplot(grid[0, 0])
    _style_axis(ax, "{:.1f}")
    pos = np.arange(1, len(m["intervalo_por_posicion_s"]) + 1)
    mean, sd = np.asarray(m["intervalo_por_posicion_s"]), np.asarray(m["intervalo_por_posicion_sd_s"])
    band(ax, REFERENCES["intervalo_saturacion_s"], "y", "reference saturation headway")
    ax.fill_between(pos, mean - sd, mean + sd, color=color, alpha=0.15, linewidth=0, zorder=1)
    ax.plot(pos, mean, color=color, linewidth=2, marker="o", markersize=5, markeredgewidth=0, zorder=2)
    sat = m["intervalo_saturacion_s"]
    ax.axhline(sat, color=INK_2, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    ax.annotate(f"simulated saturation {sat:.2f} s", (1.0, sat), xycoords=("axes fraction", "data"), xytext=(-4, 4),
                textcoords="offset points", ha="right", va="bottom", fontsize=8.5, color=INK_2)  # fmt: skip
    ax.set_ylim(bottom=0)
    ax.set_xticks(pos)
    ax.set_xlabel("position in the queue when the light turns green", color=INK_2, fontsize=9)
    ax.set_ylabel("s since the previous one crossed (mean ± σ)", color=INK_2, fontsize=9)
    ax.set_title("Queue discharge on green", loc="left", fontsize=10.5, color=INK, pad=8)

    # Flow against density in continuous flow.
    ax = fig.add_subplot(grid[0, 1])
    _style_axis(ax, "{:,.0f}")
    k, q = np.asarray(m["densidad_veh_km"]), np.asarray(m["flujo_veh_h"])
    band(ax, REFERENCES["capacidad_marcha_veh_h"], "y", "reference capacity")
    jam_ref = REFERENCES["densidad_embotellamiento_veh_km"]
    band(ax, jam_ref, "x", "reference jam density")
    ax.plot(k, q, color=color, linewidth=2, marker="o", markersize=5, markeredgewidth=0, zorder=2)
    jam = m["densidad_embotellamiento_veh_km"]
    ax.axvline(jam, color=INK_2, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    ax.annotate(f"simulated jam density {jam:,.0f}", (jam, 0.02), xycoords=("data", "axes fraction"),
                xytext=(-4, 0), textcoords="offset points", ha="right", va="bottom", rotation=90, fontsize=8.5,
                color=INK_2)  # fmt: skip
    ax.set_xlim(0, max(jam, jam_ref.high) * 1.08)
    ax.set_ylim(0, max(q.max(), REFERENCES["capacidad_marcha_veh_h"].high) * 1.12)
    ax.set_xlabel("density on the segment (veh/km/lane)", color=INK_2, fontsize=9)
    ax.set_ylabel("flow (veh/h/lane)", color=INK_2, fontsize=9)
    ax.set_title("Continuous flow with increasing demand (no traffic light)", loc="left", fontsize=10.5, color=INK, pad=8)

    out = [r for r in rows if r["estado"] in ("abajo", "arriba")]
    judged = [r for r in rows if r["estado"] is not None]
    if judged:
        verdict = (f"{m['nombre'].capitalize()}: {len(out)} of {len(judged)} measurements outside the reference" if out
                   else f"{m['nombre'].capitalize()}: all measurements within the reference")  # fmt: skip
    else:
        verdict = f"Calibration of {m['nombre']} (the references are for cars: not compared)"
    fig.suptitle(verdict, x=0.06, y=0.975, ha="left", fontsize=15, color=INK, fontweight="bold")
    p = m["parametros"]
    accel = "instantaneous" if p["accel"] is None else f"{p['accel']:g} m/s²"
    decel = "dead stop" if p["decel"] is None else f"{p['decel']:g} m/s²"
    def item(r: dict) -> str:
        label, unit_, fmt = LABELS[r["medida"]]
        value = "—" if r["valor"] is None else fmt.format(r["valor"])
        return f"{label} {value} {unit_}" + {"abajo": " (low)", "arriba": " (high)", "dentro": "", None: ""}[r["estado"]]

    lines = [
        "  ·  ".join(item(r) for r in rows[:3]),
        "  ·  ".join(item(r) for r in rows[3:]),
        f"One lane with only {m['nombre']} vehicles: {p['speed_kmh']:g} km/h (lane at {p['limite_kmh']:g}), length {p['largo_m']:g} m, "
        f"gap_run {p['gap_run_m']:g} m, gap_stop {p['gap_stop_m']:g} m, accel {accel}, decel {decel}, reaction "
        f"{p['reaccion_s']} s · {m['colas']} queues",
    ]  # fmt: skip
    for i, line in enumerate(lines):
        fig.text(0.06, 0.905 - 0.04 * i, line, ha="left", va="top", fontsize=9, color=INK_2)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
