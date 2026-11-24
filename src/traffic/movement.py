"""Visualization of the vehicles' movement in one replica, to tune the model.

A replica is simulated again (same seed as in the run, so it is identical) recording
the state of each vehicle in a time window, and it produces:

  * a space-time diagram (PNG): one line per vehicle, position against time, one panel
    per lane. The slope is the speed; a horizontal stretch, a stopped vehicle; the
    fan when the light turns green shows the reaction and the stretching.
  * a top-down video: WebM (VP9) by default, MP4 (H.264) or GIF according to
    [animation] format. WebM is patent-free and plays on Fedora without extra
    codecs (Fedora's VLC, GNOME Videos/Showtime, browsers); without ffmpeg, GIF.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from traffic.config import DT, GREEN, RED, YELLOW, SimConfig, StopLine
from traffic.engine import Simulation
from traffic.plotting import (
    BASELINE, INK, INK_2, MUTED, RED_PHASE, SURFACE, TYPE_COLORS, YELLOW_PHASE,
    _lane_assignments, _lane_label, _style_axis, phase_handles, shade_phases,
)  # fmt: skip
from traffic.settings import AnimationOptions

GREEN_LIGHT = "#1f9d55"
BOTTLENECK_COLOR = (0.0, 0.0, 0.0, 1.0)  # black: stopped by a bottleneck
LIGHT_COLORS = {RED: RED_PHASE, GREEN: GREEN_LIGHT, YELLOW: YELLOW_PHASE}  # colour of the traffic light in the video
ROAD = "#ecebe6"
MAX_PLOT_FRAMES = 1500  # maximum number of instants in the space-time diagram


@dataclass(slots=True)
class Trajectories:
    """State of the vehicles at each recorded instant (concatenated arrays, CSR style)."""

    cfg: SimConfig
    t: np.ndarray  # (F,) simulated s of each instant
    phase: np.ndarray  # (F,) phase of the light at the end of the segment (RED, GREEN or YELLOW)
    inner_phase: np.ndarray  # (F, intermediate stop lines) phase of each one (`cfg.stop_lines`)
    offsets: np.ndarray  # (F+1,) rows of instant f: offsets[f]:offsets[f+1]
    vid: np.ndarray
    vtype: np.ndarray
    lane: np.ndarray
    target: np.ndarray  # target lane if changing lanes; -1 otherwise
    x: np.ndarray  # front of the vehicle (m)
    stopped: np.ndarray
    row_rank: np.ndarray  # place in its side-by-side row (0 = the one in front)
    row_size: np.ndarray  # vehicles in that row (1 = alone)
    vlen: np.ndarray  # length of the vehicle (m)
    cargo: np.ndarray  # carries freight (no passengers)
    bottleneck: np.ndarray  # stopped by a [bottleneck]
    crossed_pax: np.ndarray  # (F, types) passengers that crossed the light since the start of the window
    crossed_veh: np.ndarray  # (F, types) vehicles that crossed it since the start of the window
    queued: np.ndarray  # (F, lanes) vehicles in the entry queue of each lane (still outside the segment)
    exit_queued: np.ndarray  # (F, lanes) m occupied in the exit queue of each lane (they already crossed the light)
    exit_closed: np.ndarray  # (F, lanes) the lane's next vehicle does not fit in the exit queue: line closed

    @property
    def n_frames(self) -> int:
        return self.t.size

    def frame(self, f: int) -> slice:
        return slice(self.offsets[f], self.offsets[f + 1])


def record(cfg: SimConfig, seed: int, replica: int, start: float, end: float) -> Trajectories:
    """Simulate replica `replica` (1 = the first) of the run with seed `seed` up to `end` and
    record every DT step in [start, end]. It is identical to the run_parallel replica."""
    seq = np.random.SeedSequence(seed).spawn(replica)[replica - 1]
    sim = Simulation(cfg, np.random.default_rng(seq))
    first, last = round(start / DT), round(end / DT)
    ts, phases, counts, parts, pax, veh, queued, exit_queued, exit_closed = [], [], [], [], [], [], [], [], []
    inner = []
    base_pax, base_veh = sim.cum_pax.copy(), sim.cum_veh.copy()
    while sim.tick < last:
        if sim.tick < first:  # what crossed before the window does not count in the video
            base_pax, base_veh = sim.cum_pax.copy(), sim.cum_veh.copy()
        sim.step()
        if sim.tick < first:
            continue
        n = sim.n
        ts.append(sim.tick * DT)
        pax.append(sim.cum_pax - base_pax)
        queued.append(sim.queue_len)
        exit_queued.append(sim.exit_q.copy())
        exit_closed.append(sim.exit_closed.copy())
        veh.append(sim.cum_veh - base_veh)
        phases.append(sim.exit_phase(sim.tick))
        inner.append(sim.line_phases(sim.tick))
        counts.append(n)
        rank, size = sim.row_positions()
        parts.append((sim.vid[:n].copy(), sim.vtype[:n].copy(), sim.lane[:n].copy(),
                      sim.lc_target[:n].copy(), sim.x[:n].astype(np.float32), sim.stopped[:n].copy(),
                      rank, size, sim.vlen[:n].astype(np.float32), sim.pax[:n] == 0,
                      (sim.stop_state[:n] == sim.AT_STOP) & (sim.stop_kind[:n] == sim.BOTTLENECK_STOP)))  # fmt: skip
    cols = [np.concatenate(c) if c else np.zeros(0) for c in zip(*parts)] if parts else [np.zeros(0)] * 11
    return Trajectories(
        cfg=cfg, t=np.array(ts), phase=np.array(phases, np.int8),
        inner_phase=np.array(inner, np.int8).reshape(len(ts), len(cfg.stop_lines)),
        offsets=np.concatenate(([0], np.cumsum(counts))).astype(np.int64),
        vid=cols[0], vtype=cols[1], lane=cols[2], target=cols[3], x=cols[4], stopped=cols[5],
        row_rank=cols[6], row_size=cols[7], vlen=cols[8], cargo=cols[9].astype(np.bool_),
        bottleneck=cols[10].astype(np.bool_),
        crossed_pax=np.array(pax).reshape(-1, cfg.n_types), crossed_veh=np.array(veh).reshape(-1, cfg.n_types),
        queued=np.array(queued, np.int64).reshape(-1, cfg.lanes),
        exit_queued=np.array(exit_queued, np.float32).reshape(-1, cfg.lanes),
        exit_closed=np.array(exit_closed, np.bool_).reshape(-1, cfg.lanes),
    )  # fmt: skip


def _active(cfg: SimConfig) -> list[int]:
    return [k for k, rate in enumerate(cfg.rates) if rate > 0]


def _phase_spans(traj: Trajectories, phases: np.ndarray) -> list[tuple[float, float, str]]:
    """Red and yellow phases of `phases` (one per recorded instant) within the window: (start, end, colour).
    They come from what was recorded, not from the config's cycle, because a pedestrian crossing's phase depends on
    the pedestrians."""
    if not phases.size:
        return []
    change = np.flatnonzero(np.diff(phases)) + 1
    first = np.r_[0, change]
    last = np.r_[change - 1, phases.size - 1]
    t1 = float(traj.t[-1])
    colors = {RED: RED_PHASE, YELLOW: YELLOW_PHASE}
    return [(float(traj.t[a]), min(float(traj.t[b]) + DT, t1), colors[int(phases[a])])
            for a, b in zip(first, last) if int(phases[a]) in colors]  # fmt: skip


def _stops(cfg: SimConfig, lane: int) -> list[tuple[int, float]]:
    """(type, position) of the stops of the participating types that can use that lane."""
    return [
        (k, cfg.specs[k].stop_position) for k in _active(cfg)
        if cfg.specs[k].stop_position is not None and lane in cfg.allowed_lanes(k)
    ]  # fmt: skip


def _type_legend(fig, cfg: SimConfig, y: float, extra: list | None = None) -> None:
    from matplotlib.patches import Patch

    handles = [Patch(facecolor=TYPE_COLORS[k], label=cfg.specs[k].name) for k in _active(cfg)] + (extra or [])
    fig.legend(
        handles=handles, loc="upper left", bbox_to_anchor=(0.068, y), ncol=len(handles), frameon=False,
        fontsize=9, labelcolor=INK_2, handlelength=1.2,
    )  # fmt: skip


def plot_space_time(traj: Trajectories, path: Path, title_note: str) -> None:
    """Space-time diagram: one panel per lane (the left one on top, the right one at the bottom)."""
    from matplotlib.collections import LineCollection
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    cfg = traj.cfg
    lanes = cfg.lanes
    step = max(1, -(-traj.n_frames // MAX_PLOT_FRAMES))
    frames = np.arange(0, traj.n_frames, step)
    # Rows of the chosen instants; a vehicle changing lanes also occupies the target lane.
    rows = np.concatenate([np.arange(traj.offsets[f], traj.offsets[f + 1]) for f in frames])
    frame_of = np.repeat(frames, np.diff(traj.offsets)[frames])
    changing = traj.target[rows] >= 0
    occ_rows = np.concatenate((rows, rows[changing]))
    occ_frame = np.concatenate((frame_of, frame_of[changing]))
    occ_lane = np.concatenate((traj.lane[rows], traj.target[rows][changing])).astype(np.int64)

    fig = Figure(figsize=(12, 1.5 + 2.2 * lanes), facecolor=SURFACE)
    axes = fig.subplots(lanes, 1, sharex=True, squeeze=False, gridspec_kw={"hspace": 0.35})[:, 0]
    assigned = _lane_assignments(cfg, _active(cfg))
    # With intermediate traffic lights or pedestrian crossings, the phases are marked on each one's line (and that of the
    # end of the segment); with only the one at the end, they shade the whole panel.
    marked = bool(cfg.stop_lines)
    exit_spans = _phase_spans(traj, traj.phase) if cfg.has_light else []
    line_spans = [(s, _phase_spans(traj, traj.inner_phase[:, j])) for j, s in enumerate(cfg.stop_lines)]
    if cfg.has_light:
        line_spans.append((StopLine(cfg.length, cfg.exit_light), exit_spans))
    crossings = {b.position for b in cfg.crossings}
    for lane in range(lanes):
        ax = axes[lanes - 1 - lane]  # left lane on top, as in a top-down view
        _style_axis(ax, "{:,.0f}")
        if not marked and lane not in cfg.free_lanes:  # without other lines, the light's phases shade the panel
            shade_phases(ax, exit_spans)
        ax.axhline(cfg.length, color=INK_2, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
        for line, spans in line_spans if marked else ():
            if line.light is not None:  # a bump's pedestrian crossing is drawn with it, below
                ax.axhline(line.position, color=INK_2, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
            if lane in line.free_lanes or (line.lanes is not None and lane not in line.lanes):
                continue
            for a, b, color in spans:
                ax.hlines(line.position, a, b, colors=color, linewidth=3, zorder=3)
        for bump in cfg.bumps:
            if lane in cfg.bump_lanes(bump):
                pos = bump.position
                ax.axhline(pos, color=MUTED, linewidth=1.5, linestyle=(0, (6, 2, 1, 2)), zorder=1)
                ax.annotate("bump · pedestrians" if pos in crossings else "bump", (0, pos),
                            xycoords=("axes fraction", "data"), xytext=(4, 3), textcoords="offset points", ha="left",
                            fontsize=8, color=INK_2)  # fmt: skip
        for k, pos in _stops(cfg, lane):
            ax.axhline(pos, color=TYPE_COLORS[k], linewidth=1, linestyle=(0, (1, 2)), zorder=1)
            ax.annotate(f"stop {cfg.specs[k].name}", (1, pos), xycoords=("axes fraction", "data"),
                        xytext=(-4, 3), textcoords="offset points", ha="right", fontsize=8, color=INK_2)  # fmt: skip
        sel = occ_lane == lane
        r, fr = occ_rows[sel], occ_frame[sel]
        order = np.lexsort((fr, traj.vid[r]))
        r, fr = r[order], fr[order]
        # Segment between two consecutive instants of the same vehicle in this lane.
        link = (traj.vid[r][1:] == traj.vid[r][:-1]) & (fr[1:] - fr[:-1] == step)
        a, b = r[:-1][link], r[1:][link]
        segs = np.stack(
            (np.column_stack((traj.t[fr[:-1][link]], traj.x[a])), np.column_stack((traj.t[fr[1:][link]], traj.x[b]))),
            axis=1,
        )  # fmt: skip
        colors = np.array(TYPE_COLORS)[traj.vtype[a].astype(np.int64)] if a.size else []
        ax.add_collection(LineCollection(segs, colors=colors, linewidths=0.9, zorder=2))
        ax.set_ylim(0, cfg.length * 1.04)
        ax.set_title(_lane_label(lane, lanes, assigned[lane]), loc="left", fontsize=10, color=INK, pad=6)
        ax.set_ylabel("position (m)", color=INK_2, fontsize=9)
    axes[-1].set_xlim(traj.t[0], traj.t[-1])
    axes[-1].set_xlabel("Simulated time (s)", color=INK_2, fontsize=9)

    fh = fig.get_figheight()
    fig.suptitle("Space-time diagram per lane", x=0.075, y=1 - 0.25 / fh, ha="left",
                 fontsize=14, color=INK, fontweight="bold")  # fmt: skip
    fig.text(
        0.075, 1 - 0.56 / fh,
        f"{title_note} · slope = speed, horizontal stretch = stopped, "
        f"dashed line = {'traffic light' if cfg.has_light else 'end of the segment'} ({cfg.length:g} m)",
        ha="left", fontsize=9, color=INK_2,
    )  # fmt: skip
    _type_legend(fig, cfg, 1 - 0.7 / fh, [*phase_handles(cfg, per_light=True),
                                          Line2D([], [], color=INK_2, linewidth=1, linestyle=(0, (4, 3)),
                                                 label="stop line")])  # fmt: skip
    fig.subplots_adjust(left=0.075, right=0.975, top=1 - 1.3 / fh, bottom=0.55 / fh)
    fig.savefig(path, dpi=150, facecolor=SURFACE)


def _vehicle_verts(traj: Trajectories, f: int, lane_h: float) -> tuple[np.ndarray, np.ndarray]:
    """Rectangles (N, 4, 2) of the vehicles of instant f and their type."""
    s = traj.frame(f)
    vt = traj.vtype[s].astype(np.int64)
    front = traj.x[s].astype(np.float64)
    rear = front - traj.vlen[s].astype(np.float64)
    lane = traj.lane[s].astype(np.float64)
    tgt = traj.target[s]
    y = np.where(tgt >= 0, (lane + tgt) / 2, lane)  # halfway while changing lanes
    # Side-by-side row: the lane is shared among its vehicles (the one in front on the right = below).
    size = traj.row_size[s].astype(np.float64)
    slot = 0.9 / size
    y = y + np.where(size > 1, (traj.row_rank[s] - (size - 1) / 2) * slot, 0.0)
    half = np.minimum(lane_h, 0.85 * slot) / 2
    lo, hi = y - half, y + half
    verts = np.stack(
        (np.column_stack((rear, lo)), np.column_stack((front, lo)),
         np.column_stack((front, hi)), np.column_stack((rear, hi))), axis=1,
    )  # fmt: skip
    return verts, vt


def render_video(traj: Trajectories, path: Path, anim: AnimationOptions, title_note: str) -> Path:
    """Top-down video at `path` + the format's extension ([animation] format), without
    overwriting. With ffmpeg (WebM or MP4) only the vehicles, the traffic light and the clock are redrawn
    over a fixed background; GIF, or any format without ffmpeg, uses matplotlib's writer.
    Returns the path written."""
    import shutil
    import subprocess

    import matplotlib
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.collections import PolyCollection
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle

    cfg = traj.cfg
    lanes = cfg.lanes
    assigned = _lane_assignments(cfg, _active(cfg))
    labels = [_lane_label(k, lanes, assigned[k]) for k in range(lanes)]
    lane_in = 0.34  # inches per lane
    width_in = 12.0
    fig = Figure(figsize=(width_in, 1.75 + lane_in * lanes), facecolor=SURFACE)
    FigureCanvasAgg(fig)
    fh = fig.get_figheight()
    left = min(0.35, (0.2 + 0.062 * max(len(label) for label in labels)) / width_in)  # ~0.062 in per character
    ax = fig.add_axes((left, 0.75 / fh, 0.975 - left, lane_in * lanes / fh))
    ax.set_facecolor(SURFACE)
    margin = max(sp.longest for sp in cfg.specs) + 5
    # Left of the segment, room for the entry queue counter (~0.45 in); on the right, if some lane has
    # an exit queue ([exit]), for its "m occupied/length" counter (~0.75 in).
    exit_lanes = [lane for lane, c in enumerate(cfg.lane_exit_capacity) if c > 0]
    axes_in = (0.975 - left) * width_in
    side_in = 0.45 + (0.75 if exit_lanes else 0.0)
    m_per_in = (cfg.length + margin) / (axes_in - side_in)
    queue_pad = 0.45 * m_per_in
    exit_pad = 0.75 * m_per_in if exit_lanes else 0.0
    ax.set_xlim(-queue_pad, cfg.length + margin + exit_pad)
    ax.set_ylim(-0.5, lanes - 0.5)
    ax.add_patch(Rectangle((0, -0.5), cfg.length, lanes, facecolor=ROAD, edgecolor="none", zorder=0))
    for k in range(1, lanes):
        ax.axhline(k - 0.5, color=BASELINE, linewidth=0.8, linestyle=(0, (6, 4)), zorder=1)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=8, length=0)
    ax.set_yticks(range(lanes), labels)
    ax.set_xlabel("position (m)", color=INK_2, fontsize=9)
    for bump in cfg.bumps:
        pos, bump_lanes = bump.position, cfg.bump_lanes(bump)
        for lane in bump_lanes:  # bump: hatched band across the width of the lane
            ax.add_patch(Rectangle((pos - 0.4, lane - 0.45), 0.8, 0.9, facecolor=YELLOW_PHASE, edgecolor=INK_2,
                                   hatch="////", linewidth=0.6, zorder=1))  # fmt: skip
        if bump_lanes:  # a single label per bump, over the topmost lane with a bump
            label = "bump · pedestrians" if bump.pedestrian else "bump"
            ax.annotate(label, (pos, max(bump_lanes) + 0.45), xytext=(0, 1),
                        textcoords="offset points", ha="center", va="bottom", fontsize=7, color=INK_2, zorder=1)  # fmt: skip
    for lane in range(lanes):
        for k, pos in _stops(cfg, lane):
            ax.plot([pos, pos], [lane - 0.45, lane + 0.45], color=TYPE_COLORS[k], linewidth=2, zorder=1,
                    solid_capstyle="butt")  # fmt: skip
            ax.annotate("stop", (pos, lane + 0.45), xytext=(0, 1), textcoords="offset points", ha="center",
                        va="bottom", fontsize=7, color=INK_2, zorder=1)  # fmt: skip
    # Entry queue: vehicles that already arrived at each lane but do not fit in the segment yet.
    queue_x = -0.12 * queue_pad
    ax.text(queue_x, lanes - 0.5, "queued", ha="right", va="bottom", fontsize=7, color=INK_2)
    queue_texts = [
        ax.text(queue_x, lane, "", ha="right", va="center", fontsize=9, color=INK, family="monospace", animated=True)
        for lane in range(lanes)
    ]  # fmt: skip
    # Exit queue: m occupied by those that crossed and have not left yet, against its length; in red if the next one
    # does not fit (that lane's line closes even if the light is green).
    exit_x = cfg.length + margin + exit_pad
    exit_texts = []
    if exit_lanes:
        ax.text(exit_x, lanes - 0.5, "exit", ha="right", va="bottom", fontsize=7, color=INK_2)
        exit_texts = [
            (lane, ax.text(exit_x, lane, "", ha="right", va="center", fontsize=9, color=INK, family="monospace",
                           animated=True))
            for lane in exit_lanes
        ]  # fmt: skip
        ax.annotate("exit: occupied m/length\nin red, the next one does not fit: line closed", (exit_x, -0.5), xytext=(0, -20),
                    textcoords="offset points", ha="right", va="top", fontsize=7, color=INK_2)  # fmt: skip
    storage = cfg.lane_exit_storage
    # Traffic light: one bar per lane; lanes without a light always green. Without a light nothing is drawn.
    light_w = max(1.5, cfg.length * 0.006)
    sources = [(line, traj.inner_phase[:, j]) for j, line in enumerate(cfg.stop_lines)]
    if cfg.has_light:
        sources.append((StopLine(cfg.length, cfg.exit_light), traj.phase))
    lights = [(Rectangle((line.position, lane - 0.5), light_w, 1, zorder=3, animated=True), lane, phases,
               line.free_lanes)
              for line, phases in sources for lane in range(lanes)
              if line.lanes is None or lane in line.lanes]  # fmt: skip
    for light, *_ in lights:
        ax.add_patch(light)
    # Legend of the pedestrian lights (green for vehicles except with pedestrians waiting), over their line.
    walkers = [line for line, _ in sources if line.light is not None and line.light.pedestrian]
    for line in walkers:
        ax.annotate("pedestrian light", (line.position, lanes - 0.5), xytext=(0, 1), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7, color=INK_2, zorder=1)  # fmt: skip
    cars = PolyCollection([], linewidths=0.9, zorder=2, animated=True)
    ax.add_collection(cars)
    # Clock: simulated time and real (process) time = simulated / time_scale, as in --run.
    clock = fig.text(0.975, 1 - 0.3 / fh, "", ha="right", fontsize=11, color=INK, family="monospace",
                     animated=True)  # fmt: skip
    fig.text(0.02, 1 - 0.3 / fh, "Vehicle movement", ha="left", fontsize=13, color=INK,
             fontweight="bold")  # fmt: skip
    fig.text(0.02, 1 - 0.55 / fh,
             f"{title_note} · video ×{anim.speed:g} · dark border = stopped"
             + (" · light = goods" if any(sp.cargo_prob > 0 for sp in cfg.specs) else "")
             + (" · black = bottleneck" if cfg.bottleneck_active else "")
             + (" · pedestrian light: red only with pedestrians waiting" if walkers else ""),
             ha="left", fontsize=9, color=INK_2)  # fmt: skip
    counters = _crossing_counters(fig, traj, width_in, 1 - 0.68 / fh)

    colors = np.array([matplotlib.colors.to_rgba(c) for c in TYPE_COLORS])
    edge_stop = matplotlib.colors.to_rgba(INK)
    stride = max(1, round(anim.speed / anim.fps / DT))
    frames = range(0, traj.n_frames, stride)

    def update(f: int) -> None:
        verts, vt = _vehicle_verts(traj, f, 0.42)
        cars.set_verts(list(verts))
        face = colors[vt]
        face[traj.cargo[traj.frame(f)], 3] = 0.4  # those carrying freight, lighter
        face[traj.bottleneck[traj.frame(f)]] = BOTTLENECK_COLOR  # stopped by [bottleneck], while it lasts
        cars.set_facecolor(face)
        edges = np.zeros((vt.size, 4))
        edges[traj.stopped[traj.frame(f)]] = edge_stop
        cars.set_edgecolor(edges)
        for light, lane, phases, free_lanes in lights:
            light.set_facecolor(GREEN_LIGHT if lane in free_lanes else LIGHT_COLORS[phases[f]])
        clock.set_text(_clock_text(float(traj.t[f])))
        for lane, text in enumerate(queue_texts):
            text.set_text(f"{traj.queued[f, lane]:,}")
        for lane, text in exit_texts:
            text.set_text(f"{traj.exit_queued[f, lane]:.0f}/{storage[lane]:g} m")
            text.set_color(RED_PHASE if traj.exit_closed[f, lane] else INK)
        for k, text in counters:
            text.set_text(_crossing_text(traj, f, k))

    def progress(i: int) -> None:
        if (i + 1) % 20 == 0 or i + 1 == len(frames):
            sys.stderr.write(f"\r  video: frame {i + 1}/{len(frames)}")
            sys.stderr.flush()

    dpi = 110
    fig.set_dpi(dpi)
    ffmpeg = shutil.which(matplotlib.rcParams["animation.ffmpeg_path"]) or shutil.which("ffmpeg")
    fmt = anim.format
    if ffmpeg is None and fmt != "gif":
        sys.stderr.write(f"  Warning: ffmpeg not found; writing the video as a GIF instead of {fmt}\n")
        fmt = "gif"
    path = _free_path(path.with_name(f"{path.name}.{fmt}"))
    if fmt == "gif":
        from matplotlib import animation

        for artist in (*(lt[0] for lt in lights), cars, clock, *queue_texts, *(t for _, t in exit_texts), *(text for _, text in counters)):
            artist.set_animated(False)
        writer = animation.PillowWriter(fps=anim.fps)
        with writer.saving(fig, str(path), dpi=dpi):
            for i, f in enumerate(frames):
                update(f)
                writer.grab_frame(facecolor=SURFACE)
                progress(i)
        sys.stderr.write("\n")
        return path

    # Fixed background drawn once; each frame restores it and draws only the animated artists.
    canvas = fig.canvas
    canvas.draw()
    background = canvas.copy_from_bbox(fig.bbox)
    w, h = canvas.get_width_height()
    if fmt == "webm":  # patent-free VP9; 'realtime' to encode as fast as it is drawn
        codec = ["-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "30", "-deadline", "realtime", "-cpu-used", "8",
                 "-row-mt", "1"]  # fmt: skip
    else:
        codec = ["-c:v", _h264_encoder(ffmpeg), "-b:v", "8M", "-g", str(anim.fps)]
    cmd = [
        ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{w}x{h}",
        "-r", str(anim.fps), "-i", "-", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        *codec, "-pix_fmt", "yuv420p", str(path),
    ]  # fmt: skip
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for i, f in enumerate(frames):
            update(f)
            canvas.restore_region(background)
            for artist in (cars, *(lt[0] for lt in lights), clock, *queue_texts, *(t for _, t in exit_texts), *(text for _, text in counters)):
                fig.draw_artist(artist)
            proc.stdin.write(canvas.buffer_rgba())
            progress(i)
    finally:
        proc.stdin.close()
        proc.wait()
    sys.stderr.write("\n")
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg exited with code {proc.returncode}")
    return path


def _crossing_text(traj: Trajectories, f: int, k: int) -> str:
    """Counter of a type in frame f: passengers that crossed; vehicles if it only carries freight."""
    spec = traj.cfg.specs[k]
    if not spec.carries_passengers:
        return f"{spec.name} {int(traj.crossed_veh[f, k]):,} veh"
    return f"{spec.name} {int(traj.crossed_pax[f, k]):,} pax"


def _crossing_counters(fig, traj: Trajectories, width_in: float, y: float) -> list:
    """Video legend: a colour box per type with its crossing counter (animated text).
    Each entry reserves the width of its final text so the counter does not overlap as it grows."""
    from matplotlib.patches import Rectangle

    fh = fig.get_figheight()
    square = 0.12  # pulgadas
    title = f"Crossed {traj.cfg.line_name} in the video:"
    fig.text(0.02, y, title, ha="left", va="center", fontsize=9, color=INK_2)
    x = 0.02 + (0.075 * len(title) + 0.15) / width_in
    counters = []
    for k in _active(traj.cfg):
        widest = _crossing_text(traj, traj.n_frames - 1, k) if traj.n_frames else traj.cfg.specs[k].name
        fig.add_artist(Rectangle((x, y - square / 2 / fh), square / width_in, square / fh, transform=fig.transFigure,
                                 facecolor=TYPE_COLORS[k], edgecolor="none"))  # fmt: skip
        text = fig.text(x + (square + 0.06) / width_in, y, "", ha="left", va="center", fontsize=9, color=INK_2,
                        animated=True)  # fmt: skip
        counters.append((k, text))
        x += (square + 0.06 + 0.075 * (len(widest) + 2) + 0.2) / width_in  # ~0.075 in per character at 9 pt
    return counters


def _h264_encoder(ffmpeg: str) -> str:
    """Available H.264 encoder (libx264 or, e.g. on Fedora, libopenh264); if there is none, MPEG-4."""
    import subprocess

    listing = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    return next((enc for enc in ("libx264", "libopenh264") if f" {enc} " in listing), "mpeg4")


def _hms(seconds: float) -> str:
    """Simulated seconds as hh:mm:ss (no tenths)."""
    hours, rest = divmod(int(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _clock_text(t_sim: float) -> str:
    """Video clock: simulated time in hours, minutes and seconds."""
    return f"simulated {_hms(t_sim)}"


def _free_path(path: Path) -> Path:
    """Do not overwrite: if the file exists, add a numeric suffix."""
    k = 2
    candidate = path
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}-{k}{path.suffix}")
        k += 1
    return candidate


def visualize(cfg: SimConfig, seed: int, anim: AnimationOptions, out_dir: Path, run_name: str) -> list[Path]:
    """Record the chosen replica in the [animation] window and write the diagram and the video."""
    start, end = anim.window(cfg)
    print(f"Movement: replica {anim.replica}, t = {start:g}–{end:g} simulated s, video ×{anim.speed:g} "
          f"at {anim.fps} frames/s (~{(end - start) / anim.speed:.0f} s of video)")  # fmt: skip
    traj = record(cfg, seed, anim.replica, start, end)
    tag = f"{start:g}-{end:g}s"
    note = f"run {run_name} · seed {seed} · replica {anim.replica}"
    diagram = _free_path(out_dir / f"space_time_{tag}.png")
    plot_space_time(traj, diagram, f"{note} · t = {start:g}–{end:g} s")  # its x axis is in seconds
    video = render_video(traj, out_dir / f"movement_{tag}", anim, f"{note} · t = {_hms(start)}–{_hms(end)}")
    return [diagram, video]
