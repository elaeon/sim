"""Landscape (16:9) visualization of a train journey: map on the left, elevation profile on the right.

Same approach as the traffic video (trafico/movement.py): matplotlib draws the static background
once, each frame redraws only the moving artists, and the RGBA buffer is piped to ffmpeg.
Without ffmpeg (or with fmt="gif") frames are written as a GIF with Pillow.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from rail.common import BASELINE, GRID, INK, INK_2, MUTED, SURFACE, free_path, h264_encoder
from rail.route import Point, Route
from rail.sim import Journey, clock, duration

TRACK = "#c3c2b7"
TRAVELLED = "#2a78d6"
TRAIN = "#e34948"
STATION = "#0b0b0b"
HIGHLIGHT = "#1baf7a"  # highlighted stations (e.g. Chepe Express stops)
HIGHLIGHT_INK = "#0e7a52"  # their labels: darker, readable on the light background
LANDMARK = "#eb6834"
TERRAIN = "#e8e2d0"
SIZE_IN = (12.8, 7.2)  # 1280x720 at 100 dpi
DPI = 100
NOTE_RANGE_KM = 25.0  # a point of interest stays "current" this far after passing it
DENSE_SAMPLES = 1500  # samples of the elevation profile
INSET_BOX = (0.03, 0.62, 0.33, 0.35)  # zoom of the slow section, in map-axes fractions (empty upper-left corner)
INSET_PAD_DEG = 0.05
INSET_LABEL_GAP_KM = 4.0  # closer stations share one label in the inset
WARP_SAMPLES = 20001  # time grid used to build the slow-motion frame times


def _points_of_interest(route: Route) -> list[Point]:
    return [p for p in route.points if p.kind in ("station", "landmark")]


def _station_style(p: Point) -> dict:
    """Marker colours of a station: filled green when highlighted, hollow otherwise."""
    return {"mfc": HIGHLIGHT, "mec": HIGHLIGHT_INK} if p.highlight else {"mfc": SURFACE, "mec": STATION}


def _label_style(p: Point) -> dict:
    return {"color": HIGHLIGHT_INK, "weight": "bold"} if p.highlight else {"color": INK}


def _add_legend(route: Route, ax) -> None:
    """Map legend (lower right, empty in the Chepe map), only when some station is highlighted."""
    from matplotlib.lines import Line2D

    if not any(p.highlight for p in route.stations):
        return
    handles = [
        Line2D([], [], ls="", marker="o", ms=6, mew=1.3, mfc=HIGHLIGHT, mec=HIGHLIGHT_INK, label=route.highlight_label),
        Line2D([], [], ls="", marker="o", ms=6, mew=1.3, mfc=SURFACE, mec=STATION, label="other station"),
        Line2D([], [], ls="", marker="^", ms=6, color=LANDMARK, label="landmark"),
    ]
    legend = ax.legend(handles=handles, loc="lower right", fontsize=7.5, frameon=True, framealpha=0.9,
                       edgecolor=BASELINE, facecolor=SURFACE, borderpad=0.6)  # fmt: skip
    legend.set_zorder(7)


def _build_figure(route: Route):
    """Static figure: titles, map, profile and, with a real track and a slow section, a zoomed inset.

    Returns (fig, ax_map, ax_prof, ax_inset); ax_inset is None without an inset.
    """
    from matplotlib.figure import Figure

    fig = Figure(figsize=SIZE_IN, dpi=DPI, facecolor=SURFACE)
    fig.text(0.03, 0.955, route.name, fontsize=16, weight="bold", color=INK, ha="left", va="center")
    if route.track is None:
        source = "approximate curated data"
    elif route.track.elevation_m is None:
        source = "real track (OpenStreetMap), approximate stations and elevations"
    else:
        source = "real track (OpenStreetMap), terrain elevation (SRTM)"
    fig.text(0.03, 0.915, f"{route.length_km:.0f} km · {len(route.stations)} stations · "
             f"{len(route.landmarks)} landmarks · {source}", fontsize=9.5, color=INK_2,
             ha="left", va="center")  # fmt: skip
    if route.track is not None:
        credits = "Track: © OpenStreetMap contributors (ODbL)"
        if route.track.elevation_m is not None:
            credits += " · Elevation: SRTM (NASA) via OpenTopoData"
        fig.text(0.03, 0.012, credits, fontsize=7, color=MUTED, ha="left", va="bottom")

    km = np.linspace(0, route.length_km, DENSE_SAMPLES)
    elev = route.locate(km)[2]
    _, lon, lat = route.line()

    # Map (lon/lat with a local aspect correction, so distances look right).
    ax_map = fig.add_axes((0.03, 0.05, 0.44, 0.83))
    ax_map.set_facecolor(SURFACE)
    ax_map.set_aspect(1 / np.cos(np.radians(lat.mean())))
    ax_map.plot(lon, lat, color=TRACK, lw=3 if route.track is None else 2.2, solid_capstyle="round", zorder=1)
    pad = 0.25
    ax_map.set_xlim(lon.min() - pad, lon.max() + pad)
    ax_map.set_ylim(lat.min() - pad, lat.max() + pad)
    ax_map.tick_params(labelsize=7, colors=MUTED, length=2)
    ax_map.grid(color=GRID, lw=0.5)
    for side in ax_map.spines.values():
        side.set_color(BASELINE)
    for p in route.stations:
        ax_map.plot(p.lon, p.lat, "o", ms=6 if p.highlight else 5, mew=1.3, zorder=3, **_station_style(p))
        ax_map.annotate(p.name, (p.lon, p.lat), xytext=p.label_offset, textcoords="offset points", fontsize=7.5,
                        ha="left" if p.label_offset[0] >= 0 else "right", va="center", zorder=4,
                        **_label_style(p))  # fmt: skip
    for p in route.landmarks:
        ax_map.plot(p.lon, p.lat, "^", ms=6, color=LANDMARK, zorder=3)
        ax_map.annotate(p.name, (p.lon, p.lat), xytext=p.label_offset, textcoords="offset points", fontsize=7,
                        color=LANDMARK, style="italic", ha="left" if p.label_offset[0] >= 0 else "right",
                        va="center", zorder=4, arrowprops={"arrowstyle": "-", "color": LANDMARK, "lw": 0.5,
                                                           "shrinkA": 1, "shrinkB": 4})  # fmt: skip
    _add_legend(route, ax_map)

    # Elevation profile.
    ax_prof = fig.add_axes((0.54, 0.08, 0.43, 0.40))
    ax_prof.set_facecolor(SURFACE)
    ax_prof.fill_between(km, 0, elev, color=TERRAIN, lw=0, zorder=0)
    ax_prof.plot(km, elev, color=TRACK, lw=1.5, zorder=1)
    top = elev.max() * 1.45
    ax_prof.set_xlim(0, route.length_km)
    ax_prof.set_ylim(0, top)
    ax_prof.set_xlabel("km from " + route.points[0].name, fontsize=8, color=INK_2)
    ax_prof.set_ylabel("elevation (m)", fontsize=8, color=INK_2)
    ax_prof.tick_params(labelsize=7, colors=MUTED, length=2)
    for name in ("top", "right"):
        ax_prof.spines[name].set_visible(False)
    for name in ("left", "bottom"):
        ax_prof.spines[name].set_color(BASELINE)
    last_label = -1e9
    for p in route.stations:
        x_label = max(p.km, last_label + route.length_km * 0.02)  # shift labels that would overlap to the right
        ax_prof.plot([p.km, p.km, x_label], [p.elevation_m, top * 0.64, top * 0.7], color=BASELINE, lw=0.6, zorder=0)
        ax_prof.plot(p.km, p.elevation_m, "o", ms=4 if p.highlight else 3.5, mew=1, zorder=3, **_station_style(p))
        ax_prof.text(x_label, top * 0.71, p.name, rotation=90, fontsize=6.5, ha="center", va="bottom",
                     **_label_style(p))  # fmt: skip
        last_label = x_label
    for p in route.landmarks:
        ax_prof.plot(p.km, p.elevation_m, "^", ms=4.5, color=LANDMARK, zorder=3)
    slow = route.slow_section()
    if slow is not None:
        ax_prof.axvspan(slow[0], slow[1], color=LANDMARK, alpha=0.07, lw=0, zorder=0)

    ax_inset = _build_inset(route, ax_map) if route.track is not None and slow is not None else None
    return fig, ax_map, ax_prof, ax_inset


def _build_inset(route: Route, ax_map):
    """Zoom of the slow section on the real track, in the empty upper-left corner of the map."""
    from matplotlib.patches import Rectangle

    km_from, km_to, factor = route.slow_section()
    line_km, lon, lat = route.line()
    part = (line_km >= km_from) & (line_km <= km_to)
    x0, x1 = lon[part].min() - INSET_PAD_DEG, lon[part].max() + INSET_PAD_DEG
    y0, y1 = lat[part].min() - INSET_PAD_DEG, lat[part].max() + INSET_PAD_DEG
    ax_map.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=MUTED, lw=0.8, ls="--", zorder=2))

    ax = ax_map.inset_axes(INSET_BOX)
    ax.set_facecolor(SURFACE)
    ax.set_aspect(ax_map.get_aspect(), anchor="NW")  # the box shrinks to the zoom's shape
    ax.plot(lon, lat, color=TRACK, lw=1.8, solid_capstyle="round", zorder=1)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_color(MUTED)
    spec = route.realistic
    ax.set_title(f"{spec.slow_from} – {spec.slow_to} (×{factor:g} slower)", fontsize=7.5, color=INK_2, pad=3)
    shown = [p for p in route.points if p.kind != "waypoint" and km_from <= p.km <= km_to]
    labeled: list[Point] = []  # highlighted stations first, so a close neighbour never hides them
    for p in sorted((p for p in shown if p.kind == "station"), key=lambda p: (not p.highlight, p.km)):
        if all(abs(p.km - q.km) >= INSET_LABEL_GAP_KM for q in labeled):
            labeled.append(p)
    for p in shown:
        if p.kind != "station":
            ax.plot(p.lon, p.lat, "^", ms=4.5, color=LANDMARK, zorder=3)
            continue
        ax.plot(p.lon, p.lat, "o", ms=4.5 if p.highlight else 4, mew=1, zorder=3, **_station_style(p))
        if p in labeled:  # labels to the upper left, or lower right near the left edge
            left = p.lon > x0 + 0.25 * (x1 - x0)
            ax.annotate(p.name, (p.lon, p.lat), xytext=(-4, 4) if left else (5, -3), textcoords="offset points",
                        fontsize=6, color=HIGHLIGHT_INK if p.highlight else INK_2,
                        weight="bold" if p.highlight else "normal", ha="right" if left else "left",
                        va="bottom" if left else "top", zorder=4, clip_on=True)  # fmt: skip
    return ax


def frame_times(journey: Journey, n_frames: int, slow: tuple[float, float, float] | None = None) -> np.ndarray:
    """Simulated time of each video frame: uniform, or `factor` times denser between km_from and km_to."""
    if slow is None:
        return np.linspace(0.0, journey.duration_s, n_frames)
    km_from, km_to, factor = slow
    t = np.linspace(0.0, journey.duration_s, WARP_SAMPLES)
    km = journey.at(t)[0]
    weight = np.where((km >= km_from) & (km <= km_to), factor, 1.0)
    video = np.concatenate(([0.0], np.cumsum((weight[1:] + weight[:-1]) / 2 * np.diff(t))))
    return np.interp(np.linspace(0.0, video[-1], n_frames), video, t)


def plot_route(route: Route, path: Path) -> Path:
    """Static overview (route.png): map, profile and the list of points of interest."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    fig = _build_figure(route)[0]
    FigureCanvasAgg(fig)
    fig.text(0.54, 0.86, "Points of interest", fontsize=11, weight="bold", color=INK, ha="left", va="center")
    pois = [p for p in _points_of_interest(route) if p.note]
    step = min(0.032, 0.34 / max(len(pois), 1))
    for i, p in enumerate(pois):
        color = LANDMARK if p.kind == "landmark" else HIGHLIGHT_INK if p.highlight else INK_2
        fig.text(0.54, 0.83 - i * step, f"km {p.km:>3.0f}  {p.note}", fontsize=7, color=color, ha="left",
                 va="center", family="monospace")  # fmt: skip
    path = free_path(path)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    return path


def render_journey(route: Route, journey: Journey, path: Path, *, duration_s: float = 60.0, fps: int = 25,
                   fmt: str = "webm", depart_s: float = 6 * 3600, dpi: int = DPI) -> Path:  # fmt: skip
    """Video of the whole journey compressed to about duration_s seconds. Returns the written path."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    fig, ax_map, ax_prof, ax_inset = _build_figure(route)
    fig.set_dpi(dpi)
    FigureCanvasAgg(fig)
    n_frames = max(2, round(duration_s * fps))
    slow = route.slow_section()
    times = frame_times(journey, n_frames, slow)
    frame_km, frame_v = journey.at(times)
    dense_km = np.linspace(0, route.length_km, DENSE_SAMPLES)
    dense_elev = route.locate(dense_km)[2]
    line_km, line_lon, line_lat = route.line()
    pois = _points_of_interest(route)
    poi_km = np.array([p.km for p in pois])
    next_stop_km = np.array([s.km for s in route.stations])

    # Moving artists.
    done_map, = ax_map.plot([], [], color=TRAVELLED, lw=3, solid_capstyle="round", zorder=2, animated=True)
    ring_map, = ax_map.plot([], [], "o", ms=14, mfc="none", mec=LANDMARK, mew=1.6, zorder=5, animated=True)
    train_map, = ax_map.plot([], [], "o", ms=9, color=TRAIN, mec=SURFACE, mew=1.5, zorder=6, animated=True)
    done_prof, = ax_prof.plot([], [], color=TRAVELLED, lw=2, zorder=2, animated=True)
    train_prof, = ax_prof.plot([], [], "o", ms=7, color=TRAIN, mec=SURFACE, mew=1.2, zorder=6, animated=True)
    big = fig.text(0.54, 0.855, "", fontsize=22, weight="bold", color=INK, ha="left", va="center", animated=True)
    elapsed = fig.text(0.54, 0.808, "", fontsize=10, color=MUTED, ha="left", va="center", animated=True)
    stats = fig.text(0.54, 0.765, "", fontsize=11, color=INK_2, ha="left", va="center", family="monospace",
                     animated=True)  # fmt: skip
    nxt = fig.text(0.54, 0.715, "", fontsize=11, color=INK_2, ha="left", va="center", animated=True)
    title = fig.text(0.54, 0.65, "", fontsize=12, weight="bold", color=LANDMARK, ha="left", va="center",
                     animated=True)  # fmt: skip
    note = fig.text(0.54, 0.61, "", fontsize=10, color=INK, ha="left", va="center", animated=True)
    badge = fig.text(0.97, 0.955, "", fontsize=10, weight="bold", color=LANDMARK, ha="right", va="center",
                     animated=True)  # fmt: skip
    moving = [done_map, ring_map, train_map, done_prof, train_prof, big, elapsed, stats, nxt, title, note, badge]
    if ax_inset is not None:
        done_inset, = ax_inset.plot([], [], color=TRAVELLED, lw=2, solid_capstyle="round", zorder=2, animated=True)
        train_inset, = ax_inset.plot([], [], "o", ms=7, color=TRAIN, mec=SURFACE, mew=1.2, zorder=6, animated=True)
        moving += [done_inset, train_inset]

    def update(i: int) -> None:
        km, v = float(frame_km[i]), float(frame_v[i])
        lon, lat, elev = (float(c) for c in route.locate(km))
        cut = int(np.searchsorted(line_km, km))
        done_map.set_data(np.append(line_lon[:cut], lon), np.append(line_lat[:cut], lat))
        cut = int(np.searchsorted(dense_km, km))
        done_prof.set_data(np.append(dense_km[:cut], km), np.append(dense_elev[:cut], elev))
        if ax_inset is not None:
            done_inset.set_data(done_map.get_data())
            train_inset.set_data([lon], [lat])
        in_slow = slow is not None and slow[0] <= km <= slow[1]
        badge.set_text(f"slow motion ×{slow[2]:g}" if in_slow else "")
        train_map.set_data([lon], [lat])
        train_prof.set_data([km], [elev])
        big.set_text(clock(depart_s + times[i]))
        elapsed.set_text(f"departure {clock(depart_s)} · {duration(times[i])} elapsed")
        stats.set_text(f"km {km:5.1f} / {route.length_km:.0f}   {v:4.0f} km/h   {elev:5.0f} m")
        k = int(np.searchsorted(next_stop_km, km - 1e-3))
        if k < len(route.stations):
            ahead = route.stations[k]
            nxt.set_text(f"stopped at {ahead.name}" if abs(ahead.km - km) < 1e-3
                         else f"next stop: {ahead.name} in {ahead.km - km:.0f} km")  # fmt: skip
        j = int(np.searchsorted(poi_km, km + 1e-3)) - 1
        if j >= 0 and km - pois[j].km <= NOTE_RANGE_KM and pois[j].note:
            current = pois[j]
            title.set_text(current.name)
            note.set_text(current.note)
            ring_map.set_data([current.lon], [current.lat])
            ring_map.set_markeredgecolor(LANDMARK if current.kind == "landmark"
                                         else HIGHLIGHT_INK if current.highlight else STATION)  # fmt: skip
            title.set_color(LANDMARK if current.kind == "landmark" else INK)
        else:
            title.set_text("")
            note.set_text("")
            ring_map.set_data([], [])

    def progress(i: int) -> None:
        if (i + 1) % 50 == 0 or i + 1 == n_frames:
            sys.stderr.write(f"\r  video: frame {i + 1}/{n_frames}")
            sys.stderr.flush()

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None and fmt != "gif":
        sys.stderr.write(f"  Warning: ffmpeg not found; writing a GIF instead of {fmt}\n")
        fmt = "gif"
    path = free_path(path.with_suffix(f".{fmt}"))
    if fmt == "gif":
        from matplotlib import animation

        for artist in moving:
            artist.set_animated(False)
        writer = animation.PillowWriter(fps=fps)
        with writer.saving(fig, str(path), dpi=dpi):
            for i in range(n_frames):
                update(i)
                writer.grab_frame(facecolor=SURFACE)
                progress(i)
        sys.stderr.write("\n")
        return path

    canvas = fig.canvas
    canvas.draw()
    background = canvas.copy_from_bbox(fig.bbox)
    w, h = canvas.get_width_height()
    if fmt == "webm":
        codec = ["-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "30", "-deadline", "realtime", "-cpu-used", "8",
                 "-row-mt", "1"]  # fmt: skip
    else:
        codec = ["-c:v", h264_encoder(ffmpeg), "-b:v", "6M", "-g", str(fps)]
    cmd = [
        ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{w}x{h}",
        "-r", str(fps), "-i", "-", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", *codec, "-pix_fmt", "yuv420p", str(path),
    ]  # fmt: skip
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for i in range(n_frames):
            update(i)
            canvas.restore_region(background)
            for artist in moving:
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
