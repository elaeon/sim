"""Visualización del movimiento de los vehículos de una réplica, para ajustar el modelo.

Se vuelve a simular una réplica (misma semilla que en la corrida, así que es idéntica) grabando
el estado de cada vehículo en una ventana de tiempo, y se generan:

  * un diagrama espacio-tiempo (PNG): una línea por vehículo, posición contra tiempo, un panel
    por carril. La pendiente es la velocidad; un tramo horizontal, un vehículo detenido; el
    abanico al ponerse verde muestra la reacción y el estiramiento.
  * un video visto desde arriba: WebM (VP9) por defecto, MP4 (H.264) o GIF según
    [animation] format. WebM es libre de patentes y se reproduce en Fedora sin códecs
    adicionales (VLC de Fedora, GNOME Videos/Showtime, navegadores); sin ffmpeg, GIF.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from trafico.config import DT, SimConfig
from trafico.engine import Simulation
from trafico.plotting import (
    BASELINE, INK, INK_2, MUTED, RED_PHASE, SURFACE, TYPE_COLORS,
    _lane_assignments, _lane_label, _style_axis,
)  # fmt: skip
from trafico.settings import AnimationOptions

GREEN_LIGHT = "#1f9d55"
ROAD = "#ecebe6"
MAX_PLOT_FRAMES = 1500  # instantes como máximo en el diagrama espacio-tiempo


@dataclass(slots=True)
class Trajectories:
    """Estado de los vehículos en cada instante grabado (arreglos concatenados, estilo CSR)."""

    cfg: SimConfig
    t: np.ndarray  # (F,) s simulados de cada instante
    green: np.ndarray  # (F,) semáforo en verde
    offsets: np.ndarray  # (F+1,) filas del instante f: offsets[f]:offsets[f+1]
    vid: np.ndarray
    vtype: np.ndarray
    lane: np.ndarray
    target: np.ndarray  # carril destino si está cambiando de carril; -1 si no
    x: np.ndarray  # frente del vehículo (m)
    stopped: np.ndarray
    row_rank: np.ndarray  # lugar en su fila lado a lado (0 = el de adelante)
    row_size: np.ndarray  # vehículos en esa fila (1 = solo)

    @property
    def n_frames(self) -> int:
        return self.t.size

    def frame(self, f: int) -> slice:
        return slice(self.offsets[f], self.offsets[f + 1])


def record(cfg: SimConfig, seed: int, replica: int, start: float, end: float) -> Trajectories:
    """Simula la réplica `replica` (1 = la primera) de la corrida con semilla `seed` hasta `end` y
    graba cada paso de DT en [start, end]. Es idéntica a la réplica de run_parallel."""
    seq = np.random.SeedSequence(seed).spawn(replica)[replica - 1]
    sim = Simulation(cfg, np.random.default_rng(seq))
    first, last = round(start / DT), round(end / DT)
    ts, greens, counts, parts = [], [], [], []
    while sim.tick < last:
        sim.step()
        if sim.tick < first:
            continue
        n = sim.n
        ts.append(sim.tick * DT)
        greens.append(cfg.is_green(sim.tick))
        counts.append(n)
        rank, size = sim.row_positions()
        parts.append((sim.vid[:n].copy(), sim.vtype[:n].copy(), sim.lane[:n].copy(),
                      sim.lc_target[:n].copy(), sim.x[:n].astype(np.float32), sim.stopped[:n].copy(),
                      rank, size))  # fmt: skip
    cols = [np.concatenate(c) if c else np.zeros(0) for c in zip(*parts)] if parts else [np.zeros(0)] * 8
    return Trajectories(
        cfg=cfg, t=np.array(ts), green=np.array(greens, np.bool_),
        offsets=np.concatenate(([0], np.cumsum(counts))).astype(np.int64),
        vid=cols[0], vtype=cols[1], lane=cols[2], target=cols[3], x=cols[4], stopped=cols[5],
        row_rank=cols[6], row_size=cols[7],
    )  # fmt: skip


def _active(cfg: SimConfig) -> list[int]:
    return [k for k, rate in enumerate(cfg.rates) if rate > 0]


def _red_spans(traj: Trajectories) -> list[tuple[float, float]]:
    t0, t1 = float(traj.t[0]), float(traj.t[-1])
    return [(max(a, t0), min(b, t1)) for a, b in traj.cfg.red_intervals() if b > t0 and a < t1]


def _stops(cfg: SimConfig, lane: int) -> list[tuple[int, float]]:
    """(tipo, posición) de las paradas de los tipos que participan y pueden usar ese carril."""
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
    """Diagrama espacio-tiempo: un panel por carril (el izquierdo arriba, el derecho abajo)."""
    from matplotlib.collections import LineCollection
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    cfg = traj.cfg
    lanes = cfg.lanes
    step = max(1, -(-traj.n_frames // MAX_PLOT_FRAMES))
    frames = np.arange(0, traj.n_frames, step)
    # Filas de los instantes elegidos; quien cambia de carril ocupa también el carril destino.
    rows = np.concatenate([np.arange(traj.offsets[f], traj.offsets[f + 1]) for f in frames])
    frame_of = np.repeat(frames, np.diff(traj.offsets)[frames])
    changing = traj.target[rows] >= 0
    occ_rows = np.concatenate((rows, rows[changing]))
    occ_frame = np.concatenate((frame_of, frame_of[changing]))
    occ_lane = np.concatenate((traj.lane[rows], traj.target[rows][changing])).astype(np.int64)

    fig = Figure(figsize=(12, 1.5 + 2.2 * lanes), facecolor=SURFACE)
    axes = fig.subplots(lanes, 1, sharex=True, squeeze=False, gridspec_kw={"hspace": 0.35})[:, 0]
    assigned = _lane_assignments(cfg, _active(cfg))
    spans = _red_spans(traj)
    for lane in range(lanes):
        ax = axes[lanes - 1 - lane]  # carril izquierdo arriba, como en una vista desde arriba
        _style_axis(ax, "{:,.0f}")
        for a, b in spans:
            ax.axvspan(a, b, color=RED_PHASE, alpha=0.07, linewidth=0, zorder=0)
        ax.axhline(cfg.length, color=INK_2, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
        for k, pos in _stops(cfg, lane):
            ax.axhline(pos, color=TYPE_COLORS[k], linewidth=1, linestyle=(0, (1, 2)), zorder=1)
            ax.annotate(f"parada {cfg.specs[k].name}", (1, pos), xycoords=("axes fraction", "data"),
                        xytext=(-4, 3), textcoords="offset points", ha="right", fontsize=8, color=INK_2)  # fmt: skip
        sel = occ_lane == lane
        r, fr = occ_rows[sel], occ_frame[sel]
        order = np.lexsort((fr, traj.vid[r]))
        r, fr = r[order], fr[order]
        # Segmento entre dos instantes consecutivos del mismo vehículo en este carril.
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
        ax.set_ylabel("posición (m)", color=INK_2, fontsize=9)
    axes[-1].set_xlim(traj.t[0], traj.t[-1])
    axes[-1].set_xlabel("Tiempo simulado (s)", color=INK_2, fontsize=9)

    fh = fig.get_figheight()
    fig.suptitle("Diagrama espacio-tiempo por carril", x=0.075, y=1 - 0.25 / fh, ha="left",
                 fontsize=14, color=INK, fontweight="bold")  # fmt: skip
    fig.text(
        0.075, 1 - 0.56 / fh,
        f"{title_note} · pendiente = velocidad, tramo horizontal = detenido, "
        f"línea punteada = semáforo ({cfg.length:g} m)",
        ha="left", fontsize=9, color=INK_2,
    )  # fmt: skip
    _type_legend(fig, cfg, 1 - 0.7 / fh, [Patch(facecolor=RED_PHASE, alpha=0.2, label="semáforo en rojo"),
                                          Line2D([], [], color=INK_2, linewidth=1, linestyle=(0, (4, 3)),
                                                 label="línea de alto")])  # fmt: skip
    fig.subplots_adjust(left=0.075, right=0.975, top=1 - 1.3 / fh, bottom=0.55 / fh)
    fig.savefig(path, dpi=150, facecolor=SURFACE)


def _vehicle_verts(traj: Trajectories, f: int, lane_h: float) -> tuple[np.ndarray, np.ndarray]:
    """Rectángulos (N, 4, 2) de los vehículos del instante f y su tipo."""
    s = traj.frame(f)
    vt = traj.vtype[s].astype(np.int64)
    front = traj.x[s].astype(np.float64)
    rear = front - np.array([sp.length for sp in traj.cfg.specs])[vt]
    lane = traj.lane[s].astype(np.float64)
    tgt = traj.target[s]
    y = np.where(tgt >= 0, (lane + tgt) / 2, lane)  # a medio camino mientras cambia de carril
    # Fila lado a lado: el carril se reparte entre sus vehículos (el de adelante, a la derecha = abajo).
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
    """Video visto desde arriba en `path` + extensión del formato ([animation] format), sin
    sobrescribir. Con ffmpeg (WebM o MP4) solo se redibujan los vehículos, el semáforo y el reloj
    sobre un fondo fijo; el GIF, o cualquier formato sin ffmpeg, usa el escritor de matplotlib.
    Devuelve la ruta escrita."""
    import shutil
    import subprocess

    import matplotlib
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.collections import PolyCollection
    from matplotlib.figure import Figure
    from matplotlib.patches import Patch, Rectangle

    cfg = traj.cfg
    lanes = cfg.lanes
    assigned = _lane_assignments(cfg, _active(cfg))
    labels = [_lane_label(k, lanes, assigned[k]) for k in range(lanes)]
    lane_in = 0.34  # pulgadas por carril
    width_in = 12.0
    fig = Figure(figsize=(width_in, 1.75 + lane_in * lanes), facecolor=SURFACE)
    FigureCanvasAgg(fig)
    fh = fig.get_figheight()
    left = min(0.35, (0.2 + 0.062 * max(len(label) for label in labels)) / width_in)  # ~0.062 in por carácter
    ax = fig.add_axes((left, 0.75 / fh, 0.975 - left, lane_in * lanes / fh))
    ax.set_facecolor(SURFACE)
    margin = max(sp.length for sp in cfg.specs) + 5
    ax.set_xlim(-margin * 0.3, cfg.length + margin)
    ax.set_ylim(-0.5, lanes - 0.5)
    ax.add_patch(Rectangle((0, -0.5), cfg.length, lanes, facecolor=ROAD, edgecolor="none", zorder=0))
    for k in range(1, lanes):
        ax.axhline(k - 0.5, color=BASELINE, linewidth=0.8, linestyle=(0, (6, 4)), zorder=1)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=8, length=0)
    ax.set_yticks(range(lanes), labels)
    ax.set_xlabel("posición (m)", color=INK_2, fontsize=9)
    for lane in range(lanes):
        for k, pos in _stops(cfg, lane):
            ax.plot([pos, pos], [lane - 0.45, lane + 0.45], color=TYPE_COLORS[k], linewidth=2, zorder=1,
                    solid_capstyle="butt")  # fmt: skip
            ax.annotate("parada", (pos, lane + 0.45), xytext=(0, 1), textcoords="offset points", ha="center",
                        va="bottom", fontsize=7, color=INK_2, zorder=1)  # fmt: skip
    light = Rectangle((cfg.length, -0.5), max(1.5, cfg.length * 0.006), lanes, zorder=3, animated=True)
    ax.add_patch(light)
    cars = PolyCollection([], linewidths=0.9, zorder=2, animated=True)
    ax.add_collection(cars)
    # Reloj: tiempo simulado y tiempo real (de proceso) = simulado / time_scale, como en --run.
    clock = fig.text(0.975, 1 - 0.3 / fh, "", ha="right", fontsize=11, color=INK, family="monospace",
                     animated=True)  # fmt: skip
    fig.text(0.02, 1 - 0.3 / fh, "Movimiento de los vehículos", ha="left", fontsize=13, color=INK,
             fontweight="bold")  # fmt: skip
    fig.text(0.02, 1 - 0.55 / fh,
             f"{title_note} · video ×{anim.speed:g} · borde oscuro = detenido",
             ha="left", fontsize=9, color=INK_2)  # fmt: skip
    _type_legend(fig, cfg, 1 - 0.68 / fh,
                 [Patch(facecolor=RED_PHASE, label="rojo"), Patch(facecolor=GREEN_LIGHT, label="verde")])  # fmt: skip

    colors = np.array([matplotlib.colors.to_rgba(c) for c in TYPE_COLORS])
    edge_stop = matplotlib.colors.to_rgba(INK)
    stride = max(1, round(anim.speed / anim.fps / DT))
    frames = range(0, traj.n_frames, stride)

    def update(f: int) -> None:
        verts, vt = _vehicle_verts(traj, f, 0.42)
        cars.set_verts(list(verts))
        cars.set_facecolor(colors[vt])
        edges = np.zeros((vt.size, 4))
        edges[traj.stopped[traj.frame(f)]] = edge_stop
        cars.set_edgecolor(edges)
        light.set_facecolor(GREEN_LIGHT if traj.green[f] else RED_PHASE)
        clock.set_text(_clock_text(float(traj.t[f])))

    def progress(i: int) -> None:
        if (i + 1) % 20 == 0 or i + 1 == len(frames):
            sys.stderr.write(f"\r  video: cuadro {i + 1}/{len(frames)}")
            sys.stderr.flush()

    dpi = 110
    fig.set_dpi(dpi)
    ffmpeg = shutil.which(matplotlib.rcParams["animation.ffmpeg_path"]) or shutil.which("ffmpeg")
    fmt = anim.format
    if ffmpeg is None and fmt != "gif":
        sys.stderr.write(f"  Aviso: no se encontró ffmpeg; el video se escribe como GIF en vez de {fmt}\n")
        fmt = "gif"
    path = _free_path(path.with_name(f"{path.name}.{fmt}"))
    if fmt == "gif":
        from matplotlib import animation

        for artist in (light, cars, clock):
            artist.set_animated(False)
        writer = animation.PillowWriter(fps=anim.fps)
        with writer.saving(fig, str(path), dpi=dpi):
            for i, f in enumerate(frames):
                update(f)
                writer.grab_frame(facecolor=SURFACE)
                progress(i)
        sys.stderr.write("\n")
        return path

    # Fondo fijo dibujado una vez; en cada cuadro se restaura y se dibujan solo los animados.
    canvas = fig.canvas
    canvas.draw()
    background = canvas.copy_from_bbox(fig.bbox)
    w, h = canvas.get_width_height()
    if fmt == "webm":  # VP9 libre de patentes; 'realtime' para codificar tan rápido como se dibuja
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
            for artist in (cars, light, clock):
                fig.draw_artist(artist)
            proc.stdin.write(canvas.buffer_rgba())
            progress(i)
    finally:
        proc.stdin.close()
        proc.wait()
    sys.stderr.write("\n")
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg terminó con código {proc.returncode}")
    return path


def _h264_encoder(ffmpeg: str) -> str:
    """Codificador H.264 disponible (libx264 o, p. ej. en Fedora, libopenh264); si no hay, MPEG-4."""
    import subprocess

    listing = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    return next((enc for enc in ("libx264", "libopenh264") if f" {enc} " in listing), "mpeg4")


def _hms(seconds: float) -> str:
    """Segundos simulados como hh:mm:ss (sin décimos)."""
    hours, rest = divmod(int(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _clock_text(t_sim: float) -> str:
    """Reloj del video: tiempo simulado en horas, minutos y segundos."""
    return f"simulado {_hms(t_sim)}"


def _free_path(path: Path) -> Path:
    """No sobrescribe: si el archivo existe, agrega un sufijo numérico."""
    k = 2
    candidate = path
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}-{k}{path.suffix}")
        k += 1
    return candidate


def visualize(cfg: SimConfig, seed: int, anim: AnimationOptions, out_dir: Path, run_name: str) -> list[Path]:
    """Graba la réplica elegida en la ventana de [animation] y escribe el diagrama y el video."""
    start, end = anim.window(cfg)
    print(f"Movimiento: réplica {anim.replica}, t = {start:g}–{end:g} s simulados, video ×{anim.speed:g} "
          f"a {anim.fps} cuadros/s (~{(end - start) / anim.speed:.0f} s de video)")  # fmt: skip
    traj = record(cfg, seed, anim.replica, start, end)
    tag = f"{start:g}-{end:g}s"
    note = f"corrida {run_name} · semilla {seed} · réplica {anim.replica}"
    diagram = _free_path(out_dir / f"espacio_tiempo_{tag}.png")
    plot_space_time(traj, diagram, f"{note} · t = {start:g}–{end:g} s")  # su eje x está en segundos
    video = render_video(traj, out_dir / f"movimiento_{tag}", anim, f"{note} · t = {_hms(start)}–{_hms(end)}")
    return [diagram, video]
