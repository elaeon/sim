from pathlib import Path

import numpy as np
import pytest

from trafico.cli import CONFIG_NAME, run, view
from trafico.config import GREEN, SimConfig
from trafico.engine import Simulation
from trafico.movement import record
from trafico.settings import AnimationOptions, ConfigError, parse_settings


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr("trafico.settings.project_root", lambda: tmp_path)
    (tmp_path / CONFIG_NAME).write_text(
        "[road]\nlength = 150\n[traffic_light]\nred = 10\ngreen = 10\n"
        "[execution]\nrun = 3\nreplicas = 2\nworkers = 1\n[output]\nprogress = false\n"
        "[animation]\nstart = 5\nduration = 4\nspeed = 2\nfps = 10\n",
        encoding="utf-8",
    )
    return tmp_path


def test_recorded_replica_is_identical_to_the_run():
    """La réplica que se visualiza es la misma que corrió run_parallel (misma semilla derivada)."""
    cfg = SimConfig(length=150, lanes=2, red=10, green=10, run=3)
    traj = record(cfg, seed=5, replica=2, start=0, end=cfg.sim_seconds)
    sim = Simulation(cfg, np.random.default_rng(np.random.SeedSequence(5).spawn(4)[1]))
    sim.run()
    last = traj.frame(traj.n_frames - 1)
    np.testing.assert_array_equal(traj.vid[last], sim.vid[: sim.n])
    np.testing.assert_allclose(traj.x[last], sim.x[: sim.n].astype(np.float32))
    assert traj.n_frames == cfg.n_ticks


def test_trajectories_follow_each_vehicle():
    cfg = SimConfig(length=150, lanes=2, rates=(30, 8, 2), red=10, green=10, run=3)
    traj = record(cfg, seed=1, replica=1, start=5, end=25)
    assert traj.t[0] == pytest.approx(5.0) and traj.t[-1] == pytest.approx(25.0)
    frame = np.repeat(np.arange(traj.n_frames), np.diff(traj.offsets))
    for vid in np.unique(traj.vid):
        rows = np.flatnonzero(traj.vid == vid)
        assert np.unique(traj.vtype[rows]).size == 1  # un identificador, un vehículo
        assert np.all(np.diff(frame[rows]) == 1)  # aparece en instantes consecutivos
        assert np.all(np.diff(traj.x[rows]) >= -1e-4)  # nunca retrocede
    assert (traj.phase == GREEN).any() and not (traj.phase == GREEN).all()


def test_animation_settings():
    s = parse_settings("[traffic_light]\nred = 25\ngreen = 35\n[execution]\nrun = 100\n", Path("x.toml"))
    assert s.animation == AnimationOptions() and not s.run.animation
    assert s.animation.window(s.sim) == (0.0, 120.0)  # 2 ciclos por defecto
    s = parse_settings("[execution]\nrun = 10\n[animation]\nstart = 90\nduration = 50\n", Path("x.toml"))
    assert s.animation.window(s.sim) == (90.0, 100.0)  # no pasa del final de la corrida
    for text, message in [
        ("[animation]\nstart = 100\n", r"\[animation\] start debe estar entre 0 y 100"),
        ("[animation]\nduration = 0\n", r"\[animation\] duration debe ser mayor que 0"),
        ("[animation]\nfps = 0\n", r"\[animation\] fps debe estar entre 1 y 60"),
        ("[animation]\nreplica = 9\n", r"\[animation\] replica debe estar entre 1 y 8"),
        ("[animation]\nzoom = 2\n", r"claves desconocidas: \[animation\] zoom"),
    ]:
        with pytest.raises(ConfigError, match=message):
            parse_settings("[execution]\nrun = 10\n" + text, Path("x.toml"))


def test_viewer_command_writes_into_run_folder(root, capsys):
    run_dir = run(["vista"])
    before = {p.name for p in run_dir.iterdir()}
    paths = view([])  # sin argumento: la corrida más reciente
    assert [p.parent for p in paths] == [run_dir, run_dir]
    assert {p.name for p in paths} == {"espacio_tiempo_5-9s.png", "movimiento_5-9s.webm"} or paths[1].suffix == ".gif"
    assert all(p.stat().st_size > 1000 for p in paths)
    # Una segunda vista con otra ventana no sobrescribe nada; con la misma, agrega un sufijo.
    again = view([str(run_dir), "--inicio", "5"])
    assert {p.name for p in again}.isdisjoint({p.name for p in paths})
    assert before <= {p.name for p in run_dir.iterdir()}
    with pytest.raises(SystemExit):
        view([str(root / "no_existe")])
    assert "no es una carpeta de corrida" in capsys.readouterr().err


def test_animation_option_in_run(root):
    (root / CONFIG_NAME).write_text(
        (root / CONFIG_NAME).read_text().replace("progress = false\n", "progress = false\nanimation = true\n"),
        encoding="utf-8",
    )
    run_dir = run([])
    names = {p.name for p in run_dir.iterdir()}
    assert "espacio_tiempo_5-9s.png" in names
    assert "movimiento_5-9s.webm" in names or "movimiento_5-9s.gif" in names


def test_gif_without_ffmpeg(root, monkeypatch):
    run_dir = run([])
    monkeypatch.setattr("shutil.which", lambda *a, **k: None)
    paths = view([str(run_dir), "--duracion", "1"])
    assert paths[1].suffix == ".gif" and paths[1].stat().st_size > 1000


@pytest.mark.parametrize("fmt", ["webm", "mp4", "gif"])
def test_video_formats(root, fmt):
    import shutil
    import subprocess

    run_dir = run([])
    paths = view([str(run_dir), "--duracion", "1", "--formato", fmt])
    video = paths[1]
    if shutil.which("ffmpeg") is None:
        assert video.suffix == ".gif"
        return
    assert video.suffix == f".{fmt}" and video.stat().st_size > 1000
    if fmt != "gif":  # el archivo se decodifica completo
        probe = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-f", "null", "-"], capture_output=True)
        assert probe.returncode == 0, probe.stderr


def test_video_format_setting():
    s = parse_settings("[execution]\nrun = 10\n", Path("x.toml"))
    assert s.animation.format == "webm"
    with pytest.raises(ConfigError, match=r"\[animation\] format debe ser 'webm', 'mp4', 'gif'"):
        parse_settings('[execution]\nrun = 10\n[animation]\nformat = "avi"\n', Path("x.toml"))


def test_video_clock_shows_simulated_time_as_hms():
    from trafico.movement import _clock_text

    assert _clock_text(360.0) == "simulado 00:06:00"
    assert _clock_text(3725.9) == "simulado 01:02:05"  # los décimos no se muestran
    assert _clock_text(0.0) == "simulado 00:00:00"


def test_crossing_counters_start_with_the_window():
    """Los contadores del video cuentan lo cruzado desde el inicio de la ventana y nunca bajan."""
    cfg = SimConfig(length=150, lanes=2, rates=(30, 8, 2), red=10, green=10, run=3)
    full = record(cfg, seed=3, replica=1, start=0, end=25)
    win = record(cfg, seed=3, replica=1, start=5, end=25)
    assert np.all(np.diff(win.crossed_pax, axis=0) >= 0) and np.all(win.crossed_pax[0] >= 0)
    before = np.flatnonzero(np.isclose(full.t, 4.9))[0]  # último instante antes de la ventana
    np.testing.assert_allclose(win.crossed_pax[-1], full.crossed_pax[-1] - full.crossed_pax[before])
    np.testing.assert_allclose(win.crossed_veh[-1], full.crossed_veh[-1] - full.crossed_veh[before])
    assert full.crossed_veh[-1].sum() > 0


def test_bottleneck_stops_are_recorded_while_they_last():
    """La marca de detención por [bottleneck] (negro en el video) dura exactamente la detención."""
    from trafico.config import Bottleneck

    from dataclasses import replace

    from trafico.config import DEFAULT_SPECS

    car, bike, bus = DEFAULT_SPECS
    cfg = SimConfig(length=200, lanes=2, rates=(30, 0, 0), red=0, green=30, run=6,
                    specs=(replace(car, bottleneck_prob=0.5, bottleneck_time_mean=4.0), bike, bus),
                    bottleneck=Bottleneck(stop_zone=(50.0, 150.0)))  # fmt: skip
    traj = record(cfg, seed=2, replica=1, start=0, end=60)
    marked = traj.bottleneck
    assert marked.any()
    assert np.all(traj.stopped[marked])  # solo mientras está detenido
    frame = np.repeat(np.arange(traj.n_frames), np.diff(traj.offsets))
    complete = 0
    for vid in np.unique(traj.vid[marked]):
        rows = np.flatnonzero((traj.vid == vid) & marked)
        if frame[rows[-1]] == traj.n_frames - 1:
            continue  # la detención sigue al terminar la grabación
        complete += 1
        assert rows.size == round(4.0 / 0.1)  # los 4 s de la detención, en instantes consecutivos
        assert np.all(np.diff(frame[rows]) == 1)
        after = np.flatnonzero(traj.vid == vid)
        assert not traj.bottleneck[after[after > rows[-1]]].any()  # después, su color original
    assert complete > 0


def test_entry_queue_is_recorded_per_lane():
    """La cola de entrada de cada carril en cada instante es la de la simulación (demanda alta: crece)."""
    cfg = SimConfig(length=150, lanes=2, rates=(120, 20, 2), red=20, green=10, run=3)
    traj = record(cfg, seed=4, replica=1, start=0, end=30)
    sim = Simulation(cfg, np.random.default_rng(np.random.SeedSequence(4).spawn(1)[0]))
    sim.run()
    assert traj.queued.shape == (traj.n_frames, cfg.lanes)
    np.testing.assert_array_equal(traj.queued[-1], [len(q) for q in sim.queues])
    assert traj.queued[-1].sum() > 0
