from pathlib import Path

import numpy as np
import pytest

from traffic.cli import CONFIG_NAME, run, view
from traffic.config import DT, GREEN, RED, SimConfig
from traffic.engine import Simulation
from traffic.movement import record
from traffic.settings import AnimationOptions, ConfigError, parse_settings


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr("traffic.settings.project_root", lambda: tmp_path)
    (tmp_path / CONFIG_NAME).write_text(
        "[road]\nlength = 150\n[traffic_light]\nred = 10\ngreen = 10\n"
        "[execution]\nrun = 3\nreplicas = 2\nworkers = 1\n[output]\nprogress = false\n"
        "[animation]\nstart = 5\nduration = 4\nspeed = 2\nfps = 10\n",
        encoding="utf-8",
    )
    return tmp_path


def test_recorded_replica_is_identical_to_the_run():
    """The replica shown is the same one run_parallel ran (the same derived seed)."""
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
        assert np.unique(traj.vtype[rows]).size == 1  # one identifier, one vehicle
        assert np.all(np.diff(frame[rows]) == 1)  # appears in consecutive instants
        assert np.all(np.diff(traj.x[rows]) >= -1e-4)  # never moves back
    assert (traj.phase == GREEN).any() and not (traj.phase == GREEN).all()


def test_animation_settings():
    s = parse_settings("[traffic_light]\nred = 25\ngreen = 35\n[execution]\nrun = 100\n", Path("x.toml"))
    assert s.animation == AnimationOptions() and not s.run.animation
    assert s.animation.window(s.sim) == (0.0, 120.0)  # 2 cycles by default
    s = parse_settings("[execution]\nrun = 10\n[animation]\nstart = 90\nduration = 50\n", Path("x.toml"))
    assert s.animation.window(s.sim) == (90.0, 100.0)  # not past the end of the run
    for text, message in [
        ("[animation]\nstart = 100\n", r"\[animation\] start must be between 0 and 100"),
        ("[animation]\nduration = 0\n", r"\[animation\] duration must be greater than 0"),
        ("[animation]\nfps = 0\n", r"\[animation\] fps must be between 1 and 60"),
        ("[animation]\nreplica = 9\n", r"\[animation\] replica must be between 1 and 8"),
        ("[animation]\nzoom = 2\n", r"unknown keys: \[animation\] zoom"),
    ]:
        with pytest.raises(ConfigError, match=message):
            parse_settings("[execution]\nrun = 10\n" + text, Path("x.toml"))


def test_viewer_command_writes_into_run_folder(root, capsys):
    run_dir = run(["vista"])
    before = {p.name for p in run_dir.iterdir()}
    paths = view([])  # no argument: the most recent run
    assert [p.parent for p in paths] == [run_dir, run_dir]
    assert {p.name for p in paths} == {"space_time_5-9s.png", "movement_5-9s.webm"} or paths[1].suffix == ".gif"
    assert all(p.stat().st_size > 1000 for p in paths)
    # A second view with another window overwrites nothing; with the same one, it adds a suffix.
    again = view([str(run_dir), "--start", "5"])
    assert {p.name for p in again}.isdisjoint({p.name for p in paths})
    assert before <= {p.name for p in run_dir.iterdir()}
    with pytest.raises(SystemExit):
        view([str(root / "no_existe")])
    assert "is not a run folder" in capsys.readouterr().err


def test_animation_option_in_run(root):
    (root / CONFIG_NAME).write_text(
        (root / CONFIG_NAME).read_text().replace("progress = false\n", "progress = false\nanimation = true\n"),
        encoding="utf-8",
    )
    run_dir = run([])
    names = {p.name for p in run_dir.iterdir()}
    assert "space_time_5-9s.png" in names
    assert "movement_5-9s.webm" in names or "movement_5-9s.gif" in names


def test_gif_without_ffmpeg(root, monkeypatch):
    run_dir = run([])
    monkeypatch.setattr("shutil.which", lambda *a, **k: None)
    paths = view([str(run_dir), "--duration", "1"])
    assert paths[1].suffix == ".gif" and paths[1].stat().st_size > 1000


@pytest.mark.parametrize("fmt", ["webm", "mp4", "gif"])
def test_video_formats(root, fmt):
    import shutil
    import subprocess

    run_dir = run([])
    paths = view([str(run_dir), "--duration", "1", "--format", fmt])
    video = paths[1]
    if shutil.which("ffmpeg") is None:
        assert video.suffix == ".gif"
        return
    assert video.suffix == f".{fmt}" and video.stat().st_size > 1000
    if fmt != "gif":  # the whole file decodes
        probe = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-f", "null", "-"], capture_output=True)
        assert probe.returncode == 0, probe.stderr


def test_video_format_setting():
    s = parse_settings("[execution]\nrun = 10\n", Path("x.toml"))
    assert s.animation.format == "webm"
    with pytest.raises(ConfigError, match=r"\[animation\] format must be 'webm', 'mp4', 'gif'"):
        parse_settings('[execution]\nrun = 10\n[animation]\nformat = "avi"\n', Path("x.toml"))


def test_video_clock_shows_simulated_time_as_hms():
    from traffic.movement import _clock_text

    assert _clock_text(360.0) == "simulated 00:06:00"
    assert _clock_text(3725.9) == "simulated 01:02:05"  # tenths are not shown
    assert _clock_text(0.0) == "simulated 00:00:00"


def test_crossing_counters_start_with_the_window():
    """The video counters count what crossed since the start of the window and never go down."""
    cfg = SimConfig(length=150, lanes=2, rates=(30, 8, 2), red=10, green=10, run=3)
    full = record(cfg, seed=3, replica=1, start=0, end=25)
    win = record(cfg, seed=3, replica=1, start=5, end=25)
    assert np.all(np.diff(win.crossed_pax, axis=0) >= 0) and np.all(win.crossed_pax[0] >= 0)
    before = np.flatnonzero(np.isclose(full.t, 4.9))[0]  # last instant before the window
    np.testing.assert_allclose(win.crossed_pax[-1], full.crossed_pax[-1] - full.crossed_pax[before])
    np.testing.assert_allclose(win.crossed_veh[-1], full.crossed_veh[-1] - full.crossed_veh[before])
    assert full.crossed_veh[-1].sum() > 0


def test_bottleneck_stops_are_recorded_while_they_last():
    """The [bottleneck] halt mark (black in the video) lasts exactly as long as the halt."""
    from traffic.config import Bottleneck

    from dataclasses import replace

    from traffic.config import DEFAULT_SPECS

    car, bike, bus = DEFAULT_SPECS
    cfg = SimConfig(length=200, lanes=2, rates=(30, 0, 0), red=0, green=30, run=6,
                    specs=(replace(car, bottleneck_prob=0.5, bottleneck_time_mean=4.0), bike, bus),
                    bottleneck=Bottleneck(stop_zone=(50.0, 150.0)))  # fmt: skip
    traj = record(cfg, seed=2, replica=1, start=0, end=60)
    marked = traj.bottleneck
    assert marked.any()
    assert np.all(traj.stopped[marked])  # only while it is stopped
    frame = np.repeat(np.arange(traj.n_frames), np.diff(traj.offsets))
    complete = 0
    for vid in np.unique(traj.vid[marked]):
        rows = np.flatnonzero((traj.vid == vid) & marked)
        if frame[rows[-1]] == traj.n_frames - 1:
            continue  # the halt continues when the recording ends
        complete += 1
        assert rows.size == round(4.0 / 0.1)  # the 4 s of the halt, in consecutive instants
        assert np.all(np.diff(frame[rows]) == 1)
        after = np.flatnonzero(traj.vid == vid)
        assert not traj.bottleneck[after[after > rows[-1]]].any()  # afterwards, its original colour
    assert complete > 0


def test_entry_queue_is_recorded_per_lane():
    """The entry queue of each lane at each instant is the simulation's (high demand: it grows)."""
    cfg = SimConfig(length=150, lanes=2, rates=(120, 20, 2), red=20, green=10, run=3)
    traj = record(cfg, seed=4, replica=1, start=0, end=30)
    sim = Simulation(cfg, np.random.default_rng(np.random.SeedSequence(4).spawn(1)[0]))
    sim.run()
    assert traj.queued.shape == (traj.n_frames, cfg.lanes)
    np.testing.assert_array_equal(traj.queued[-1], [len(q) for q in sim.queues])
    assert traj.queued[-1].sum() > 0


def test_exit_queue_is_recorded_per_lane():
    """The exit queue of each lane at each instant is the simulation's; without [exit], always empty."""
    cfg = SimConfig(length=150, lanes=2, rates=(40, 0, 0), red=2, green=60, run=6, exit_capacity=(0, 6),
                    exit_storage=16.5)  # fmt: skip
    traj = record(cfg, seed=4, replica=1, start=0, end=60)
    sim = Simulation(cfg, np.random.default_rng(np.random.SeedSequence(4).spawn(1)[0]))
    sim.run()
    assert traj.exit_queued.shape == (traj.n_frames, cfg.lanes)
    np.testing.assert_allclose(traj.exit_queued[-1], sim.exit_q, rtol=1e-6)  # m occupied
    np.testing.assert_array_equal(traj.exit_closed[-1], sim.exit_closed)
    assert not traj.exit_queued[:, 0].any() and traj.exit_queued[:, 1].max() >= 16.5 - 1e-4  # 3 cars of 5.5 m
    assert not traj.exit_closed[:, 0].any() and traj.exit_closed[:, 1].any()


def test_outputs_with_several_lights(root):
    """With several traffic lights: the header lists them, the run writes the plot, summary and CSV, and the view
    (space-time diagram and video) draws each one with its phases."""
    text = (root / CONFIG_NAME).read_text().replace(
        "[traffic_light]\nred = 10\ngreen = 10\n",
        "[[traffic_light]]\nposition = 60\nred = 8\ngreen = 12\nstart_phase = \"green\"\n"
        "[[traffic_light]]\nred = 10\ngreen = 10\n",
    )
    (root / CONFIG_NAME).write_text(text, encoding="utf-8")
    run_dir = run([])
    summary = (run_dir / "summary.txt").read_text(encoding="utf-8")
    assert "traffic lights at 60 m: red 8 s / green 12 s, starts green · at 150 m: red 10 s / green 10 s" in summary
    assert (run_dir / "passenger_mobility.png").stat().st_size > 10_000
    paths = view([str(run_dir), "--duration", "2"])
    assert all(p.stat().st_size > 1000 for p in paths)
    cfg = parse_settings((root / CONFIG_NAME).read_text(encoding="utf-8"), Path("x.toml")).sim
    traj = record(cfg, seed=1, replica=1, start=0, end=25)
    assert traj.inner_phase.shape == (traj.n_frames, 1)
    assert traj.inner_phase[0, 0] == GREEN and (traj.inner_phase[:, 0] != GREEN).any()  # starts on green


def test_outputs_with_pedestrians(root):
    """With a pedestrian bump and a pedestrian light: the header describes them, the summary has the «Pedestrians» block,
    and the view (space-time diagram and video) draws the recorded phases of each one."""
    text = (root / CONFIG_NAME).read_text().replace(
        "[traffic_light]\nred = 10\ngreen = 10\n",
        "[[traffic_light]]\nposition = 100\nred = 10\ngreen = 10\nyellow = 2\n"
        "[[traffic_light]]\nposition = 60\npedestrian = true\nred = 6\ngreen = 4\nyellow = 2\n"
        "pedestrian_crossing = {min = 6, max = 6, mean = 6, std = 0}\n"
        "[speed_bump]\nposition = 30\npedestrian = true\npedestrian_time = 5\n"
        "pedestrian_crossing = {min = 6, max = 6, mean = 6, std = 0}\n"
        "[vehicles.car]\naccel = 2.5\ndecel = 4.5\n",
    )
    (root / CONFIG_NAME).write_text(text, encoding="utf-8")
    run_dir = run([])
    summary = (run_dir / "summary.txt").read_text(encoding="utf-8")
    assert "pedestrian: red 6 s with pedestrians waiting, minimum green 4 s, yellow 2 s" in summary
    assert "Pedestrian crossing at the bump" in summary and "Pedestrian light at 60 m" in summary
    assert "with the light at" not in summary.split("Pedestrian light at 60 m")[0]  # there is no previous light at 60 m
    block = summary.split("Pedestrians (mean per replica):")[1].splitlines()
    assert "pedestrian crossing at the bump at 30 m" in block[1] and "pedestrian light at 60 m" in block[2]
    paths = view([str(run_dir), "--duration", "2"])
    assert all(p.stat().st_size > 1000 for p in paths)
    cfg = parse_settings((root / CONFIG_NAME).read_text(encoding="utf-8"), Path("x.toml")).sim
    traj = record(cfg, seed=1, replica=1, start=0, end=30)
    assert traj.inner_phase.shape == (traj.n_frames, 3)  # bump at 30 m, pedestrian light at 60 m, traffic light at 100 m
    assert (traj.inner_phase[:, 0] == RED).any() and (traj.inner_phase[:, 1] == RED).any()
    assert (traj.inner_phase[:, 2] == RED).any()
    # The recorded phases are the simulation's (not the config's): they match the replica that ran.
    sim = Simulation(cfg, np.random.default_rng(np.random.SeedSequence(1).spawn(1)[0]))
    assert [int(sim.line_phases(round(t / DT))[0]) for t in traj.t[:50]] == traj.inner_phase[:50, 0].tolist()
