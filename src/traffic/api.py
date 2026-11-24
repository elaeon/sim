"""Python API of `traffic`: the same runs as the commands, with named arguments and structured results.

    from traffic import api

    run = api.simulate()                                   # like `uv run traffic`
    run.results["metricas"]["tipos"]["car"]["tiempo_recorrido_s"]
    run = api.compare_emissions(bumps=["none", 35, (35, 70)], replicas=4)
    run = api.bump_spacing(distances=[20, 40, 60], chain=3)
    run = api.light_spacing(distances=[50, 100, 200], cycle=(30, 30, 3))
    run = api.light_spacing(distances=[25, 50], chain=3, fixed=[150], offset=["green", "red"])
    api.list_runs(mode="separacion", limit=5)

Contract (`API_VERSION`):

  * Each run function (`simulate`, `compare_emissions`, `bump_spacing`, `light_spacing`, `compare_variants`,
    `calibrate`) does the same as its command, writes the same results folder and returns a `Run` with the
    contents of its `results.json` (format in `traffic.results`). It prints nothing and does not read `sys.argv`;
    what the command would have printed goes to `Run.log`.
  * Configuration or argument errors are `ConfigError` (a subclass of `ValueError`); a folder that does not
    exist or is not a results folder, `FileNotFoundError`.
  * `config` is None (the `config.toml` at the project root), a folder with its `config.toml` (e.g. a results
    folder, to repeat a run) or the path of a `config.toml`. `name` (the suffix of the results folder) is only
    accepted without `config`, as on the command line.
  * The names and meaning of the arguments and of the `results.json` keys do not change within the same
    `API_VERSION`; optional arguments and new keys may be added. A release that changes or removes something bumps
    the version.
  * The run functions capture the process's stdout and stderr while they run: do not call them from several
    threads at once (several processes are fine).
"""

from __future__ import annotations

import contextlib
import io
import math
import re
from dataclasses import dataclass
from pathlib import Path

from traffic.results import MODES, RESULTS_NAME, SCHEMA, list_runs, mode_of, read_results
from traffic.settings import ConfigError, default_config_path, load_settings, resolve_target

API_VERSION = 1

__all__ = [
    "API_VERSION", "MODES", "RESULTS_NAME", "SCHEMA", "ConfigError", "Run", "bump_spacing", "calibrate", "compare_emissions",
    "compare_variants", "default_config", "describe_config", "light_spacing", "list_runs", "read_results", "redraw",
    "simulate",
]  # fmt: skip


@dataclass(frozen=True, slots=True)
class Run:
    """A finished run."""

    folder: Path
    mode: str  # "corrida" | "variantes" | "emisiones" | "separacion"
    results: dict  # the folder's results.json
    log: str  # what the command would have printed

    @property
    def files(self) -> list[str]:
        return list(self.results["archivos"])

    def path(self, name: str) -> Path:
        """Path of a file in the folder (e.g. `run.path("summary.txt")`)."""
        return self.folder / name


def _call(command, argv: list[str]) -> tuple[Path, str]:
    """Run a `traffic.cli` command without printing: (folder, printed text). `parser.error` becomes ConfigError."""
    out, err = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            folder = command(argv)
    except SystemExit as exc:
        message = err.getvalue().strip().splitlines()
        text = re.sub(r"^.*?error: ", "", message[-1]) if message else f"the command exited with code {exc.code}"
        raise ConfigError(text) from None
    return Path(folder), out.getvalue()


def _target(config: str | Path | None, name: str | None) -> list[str]:
    if config is not None and name is not None:
        raise ConfigError("name is only accepted without config (with config, the results folder takes the name of the "
                          "configuration folder)")  # fmt: skip
    value = config if config is not None else name
    return [] if value is None else [str(value)]


def _opt(flag: str, value) -> list[str]:
    """`--flag value` (or `--flag v1 v2 …` with a list); nothing if the value is None or an empty list."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [flag, *(str(v) for v in value)] if value else []
    return [flag, str(value)]


def _finish(command, argv: list[str], mode: str) -> Run:
    folder, log = _call(command, argv)
    return Run(folder, mode, read_results(folder), log)


def _bump(b) -> str:
    """A bump scenario of `compare_emissions` as `--bumps` expects it."""
    if b is None:
        return "none"
    if isinstance(b, str):
        return "none" if b.lower() == "sin" else b
    if isinstance(b, (list, tuple)):
        return ",".join(f"{x:g}" for x in b) or "none"
    return f"{b:g}"


def _light(value) -> str:
    if value is True or value is False or value is None:
        return {True: "yes", False: "no", None: "config"}[value]
    return {"si": "yes", "ambos": "both"}.get(str(value), str(value))  # also the Spanish values of earlier versions


def simulate(config: str | Path | None = None, *, name: str | None = None) -> Run:
    """Full `traffic` run: Monte Carlo replicas with the configuration, plots, tables and summary."""
    from traffic.cli import run

    return _finish(run, _target(config, name), "corrida")


def compare_emissions(
    config: str | Path | None = None, *, name: str | None = None, bumps: list | None = None, lights="config",
    lanes: list[int] | None = None, without: list[str] = (), length: float | None = None,
    bump_lanes: list[int] | None = None, replicas: int | None = None, run: float | None = None,
) -> Run:  # fmt: skip
    """`traffic-emissions`: emissions, fuel and travel time of each scenario with the same seed.

    `bumps`: the bumps of each scenario; each item is None or "none" (no bump), a number (position in m) or a list
    of positions (several bumps); None = no bump and the configuration's own. `lights`: "config", "yes", "no",
    "both" (or True/False). `lanes`: lanes that are kept; `without`: keys of types that do not take part;
    `length`: segment length; `bump_lanes`: lanes with a bump; `replicas` and `run` (process s) replace those of the
    configuration."""
    from traffic.cli import emissions

    argv = _target(config, name) + _opt("--bumps", None if bumps is None else [_bump(b) for b in bumps])
    argv += ["--light", _light(lights)] + _opt("--lanes", lanes) + _opt("--without", list(without))
    argv += _opt("--length", length) + _opt("--bump-lanes", bump_lanes) + _opt("--replicas", replicas)
    argv += _opt("--run", run)
    return _finish(emissions, argv, "emisiones")


def bump_spacing(
    config: str | Path | None = None, *, name: str | None = None, distances: list[float] | None = None,
    chain: int | None = None, first: float | None = None, tolerance: float | None = None,
    threshold: float | None = None, lights="config", lanes: list[int] | None = None, without: list[str] = (),
    length: float | None = None, bump_lanes: list[int] | None = None, replicas: int | None = None,
    run: float | None = None,
) -> Run:  # fmt: skip
    """`traffic-emissions --spacing`: minimum spacing between bumps for the emission footprint to dilute.

    `distances`: spacings in m (None = 10 15 20 25 30 40 50 60 80 100); `chain`: bumps per chain (2 = a pair);
    `first`: position of the first bump; `tolerance` and `threshold`: see the documentation. The other arguments, as in
    `compare_emissions` (here `lights` does not accept "both")."""
    from traffic.cli import emissions

    argv = _target(config, name) + ["--spacing"] + [str(d) for d in distances or ()]
    argv += _opt("--chain", chain) + _opt("--first-bump", first) + _opt("--tolerance", tolerance)
    argv += _opt("--threshold", threshold) + ["--light", _light(lights)] + _opt("--lanes", lanes)
    argv += _opt("--without", list(without)) + _opt("--length", length) + _opt("--bump-lanes", bump_lanes)
    argv += _opt("--replicas", replicas) + _opt("--run", run)
    return _finish(emissions, argv, "separacion")


def light_spacing(
    config: str | Path | None = None, *, name: str | None = None, distances: list[float] | None = None,
    chain: int | None = None, first: float | None = None, cycle: tuple | str | None = None,
    offset: str | list[str] | None = None, fixed: list[float] | None = None, bumps: list[float] | None = None,
    tolerance: float | None = None, threshold: float | None = None,
    lanes: list[int] | None = None, without: list[str] = (), length: float | None = None,
    bump_lanes: list[int] | None = None, replicas: int | None = None, run: float | None = None,
) -> Run:  # fmt: skip
    """`traffic-emissions --light-spacing`: minimum spacing between fixed-cycle traffic lights for the emission
    footprint of the first one to stop changing as the second moves away.

    `distances`: spacings in m (None = 25 50 75 100 150 200 250 300); `first`: position of the first light (None =
    100 m); `cycle`: (red, green[, yellow]) in s or a text "30/30/3" (None = the configuration's first fixed-cycle
    light, or 30/30/3); `offset`: "red" (the next one is red when the platoon arrives; the default), "green" (green
    wave) or "same", or a list with one per gap of the chain; `fixed`: fixed spacings (m) of the gaps after the
    first one, with `chain` ≥ 3; `bumps`: bumps fixed in every scenario (m; None = no bumps). The rest, as in
    `bump_spacing`."""
    from traffic.cli import emissions

    text = cycle if cycle is None or isinstance(cycle, str) else "/".join(f"{v:g}" for v in cycle)
    argv = _target(config, name) + ["--light-spacing"] + [str(d) for d in distances or ()]
    argv += _opt("--chain", chain) + _opt("--first-light", first) + _opt("--cycle", text)
    offset = ",".join(offset) if isinstance(offset, (list, tuple)) else offset
    if offset is not None:  # also the Spanish values of earlier versions (rojo, verde, igual)
        spanish = {"rojo": "red", "verde": "green", "igual": "same"}
        offset = ",".join(spanish.get(part.strip(), part.strip()) for part in offset.split(","))
    argv += _opt("--offset", offset) + _opt("--fixed-spacings", fixed)
    argv += ["--bumps", ",".join(f"{b:g}" for b in bumps)] if bumps else []
    argv += _opt("--tolerance", tolerance) + _opt("--threshold", threshold) + _opt("--lanes", lanes)
    argv += _opt("--without", list(without)) + _opt("--length", length) + _opt("--bump-lanes", bump_lanes)
    argv += _opt("--replicas", replicas) + _opt("--run", run)
    return _finish(emissions, argv, "separacion")


def compare_variants(
    config: str | Path | None = None, *, name: str | None = None, lengths: list[float] | None = None,
    lights: list | None = None, lanes: list[int] | None = None, without: list[str] = (),
    queue: list[str] | None = None, replicas: int | None = None, run: float | None = None,
) -> Run:  # fmt: skip
    """`traffic-variants`: speed per lane and entry queue for each segment length and traffic light split.

    `lights`: splits as (red, green) pairs in s or texts "40/20" (None = three splits of the configuration's
    cycle); `queue`: keys of the types counted in the entry queue."""
    from traffic.cli import variants

    split = None if lights is None else [x if isinstance(x, str) else f"{x[0]:g}/{x[1]:g}" for x in lights]
    argv = _target(config, name) + _opt("--lengths", lengths) + _opt("--lights", split) + _opt("--lanes", lanes)
    argv += _opt("--without", list(without)) + _opt("--queue", queue) + _opt("--replicas", replicas) + _opt("--run", run)
    return _finish(variants, argv, "variantes")


def calibrate(
    config: str | Path | None = None, *, name: str | None = None, vehicle: str = "car", queues: int | None = None,
) -> Run:  # fmt: skip
    """`traffic-calibrate`: measures the type `vehicle` (its key) in standard situations (discharge of a queue on green,
    continuous flow, stopped queue) and compares with published references for cars; `queues`: queues that are
    discharged (30 by default). `results["metricas"]["medidas"]` holds each measurement with its value, range and status."""
    from traffic.cli import calibrate as command

    argv = _target(config, name) + ["--vehicle", vehicle] + _opt("--queues", queues)
    return _finish(command, argv, "calibracion")


def redraw(folder: str | Path) -> Path:
    """Redraw the plot of a comparison or sweep that has already run, without simulating or overwriting; returns
    the path of the new file."""
    from traffic.cli import emissions, variants

    folder = Path(folder).expanduser()
    mode = mode_of(folder)
    if mode in ("emisiones", "separacion"):
        command = emissions
    elif mode == "variantes":
        command = variants
    else:
        raise ConfigError(f"{folder} is neither a comparison nor a sweep: only those can be redrawn")
    _, log = _call(command, ["--redraw", str(folder)])
    match = re.search(r"Plot in (.+)", log)
    if match is None:
        raise ConfigError(f"could not redraw {folder}")
    return Path(match.group(1).strip())


def describe_config(config: str | Path | None = None) -> dict:
    """The configuration, read and validated, as a dictionary: segment, run, vehicle types, traffic lights and bumps.
    Useful to check a configuration before running it (errors come out as `ConfigError`)."""
    target = resolve_target(None if config is None else str(config))
    settings = load_settings(target.config)
    cfg, opts = settings.sim, settings.run
    limits = cfg.lane_max_kmh
    return {
        "config": str(target.config.resolve()),
        "tramo": {"largo_m": cfg.length, "carriles": cfg.lanes, "limite_kmh": [v if math.isfinite(v) else None for v in limits]},
        "ejecucion": {"replicas": opts.replicas, "procesos": opts.workers, "semilla": opts.seed,
                      "run_s": cfg.run, "time_scale": cfg.time_scale, "s_simulados": cfg.sim_seconds,
                      "carpeta_resultados": opts.output_dir},  # fmt: skip
        "tipos": [{"clave": sp.key, "nombre": sp.name, "tasa_veh_min": cfg.rates[k], "emite": sp.emits}
                  for k, sp in enumerate(cfg.specs)],  # fmt: skip
        "semaforo": {"activo": cfg.any_light, "descripcion": cfg.lights_label},
        "topes": [{"posicion_m": b.position, "carriles": list(cfg.bump_lanes(b)), "peatones": b.pedestrian}
                  for b in cfg.bumps],  # fmt: skip
        "avisos": list(settings.notices),
    }


def default_config() -> Path:
    """Path of the `config.toml` the functions use when `config` is None."""
    return default_config_path()
