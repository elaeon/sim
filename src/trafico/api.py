"""API de Python de `trafico`: las mismas corridas que los comandos, con argumentos con nombre y resultados estructurados.

    from trafico import api

    run = api.simulate()                                   # como `uv run trafico`
    run.results["metricas"]["tipos"]["car"]["tiempo_recorrido_s"]
    run = api.compare_emissions(bumps=["sin", 35, (35, 70)], replicas=4)
    run = api.bump_spacing(distances=[20, 40, 60], chain=3)
    api.list_runs(mode="separacion", limit=5)

Contrato (`API_VERSION`):

  * Cada función de corrida (`simulate`, `compare_emissions`, `bump_spacing`, `compare_variants`) hace lo mismo que su
    comando, escribe la misma carpeta de resultados y devuelve un `Run` con el contenido de su `resultados.json`
    (formato en `trafico.results`). No imprime nada ni lee `sys.argv`; lo que el comando habría impreso queda en
    `Run.log`.
  * Los errores de configuración o de argumentos son `ConfigError` (subclase de `ValueError`); una carpeta que no
    existe o no es de resultados, `FileNotFoundError`.
  * `config` es None (el `config.toml` de la raíz del proyecto), una carpeta con su `config.toml` (p. ej. una carpeta
    de resultados, para repetir una corrida) o la ruta de un `config.toml`. `name` (el sufijo de la carpeta de
    resultados) solo se admite sin `config`, como en la línea de comandos.
  * Los nombres y el significado de los argumentos y de las claves de `resultados.json` no cambian dentro de la misma
    `API_VERSION`; se pueden añadir argumentos opcionales y claves nuevas. Una corrida que cambia o quita algo sube la
    versión.
  * Las funciones de corrida capturan stdout y stderr del proceso mientras corren: no las llame desde varios hilos a
    la vez (varios procesos sí).
"""

from __future__ import annotations

import contextlib
import io
import math
import re
from dataclasses import dataclass
from pathlib import Path

from trafico.results import MODES, RESULTS_NAME, SCHEMA, list_runs, mode_of, read_results
from trafico.settings import ConfigError, default_config_path, load_settings, resolve_target

API_VERSION = 1

__all__ = [
    "API_VERSION", "MODES", "RESULTS_NAME", "SCHEMA", "ConfigError", "Run", "bump_spacing", "compare_emissions",
    "compare_variants", "default_config", "describe_config", "list_runs", "read_results", "redraw", "simulate",
]  # fmt: skip


@dataclass(frozen=True, slots=True)
class Run:
    """Una corrida terminada."""

    folder: Path
    mode: str  # "corrida" | "variantes" | "emisiones" | "separacion"
    results: dict  # el resultados.json de la carpeta
    log: str  # lo que el comando habría impreso

    @property
    def files(self) -> list[str]:
        return list(self.results["archivos"])

    def path(self, name: str) -> Path:
        """Ruta de un archivo de la carpeta (p. ej. `run.path("resumen.txt")`)."""
        return self.folder / name


def _call(command, argv: list[str]) -> tuple[Path, str]:
    """Corre un comando de `trafico.cli` sin imprimir: (carpeta, texto impreso). `parser.error` pasa a ConfigError."""
    out, err = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            folder = command(argv)
    except SystemExit as exc:
        message = err.getvalue().strip().splitlines()
        text = re.sub(r"^.*?error: ", "", message[-1]) if message else f"el comando terminó con código {exc.code}"
        raise ConfigError(text) from None
    return Path(folder), out.getvalue()


def _target(config: str | Path | None, name: str | None) -> list[str]:
    if config is not None and name is not None:
        raise ConfigError("name solo se admite sin config (con config, la carpeta de resultados toma el nombre de la "
                          "carpeta de la configuración)")  # fmt: skip
    value = config if config is not None else name
    return [] if value is None else [str(value)]


def _opt(flag: str, value) -> list[str]:
    """`--flag valor` (o `--flag v1 v2 …` con una lista); nada si el valor es None o una lista vacía."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [flag, *(str(v) for v in value)] if value else []
    return [flag, str(value)]


def _finish(command, argv: list[str], mode: str) -> Run:
    folder, log = _call(command, argv)
    return Run(folder, mode, read_results(folder), log)


def _bump(b) -> str:
    """Un escenario de topes de `compare_emissions` como lo pide `--topes`."""
    if b is None:
        return "sin"
    if isinstance(b, str):
        return b
    if isinstance(b, (list, tuple)):
        return ",".join(f"{x:g}" for x in b) or "sin"
    return f"{b:g}"


def _light(value) -> str:
    return "si" if value is True else "no" if value is False else "config" if value is None else str(value)


def simulate(config: str | Path | None = None, *, name: str | None = None) -> Run:
    """Corrida completa de `trafico`: réplicas Monte Carlo con la configuración, gráficas, tablas y resumen."""
    from trafico.cli import run

    return _finish(run, _target(config, name), "corrida")


def compare_emissions(
    config: str | Path | None = None, *, name: str | None = None, bumps: list | None = None, lights="config",
    lanes: list[int] | None = None, without: list[str] = (), length: float | None = None,
    bump_lanes: list[int] | None = None, replicas: int | None = None, run: float | None = None,
) -> Run:  # fmt: skip
    """`trafico-emisiones`: emisiones, consumo y tiempo de cada escenario con la misma semilla.

    `bumps`: topes de cada escenario; cada elemento es None o "sin" (sin tope), un número (posición en m) o una lista
    de posiciones (varios topes); None = sin tope y con los de la configuración. `lights`: "config", "si", "no",
    "ambos" (o True/False). `lanes`: carriles que se conservan; `without`: claves de tipos que no participan;
    `length`: largo del tramo; `bump_lanes`: carriles con tope; `replicas` y `run` (s de proceso) reemplazan a los de la
    configuración."""
    from trafico.cli import emissions

    argv = _target(config, name) + _opt("--topes", None if bumps is None else [_bump(b) for b in bumps])
    argv += ["--semaforo", _light(lights)] + _opt("--carriles", lanes) + _opt("--sin", list(without))
    argv += _opt("--largo", length) + _opt("--carriles-tope", bump_lanes) + _opt("--replicas", replicas)
    argv += _opt("--run", run)
    return _finish(emissions, argv, "emisiones")


def bump_spacing(
    config: str | Path | None = None, *, name: str | None = None, distances: list[float] | None = None,
    chain: int | None = None, first: float | None = None, tolerance: float | None = None,
    threshold: float | None = None, lights="config", lanes: list[int] | None = None, without: list[str] = (),
    length: float | None = None, bump_lanes: list[int] | None = None, replicas: int | None = None,
    run: float | None = None,
) -> Run:  # fmt: skip
    """`trafico-emisiones --separacion`: separación mínima entre topes para diluir la huella de emisiones.

    `distances`: separaciones en m (None = 10 15 20 25 30 40 50 60 80 100); `chain`: topes por cadena (2 = una pareja);
    `first`: posición del primer tope; `tolerance` y `threshold`: ver la documentación. Los demás argumentos, como en
    `compare_emissions` (aquí `lights` no admite "ambos")."""
    from trafico.cli import emissions

    argv = _target(config, name) + ["--separacion"] + [str(d) for d in distances or ()]
    argv += _opt("--cadena", chain) + _opt("--primer-tope", first) + _opt("--tolerancia", tolerance)
    argv += _opt("--umbral", threshold) + ["--semaforo", _light(lights)] + _opt("--carriles", lanes)
    argv += _opt("--sin", list(without)) + _opt("--largo", length) + _opt("--carriles-tope", bump_lanes)
    argv += _opt("--replicas", replicas) + _opt("--run", run)
    return _finish(emissions, argv, "separacion")


def compare_variants(
    config: str | Path | None = None, *, name: str | None = None, lengths: list[float] | None = None,
    lights: list | None = None, lanes: list[int] | None = None, without: list[str] = (),
    queue: list[str] | None = None, replicas: int | None = None, run: float | None = None,
) -> Run:  # fmt: skip
    """`trafico-variantes`: velocidad por carril y cola de entrada con cada largo de tramo y reparto del semáforo.

    `lights`: repartos como pares (rojo, verde) en s o textos "40/20" (None = tres repartos del ciclo de la
    configuración); `queue`: claves de los tipos que se cuentan en la cola de entrada."""
    from trafico.cli import variants

    split = None if lights is None else [x if isinstance(x, str) else f"{x[0]:g}/{x[1]:g}" for x in lights]
    argv = _target(config, name) + _opt("--largos", lengths) + _opt("--semaforos", split) + _opt("--carriles", lanes)
    argv += _opt("--sin", list(without)) + _opt("--cola", queue) + _opt("--replicas", replicas) + _opt("--run", run)
    return _finish(variants, argv, "variantes")


def redraw(folder: str | Path) -> Path:
    """Dibuja de nuevo la gráfica de una comparación o de un barrido ya corridos, sin simular ni sobrescribir; devuelve
    la ruta del archivo nuevo."""
    from trafico.cli import emissions, variants

    folder = Path(folder).expanduser()
    mode = mode_of(folder)
    if mode in ("emisiones", "separacion"):
        command = emissions
    elif mode == "variantes":
        command = variants
    else:
        raise ConfigError(f"{folder} no es una comparación ni un barrido: solo esas se redibujan")
    _, log = _call(command, ["--redibujar", str(folder)])
    match = re.search(r"Gráfica en (.+)", log)
    if match is None:
        raise ConfigError(f"no se pudo redibujar {folder}")
    return Path(match.group(1).strip())


def describe_config(config: str | Path | None = None) -> dict:
    """La configuración ya leída y validada, en un diccionario: tramo, ejecución, tipos de vehículo, semáforos y topes.
    Sirve para revisar una configuración antes de correrla (los errores salen como `ConfigError`)."""
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
    """Ruta del `config.toml` que usan las funciones cuando `config` es None."""
    return default_config_path()
