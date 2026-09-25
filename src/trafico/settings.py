"""Lectura y validación del archivo de configuración TOML de una corrida."""

from __future__ import annotations

import json
import re
import tomllib
from dataclasses import dataclass, fields
from datetime import datetime
from pathlib import Path

from trafico import __version__
from trafico.config import DEFAULT_RATES, DEFAULT_SPECS, DT, MAX_TYPES, Behavior, Bottleneck, SimConfig, VehicleSpec

CONFIG_NAME = "config.toml"
DEFAULT_RUN_NAME = "corrida"
# Parámetros numéricos de un vehículo: obligatorios para los tipos nuevos.
_SPEC_FIELDS = {
    "speed_kmh": float, "length": float, "gap_run": float, "gap_stop": float,
    "pax_min": int, "pax_max": int, "pax_mean": float, "pax_std": float,
}  # fmt: skip
# Umbrales de cambio de carril que un tipo puede fijar para sí; sin ellos, los de [behavior].
_LANE_CHANGE_FIELDS = ("lookahead", "min_advantage", "lane_change_cooldown")
# Detenciones de [bottleneck] de cada tipo: probabilidad y duración.
_BOTTLENECK_FIELDS = ("bottleneck_prob", "bottleneck_time_mean", "bottleneck_time_std")
# Claves que [bottleneck] ya no admite y a dónde se movieron (para el mensaje de error).
_MOVED_BOTTLENECK = {
    "[bottleneck] stop_prob": "bottleneck_prob", "[bottleneck] stop_time_mean": "bottleneck_time_mean",
    "[bottleneck] stop_time_std": "bottleneck_time_std", "[bottleneck] stop_types": "bottleneck_prob",
}  # fmt: skip
# Pasajeros de un tipo que solo lleva mercancía (cargo_prob = 1): sus claves son opcionales.
_CARGO_PAX = {"pax_min": 1, "pax_max": 1, "pax_mean": 1.0, "pax_std": 0.0}
# Valores por defecto de VehicleSpec (con slots, VehicleSpec.<campo> es un descriptor, no el valor).
_SPEC_DEFAULTS = {f.name: f.default for f in fields(VehicleSpec)}
_BUILTIN = {spec.key: (spec, rate) for spec, rate in zip(DEFAULT_SPECS, DEFAULT_RATES)}
_REQUIRED = object()
FLOATS = "float_o_lista"
"""Tipo de clave que acepta un número o una lista no vacía de números (se lee como tupla)."""
INTS = "lista_de_enteros"
_KIND_NAMES = {float: "un número", int: "un entero", str: "un texto", bool: "true o false",
               FLOATS: "un número o una lista de números", INTS: "una lista de enteros"}  # fmt: skip


VIDEO_FORMATS = ("webm", "mp4", "gif")


class ConfigError(ValueError):
    """Error en el contenido del archivo de configuración."""


@dataclass(frozen=True, slots=True)
class RunOptions:
    """Parámetros de ejecución y salida que no afectan al modelo."""

    replicas: int = 8
    workers: int = 0  # 0 = min(réplicas, CPUs disponibles)
    seed: int | None = None  # None = aleatoria (se registra en la copia de la configuración)
    output_dir: str = "resultados"
    name: str = ""  # sufijo de la carpeta de resultados; el nombre dado en el comando tiene prioridad
    csv: bool = True
    show: bool = False
    progress: bool = True
    animation: bool = False  # genera también el diagrama espacio-tiempo y el video de movimiento


@dataclass(frozen=True, slots=True)
class AnimationOptions:
    """Ventana y ritmo de la visualización del movimiento ([animation]); no afectan al modelo."""

    start: float = 0.0  # s simulados
    duration: float | None = None  # s simulados; None = 2 ciclos del semáforo
    speed: float = 4.0  # s simulados por s de video
    fps: int = 20
    replica: int = 1  # réplica que se visualiza (1 = la primera)
    format: str = "webm"  # "webm" (VP9) | "mp4" (H.264) | "gif"

    def window(self, sim: SimConfig) -> tuple[float, float]:
        """(inicio, fin) en s simulados, sin pasar del final de la corrida."""
        duration = self.duration if self.duration is not None else 2 * sim.cycle
        return self.start, min(self.start + duration, sim.sim_seconds)


@dataclass(frozen=True, slots=True)
class Settings:
    sim: SimConfig
    run: RunOptions
    path: Path
    text: str  # contenido original del archivo
    notices: tuple[str, ...] = ()  # avisos para la cabecera de la corrida
    animation: AnimationOptions = AnimationOptions()


def project_root() -> Path:
    """Raíz del proyecto (donde está pyproject.toml); si el paquete no está en modo editable, el directorio actual."""
    root = Path(__file__).resolve().parents[2]
    if (root / "pyproject.toml").is_file() and (root / "src" / "trafico").is_dir():
        return root
    return Path.cwd()


def default_config_path() -> Path:
    return project_root() / CONFIG_NAME


def resolve_output_dir(output_dir: str) -> Path:
    """Las rutas relativas de salida se toman desde la raíz del proyecto."""
    path = Path(output_dir).expanduser()
    return path if path.is_absolute() else project_root() / path


class _Reader:
    """Lee claves tipadas y recuerda cuáles se usaron para detectar claves desconocidas."""

    def __init__(self, data: dict):
        self.data = data
        self.used: set[tuple[str, ...]] = set()

    def get(self, keypath: tuple[str, ...], kind: type, default):
        *sections, key = keypath
        table = self.data
        for i, section in enumerate(sections):
            table = table.get(section, {})
            if not isinstance(table, dict):
                raise ConfigError(f"[{'.'.join(sections[: i + 1])}] debe ser una sección")
        self.used.add(keypath)
        if key not in table:
            if default is _REQUIRED:
                raise ConfigError(f"falta la clave obligatoria {_label(keypath)}")
            return default
        value = table[key]
        if kind is FLOATS:
            items = value if type(value) is list else [value]
            if not items or any(type(v) not in (int, float) for v in items):
                raise ConfigError(f"{_label(keypath)} debe ser {_KIND_NAMES[kind]}; se leyó {value!r}")
            floats = tuple(float(v) for v in items)
            return floats if type(value) is list else floats[0]
        if kind is INTS:
            if type(value) is not list or any(type(v) is not int for v in value):
                raise ConfigError(f"{_label(keypath)} debe ser {_KIND_NAMES[kind]}; se leyó {value!r}")
            return tuple(value)
        if kind is float and type(value) is int:
            value = float(value)
        if type(value) is not kind:  # type() exacto: un bool no cuenta como entero
            raise ConfigError(f"{_label(keypath)} debe ser {_KIND_NAMES[kind]}; se leyó {value!r}")
        return value

    def unknown(self) -> list[str]:
        out: list[str] = []

        def walk(table: dict, prefix: tuple[str, ...]) -> None:
            for key, value in table.items():
                keypath = prefix + (key,)
                if isinstance(value, dict):
                    walk(value, keypath)
                elif keypath not in self.used:
                    out.append(_label(keypath))

        walk(self.data, ())
        return out


def _label(keypath: tuple[str, ...]) -> str:
    *sections, key = keypath
    return f"[{'.'.join(sections)}] {key}" if sections else key


def parse_settings(text: str, path: Path) -> Settings:
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"TOML inválido: {exc}") from exc
    r = _Reader(data)

    specs, rates = _parse_vehicles(r)
    behavior = Behavior(
        **{
            f.name: r.get(
                ("behavior", f.name), FLOATS if f.name == "congestion_factor" else float, getattr(Behavior(), f.name)
            )
            for f in fields(Behavior)
        }
    )

    bn = Bottleneck()
    bottleneck = Bottleneck(
        stop_lanes=r.get(("bottleneck", "stop_lanes"), INTS, bn.stop_lanes),
        stop_zone=_zone(r.get(("bottleneck", "stop_zone"), FLOATS, bn.stop_zone)),
    )

    d = SimConfig()
    occupancy = r.get(("initial", "occupancy"), FLOATS, d.initial_occupancy)
    lanes, speed_limit, notices = _lanes(r, behavior, occupancy, d.lanes)
    sim = SimConfig(
        length=r.get(("road", "length"), float, d.length),
        lanes=lanes,
        lane_speed_limit=speed_limit,
        red=r.get(("traffic_light", "red"), float, d.red),
        green=r.get(("traffic_light", "green"), float, d.green),
        yellow=r.get(("traffic_light", "yellow"), float, d.yellow),
        start_phase=r.get(("traffic_light", "start_phase"), str, d.start_phase),
        rates=rates,
        slow_lane=r.get(("demand", "slow_lane"), str, d.slow_lane),
        initial_occupancy=occupancy,
        run=r.get(("execution", "run"), float, d.run),
        time_scale=r.get(("execution", "time_scale"), float, d.time_scale),
        sample=r.get(("execution", "sample"), float, d.sample),
        specs=specs,
        behavior=behavior,
        bottleneck=bottleneck,
    )

    o = RunOptions()
    opts = RunOptions(
        replicas=r.get(("execution", "replicas"), int, o.replicas),
        workers=r.get(("execution", "workers"), int, o.workers),
        seed=r.get(("execution", "seed"), int, o.seed),
        output_dir=r.get(("output", "dir"), str, o.output_dir),
        name=r.get(("output", "name"), str, o.name),
        csv=r.get(("output", "csv"), bool, o.csv),
        show=r.get(("output", "show"), bool, o.show),
        progress=r.get(("output", "progress"), bool, o.progress),
        animation=r.get(("output", "animation"), bool, o.animation),
    )
    a = AnimationOptions()
    anim = AnimationOptions(
        start=r.get(("animation", "start"), float, a.start),
        duration=r.get(("animation", "duration"), float, a.duration),
        speed=r.get(("animation", "speed"), float, a.speed),
        fps=r.get(("animation", "fps"), int, a.fps),
        replica=r.get(("animation", "replica"), int, a.replica),
        format=r.get(("animation", "format"), str, a.format),
    )

    unknown = r.unknown()
    if unknown:
        msg = "claves desconocidas: " + ", ".join(unknown)
        moved = [u for u in unknown if u in _MOVED_BOTTLENECK]
        if moved:
            msg += ("; en [bottleneck] solo quedan stop_lanes y stop_zone: la probabilidad y la duración van en cada "
                    "[vehicles.<clave>] como " + ", ".join(sorted({_MOVED_BOTTLENECK[u] for u in moved})))
        orphan_rates = [u for u in unknown if u.startswith("[demand] ") and u.endswith("_rate")]
        if orphan_rates:
            key = orphan_rates[0].removeprefix("[demand] ").removesuffix("_rate")
            msg += f" (para un vehículo nuevo define también su sección [vehicles.{key}])"
        raise ConfigError(msg)
    validate_config(sim, opts)
    validate_animation(anim, sim, opts.replicas)
    return Settings(sim=sim, run=opts, path=path, text=text, notices=notices, animation=anim)


def _zone(value):
    """[inicio, fin] de [bottleneck] stop_zone: una lista de dos números."""
    if value is None:
        return None
    if not isinstance(value, tuple) or len(value) != 2:
        raise ConfigError(f"[bottleneck] stop_zone debe ser una lista [inicio, fin] en m; se leyó {value!r}")
    return value


def validate_animation(anim: AnimationOptions, sim: SimConfig, replicas: int) -> None:
    def check(ok: bool, msg: str) -> None:
        if not ok:
            raise ConfigError(msg)

    check(0 <= anim.start < sim.sim_seconds,
          f"[animation] start debe estar entre 0 y {sim.sim_seconds:g} s simulados (la duración de la corrida)")  # fmt: skip
    check(anim.duration is None or anim.duration > 0, "[animation] duration debe ser mayor que 0")
    check(anim.speed > 0, "[animation] speed debe ser mayor que 0")
    check(1 <= anim.fps <= 60, "[animation] fps debe estar entre 1 y 60")
    check(anim.format in VIDEO_FORMATS, f"[animation] format debe ser {', '.join(map(repr, VIDEO_FORMATS))}")
    check(1 <= anim.replica <= replicas, f"[animation] replica debe estar entre 1 y {replicas} ([execution] replicas)")


def _lanes(r: _Reader, behavior: Behavior, occupancy, default: int):
    """Número de carriles: la longitud de las listas por carril ([behavior] congestion_factor,
    [road] max_line_speed e [initial] occupancy), que deben coincidir. Un número suelto aplica a todos los carriles;
    sin ninguna lista se usan `default` carriles.

    `[road] lanes` está obsoleto: solo se acepta para replicar copias anteriores, y en ese caso
    las listas se ajustan a él (los carriles extra usan el último valor; los sobrantes se ignoran).
    """
    speed = r.get(("road", "max_line_speed"), FLOATS, None)
    legacy = r.get(("road", "lanes"), int, None)
    lists = {
        "[behavior] congestion_factor": behavior.congestion_factor,
        "[road] max_line_speed": speed,
        "[initial] occupancy": occupancy,
    }
    lengths = {name: len(v) for name, v in lists.items() if isinstance(v, tuple)}
    notices: tuple[str, ...] = ()
    if legacy is not None:
        lanes = legacy
        notices = ("Aviso: [road] lanes está obsoleto; el número de carriles sale de la longitud de "
                   "congestion_factor o max_line_speed",)  # fmt: skip
    elif lengths:
        if len(set(lengths.values())) > 1:
            detail = ", ".join(f"{name} tiene {n}" for name, n in lengths.items())
            raise ConfigError(f"las listas por carril deben tener el mismo largo (un valor por carril): {detail}")
        lanes = next(iter(lengths.values()))
    else:
        lanes = default
    return lanes, speed, notices


def _parse_vehicles(r: _Reader) -> tuple[tuple[VehicleSpec, ...], tuple[float, ...]]:
    """Tipos de vehículo: los incorporados (car, bike, bus) y cada [vehicles.<clave>] nuevo.

    Los incorporados toman sus valores por defecto para las claves que falten. Un tipo nuevo
    debe definir todos los parámetros numéricos y su tasa `[demand] <clave>_rate`; `name`
    (por defecto, la clave), `lane_change` (por defecto, true), `lane` (carril fijo de un tipo que
    no cambia de carril; por defecto, según [demand] slow_lane), `exclusive` (reserva ese carril
    para los tipos que lo tienen como fijo; por defecto, false), la parada antes del semáforo
    (`stop_position`, `stop_time_mean`, `stop_time_std`; por defecto, sin parada) y `abreast`
    (cuántos se detienen lado a lado en un carril; por defecto, 1) son opcionales, igual que el
    rebase agresivo (`overtake`; por defecto, false), los umbrales de cambio de carril propios
    del tipo (`lookahead`, `min_advantage`, `lane_change_cooldown`; por defecto, los de [behavior]),
    la probabilidad de llevar mercancía (`cargo_prob`; por defecto, 0; con 1, los parámetros de
    pasajeros son opcionales) y el largo variable (`length_std`, `length_min`, `length_max`; por
    defecto, fijo).
    """
    table = r.data.get("vehicles", {})
    if not isinstance(table, dict):
        raise ConfigError("[vehicles] debe ser una sección")
    for key, value in table.items():
        if not isinstance(value, dict):
            raise ConfigError(f"[vehicles] {key} debe ser una sección [vehicles.{key}]")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", key):
            raise ConfigError(f"[vehicles.{key}] la clave solo admite letras, dígitos, '_' y '-'")

    specs, rates = [], []
    for key in [*_BUILTIN, *(k for k in table if k not in _BUILTIN)]:
        base, base_rate = _BUILTIN.get(key, (None, _REQUIRED))
        section = ("vehicles", key)
        try:
            cargo_prob = r.get(section + ("cargo_prob",), float, base.cargo_prob if base else 0.0)
            optional = _CARGO_PAX if cargo_prob == 1 else {}  # sin pasajeros, no hace falta describirlos
            values = {
                f: r.get(section + (f,), kind, getattr(base, f) if base else optional.get(f, _REQUIRED))
                for f, kind in _SPEC_FIELDS.items()
            }
            rate = r.get(("demand", f"{key}_rate"), float, base_rate)
        except ConfigError as exc:
            if base is None and str(exc).startswith("falta"):
                raise ConfigError(f"{exc} (los vehículos nuevos deben definir todos sus parámetros)") from None
            raise
        specs.append(
            VehicleSpec(
                key=key,
                name=r.get(section + ("name",), str, base.name if base else key),
                can_change_lane=r.get(section + ("lane_change",), bool, base.can_change_lane if base else True),
                lane=r.get(section + ("lane",), int, base.lane if base else None),
                exclusive=r.get(section + ("exclusive",), bool, base.exclusive if base else False),
                stop_position=r.get(section + ("stop_position",), float, base.stop_position if base else None),
                stop_time_mean=r.get(section + ("stop_time_mean",), float, base.stop_time_mean if base else _SPEC_DEFAULTS["stop_time_mean"]),
                stop_time_std=r.get(section + ("stop_time_std",), float, base.stop_time_std if base else _SPEC_DEFAULTS["stop_time_std"]),
                abreast=r.get(section + ("abreast",), int, base.abreast if base else 1),
                overtake=r.get(section + ("overtake",), bool, base.overtake if base else False),
                **{f: r.get(section + (f,), float, getattr(base, f) if base else None) for f in _LANE_CHANGE_FIELDS},
                cargo_prob=cargo_prob,
                length_std=r.get(section + ("length_std",), float, base.length_std if base else 0.0),
                length_min=r.get(section + ("length_min",), float, base.length_min if base else None),
                length_max=r.get(section + ("length_max",), float, base.length_max if base else None),
                **{f: r.get(section + (f,), float, getattr(base, f) if base else _SPEC_DEFAULTS[f]) for f in _BOTTLENECK_FIELDS},
                **values,
            )
        )
        rates.append(rate)
    return tuple(specs), tuple(rates)


def load_settings(path: Path) -> Settings:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ConfigError(f"no existe el archivo de configuración {path}") from None
    try:
        return parse_settings(text, path)
    except ConfigError as exc:
        raise ConfigError(f"{path}: {exc}") from None


def validate_config(sim: SimConfig, opts: RunOptions) -> None:
    """Valida la configuración de una corrida; lanza ConfigError con la clave que falla."""
    def check(ok: bool, msg: str) -> None:
        if not ok:
            raise ConfigError(msg)

    check(sim.n_types <= MAX_TYPES, f"[vehicles] se admiten hasta {MAX_TYPES} tipos de vehículo; hay {sim.n_types}")
    for spec in sim.specs:
        _validate_spec(spec, f"[vehicles.{spec.key}]", check)
        if spec.lane is not None:
            label = f"[vehicles.{spec.key}] lane"
            check(not spec.can_change_lane, f"{label} solo aplica a tipos con lane_change = false")
            check(0 <= spec.lane < sim.lanes, f"{label} debe estar entre 0 (carril derecho) y {sim.lanes - 1}")
        check(not spec.exclusive or spec.lane is not None,
              f"[vehicles.{spec.key}] exclusive = true requiere un carril fijo (lane)")  # fmt: skip
        check(1 <= spec.abreast <= 4, f"[vehicles.{spec.key}] abreast debe estar entre 1 y 4")
        overrides = {f: getattr(spec, f) for f in _LANE_CHANGE_FIELDS if getattr(spec, f) is not None}
        for name in ([*overrides, "overtake"] if spec.overtake else overrides):
            check(spec.can_change_lane, f"[vehicles.{spec.key}] {name} solo aplica a tipos con lane_change = true")
        for name, value in overrides.items():
            check(value >= 0, f"[vehicles.{spec.key}] {name} no puede ser negativo")
        if spec.stop_position is not None:
            label = f"[vehicles.{spec.key}]"
            check(spec.longest <= spec.stop_position <= sim.length,
                  f"{label} stop_position debe estar entre {spec.longest:g} (el largo máximo del vehículo, que entra con "
                  f"la parte trasera en 0) y {sim.length:g} m (el semáforo)")  # fmt: skip
            check(spec.stop_time_mean >= 0 and spec.stop_time_std >= 0,
                  f"{label} stop_time_mean y stop_time_std no pueden ser negativos")  # fmt: skip
            check(spec.stop_time_mean + 6 * spec.stop_time_std <= 3600,
                  f"{label} la parada no puede durar más de una hora (stop_time_mean + 6·stop_time_std ≤ 3600 s)")  # fmt: skip
    for k, spec in enumerate(sim.specs):
        check(sim.rates[k] == 0 or bool(sim.allowed_lanes(k)),
              f"[vehicles.{spec.key}] no le queda carril: todos son exclusivos de otros tipos "
              f"(carriles reservados: {', '.join(map(str, sorted(sim.reserved_lanes)))})")  # fmt: skip
    names = [spec.name for spec in sim.specs]
    dupes = sorted({n for n in names if names.count(n) > 1})
    check(not dupes, f"[vehicles] los nombres deben ser distintos; se repite {', '.join(dupes)}")
    b = sim.behavior
    check(0 < b.reaction_min <= b.reaction_max <= 600, "[behavior] requiere 0 < reaction_min ≤ reaction_max ≤ 600")
    check(b.reaction_mean is None or b.reaction_std is not None,
          "[behavior] reaction_mean requiere reaction_std (sin ella, la reacción es uniforme en [min, max])")  # fmt: skip
    check(b.reaction_std is None or b.reaction_std >= 0, "[behavior] reaction_std no puede ser negativo")
    check(b.reaction_mean is None or b.reaction_min <= b.reaction_mean <= b.reaction_max,
          "[behavior] reaction_mean debe estar entre reaction_min y reaction_max")  # fmt: skip
    check(
        0 < b.lane_change_min <= b.lane_change_max <= 600,
        "[behavior] requiere 0 < lane_change_min ≤ lane_change_max ≤ 600",
    )
    check(0 < b.lane_change_speed_factor <= 1, "[behavior] lane_change_speed_factor debe estar en (0, 1]")
    check(0 <= b.leader_slowdown < 1, "[behavior] leader_slowdown debe estar en [0, 1)")
    check(0 < b.yellow_speed_factor <= 1, "[behavior] yellow_speed_factor debe estar en (0, 1]")
    for name in ("lane_change_cooldown", "lookahead", "min_advantage", "release_space", "yellow_approach"):
        check(getattr(b, name) >= 0, f"[behavior] {name} no puede ser negativo")
    check(b.lane_change_interval >= DT, f"[behavior] lane_change_interval debe ser ≥ {DT} s")
    cf = b.congestion_factor if isinstance(b.congestion_factor, tuple) else (b.congestion_factor,)
    check(all(0 <= v < 1 for v in cf), "[behavior] congestion_factor: cada valor debe estar en [0, 1)")

    _validate_bottleneck(sim, check)

    longest = max(s.longest + s.gap_run for s in sim.specs)
    check(sim.length >= 2 * longest, f"[road] length debe ser al menos {2 * longest:g} m (dos veces el vehículo más largo con su gap)")
    check(1 <= sim.lanes <= 50, "se admiten entre 1 y 50 carriles (largo de las listas por carril)")
    check(all(v > 0 for v in sim.lane_max_kmh), "[road] max_line_speed: cada valor debe ser mayor que 0")
    check(sim.red >= 0, "[traffic_light] red no puede ser negativo")
    check(sim.green > 0, "[traffic_light] green debe ser mayor que 0")
    check(sim.yellow >= 0, "[traffic_light] yellow no puede ser negativo")
    check(sim.start_phase in ("red", "green"), '[traffic_light] start_phase debe ser "red" o "green"')
    check(all(rate >= 0 for rate in sim.rates), "[demand] las tasas no pueden ser negativas")
    check(any(rate > 0 for rate in sim.rates), "[demand] al menos un tipo de vehículo debe tener tasa > 0")
    check(sim.slow_lane in ("right", "random"), '[demand] slow_lane debe ser "right" o "random"')
    check(all(0 <= v <= 1 for v in sim.lane_initial_occupancy), "[initial] occupancy: cada valor debe estar en [0, 1]")
    check(sim.run > 0, "[execution] run debe ser mayor que 0")
    check(sim.time_scale > 0, "[execution] time_scale debe ser mayor que 0")
    check(sim.sample > 0, "[execution] sample debe ser mayor que 0")
    for label, value in (
        ("[traffic_light] red", sim.red),
        ("[traffic_light] green", sim.green),
        ("[traffic_light] yellow", sim.yellow),
        ("[execution] sample", sim.sample),
    ):
        check(abs(value / DT - round(value / DT)) < 1e-9, f"{label} debe ser múltiplo de {DT} s")
    check(sim.n_samples >= 2, "[execution] la corrida debe cubrir al menos dos muestras (run × time_scale ≥ 2 × sample)")
    check(opts.replicas >= 1, "[execution] replicas debe ser al menos 1")
    check(opts.workers >= 0, "[execution] workers no puede ser negativo (0 = automático)")
    check(opts.seed is None or 0 <= opts.seed < 2**63, "[execution] seed debe ser un entero no negativo")
    check(bool(opts.output_dir.strip()), "[output] dir no puede estar vacío")


def _validate_bottleneck(sim: SimConfig, check) -> None:
    for spec in sim.specs:
        label = f"[vehicles.{spec.key}]"
        check(0 <= spec.bottleneck_prob <= 1, f"{label} bottleneck_prob debe estar entre 0 y 1")
        check(spec.bottleneck_prob == 0 or spec.stop_position is None,
              f"{label} bottleneck_prob no aplica a un tipo con parada propia (stop_position); solo se admite una")  # fmt: skip
        mean, std = spec.bottleneck_time_mean, spec.bottleneck_time_std
        check(mean >= 0 and std >= 0, f"{label} bottleneck_time_mean y bottleneck_time_std no pueden ser negativos")
        check(mean + 6 * std <= 3600,
              f"{label} la detención no puede durar más de una hora (media + 6·σ ≤ 3600 s)")  # fmt: skip
    for lane in sim.bottleneck.stop_lanes or ():
        check(0 <= lane < sim.lanes, f"[bottleneck] stop_lanes: cada carril debe estar entre 0 y {sim.lanes - 1}")
    lo, hi = sim.bottleneck_zone()
    check(0 <= lo < hi <= sim.length, f"[bottleneck] stop_zone requiere 0 ≤ inicio < fin ≤ {sim.length:g} ([road] length)")


def _validate_spec(s: VehicleSpec, label: str, check) -> None:
    check(bool(s.name.strip()), f"{label} name no puede estar vacío")
    check(s.speed_kmh > 0 and s.length > 0, f"{label} speed_kmh y length deben ser mayores que 0")
    check(0 < s.gap_stop <= s.gap_run, f"{label} requiere 0 < gap_stop ≤ gap_run")
    check(1 <= s.pax_min <= s.pax_max <= 255, f"{label} requiere 1 ≤ pax_min ≤ pax_max ≤ 255")
    check(s.pax_min <= s.pax_mean <= s.pax_max, f"{label} pax_mean debe estar entre pax_min y pax_max")
    check(s.pax_std >= 0, f"{label} pax_std no puede ser negativo")
    check(0 <= s.cargo_prob <= 1, f"{label} cargo_prob debe estar entre 0 y 1")
    check(s.length_std >= 0, f"{label} length_std no puede ser negativo")
    if s.length_std > 0:
        check(0 < s.shortest <= s.length <= s.longest,
              f"{label} requiere 0 < length_min ≤ length ≤ length_max (sin ellos, length ∓ 3·length_std); "
              f"quedan {s.shortest:g} ≤ {s.length:g} ≤ {s.longest:g}")  # fmt: skip


# ------------------------------------------------------------ carpeta de corrida


@dataclass(frozen=True, slots=True)
class Target:
    """Qué configuración leer y con qué nombre guardar la corrida."""

    config: Path  # ruta del config.toml
    name: str | None  # nombre dado en el comando (tiene prioridad sobre [output] name)
    fallback_name: str  # si no hay nombre en el comando ni en [output] name


def resolve_target(arg: str | None) -> Target:
    """Interpreta el único argumento del comando.

    - Sin argumento: config.toml de la raíz del proyecto.
    - Una carpeta existente (p. ej. resultados/<ID>_<nombre>): su config.toml.
    - Cualquier otro texto sin separadores de ruta: el nombre de la corrida, con el
      config.toml de la raíz del proyecto.
    """
    if arg is None:
        return Target(default_config_path(), None, DEFAULT_RUN_NAME)
    path = Path(arg).expanduser()
    if path.is_dir():
        return Target(path / CONFIG_NAME, None, path.resolve().name)
    if path.is_file():
        if path.name != CONFIG_NAME:
            raise ConfigError(f"el archivo de configuración debe llamarse {CONFIG_NAME}: {path}")
        return Target(path, None, path.resolve().parent.name)
    if "/" in arg or "\\" in arg or arg.endswith(".toml"):
        raise ConfigError(f"no existe la carpeta {path}")
    return Target(default_config_path(), arg, DEFAULT_RUN_NAME)


def safe_name(name: str) -> str:
    """Nombre apto para carpeta: letras, dígitos, '.', '_' y '-'."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or DEFAULT_RUN_NAME


def make_run_dir(base: Path, name: str, now: datetime) -> Path:
    """Crea `base/<fecha-hora>_<nombre>`; si ya existe, agrega un sufijo numérico."""
    stem = f"{now:%Y%m%d-%H%M%S}_{safe_name(name)}"
    run_dir = base / stem
    k = 2
    while run_dir.exists():
        run_dir = base / f"{stem}-{k}"
        k += 1
    run_dir.mkdir(parents=True)
    return run_dir


def config_copy_text(settings: Settings, seed: int, name: str, run_dir: Path, now: datetime) -> str:
    """Copia de la configuración para replicar la corrida.

    Es el texto original con un encabezado de comentario, más lo necesario para
    que la copia reproduzca la corrida: la semilla, si era aleatoria, y el nombre
    usado en [output] name, para que la réplica conserve el sufijo de la carpeta.
    """
    try:
        shown = run_dir.relative_to(Path.cwd())
    except ValueError:
        shown = run_dir
    header = (
        f"# Copia de la configuración usada en la corrida {run_dir.name}\n"
        f"# ({now:%Y-%m-%d %H:%M:%S}, trafico {__version__}). Para replicarla:\n"
        f"#   uv run trafico {shown}\n\n"
    )
    text = _COPY_HEADER.sub("", settings.text, count=1)  # al replicar una copia, no apilar encabezados
    if settings.run.seed is None:
        text = _set_key(text, "execution", "seed", seed, f"semilla aleatoria generada en la corrida {run_dir.name}")
    if settings.run.name != name:
        text = _set_key(text, "output", "name", name, f"nombre usado en la corrida {run_dir.name}")
    return header + text


_COPY_HEADER = re.compile(r"\A# Copia de .*?\n\n", flags=re.DOTALL)


def _set_key(text: str, section: str, key: str, value: int | str, comment: str) -> str:
    """Escribe `key = value` en [section] conservando el resto del texto tal cual.

    Reemplaza la línea de la clave si ya existe, la agrega al inicio de la sección
    si no, y crea la sección al final si falta. El resultado se verifica
    releyéndolo: debe ser igual al original salvo por esa clave. Si el formato no
    lo permite (p. ej. una tabla en línea), se deja el texto y se anota el valor.
    """
    literal = json.dumps(value)  # un entero o una cadena JSON también son TOML válido
    line = f"{key} = {literal}  # {comment}\n"
    if not text.endswith("\n"):
        text += "\n"
    header = re.search(rf"^[ \t]*\[{re.escape(section)}\][ \t]*(#.*)?\n", text, flags=re.MULTILINE)
    if header is None:
        candidate = text.rstrip("\n") + f"\n\n[{section}]\n" + line
    else:
        start = header.end()
        nxt = re.search(r"^[ \t]*\[", text[start:], flags=re.MULTILINE)
        end = start + nxt.start() if nxt else len(text)
        body = text[start:end]
        old = re.search(rf"^[ \t]*{re.escape(key)}[ \t]*=.*\n", body, flags=re.MULTILINE)
        body = body[: old.start()] + line + body[old.end() :] if old else line + body
        candidate = text[:start] + body + text[end:]
    try:
        expected = tomllib.loads(text)
        expected.setdefault(section, {})[key] = value
        if tomllib.loads(candidate) == expected:
            return candidate
    except (tomllib.TOMLDecodeError, AttributeError, TypeError):
        pass
    return text + f"\n# {comment}: {key} = {literal}\n"
