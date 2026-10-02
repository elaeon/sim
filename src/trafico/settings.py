"""Lectura y validación del archivo de configuración TOML de una corrida."""

from __future__ import annotations

import json
import re
import tomllib
from dataclasses import dataclass, fields, replace
from datetime import datetime
from pathlib import Path

from trafico import __version__
from trafico.config import (
    DEFAULT_RATES, DEFAULT_SPECS, DT, FUELS, MAX_TYPES, PEDESTRIAN_RATE, PEDESTRIAN_TIME, POLLUTANTS, Behavior,
    Bottleneck, Light, Rate, SimConfig, SpeedBump, VehicleSpec,
)
from trafico.emission_sets import EMISSION_SETS

CONFIG_NAME = "config.toml"
DEFAULT_RUN_NAME = "corrida"
# Parámetros numéricos de un vehículo: obligatorios para los tipos nuevos.
_SPEC_FIELDS = {
    "gap_run": float, "gap_stop": float,
}  # fmt: skip
# Diccionarios {min, max, mean, std}: pasajeros por vehículo ([vehicles.<clave>] pax), velocidad máxima, largo
# y tasa de llegada.
_PAX_KINDS = {"min": int, "max": int, "mean": float, "std": float}
_RATE_KINDS = dict.fromkeys(("min", "max", "mean", "std"), float)
_FLOAT_DIST_KINDS = _RATE_KINDS  # velocidad máxima y largo
# Campos de VehicleSpec que salen de [vehicles.<clave>] length.
_LENGTH_FIELDS = {"mean": "length", "std": "length_std", "min": "length_min", "max": "length_max"}
# Campos de VehicleSpec que salen de [vehicles.<clave>] speed_kmh.
_SPEED_FIELDS = {"mean": "speed_kmh", "std": "speed_std", "min": "speed_min", "max": "speed_max"}
# Umbrales de cambio de carril que un tipo puede fijar para sí; sin ellos, los de [behavior].
_LANE_CHANGE_FIELDS = ("lookahead", "min_advantage", "lane_change_cooldown")
# Aceleración y frenado graduales (m/s²); sin ellos, cambios de velocidad instantáneos.
_DYNAMICS_FIELDS = ("accel", "decel", "time_headway")
# Detenciones de [bottleneck] de cada tipo: probabilidad y duración.
_BOTTLENECK_FIELDS = ("bottleneck_prob", "bottleneck_time_mean", "bottleneck_time_std")
# Claves que [bottleneck] ya no admite y a dónde se movieron (para el mensaje de error).
_MOVED_BOTTLENECK = {
    "[bottleneck] stop_prob": "bottleneck_prob", "[bottleneck] stop_time_mean": "bottleneck_time_mean",
    "[bottleneck] stop_time_std": "bottleneck_time_std", "[bottleneck] stop_types": "bottleneck_prob",
}  # fmt: skip
# Pasajeros de un tipo que solo lleva mercancía (cargo_prob = 1): su clave pax es opcional.
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
    series_csv: bool = True  # genera series.csv
    emisiones_csv: bool = True  # genera emisiones_posicion.csv (si algún tipo emite)
    progress: bool = True
    animation: bool = False  # genera también el diagrama espacio-tiempo y el video de movimiento
    mobility_plot: bool = True  # genera movilidad_pasajeros.png
    passengers_plot: bool = True  # genera distribucion_pasajeros.png (si algún tipo lleva pasajeros)
    emissions_plot: bool = True  # genera emisiones_posicion.png (si algún tipo emite)


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
        duration = self.duration if self.duration is not None else 2 * sim.flow_window
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
    kinds = {"queue_reaction": bool}
    behavior = Behavior(
        **{
            f.name: r.get(("behavior", f.name), kinds.get(f.name, float), getattr(Behavior(), f.name))
            for f in fields(Behavior)
            if not f.name.startswith("reaction_")
        },
        **_reaction(r),
    )

    bn = Bottleneck()
    bottleneck = Bottleneck(
        stop_lanes=r.get(("bottleneck", "stop_lanes"), INTS, bn.stop_lanes),
        stop_zone=_zone(r.get(("bottleneck", "stop_zone"), FLOATS, bn.stop_zone)),
    )
    speed_bumps, bump_notices = _speed_bumps(r)

    d = SimConfig()
    occupancy = r.get(("initial", "occupancy"), FLOATS, d.initial_occupancy)
    exit_capacity = r.get(("exit", "capacity"), FLOATS, d.exit_capacity)
    exit_storage = r.get(("exit", "storage"), FLOATS, d.exit_storage)
    # [behavior] congestion_factor se eliminó (casi no cambiaba los resultados y duplicaba lo que ya hacen la
    # dinámica gradual y la reacción en cadena); las copias anteriores la traen. Una lista sigue contando para el
    # número de carriles, para que esas copias carguen con los mismos carriles.
    congestion = r.get(("behavior", "congestion_factor"), FLOATS, None)
    lanes, speed_limit, notices = _lanes(r, occupancy, d.lanes, exit_capacity, exit_storage, congestion)
    notices += bump_notices
    if congestion is not None:
        notices += ("Aviso: [behavior] congestion_factor se eliminó y se ignora (no cambia la velocidad); puedes "
                    "quitarla del archivo",)  # fmt: skip
    # [demand] slow_lane se eliminó; las copias anteriores la traen con "right", que es lo que se hace ahora.
    slow_lane = r.get(("demand", "slow_lane"), str, None)
    if slow_lane is not None:
        if slow_lane != "right":
            raise ConfigError('[demand] slow_lane se eliminó: los tipos que no cambian de carril ni tienen lane van por '
                              'el carril libre más a la derecha; para otro carril usa lane en [vehicles.<clave>]')
        notices += ("Aviso: [demand] slow_lane se eliminó y se ignora; puedes quitarla del archivo",)
    # [output] csv se renombró series_csv; las copias anteriores la traen y conservan su valor.
    legacy_csv = r.get(("output", "csv"), bool, None)
    if legacy_csv is not None:
        notices += ("Aviso: [output] csv se renombró series_csv; se usa su valor, pero cámbiala en el archivo",)
    # [output] show se eliminó (abría la ventana de matplotlib); las copias anteriores la traen.
    if r.get(("output", "show"), bool, None) is not None:
        notices += ("Aviso: [output] show se eliminó y se ignora; puedes quitarla del archivo",)
    # [output] saturation_threshold se eliminó (solo marcaba la gráfica); las copias anteriores la traen.
    if r.get(("output", "saturation_threshold"), float, None) is not None:
        notices += ("Aviso: [output] saturation_threshold se eliminó y se ignora; puedes quitarla del archivo",)
    length = r.get(("road", "length"), float, d.length)
    exit_light, extra_lights = _lights(r, length, d)
    sim = SimConfig(
        length=length,
        lanes=lanes,
        lane_speed_limit=speed_limit,
        red=exit_light.red,
        green=exit_light.green,
        yellow=exit_light.yellow,
        start_phase=exit_light.start_phase,
        traffic_light=exit_light.enabled,
        free_lanes=exit_light.free_lanes,
        light_pedestrian=exit_light.pedestrian,
        light_pedestrian_crossing=exit_light.pedestrian_crossing,
        extra_lights=extra_lights,
        # Solo con alguna tasa variable hace falta la distribución; si todas son fijas, basta su valor.
        **({"rate_dists": rates} if any(rate.variable for rate in rates) else {"rates": tuple(r.expected for r in rates)}),
        rate_interval=r.get(("demand", "rate_interval"), float, d.rate_interval),
        initial_occupancy=occupancy,
        exit_capacity=exit_capacity,
        exit_storage=exit_storage,
        run=r.get(("execution", "run"), float, d.run),
        time_scale=r.get(("execution", "time_scale"), float, d.time_scale),
        sample=r.get(("execution", "sample"), float, d.sample),
        specs=specs,
        behavior=behavior,
        bottleneck=bottleneck,
        speed_bumps=speed_bumps,
        **_fuel(r, d),
    )

    o = RunOptions()
    opts = RunOptions(
        replicas=r.get(("execution", "replicas"), int, o.replicas),
        workers=r.get(("execution", "workers"), int, o.workers),
        seed=r.get(("execution", "seed"), int, o.seed),
        output_dir=r.get(("output", "dir"), str, o.output_dir),
        name=r.get(("output", "name"), str, o.name),
        series_csv=r.get(("output", "series_csv"), bool, legacy_csv if legacy_csv is not None else o.series_csv),
        emisiones_csv=r.get(("output", "emisiones_csv"), bool, o.emisiones_csv),
        progress=r.get(("output", "progress"), bool, o.progress),
        animation=r.get(("output", "animation"), bool, o.animation),
        mobility_plot=r.get(("output", "mobility_plot"), bool, o.mobility_plot),
        passengers_plot=r.get(("output", "passengers_plot"), bool, o.passengers_plot),
        emissions_plot=r.get(("output", "emissions_plot"), bool, o.emissions_plot),
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
        if any(re.fullmatch(r"\[vehicles\.[^\]]+\] pax_(min|max|mean|std)", u) for u in unknown):
            msg += "; los pasajeros van ahora en un diccionario: pax = {min = 1, max = 6, mean = 1.5, std = 0.8}"
        if any(re.fullmatch(r"\[vehicles\.[^\]]+\] length_(min|max|std)", u) for u in unknown):
            msg += "; el largo va ahora en un diccionario: length = {min = 8, max = 16, mean = 10, std = 2}"
        if any(re.fullmatch(r"\[behavior\] reaction_(min|max|mean|std)", u) for u in unknown):
            msg += "; la reacción va ahora en un diccionario: reaction = {min = 0.7, max = 3, mean = 1.5, std = 0.5}"
        orphan_rates = [m for u in unknown if (m := re.match(r"\[demand\.(.+)_rate\] |\[demand\] (.+)_rate$", u))]
        if orphan_rates:
            key = orphan_rates[0].group(1) or orphan_rates[0].group(2)
            msg += f" (para un vehículo nuevo define también su sección [vehicles.{key}])"
        raise ConfigError(msg)
    validate_config(sim, opts)
    validate_animation(anim, sim, opts.replicas)
    # Aviso solo para las emisiones escritas en el archivo (el auto y el autobús incorporados traen coeficientes).
    written = {key for key, value in r.data.get("vehicles", {}).items() if isinstance(value, dict) and "emissions" in value}
    silent = [sp.name for k, sp in enumerate(sim.specs)
              if sp.key in written and sp.emissions and not sp.emits and sim.rates[k] > 0]  # fmt: skip
    if silent:
        notices += (f"Aviso: sin accel y decel no se calculan las emisiones de {', '.join(silent)} (el modelo usa la "
                    "aceleración real)",)  # fmt: skip
    unpriced = sorted({sp.fuel for k, sp in enumerate(sim.specs) if sp.burns and sim.rates[k] > 0
                       and sim.price(sp.fuel) is None})  # fmt: skip
    if unpriced:
        notices += (f"Aviso: sin precio en [fuel] price para {', '.join(unpriced)}: su consumo solo se da en litros",)
    return Settings(sim=sim, run=opts, path=path, text=text, notices=notices, animation=anim)


def _emissions(r: _Reader, key: str, base: VehicleSpec | None) -> tuple[tuple, str]:
    """[vehicles.<clave>.emissions]: (coeficientes, conjunto del que salen). `source` toma los de un conjunto con
    nombre (`EMISSION_SETS`) para la clave del tipo o para `source_type`; encima, por contaminante, los
    coeficientes [f1..f6] escritos a mano del modelo de Int Panis et al. (2006) y, opcional, `<contaminante>_decel`
    para a < −0.5 m/s² (por defecto, los mismos). Si la sección está, reemplaza a la del tipo incorporado; vacía,
    el tipo no emite."""
    table = r.data.get("vehicles", {}).get(key, {}).get("emissions")
    if table is None:
        return (base.emissions, base.emission_source) if base else ((), "")
    label = f"[vehicles.{key}.emissions]"
    if not isinstance(table, dict):
        raise ConfigError(f"[vehicles.{key}] emissions debe ser una sección {label}")
    section = ("vehicles", key, "emissions")
    source = r.get(section + ("source",), str, None)
    entry = r.get(section + ("source_type",), str, None)
    found: dict[str, tuple] = {}
    if source is not None:
        if source not in EMISSION_SETS:
            raise ConfigError(f"{label} source: no existe el conjunto {source!r}; hay {', '.join(EMISSION_SETS)}")
        entry = entry or key
        if entry not in EMISSION_SETS[source]:
            raise ConfigError(f"{label} el conjunto {source!r} no tiene el tipo {entry!r}"
                              f"{'' if key == entry else ' (source_type)'}; tiene {', '.join(EMISSION_SETS[source])} "
                              "(elige uno con source_type)")  # fmt: skip
        found = {pol: (acc, dec) for pol, acc, dec in EMISSION_SETS[source][entry]}
    elif entry is not None:
        raise ConfigError(f"{label} source_type requiere source (el conjunto de coeficientes)")
    out = []
    for pol in POLLUTANTS:
        coefs = {}
        for name in (pol, f"{pol}_decel"):
            value = r.get(section + (name,), FLOATS, None)
            if value is not None and (not isinstance(value, tuple) or len(value) != 6):
                raise ConfigError(f"{label} {name} debe ser una lista de 6 números [f1, f2, f3, f4, f5, f6]")
            coefs[name] = value
        acc, dec = coefs[pol], coefs[f"{pol}_decel"]
        if acc is None and pol in found:  # del conjunto; el _decel escrito a mano lo reemplaza
            acc, dec = found[pol][0], dec or found[pol][1]
        if acc is None:
            if dec is not None:
                raise ConfigError(f"{label} {pol}_decel requiere {pol} (los coeficientes para a ≥ −0.5 m/s²)")
            continue
        out.append((pol, acc, dec or acc))
    return tuple(out), source or ""


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


def _lanes(r: _Reader, occupancy, default: int, exit_capacity=0.0, exit_storage=60.0, congestion=None):
    """Número de carriles: la longitud de las listas por carril ([road] max_line_speed, [initial] occupancy,
    [exit] capacity y storage y, en copias anteriores, [behavior] congestion_factor), que deben coincidir. Un
    número suelto aplica a todos los carriles; sin ninguna lista se usan `default` carriles.

    `[road] lanes` está obsoleto: solo se acepta para replicar copias anteriores, y en ese caso
    las listas se ajustan a él (los carriles extra usan el último valor; los sobrantes se ignoran).
    """
    speed = r.get(("road", "max_line_speed"), FLOATS, None)
    legacy = r.get(("road", "lanes"), int, None)
    lists = {
        "[behavior] congestion_factor": congestion,
        "[road] max_line_speed": speed,
        "[initial] occupancy": occupancy,
        "[exit] capacity": exit_capacity,
        "[exit] storage": exit_storage,
    }
    lengths = {name: len(v) for name, v in lists.items() if isinstance(v, tuple)}
    notices: tuple[str, ...] = ()
    if legacy is not None:
        lanes = legacy
        notices = ("Aviso: [road] lanes está obsoleto; el número de carriles sale de la longitud de "
                   "max_line_speed (o de las demás listas por carril)",)  # fmt: skip
    elif lengths:
        if len(set(lengths.values())) > 1:
            detail = ", ".join(f"{name} tiene {n}" for name, n in lengths.items())
            raise ConfigError(f"las listas por carril deben tener el mismo largo (un valor por carril): {detail}")
        lanes = next(iter(lengths.values()))
    else:
        lanes = default
    return lanes, speed, notices


def _parse_vehicles(r: _Reader) -> tuple[tuple[VehicleSpec, ...], tuple[Rate, ...]]:
    """Tipos de vehículo: los incorporados (car, bike, bus) y cada [vehicles.<clave>] nuevo.

    Los incorporados toman sus valores por defecto para las claves que falten. Un tipo nuevo
    debe definir todos los parámetros numéricos, sus pasajeros `pax = {min, max, mean, std}` y su
    tasa `[demand] <clave>_rate`; `name`
    (por defecto, la clave), `lane_change` (por defecto, true), `lane` (carril fijo de un tipo que
    no cambia de carril; por defecto, el derecho libre), `exclusive` (reserva ese carril
    para los tipos que lo tienen como fijo; por defecto, false), la parada antes del semáforo
    (`stop_position`, `stop_time_mean`, `stop_time_std`; por defecto, sin parada) y `abreast`
    (cuántos se detienen lado a lado en un carril; por defecto, 1) son opcionales, igual que el
    rebase dentro del carril (`pass_in_lane`; por defecto, false), la aceleración y el frenado
    graduales (`accel`, `decel` en m/s²; por defecto, instantáneos), el intervalo de seguimiento en marcha
    (`time_headway` en s; por defecto, gap_run fijo), la velocidad al pasar el tope
    (`speed_bump_kmh`; por defecto, sin frenar), el rebase agresivo (`overtake`; por defecto, false), los umbrales de cambio de carril propios
    del tipo (`lookahead`, `min_advantage`, `lane_change_cooldown`; por defecto, los de [behavior]),
    la probabilidad de llevar mercancía (`cargo_prob`; por defecto, 0; con 1, `pax` es
    opcional). La velocidad máxima (`speed_kmh`) y el largo (`length`) son diccionarios {min, max, mean,
    std}: normal truncada a [min, max] sorteada para cada vehículo; con std = 0, todos tienen mean.
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
            values = {f: r.get(section + (f,), kind, getattr(base, f) if base else _REQUIRED) for f, kind in _SPEC_FIELDS.items()}
            for name, fields, example in (
                ("speed_kmh", _SPEED_FIELDS, "{min = 60, max = 90, mean = 80, std = 8} (km/h)"),
                ("length", _LENGTH_FIELDS, "{min = 4, max = 5, mean = 4.5, std = 0.3} (m)"),
            ):
                dist = _dist(r, section + (name,), _FLOAT_DIST_KINDS, example)
                if dist is not None:
                    values.update({fields[f]: v for f, v in dist.items()})
                elif base is not None:
                    values.update({f: getattr(base, f) for f in fields.values()})
                else:
                    raise ConfigError(f"falta la clave obligatoria [vehicles.{key}] {name}")
            pax = _dist(r, section + ("pax",), _PAX_KINDS, "{min = 1, max = 6, mean = 1.5, std = 0.8}")
            if pax is not None:
                values.update({f"pax_{f}": v for f, v in pax.items()})
            elif base is not None:
                values.update({f: getattr(base, f) for f in _CARGO_PAX})
            elif cargo_prob == 1:  # sin pasajeros, no hace falta describirlos
                values.update(_CARGO_PAX)
            else:
                raise ConfigError(f"falta la clave obligatoria [vehicles.{key}] pax")
            rate = _rate(r, key, base_rate)
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
                pass_in_lane=r.get(section + ("pass_in_lane",), bool, base.pass_in_lane if base else False),
                overtake=r.get(section + ("overtake",), bool, base.overtake if base else False),
                **{f: r.get(section + (f,), float, getattr(base, f) if base else None) for f in _LANE_CHANGE_FIELDS},
                **{f: r.get(section + (f,), float, getattr(base, f) if base else None) for f in _DYNAMICS_FIELDS},
                speed_bump_kmh=r.get(section + ("speed_bump_kmh",), float, base.speed_bump_kmh if base else None),
                cargo_prob=cargo_prob,
                **{f: r.get(section + (f,), float, getattr(base, f) if base else _SPEC_DEFAULTS[f]) for f in _BOTTLENECK_FIELDS},
                **dict(zip(("emissions", "emission_source"), _emissions(r, key, base))),
                fuel=r.get(section + ("fuel",), str, base.fuel if base else None),
                mass_kg=r.get(section + ("mass_kg",), float, base.mass_kg if base else None),
                **values,
            )
        )
        rates.append(rate)
    return tuple(specs), tuple(rates)


def _lights(r: _Reader, length: float, d: SimConfig) -> tuple[Light, tuple[Light, ...]]:
    """[traffic_light]: un semáforo al final del tramo (una sección) o varios ([[traffic_light]], uno por semáforo).
    Cada uno lleva `position` (m desde el inicio; sin ella o igual a `length`, el del final del tramo), `enabled`,
    `red`, `green`, `yellow`, `start_phase`, `free_lanes` y, si es peatonal, `pedestrian` y `pedestrian_crossing`.
    Devuelve (el del final del tramo, los intermedios); si hay semáforos pero ninguno en el final del tramo, este
    queda desactivado."""
    table = r.data.get("traffic_light")
    if table is None:
        return d.exit_light, ()
    if isinstance(table, dict):
        readers, checks = [r], []
    elif isinstance(table, list) and all(isinstance(e, dict) for e in table):
        r.used.add(("traffic_light",))
        readers, checks = [], []
        for i, entry in enumerate(table, 1):
            sub = _Reader({"traffic_light": entry})
            readers.append(sub)
            checks.append((i, sub))
    else:
        raise ConfigError("[traffic_light] debe ser una sección o una lista de secciones ([[traffic_light]])")
    lights = []
    for reader in readers:
        def read(key, kind, default, reader=reader):
            return reader.get(("traffic_light", key), kind, default)

        position = read("position", float, None)
        pedestrian = read("pedestrian", bool, False)
        start_phase = read("start_phase", str, None)
        if pedestrian and start_phase is not None:
            raise ConfigError("[traffic_light] start_phase no aplica a un semáforo con pedestrian = true (siempre "
                              "empieza en verde): quítala")  # fmt: skip
        lights.append(Light(
            position=length if position is None else position,
            red=read("red", float, d.red), green=read("green", float, d.green),
            yellow=read("yellow", float, d.yellow), start_phase=start_phase or d.start_phase,
            enabled=read("enabled", bool, d.traffic_light), free_lanes=read("free_lanes", INTS, d.free_lanes),
            pedestrian=pedestrian, pedestrian_crossing=_crossing(reader, ("traffic_light",)),
        ))  # fmt: skip
    for i, sub in checks:
        unknown = sub.unknown()
        if unknown:
            raise ConfigError(f"claves desconocidas en el semáforo {i} de [[traffic_light]]: "
                              + ", ".join(u.replace("[traffic_light] ", "") for u in unknown))  # fmt: skip
    at_exit = [lt for lt in lights if abs(lt.position - length) < 1e-9]
    if len(at_exit) > 1:
        raise ConfigError(f"[[traffic_light]] hay {len(at_exit)} semáforos en el final del tramo ({length:g} m, o sin "
                          "position): solo puede haber uno")  # fmt: skip
    extra = tuple(sorted((lt for lt in lights if lt not in at_exit), key=lambda lt: lt.position))
    return (at_exit[0] if at_exit else replace(d.exit_light, enabled=False)), extra


def _speed_bumps(r: _Reader) -> tuple[tuple[SpeedBump, ...], tuple[str, ...]]:
    """[speed_bump]: un tope (una sección) o varios ([[speed_bump]], uno por tope), cada uno con su `position` (m desde
    el inicio, un número), `lanes`, `pedestrian`, `pedestrian_crossing` y `pedestrian_time`. Sin la sección no hay
    topes; una sección sin `position` ni otra clave tampoco. Devuelve (topes, avisos). `position` como lista
    (el formato anterior) se acepta con un aviso: cada posición es un tope con las demás claves de la sección."""
    table = r.data.get("speed_bump")
    if table is None:
        return (), ()
    if isinstance(table, dict):
        readers, checks = [r], []
    elif isinstance(table, list) and all(isinstance(e, dict) for e in table):
        r.used.add(("speed_bump",))
        readers, checks = [], []
        for i, entry in enumerate(table, 1):
            sub = _Reader({"speed_bump": entry})
            readers.append(sub)
            checks.append((i, sub))
    else:
        raise ConfigError("[speed_bump] debe ser una sección o una lista de secciones ([[speed_bump]])")
    bumps, notices = [], ()
    for n, reader in enumerate(readers, 1):
        legacy = isinstance(table, dict)
        value = reader.get(("speed_bump", "position"), FLOATS, None)
        lanes = reader.get(("speed_bump", "lanes"), INTS, None)
        pedestrian = reader.get(("speed_bump", "pedestrian"), bool, False)
        rate = _crossing(reader, ("speed_bump",))
        time = reader.get(("speed_bump", "pedestrian_time"), float, PEDESTRIAN_TIME)
        if isinstance(value, tuple):
            if not legacy:
                raise ConfigError(f"[[speed_bump]] position del tope {n} debe ser un número (un tope por sección); "
                                  "para varios topes, una sección [[speed_bump]] cada uno")  # fmt: skip
            notices = ("Aviso: [speed_bump] position como lista se reemplazó por una sección [[speed_bump]] por tope; "
                       "se usa cada posición como un tope con las mismas claves, pero cámbiala en el archivo",)
            positions = value
        else:
            positions = () if value is None else (value,)
        if not positions:
            if lanes is not None or pedestrian:  # claves de un tope sin dónde está
                where = "[speed_bump]" if legacy else f"[[speed_bump]] del tope {n}"
                key = "lanes" if lanes is not None else "pedestrian"
                raise ConfigError(f"{where}: {key} requiere position (dónde está el tope)")
            if not legacy:
                raise ConfigError(f"[[speed_bump]] falta position en el tope {n}")
        bumps += [SpeedBump(p, lanes, pedestrian, rate, time) for p in positions]
    for i, sub in checks:
        unknown = sub.unknown()
        if unknown:
            raise ConfigError(f"claves desconocidas en el tope {i} de [[speed_bump]]: "
                              + ", ".join(u.replace("[speed_bump] ", "") for u in unknown))  # fmt: skip
    return tuple(sorted(bumps, key=lambda b: b.position)), notices


def _crossing(r: _Reader, section: tuple[str, ...]) -> Rate:
    """`pedestrian_crossing` de una sección ([speed_bump] o un [[traffic_light]]): la aparición de peatones, un
    diccionario {min, max, mean, std} en peatones/min. Sin ella, PEDESTRIAN_RATE. Se valida solo si la sección tiene
    pedestrian = true (en `validate_config`)."""
    values = _dist(r, (*section, "pedestrian_crossing"), _RATE_KINDS,
                   "{min = 0.5, max = 3, mean = 1.5, std = 1} (peatones/min)")  # fmt: skip
    return PEDESTRIAN_RATE if values is None else Rate(**values)


def _fuel(r: _Reader, d: SimConfig) -> dict:
    """[fuel]: precio por litro de cada combustible (`price = {gasolina = 24.0, ...}`) y su moneda."""
    table = r.data.get("fuel", {})
    if isinstance(table, dict) and "price" in table and not isinstance(table["price"], dict):
        raise ConfigError(f"[fuel] price debe ser un diccionario por combustible, p. ej. price = {{gasolina = 24.0}}; "
                          f"se leyó {table['price']!r}")  # fmt: skip
    prices = tuple((f, v) for f in FUELS if (v := r.get(("fuel", "price", f), float, None)) is not None)
    return {"fuel_price": prices, "currency": r.get(("fuel", "currency"), str, d.currency)}


def _rate(r: _Reader, key: str, default) -> Rate:
    """[demand] <clave>_rate: un diccionario {min, max, mean, std} en veh/min. Los tipos incorporados
    que no la definen usan su tasa fija por defecto."""
    name = f"{key}_rate"
    if not isinstance(r.data.get("demand", {}), dict):
        raise ConfigError("[demand] debe ser una sección")
    values = _dist(r, ("demand", name), _RATE_KINDS, "{min = 10, max = 20, mean = 15, std = 5} (veh/min)")
    if values is None:
        if default is _REQUIRED:
            raise ConfigError(f"falta la clave obligatoria [demand] {name}")
        return Rate.fixed(default)
    rate = Rate(**values)
    _check_rate(f"[demand] {name}", rate)
    return rate


def _reaction(r: _Reader) -> dict[str, float]:
    """[behavior] reaction: {min, max} (uniforme) o {min, max, mean, std} (normal truncada a [min, max]).
    Sin la clave, los valores por defecto de Behavior."""
    behavior = r.data.get("behavior", {})
    value = behavior.get("reaction") if isinstance(behavior, dict) else None
    pair = {"mean", "std"} & set(value) if isinstance(value, dict) else set()
    if len(pair) == 1:
        raise ConfigError("[behavior] reaction: mean y std van juntas (con ellas, normal truncada a [min, max]; "
                          "sin ellas, uniforme en [min, max])")  # fmt: skip
    kinds = dict.fromkeys(("min", "max", *(("mean", "std") if pair else ())), float)
    values = _dist(r, ("behavior", "reaction"), kinds, "{min = 0.7, max = 3, mean = 1.5, std = 0.5}")
    return {} if values is None else {f"reaction_{f}": v for f, v in values.items()}


def _dist(r: _Reader, keypath: tuple[str, ...], kinds: dict[str, type], example: str) -> dict | None:
    """Diccionario en `keypath` con todas las claves de `kinds`; None si no está."""
    *sections, key = keypath
    table = r.data
    for section in sections:
        table = table.get(section, {})
    if key not in table:
        return None
    if not isinstance(table[key], dict):
        raise ConfigError(f"{_label(keypath)} debe ser un diccionario, p. ej. {key} = {example}; se leyó {table[key]!r}")
    return {f: r.get(keypath + (f,), kind, _REQUIRED) for f, kind in kinds.items()}


def _check_rate(label: str, rate: Rate) -> None:
    if not 0 <= rate.min <= rate.mean <= rate.max:
        raise ConfigError(f"{label}: debe cumplirse 0 ≤ min ≤ mean ≤ max")
    if rate.std < 0:
        raise ConfigError(f"{label}: std no puede ser negativa")


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
        check(not spec.pass_in_lane or spec.abreast >= 2,
              f"[vehicles.{spec.key}] pass_in_lane = true requiere abreast ≥ 2 (que quepan dos lado a lado)")  # fmt: skip
        for name in _DYNAMICS_FIELDS:
            value = getattr(spec, name)
            check(value is None or 0 < value <= 20, f"[vehicles.{spec.key}] {name} debe estar en (0, 20] m/s²")
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
    check(0 < b.reaction_min <= b.reaction_max <= 600, "[behavior] reaction: debe cumplirse 0 < min ≤ max ≤ 600")
    check(b.reaction_mean is None or b.reaction_std is not None, "[behavior] reaction: mean requiere std")
    check(b.reaction_std is None or b.reaction_std >= 0, "[behavior] reaction: std no puede ser negativa")
    check(b.reaction_mean is None or b.reaction_min <= b.reaction_mean <= b.reaction_max,
          "[behavior] reaction: mean debe estar entre min y max")  # fmt: skip
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

    _validate_bottleneck(sim, check)
    _validate_speed_bump(sim, check)
    for spec in sim.specs:
        check(spec.fuel is None or spec.fuel in FUELS,
              f"[vehicles.{spec.key}] fuel debe ser uno de {', '.join(FUELS)}; se leyó {spec.fuel!r}")  # fmt: skip
        check(spec.mass_kg is None or spec.mass_kg > 0, f"[vehicles.{spec.key}] mass_kg debe ser mayor que 0 kg")
    for fuel, price in sim.fuel_price:
        check(price >= 0, f"[fuel] price: el precio de {fuel} no puede ser negativo")
    check(bool(sim.currency.strip()), "[fuel] currency no puede estar vacía")

    longest = max(s.longest + s.gap_run for s in sim.specs)
    check(sim.length >= 2 * longest, f"[road] length debe ser al menos {2 * longest:g} m (dos veces el vehículo más largo con su gap)")
    check(1 <= sim.lanes <= 50, "se admiten entre 1 y 50 carriles (largo de las listas por carril)")
    check(all(v > 0 for v in sim.lane_max_kmh), "[road] max_line_speed: cada valor debe ser mayor que 0")
    for where, lt in [("[traffic_light]", sim.exit_light), *((f"[traffic_light] a {e.position:g} m", e)
                                                                for e in sim.extra_lights)]:  # fmt: skip
        check(lt.red >= 0, f"{where} red no puede ser negativo")
        check(lt.yellow >= 0, f"{where} yellow no puede ser negativo")
        # Con los tres en 0 no hay semáforo (como enabled = false); si no, hace falta verde. El peatonal usa
        # green como verde mínimo entre dos cruces (puede ser 0) y red como lo que dura el cruce.
        check(not lt.active or lt.pedestrian or lt.green > 0,
              f"{where} green debe ser mayor que 0 (o red, green y yellow en 0, o enabled = false, para no tener "
              "semáforo)")  # fmt: skip
        if lt.pedestrian and lt.enabled:
            check(lt.red > 0, f"{where} red debe ser mayor que 0 con pedestrian = true (es lo que dura el cruce)")
            _check_rate(f"{where} pedestrian_crossing", lt.pedestrian_crossing)
        check(lt.start_phase in ("red", "green"), f'{where} start_phase debe ser "red" o "green"')
        for name, value in (("red", lt.red), ("green", lt.green), ("yellow", lt.yellow)):
            check(abs(value / DT - round(value / DT)) < 1e-9, f"{where} {name} debe ser múltiplo de {DT} s")
    # Un semáforo desactivado (enabled = false, o sus fases en 0) no está en la calle: su posición no se revisa.
    positions = [e.position for e in sim.extra_lights if e.active]
    check(all(0 < p < sim.length for p in positions),
          f"[[traffic_light]] position: cada semáforo intermedio debe estar entre 0 y {sim.length:g} m ([road] "
          "length), sin incluirlos (el del final del tramo va sin position)")  # fmt: skip
    check(len(set(positions)) == len(positions), "[[traffic_light]] position: hay semáforos en la misma posición")
    for k, spec in enumerate(sim.specs):
        _check_rate(f"[demand] {spec.key}_rate", sim.rate(k))
    check(sim.rate_interval >= DT, f"[demand] rate_interval debe ser ≥ {DT} s")
    check(any(rate > 0 for rate in sim.rates), "[demand] al menos un tipo de vehículo debe tener tasa > 0")
    check(all(0 <= v <= 1 for v in sim.lane_initial_occupancy), "[initial] occupancy: cada valor debe estar en [0, 1]")
    check(all(v >= 0 for v in sim.lane_exit_capacity), "[exit] capacity: cada valor debe ser ≥ 0 veh/min (0 = sin cola de salida)")
    check(all(v > 0 for v in sim.lane_exit_storage), "[exit] storage: cada valor debe ser > 0 m")
    check(sim.run > 0, "[execution] run debe ser mayor que 0")
    check(sim.time_scale > 0, "[execution] time_scale debe ser mayor que 0")
    check(sim.sample > 0, "[execution] sample debe ser mayor que 0")
    for label, value in (
        ("[execution] sample", sim.sample),
        ("[demand] rate_interval", sim.rate_interval),
    ):
        check(abs(value / DT - round(value / DT)) < 1e-9, f"{label} debe ser múltiplo de {DT} s")
    check(sim.n_samples >= 2, "[execution] la corrida debe cubrir al menos dos muestras (run × time_scale ≥ 2 × sample)")
    check(opts.replicas >= 1, "[execution] replicas debe ser al menos 1")
    check(opts.workers >= 0, "[execution] workers no puede ser negativo (0 = automático)")
    check(opts.seed is None or 0 <= opts.seed < 2**63, "[execution] seed debe ser un entero no negativo")
    check(bool(opts.output_dir.strip()), "[output] dir no puede estar vacío")


def _validate_speed_bump(sim: SimConfig, check) -> None:
    for spec in sim.specs:
        check(spec.speed_bump_kmh is None or spec.speed_bump_kmh > 0,
              f"[vehicles.{spec.key}] speed_bump_kmh debe ser mayor que 0 km/h")  # fmt: skip
    positions = [b.position for b in sim.speed_bumps]
    check(all(0 < p < sim.length for p in positions),
          f"[speed_bump] position debe estar entre 0 y {sim.length:g} m ([road] length), sin incluirlos (cada tope)")  # fmt: skip
    check(len(set(positions)) == len(positions), "[speed_bump] position: hay topes en la misma posición")
    for bump in sim.speed_bumps:
        where = f"[speed_bump] a {bump.position:g} m"
        lanes = bump.lanes or ()
        for lane in lanes:
            check(0 <= lane < sim.lanes, f"{where} lanes: cada carril debe estar entre 0 y {sim.lanes - 1}")
        check(len(set(lanes)) == len(lanes), f"{where} lanes: hay carriles repetidos")
        check(bump.lanes is None or len(lanes) > 0, f"{where} lanes no puede estar vacía (sin ella, todos los carriles)")
        if bump.pedestrian:
            check(bump.pedestrian_time >= DT and abs(bump.pedestrian_time / DT - round(bump.pedestrian_time / DT)) < 1e-9,
                  f"{where} pedestrian_time debe ser un múltiplo de {DT} s, al menos {DT} s")  # fmt: skip
            _check_rate(f"{where} pedestrian_crossing", bump.pedestrian_crossing)


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
    for where, lt in [("[traffic_light]", sim.exit_light), *((f"[traffic_light] a {e.position:g} m", e)
                                                                for e in sim.extra_lights)]:  # fmt: skip
        for lane in lt.free_lanes:
            check(0 <= lane < sim.lanes, f"{where} free_lanes: cada carril debe estar entre 0 y {sim.lanes - 1}")
        check(len(set(lt.free_lanes)) == len(lt.free_lanes), f"{where} free_lanes: hay carriles repetidos")
        for lane in lt.free_lanes:
            check(lane not in range(sim.lanes) or lane in sim.reserved_lanes,
                  f"{where} free_lanes: el carril {lane} no es exclusivo de ningún tipo; free_lanes solo aplica a los "
                  "tipos con carril exclusivo ([vehicles.<clave>] lane y exclusive = true)")  # fmt: skip
    lo, hi = sim.bottleneck_zone()
    check(0 <= lo < hi <= sim.length, f"[bottleneck] stop_zone requiere 0 ≤ inicio < fin ≤ {sim.length:g} ([road] length)")


def _validate_spec(s: VehicleSpec, label: str, check) -> None:
    check(bool(s.name.strip()), f"{label} name no puede estar vacío")
    check(s.speed_std >= 0, f"{label} speed_kmh: std no puede ser negativa")
    lo = s.speed_kmh if s.speed_min is None else s.speed_min
    hi = s.speed_kmh if s.speed_max is None else s.speed_max
    check(0 < lo <= s.speed_kmh <= hi, f"{label} speed_kmh: debe cumplirse 0 < min ≤ mean ≤ max")
    check(s.slowest_kmh > 0, f"{label} speed_kmh: la velocidad mínima (mean − 3·std sin min) debe ser mayor que 0")
    check(0 < s.gap_stop <= s.gap_run, f"{label} requiere 0 < gap_stop ≤ gap_run")
    check(s.time_headway is None or 0 < s.time_headway <= 5, f"{label} time_headway debe estar entre 0 y 5 s")
    check(1 <= s.pax_min <= s.pax_max <= 255, f"{label} pax: debe cumplirse 1 ≤ min ≤ max ≤ 255")
    check(s.pax_min <= s.pax_mean <= s.pax_max, f"{label} pax: mean debe estar entre min y max")
    check(s.pax_std >= 0, f"{label} pax: std no puede ser negativa")
    check(0 <= s.cargo_prob <= 1, f"{label} cargo_prob debe estar entre 0 y 1")
    check(s.length_std >= 0, f"{label} length: std no puede ser negativa")
    lo = s.length if s.length_min is None else s.length_min
    hi = s.length if s.length_max is None else s.length_max
    check(0 < lo <= s.length <= hi, f"{label} length: debe cumplirse 0 < min ≤ mean ≤ max")
    check(s.shortest > 0, f"{label} length: el largo mínimo (mean − 3·std sin min) debe ser mayor que 0")


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
