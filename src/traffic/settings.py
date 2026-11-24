"""Reading and validation of a run's TOML configuration file."""

from __future__ import annotations

import json
import re
import tomllib
from dataclasses import dataclass, fields, replace
from datetime import datetime
from pathlib import Path

from traffic import __version__
from traffic.config import (
    DEFAULT_RATES, DEFAULT_SPECS, DT, FUELS, MAX_TYPES, PEDESTRIAN_RATE, PEDESTRIAN_TIME, POLLUTANTS, Behavior,
    Bottleneck, Light, Rate, SimConfig, SpeedBump, VehicleSpec,
)
from traffic.emission_sets import EMISSION_SETS

CONFIG_NAME = "config.toml"
DEFAULT_RUN_NAME = "run"
# Numeric parameters of a vehicle: required for new types.
_SPEC_FIELDS = {
    "gap_run": float, "gap_stop": float,
}  # fmt: skip
# {min, max, mean, std} dictionaries: passengers per vehicle ([vehicles.<key>] pax), maximum speed, length
# and arrival rate.
_PAX_KINDS = {"min": int, "max": int, "mean": float, "std": float}
_RATE_KINDS = dict.fromkeys(("min", "max", "mean", "std"), float)
_FLOAT_DIST_KINDS = _RATE_KINDS  # maximum speed and length
# VehicleSpec fields that come from [vehicles.<key>] length.
_LENGTH_FIELDS = {"mean": "length", "std": "length_std", "min": "length_min", "max": "length_max"}
# VehicleSpec fields that come from [vehicles.<key>] speed_kmh.
_SPEED_FIELDS = {"mean": "speed_kmh", "std": "speed_std", "min": "speed_min", "max": "speed_max"}
# Lane change thresholds a type can set for itself; without them, those of [behavior].
_LANE_CHANGE_FIELDS = ("lookahead", "min_advantage", "lane_change_cooldown")
# Gradual acceleration and braking (m/s²); without them, instantaneous speed changes.
_DYNAMICS_FIELDS = ("accel", "decel", "time_headway")
# Halts at [bottleneck] for each type: probability and duration.
_BOTTLENECK_FIELDS = ("bottleneck_prob", "bottleneck_time_mean", "bottleneck_time_std")
# Keys [bottleneck] no longer accepts and where they moved to (for the error message).
_MOVED_BOTTLENECK = {
    "[bottleneck] stop_prob": "bottleneck_prob", "[bottleneck] stop_time_mean": "bottleneck_time_mean",
    "[bottleneck] stop_time_std": "bottleneck_time_std", "[bottleneck] stop_types": "bottleneck_prob",
}  # fmt: skip
# Passengers of a type that only carries freight (cargo_prob = 1): its pax key is optional.
_CARGO_PAX = {"pax_min": 1, "pax_max": 1, "pax_mean": 1.0, "pax_std": 0.0}
# VehicleSpec defaults (with slots, VehicleSpec.<field> is a descriptor, not the value).
_SPEC_DEFAULTS = {f.name: f.default for f in fields(VehicleSpec)}
_BUILTIN = {spec.key: (spec, rate) for spec, rate in zip(DEFAULT_SPECS, DEFAULT_RATES)}
_REQUIRED = object()
FLOATS = "float_o_lista"
"""Tipo de clave que acepta un número o una lista no vacía de números (se lee como tupla)."""
INTS = "lista_de_enteros"
_KIND_NAMES = {float: "a number", int: "an integer", str: "a string", bool: "true or false",
               FLOATS: "a number or a list of numbers", INTS: "a list of integers"}  # fmt: skip


VIDEO_FORMATS = ("webm", "mp4", "gif")


class ConfigError(ValueError):
    """Error in the contents of the configuration file."""


@dataclass(frozen=True, slots=True)
class RunOptions:
    """Run and output parameters that do not affect the model."""

    replicas: int = 8
    workers: int = 0  # 0 = min(replicas, available CPUs)
    seed: int | None = None  # None = random (recorded in the copy of the configuration)
    output_dir: str = "results"
    name: str = ""  # suffix of the results folder; the name given on the command line takes priority
    series_csv: bool = True  # genera series.csv
    emisiones_csv: bool = True  # writes emissions_by_position.csv (if some type emits)
    progress: bool = True
    animation: bool = False  # also writes the space-time diagram and the movement video
    mobility_plot: bool = True  # writes passenger_mobility.png
    passengers_plot: bool = True  # writes passenger_distribution.png (if some type carries passengers)
    emissions_plot: bool = True  # writes emissions_by_position.png (if some type emits)


@dataclass(frozen=True, slots=True)
class AnimationOptions:
    """Window and pace of the movement visualization ([animation]); they do not affect the model."""

    start: float = 0.0  # simulated s
    duration: float | None = None  # simulated s; None = 2 traffic light cycles
    speed: float = 4.0  # simulated s per s of video
    fps: int = 20
    replica: int = 1  # replica shown (1 = the first)
    format: str = "webm"  # "webm" (VP9) | "mp4" (H.264) | "gif"

    def window(self, sim: SimConfig) -> tuple[float, float]:
        """(start, end) in simulated s, not past the end of the run."""
        duration = self.duration if self.duration is not None else 2 * sim.flow_window
        return self.start, min(self.start + duration, sim.sim_seconds)


@dataclass(frozen=True, slots=True)
class Settings:
    sim: SimConfig
    run: RunOptions
    path: Path
    text: str  # original contents of the file
    notices: tuple[str, ...] = ()  # notices for the run's header
    animation: AnimationOptions = AnimationOptions()


def project_root() -> Path:
    """Project root (where pyproject.toml is); if the package is not installed in editable mode, the current directory."""
    root = Path(__file__).resolve().parents[2]
    if (root / "pyproject.toml").is_file() and (root / "src" / "traffic").is_dir():
        return root
    return Path.cwd()


def default_config_path() -> Path:
    return project_root() / CONFIG_NAME


def resolve_output_dir(output_dir: str) -> Path:
    """Relative output paths are taken from the project root."""
    path = Path(output_dir).expanduser()
    return path if path.is_absolute() else project_root() / path


class _Reader:
    """Read typed keys and remember which ones were used, to detect unknown keys."""

    def __init__(self, data: dict):
        self.data = data
        self.used: set[tuple[str, ...]] = set()

    def get(self, keypath: tuple[str, ...], kind: type, default):
        *sections, key = keypath
        table = self.data
        for i, section in enumerate(sections):
            table = table.get(section, {})
            if not isinstance(table, dict):
                raise ConfigError(f"[{'.'.join(sections[: i + 1])}] must be a section")
        self.used.add(keypath)
        if key not in table:
            if default is _REQUIRED:
                raise ConfigError(f"missing required key {_label(keypath)}")
            return default
        value = table[key]
        if kind is FLOATS:
            items = value if type(value) is list else [value]
            if not items or any(type(v) not in (int, float) for v in items):
                raise ConfigError(f"{_label(keypath)} must be {_KIND_NAMES[kind]}; got {value!r}")
            floats = tuple(float(v) for v in items)
            return floats if type(value) is list else floats[0]
        if kind is INTS:
            if type(value) is not list or any(type(v) is not int for v in value):
                raise ConfigError(f"{_label(keypath)} must be {_KIND_NAMES[kind]}; got {value!r}")
            return tuple(value)
        if kind is float and type(value) is int:
            value = float(value)
        if type(value) is not kind:  # exact type(): a bool does not count as an integer
            raise ConfigError(f"{_label(keypath)} must be {_KIND_NAMES[kind]}; got {value!r}")
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
        raise ConfigError(f"invalid TOML: {exc}") from exc
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
    # [behavior] congestion_factor was removed (it barely changed the results and duplicated what gradual
    # dynamics and the chain reaction already do); older copies include it. A list still counts for the
    # number of lanes, so that those copies load with the same lanes.
    congestion = r.get(("behavior", "congestion_factor"), FLOATS, None)
    lanes, speed_limit, notices = _lanes(r, occupancy, d.lanes, exit_capacity, exit_storage, congestion)
    notices += bump_notices
    if congestion is not None:
        notices += ("Warning: [behavior] congestion_factor was removed and is ignored (it does not change the speed); you can "
                    "remove it from the file",)  # fmt: skip
    # [demand] slow_lane was removed; older copies include it with "right", which is what is done now.
    slow_lane = r.get(("demand", "slow_lane"), str, None)
    if slow_lane is not None:
        if slow_lane != "right":
            raise ConfigError('[demand] slow_lane was removed: types that do not change lanes and have no lane use '
                              'the rightmost free lane; for another lane use lane in [vehicles.<key>]')
        notices += ("Warning: [demand] slow_lane was removed and is ignored; you can remove it from the file",)
    # [output] csv was renamed series_csv; older copies include it and keep their value.
    legacy_csv = r.get(("output", "csv"), bool, None)
    if legacy_csv is not None:
        notices += ("Warning: [output] csv was renamed series_csv; its value is used, but change it in the file",)
    # [output] show was removed (it opened the matplotlib window); older copies include it.
    if r.get(("output", "show"), bool, None) is not None:
        notices += ("Warning: [output] show was removed and is ignored; you can remove it from the file",)
    # [output] saturation_threshold was removed (it only marked the plot); older copies include it.
    if r.get(("output", "saturation_threshold"), float, None) is not None:
        notices += ("Warning: [output] saturation_threshold was removed and is ignored; you can remove it from the file",)
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
        light_offset=exit_light.offset,
        extra_lights=extra_lights,
        # The distribution is only needed with some variable rate; if all are fixed, their value is enough.
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
        msg = "unknown keys: " + ", ".join(unknown)
        moved = [u for u in unknown if u in _MOVED_BOTTLENECK]
        if moved:
            msg += ("; [bottleneck] only keeps stop_lanes and stop_zone: the probability and duration go in each "
                    "[vehicles.<key>] as " + ", ".join(sorted({_MOVED_BOTTLENECK[u] for u in moved})))
        if any(re.fullmatch(r"\[vehicles\.[^\]]+\] pax_(min|max|mean|std)", u) for u in unknown):
            msg += "; passengers now go in a dictionary: pax = {min = 1, max = 6, mean = 1.5, std = 0.8}"
        if any(re.fullmatch(r"\[vehicles\.[^\]]+\] length_(min|max|std)", u) for u in unknown):
            msg += "; length now goes in a dictionary: length = {min = 8, max = 16, mean = 10, std = 2}"
        if any(re.fullmatch(r"\[behavior\] reaction_(min|max|mean|std)", u) for u in unknown):
            msg += "; reaction now goes in a dictionary: reaction = {min = 0.7, max = 3, mean = 1.5, std = 0.5}"
        orphan_rates = [m for u in unknown if (m := re.match(r"\[demand\.(.+)_rate\] |\[demand\] (.+)_rate$", u))]
        if orphan_rates:
            key = orphan_rates[0].group(1) or orphan_rates[0].group(2)
            msg += f" (for a new vehicle also define its [vehicles.{key}] section)"
        raise ConfigError(msg)
    validate_config(sim, opts)
    validate_animation(anim, sim, opts.replicas)
    # Warn only about emissions written in the file (the built-in car and bus come with coefficients).
    written = {key for key, value in r.data.get("vehicles", {}).items() if isinstance(value, dict) and "emissions" in value}
    silent = [sp.name for k, sp in enumerate(sim.specs)
              if sp.key in written and sp.emissions and not sp.emits and sim.rates[k] > 0]  # fmt: skip
    if silent:
        notices += (f"Warning: without accel and decel the emissions of {', '.join(silent)} are not computed (the model uses the "
                    "real acceleration)",)  # fmt: skip
    unpriced = sorted({sp.fuel for k, sp in enumerate(sim.specs) if sp.burns and sim.rates[k] > 0
                       and sim.price(sp.fuel) is None})  # fmt: skip
    if unpriced:
        notices += (f"Warning: no price in [fuel] price for {', '.join(unpriced)}: its consumption is only given in litres",)
    return Settings(sim=sim, run=opts, path=path, text=text, notices=notices, animation=anim)


def _emissions(r: _Reader, key: str, base: VehicleSpec | None) -> tuple[tuple, str]:
    """[vehicles.<key>.emissions]: (coefficients, set they come from). `source` takes those of a named set
    (`EMISSION_SETS`) for the type's key or for `source_type`; on top of that, per pollutant, the hand-written
    [f1..f6] coefficients of the model of Int Panis et al. (2006) and, optionally, `<pollutant>_decel`
    for a < −0.5 m/s² (by default, the same ones). If the section is present, it replaces that of the built-in type;
    if empty, the type does not emit."""
    table = r.data.get("vehicles", {}).get(key, {}).get("emissions")
    if table is None:
        return (base.emissions, base.emission_source) if base else ((), "")
    label = f"[vehicles.{key}.emissions]"
    if not isinstance(table, dict):
        raise ConfigError(f"[vehicles.{key}] emissions must be a section {label}")
    section = ("vehicles", key, "emissions")
    source = r.get(section + ("source",), str, None)
    entry = r.get(section + ("source_type",), str, None)
    found: dict[str, tuple] = {}
    if source is not None:
        if source not in EMISSION_SETS:
            raise ConfigError(f"{label} source: there is no set {source!r}; the sets are {', '.join(EMISSION_SETS)}")
        entry = entry or key
        if entry not in EMISSION_SETS[source]:
            raise ConfigError(f"{label} the set {source!r} has no type {entry!r}"
                              f"{'' if key == entry else ' (source_type)'}; it has {', '.join(EMISSION_SETS[source])} "
                              "(choose one with source_type)")  # fmt: skip
        found = {pol: (acc, dec) for pol, acc, dec in EMISSION_SETS[source][entry]}
    elif entry is not None:
        raise ConfigError(f"{label} source_type requires source (the coefficient set)")
    out = []
    for pol in POLLUTANTS:
        coefs = {}
        for name in (pol, f"{pol}_decel"):
            value = r.get(section + (name,), FLOATS, None)
            if value is not None and (not isinstance(value, tuple) or len(value) != 6):
                raise ConfigError(f"{label} {name} must be a list of 6 numbers [f1, f2, f3, f4, f5, f6]")
            coefs[name] = value
        acc, dec = coefs[pol], coefs[f"{pol}_decel"]
        if acc is None and pol in found:  # from the set; a hand-written _decel replaces it
            acc, dec = found[pol][0], dec or found[pol][1]
        if acc is None:
            if dec is not None:
                raise ConfigError(f"{label} {pol}_decel requires {pol} (the coefficients for a ≥ −0.5 m/s²)")
            continue
        out.append((pol, acc, dec or acc))
    return tuple(out), source or ""


def _zone(value):
    """[start, end] of [bottleneck] stop_zone: a list of two numbers."""
    if value is None:
        return None
    if not isinstance(value, tuple) or len(value) != 2:
        raise ConfigError(f"[bottleneck] stop_zone must be a list [start, end] in m; got {value!r}")
    return value


def validate_animation(anim: AnimationOptions, sim: SimConfig, replicas: int) -> None:
    def check(ok: bool, msg: str) -> None:
        if not ok:
            raise ConfigError(msg)

    check(0 <= anim.start < sim.sim_seconds,
          f"[animation] start must be between 0 and {sim.sim_seconds:g} simulated s (the run duration)")  # fmt: skip
    check(anim.duration is None or anim.duration > 0, "[animation] duration must be greater than 0")
    check(anim.speed > 0, "[animation] speed must be greater than 0")
    check(1 <= anim.fps <= 60, "[animation] fps must be between 1 and 60")
    check(anim.format in VIDEO_FORMATS, f"[animation] format must be {', '.join(map(repr, VIDEO_FORMATS))}")
    check(1 <= anim.replica <= replicas, f"[animation] replica must be between 1 and {replicas} ([execution] replicas)")


def _lanes(r: _Reader, occupancy, default: int, exit_capacity=0.0, exit_storage=60.0, congestion=None):
    """Number of lanes: the length of the per-lane lists ([road] max_line_speed, [initial] occupancy,
    [exit] capacity and storage and, in older copies, [behavior] congestion_factor), which must match. A
    single number applies to all lanes; without any list, `default` lanes are used.

    `[road] lanes` is deprecated: it is only accepted to replicate older copies, and in that case
    the lists are adjusted to it (extra lanes use the last value; surplus values are ignored).
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
        notices = ("Warning: [road] lanes is obsolete; the number of lanes comes from the length of "
                   "max_line_speed (or of the other per-lane lists)",)  # fmt: skip
    elif lengths:
        if len(set(lengths.values())) > 1:
            detail = ", ".join(f"{name} has {n}" for name, n in lengths.items())
            raise ConfigError(f"the per-lane lists must have the same length (one value per lane): {detail}")
        lanes = next(iter(lengths.values()))
    else:
        lanes = default
    return lanes, speed, notices


def _parse_vehicles(r: _Reader) -> tuple[tuple[VehicleSpec, ...], tuple[Rate, ...]]:
    """Vehicle types: the built-in ones (car, bike, bus) and each new [vehicles.<key>].

    The built-in ones take their defaults for the missing keys. A new type
    must define all numeric parameters, its passengers `pax = {min, max, mean, std}` and its
    rate `[demand] <key>_rate`; `name`
    (default: the key), `lane_change` (default: true), `lane` (fixed lane of a type that
    does not change lanes; default: the free right lane), `exclusive` (reserves that lane
    for the types that have it as fixed; default: false), the stop before the traffic light
    (`stop_position`, `stop_time_mean`, `stop_time_std`; default: no stop) and `abreast`
    (how many stop side by side in one lane; default: 1) are optional, as are
    overtaking within the lane (`pass_in_lane`; default: false), gradual acceleration and
    braking (`accel`, `decel` in m/s²; default: instantaneous), the car-following headway when moving
    (`time_headway` in s; default: fixed gap_run), the speed over the bump
    (`speed_bump_kmh`; default: no braking), aggressive overtaking (`overtake`; default: false), the type's own
    lane change thresholds (`lookahead`, `min_advantage`, `lane_change_cooldown`; default: those of [behavior]),
    and the probability of carrying freight (`cargo_prob`; default: 0; with 1, `pax` is
    optional). The maximum speed (`speed_kmh`) and the length (`length`) are {min, max, mean,
    std} dictionaries: a normal truncated to [min, max] drawn for each vehicle; with std = 0, all have mean.
    """
    table = r.data.get("vehicles", {})
    if not isinstance(table, dict):
        raise ConfigError("[vehicles] must be a section")
    for key, value in table.items():
        if not isinstance(value, dict):
            raise ConfigError(f"[vehicles] {key} must be a section [vehicles.{key}]")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", key):
            raise ConfigError(f"[vehicles.{key}] the key only accepts letters, digits, '_' and '-'")

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
                    raise ConfigError(f"missing required key [vehicles.{key}] {name}")
            pax = _dist(r, section + ("pax",), _PAX_KINDS, "{min = 1, max = 6, mean = 1.5, std = 0.8}")
            if pax is not None:
                values.update({f"pax_{f}": v for f, v in pax.items()})
            elif base is not None:
                values.update({f: getattr(base, f) for f in _CARGO_PAX})
            elif cargo_prob == 1:  # without passengers, there is no need to describe them
                values.update(_CARGO_PAX)
            else:
                raise ConfigError(f"missing required key [vehicles.{key}] pax")
            rate = _rate(r, key, base_rate)
        except ConfigError as exc:
            if base is None and str(exc).startswith("missing"):
                raise ConfigError(f"{exc} (new vehicle types must define all their parameters)") from None
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
    """[traffic_light]: one traffic light at the end of the segment (one section) or several ([[traffic_light]], one per
    light). Each has `position` (m from the start; without it or equal to `length`, the one at the end of the segment),
    `enabled`, `red`, `green`, `yellow`, `start_phase`, `free_lanes` and, if it is a pedestrian light, `pedestrian` and
    `pedestrian_crossing`. Returns (the one at the end of the segment, the intermediate ones); if there are lights but
    none at the end of the segment, that one is disabled."""
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
        raise ConfigError("[traffic_light] must be a section or a list of sections ([[traffic_light]])")
    lights = []
    for reader in readers:
        def read(key, kind, default, reader=reader):
            return reader.get(("traffic_light", key), kind, default)

        position = read("position", float, None)
        pedestrian = read("pedestrian", bool, False)
        start_phase = read("start_phase", str, None)
        if pedestrian and start_phase is not None:
            raise ConfigError("[traffic_light] start_phase does not apply to a light with pedestrian = true (it always "
                              "starts green): remove it")  # fmt: skip
        offset = read("offset", float, None)
        if pedestrian and offset is not None:
            raise ConfigError("[traffic_light] offset does not apply to a light with pedestrian = true (it has no cycle): "
                              "remove it")  # fmt: skip
        lights.append(Light(
            position=length if position is None else position,
            red=read("red", float, d.red), green=read("green", float, d.green),
            yellow=read("yellow", float, d.yellow), start_phase=start_phase or d.start_phase,
            enabled=read("enabled", bool, d.traffic_light), free_lanes=read("free_lanes", INTS, d.free_lanes),
            pedestrian=pedestrian, pedestrian_crossing=_crossing(reader, ("traffic_light",)),
            offset=d.light_offset if offset is None else offset,
        ))  # fmt: skip
    for i, sub in checks:
        unknown = sub.unknown()
        if unknown:
            raise ConfigError(f"unknown keys in traffic light {i} of [[traffic_light]]: "
                              + ", ".join(u.replace("[traffic_light] ", "") for u in unknown))  # fmt: skip
    at_exit = [lt for lt in lights if abs(lt.position - length) < 1e-9]
    if len(at_exit) > 1:
        raise ConfigError(f"[[traffic_light]] there are {len(at_exit)} lights at the end of the segment ({length:g} m, or without "
                          "position): there can be only one")  # fmt: skip
    extra = tuple(sorted((lt for lt in lights if lt not in at_exit), key=lambda lt: lt.position))
    return (at_exit[0] if at_exit else replace(d.exit_light, enabled=False)), extra


def _speed_bumps(r: _Reader) -> tuple[tuple[SpeedBump, ...], tuple[str, ...]]:
    """[speed_bump]: one bump (one section) or several ([[speed_bump]], one per bump), each with its `position` (m from
    the start, a number), `lanes`, `pedestrian`, `pedestrian_crossing` and `pedestrian_time`. Without the section there
    are no bumps; neither with a section without `position` or any other key. Returns (bumps, notices). `position` as a
    list (the old format) is accepted with a notice: each position is a bump with the section's other keys."""
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
        raise ConfigError("[speed_bump] must be a section or a list of sections ([[speed_bump]])")
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
                raise ConfigError(f"[[speed_bump]] position of bump {n} must be a number (one bump per section); "
                                  "for several bumps, one [[speed_bump]] section each")  # fmt: skip
            notices = ("Warning: [speed_bump] position as a list was replaced by one [[speed_bump]] section per bump; "
                       "each position is used as a bump with the same keys, but change it in the file",)
            positions = value
        else:
            positions = () if value is None else (value,)
        if not positions:
            if lanes is not None or pedestrian:  # keys of a bump without where it is
                where = "[speed_bump]" if legacy else f"[[speed_bump]] of bump {n}"
                key = "lanes" if lanes is not None else "pedestrian"
                raise ConfigError(f"{where}: {key} requires position (where the bump is)")
            if not legacy:
                raise ConfigError(f"[[speed_bump]] position is missing in bump {n}")
        bumps += [SpeedBump(p, lanes, pedestrian, rate, time) for p in positions]
    for i, sub in checks:
        unknown = sub.unknown()
        if unknown:
            raise ConfigError(f"unknown keys in bump {i} of [[speed_bump]]: "
                              + ", ".join(u.replace("[speed_bump] ", "") for u in unknown))  # fmt: skip
    return tuple(sorted(bumps, key=lambda b: b.position)), notices


def _crossing(r: _Reader, section: tuple[str, ...]) -> Rate:
    """`pedestrian_crossing` of a section ([speed_bump] or a [[traffic_light]]): the appearance of pedestrians, a
    {min, max, mean, std} dictionary in pedestrians/min. Without it, PEDESTRIAN_RATE. Validated only if the section has
    pedestrian = true (in `validate_config`)."""
    values = _dist(r, (*section, "pedestrian_crossing"), _RATE_KINDS,
                   "{min = 0.5, max = 3, mean = 1.5, std = 1} (pedestrians/min)")  # fmt: skip
    return PEDESTRIAN_RATE if values is None else Rate(**values)


def _fuel(r: _Reader, d: SimConfig) -> dict:
    """[fuel]: price per litre of each fuel (`price = {gasolina = 24.0, ...}`) and its currency."""
    table = r.data.get("fuel", {})
    if isinstance(table, dict) and "price" in table and not isinstance(table["price"], dict):
        raise ConfigError(f"[fuel] price must be a dictionary per fuel, e.g. price = {{gasolina = 24.0}}; "
                          f"got {table['price']!r}")  # fmt: skip
    prices = tuple((f, v) for f in FUELS if (v := r.get(("fuel", "price", f), float, None)) is not None)
    return {"fuel_price": prices, "currency": r.get(("fuel", "currency"), str, d.currency)}


def _rate(r: _Reader, key: str, default) -> Rate:
    """[demand] <key>_rate: a {min, max, mean, std} dictionary in veh/min. Built-in types
    that do not define it use their fixed default rate."""
    name = f"{key}_rate"
    if not isinstance(r.data.get("demand", {}), dict):
        raise ConfigError("[demand] must be a section")
    values = _dist(r, ("demand", name), _RATE_KINDS, "{min = 10, max = 20, mean = 15, std = 5} (veh/min)")
    if values is None:
        if default is _REQUIRED:
            raise ConfigError(f"missing required key [demand] {name}")
        return Rate.fixed(default)
    rate = Rate(**values)
    _check_rate(f"[demand] {name}", rate)
    return rate


def _reaction(r: _Reader) -> dict[str, float]:
    """[behavior] reaction: {min, max} (uniform) or {min, max, mean, std} (normal truncated to [min, max]).
    Without the key, the Behavior defaults."""
    behavior = r.data.get("behavior", {})
    value = behavior.get("reaction") if isinstance(behavior, dict) else None
    pair = {"mean", "std"} & set(value) if isinstance(value, dict) else set()
    if len(pair) == 1:
        raise ConfigError("[behavior] reaction: mean and std go together (with them, a normal truncated to [min, max]; "
                          "without them, uniform in [min, max])")  # fmt: skip
    kinds = dict.fromkeys(("min", "max", *(("mean", "std") if pair else ())), float)
    values = _dist(r, ("behavior", "reaction"), kinds, "{min = 0.7, max = 3, mean = 1.5, std = 0.5}")
    return {} if values is None else {f"reaction_{f}": v for f, v in values.items()}


def _dist(r: _Reader, keypath: tuple[str, ...], kinds: dict[str, type], example: str) -> dict | None:
    """Dictionary at `keypath` with all the keys of `kinds`; None if it is missing."""
    *sections, key = keypath
    table = r.data
    for section in sections:
        table = table.get(section, {})
    if key not in table:
        return None
    if not isinstance(table[key], dict):
        raise ConfigError(f"{_label(keypath)} must be a dictionary, e.g. {key} = {example}; got {table[key]!r}")
    return {f: r.get(keypath + (f,), kind, _REQUIRED) for f, kind in kinds.items()}


def _check_rate(label: str, rate: Rate) -> None:
    if not 0 <= rate.min <= rate.mean <= rate.max:
        raise ConfigError(f"{label}: 0 ≤ min ≤ mean ≤ max must hold")
    if rate.std < 0:
        raise ConfigError(f"{label}: std cannot be negative")


def load_settings(path: Path) -> Settings:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ConfigError(f"the configuration file {path} does not exist") from None
    try:
        return parse_settings(text, path)
    except ConfigError as exc:
        raise ConfigError(f"{path}: {exc}") from None


def validate_config(sim: SimConfig, opts: RunOptions) -> None:
    """Validate a run's configuration; raises ConfigError with the failing key."""
    def check(ok: bool, msg: str) -> None:
        if not ok:
            raise ConfigError(msg)

    check(sim.n_types <= MAX_TYPES, f"[vehicles] up to {MAX_TYPES} vehicle types are allowed; there are {sim.n_types}")
    for spec in sim.specs:
        _validate_spec(spec, f"[vehicles.{spec.key}]", check)
        if spec.lane is not None:
            label = f"[vehicles.{spec.key}] lane"
            check(not spec.can_change_lane, f"{label} only applies to types with lane_change = false")
            check(0 <= spec.lane < sim.lanes, f"{label} must be between 0 (right lane) and {sim.lanes - 1}")
        check(not spec.exclusive or spec.lane is not None,
              f"[vehicles.{spec.key}] exclusive = true requires a fixed lane (lane)")  # fmt: skip
        check(1 <= spec.abreast <= 4, f"[vehicles.{spec.key}] abreast must be between 1 and 4")
        check(not spec.pass_in_lane or spec.abreast >= 2,
              f"[vehicles.{spec.key}] pass_in_lane = true requires abreast ≥ 2 (room for two side by side)")  # fmt: skip
        for name in _DYNAMICS_FIELDS:
            value = getattr(spec, name)
            check(value is None or 0 < value <= 20, f"[vehicles.{spec.key}] {name} must be in (0, 20] m/s²")
        overrides = {f: getattr(spec, f) for f in _LANE_CHANGE_FIELDS if getattr(spec, f) is not None}
        for name in ([*overrides, "overtake"] if spec.overtake else overrides):
            check(spec.can_change_lane, f"[vehicles.{spec.key}] {name} only applies to types with lane_change = true")
        for name, value in overrides.items():
            check(value >= 0, f"[vehicles.{spec.key}] {name} cannot be negative")
        if spec.stop_position is not None:
            label = f"[vehicles.{spec.key}]"
            check(spec.longest <= spec.stop_position <= sim.length,
                  f"{label} stop_position must be between {spec.longest:g} (the maximum vehicle length, which enters with "
                  f"its rear at 0) and {sim.length:g} m (the traffic light)")  # fmt: skip
            check(spec.stop_time_mean >= 0 and spec.stop_time_std >= 0,
                  f"{label} stop_time_mean and stop_time_std cannot be negative")  # fmt: skip
            check(spec.stop_time_mean + 6 * spec.stop_time_std <= 3600,
                  f"{label} the stop cannot last more than an hour (stop_time_mean + 6·stop_time_std ≤ 3600 s)")  # fmt: skip
    for k, spec in enumerate(sim.specs):
        check(sim.rates[k] == 0 or bool(sim.allowed_lanes(k)),
              f"[vehicles.{spec.key}] has no lane left: all are exclusive to other types "
              f"(reserved lanes: {', '.join(map(str, sorted(sim.reserved_lanes)))})")  # fmt: skip
    names = [spec.name for spec in sim.specs]
    dupes = sorted({n for n in names if names.count(n) > 1})
    check(not dupes, f"[vehicles] names must be different; repeated: {', '.join(dupes)}")
    b = sim.behavior
    check(0 < b.reaction_min <= b.reaction_max <= 600, "[behavior] reaction: 0 < min ≤ max ≤ 600 must hold")
    check(b.reaction_mean is None or b.reaction_std is not None, "[behavior] reaction: mean requires std")
    check(b.reaction_std is None or b.reaction_std >= 0, "[behavior] reaction: std cannot be negative")
    check(b.reaction_mean is None or b.reaction_min <= b.reaction_mean <= b.reaction_max,
          "[behavior] reaction: mean must be between min and max")  # fmt: skip
    check(
        0 < b.lane_change_min <= b.lane_change_max <= 600,
        "[behavior] requires 0 < lane_change_min ≤ lane_change_max ≤ 600",
    )
    check(0 < b.lane_change_speed_factor <= 1, "[behavior] lane_change_speed_factor must be in (0, 1]")
    check(0 <= b.leader_slowdown < 1, "[behavior] leader_slowdown must be in [0, 1)")
    check(0 < b.yellow_speed_factor <= 1, "[behavior] yellow_speed_factor must be in (0, 1]")
    for name in ("lane_change_cooldown", "lookahead", "min_advantage", "release_space", "yellow_approach"):
        check(getattr(b, name) >= 0, f"[behavior] {name} cannot be negative")
    check(b.lane_change_interval >= DT, f"[behavior] lane_change_interval must be ≥ {DT} s")

    _validate_bottleneck(sim, check)
    _validate_speed_bump(sim, check)
    for spec in sim.specs:
        check(spec.fuel is None or spec.fuel in FUELS,
              f"[vehicles.{spec.key}] fuel must be one of {', '.join(FUELS)}; got {spec.fuel!r}")  # fmt: skip
        check(spec.mass_kg is None or spec.mass_kg > 0, f"[vehicles.{spec.key}] mass_kg must be greater than 0 kg")
    for fuel, price in sim.fuel_price:
        check(price >= 0, f"[fuel] price: the price of {fuel} cannot be negative")
    check(bool(sim.currency.strip()), "[fuel] currency cannot be empty")

    longest = max(s.longest + s.gap_run for s in sim.specs)
    check(sim.length >= 2 * longest, f"[road] length must be at least {2 * longest:g} m (twice the longest vehicle with its gap)")
    check(1 <= sim.lanes <= 50, "between 1 and 50 lanes are allowed (length of the per-lane lists)")
    check(all(v > 0 for v in sim.lane_max_kmh), "[road] max_line_speed: each value must be greater than 0")
    for where, lt in [("[traffic_light]", sim.exit_light), *((f"[traffic_light] at {e.position:g} m", e)
                                                                for e in sim.extra_lights)]:  # fmt: skip
        check(lt.red >= 0, f"{where} red cannot be negative")
        check(lt.yellow >= 0, f"{where} yellow cannot be negative")
        # With all three at 0 there is no traffic light (like enabled = false); otherwise green is required. The pedestrian
        # light uses green as the minimum green between two crossings (it may be 0) and red as how long the crossing lasts.
        check(not lt.active or lt.pedestrian or lt.green > 0,
              f"{where} green must be greater than 0 (or red, green and yellow at 0, or enabled = false, for no "
              "traffic light)")  # fmt: skip
        if lt.pedestrian and lt.enabled:
            check(lt.red > 0, f"{where} red must be greater than 0 with pedestrian = true (it is how long the crossing lasts)")
            _check_rate(f"{where} pedestrian_crossing", lt.pedestrian_crossing)
        check(lt.start_phase in ("red", "green"), f'{where} start_phase must be "red" or "green"')
        for name, value in (("red", lt.red), ("green", lt.green), ("yellow", lt.yellow), ("offset", lt.offset)):
            check(abs(value / DT - round(value / DT)) < 1e-9, f"{where} {name} must be a multiple of {DT} s")
        check(lt.offset == 0 or 0 < lt.offset < lt.cycle,
              f"{where} offset must be between 0 and the cycle (red + green + yellow), excluding it")  # fmt: skip
    # A disabled traffic light (enabled = false, or its phases at 0) is not on the street: its position is not checked.
    positions = [e.position for e in sim.extra_lights if e.active]
    check(all(0 < p < sim.length for p in positions),
          f"[[traffic_light]] position: each intermediate light must be between 0 and {sim.length:g} m ([road] "
          "length), both excluded (the one at the end of the segment has no position)")  # fmt: skip
    check(len(set(positions)) == len(positions), "[[traffic_light]] position: there are lights at the same position")
    for k, spec in enumerate(sim.specs):
        _check_rate(f"[demand] {spec.key}_rate", sim.rate(k))
    check(sim.rate_interval >= DT, f"[demand] rate_interval must be ≥ {DT} s")
    check(any(rate > 0 for rate in sim.rates), "[demand] at least one vehicle type must have a rate > 0")
    check(all(0 <= v <= 1 for v in sim.lane_initial_occupancy), "[initial] occupancy: each value must be in [0, 1]")
    check(all(v >= 0 for v in sim.lane_exit_capacity), "[exit] capacity: each value must be ≥ 0 veh/min (0 = no exit queue)")
    check(all(v > 0 for v in sim.lane_exit_storage), "[exit] storage: each value must be > 0 m")
    check(sim.run > 0, "[execution] run must be greater than 0")
    check(sim.time_scale > 0, "[execution] time_scale must be greater than 0")
    check(sim.sample > 0, "[execution] sample must be greater than 0")
    for label, value in (
        ("[execution] sample", sim.sample),
        ("[demand] rate_interval", sim.rate_interval),
    ):
        check(abs(value / DT - round(value / DT)) < 1e-9, f"{label} must be a multiple of {DT} s")
    check(sim.n_samples >= 2, "[execution] the run must cover at least two samples (run × time_scale ≥ 2 × sample)")
    check(opts.replicas >= 1, "[execution] replicas must be at least 1")
    check(opts.workers >= 0, "[execution] workers cannot be negative (0 = automatic)")
    check(opts.seed is None or 0 <= opts.seed < 2**63, "[execution] seed must be a non-negative integer")
    check(bool(opts.output_dir.strip()), "[output] dir cannot be empty")


def _validate_speed_bump(sim: SimConfig, check) -> None:
    for spec in sim.specs:
        check(spec.speed_bump_kmh is None or spec.speed_bump_kmh > 0,
              f"[vehicles.{spec.key}] speed_bump_kmh must be greater than 0 km/h")  # fmt: skip
    positions = [b.position for b in sim.speed_bumps]
    check(all(0 < p < sim.length for p in positions),
          f"[speed_bump] position must be between 0 and {sim.length:g} m ([road] length), both excluded (each bump)")  # fmt: skip
    check(len(set(positions)) == len(positions), "[speed_bump] position: there are bumps at the same position")
    for bump in sim.speed_bumps:
        where = f"[speed_bump] at {bump.position:g} m"
        lanes = bump.lanes or ()
        for lane in lanes:
            check(0 <= lane < sim.lanes, f"{where} lanes: each lane must be between 0 and {sim.lanes - 1}")
        check(len(set(lanes)) == len(lanes), f"{where} lanes: there are repeated lanes")
        check(bump.lanes is None or len(lanes) > 0, f"{where} lanes cannot be empty (without it, all lanes)")
        if bump.pedestrian:
            check(bump.pedestrian_time >= DT and abs(bump.pedestrian_time / DT - round(bump.pedestrian_time / DT)) < 1e-9,
                  f"{where} pedestrian_time must be a multiple of {DT} s, at least {DT} s")  # fmt: skip
            _check_rate(f"{where} pedestrian_crossing", bump.pedestrian_crossing)


def _validate_bottleneck(sim: SimConfig, check) -> None:
    for spec in sim.specs:
        label = f"[vehicles.{spec.key}]"
        check(0 <= spec.bottleneck_prob <= 1, f"{label} bottleneck_prob must be between 0 and 1")
        check(spec.bottleneck_prob == 0 or spec.stop_position is None,
              f"{label} bottleneck_prob does not apply to a type with its own stop (stop_position); only one is allowed")  # fmt: skip
        mean, std = spec.bottleneck_time_mean, spec.bottleneck_time_std
        check(mean >= 0 and std >= 0, f"{label} bottleneck_time_mean and bottleneck_time_std cannot be negative")
        check(mean + 6 * std <= 3600,
              f"{label} the halt cannot last more than an hour (mean + 6·σ ≤ 3600 s)")  # fmt: skip
    for lane in sim.bottleneck.stop_lanes or ():
        check(0 <= lane < sim.lanes, f"[bottleneck] stop_lanes: each lane must be between 0 and {sim.lanes - 1}")
    for where, lt in [("[traffic_light]", sim.exit_light), *((f"[traffic_light] at {e.position:g} m", e)
                                                                for e in sim.extra_lights)]:  # fmt: skip
        for lane in lt.free_lanes:
            check(0 <= lane < sim.lanes, f"{where} free_lanes: each lane must be between 0 and {sim.lanes - 1}")
        check(len(set(lt.free_lanes)) == len(lt.free_lanes), f"{where} free_lanes: there are repeated lanes")
        for lane in lt.free_lanes:
            check(lane not in range(sim.lanes) or lane in sim.reserved_lanes,
                  f"{where} free_lanes: lane {lane} is not exclusive to any type; free_lanes only applies to "
                  "types with an exclusive lane ([vehicles.<key>] lane and exclusive = true)")  # fmt: skip
    lo, hi = sim.bottleneck_zone()
    check(0 <= lo < hi <= sim.length, f"[bottleneck] stop_zone requires 0 ≤ start < end ≤ {sim.length:g} ([road] length)")


def _validate_spec(s: VehicleSpec, label: str, check) -> None:
    check(bool(s.name.strip()), f"{label} name cannot be empty")
    check(s.speed_std >= 0, f"{label} speed_kmh: std cannot be negative")
    lo = s.speed_kmh if s.speed_min is None else s.speed_min
    hi = s.speed_kmh if s.speed_max is None else s.speed_max
    check(0 < lo <= s.speed_kmh <= hi, f"{label} speed_kmh: 0 < min ≤ mean ≤ max must hold")
    check(s.slowest_kmh > 0, f"{label} speed_kmh: the minimum speed (mean − 3·std without min) must be greater than 0")
    check(0 < s.gap_stop <= s.gap_run, f"{label} requires 0 < gap_stop ≤ gap_run")
    check(s.time_headway is None or 0 < s.time_headway <= 5, f"{label} time_headway must be between 0 and 5 s")
    check(1 <= s.pax_min <= s.pax_max <= 255, f"{label} pax: 1 ≤ min ≤ max ≤ 255 must hold")
    check(s.pax_min <= s.pax_mean <= s.pax_max, f"{label} pax: mean must be between min and max")
    check(s.pax_std >= 0, f"{label} pax: std cannot be negative")
    check(0 <= s.cargo_prob <= 1, f"{label} cargo_prob must be between 0 and 1")
    check(s.length_std >= 0, f"{label} length: std cannot be negative")
    lo = s.length if s.length_min is None else s.length_min
    hi = s.length if s.length_max is None else s.length_max
    check(0 < lo <= s.length <= hi, f"{label} length: 0 < min ≤ mean ≤ max must hold")
    check(s.shortest > 0, f"{label} length: the minimum length (mean − 3·std without min) must be greater than 0")


# ------------------------------------------------------------ run folder


@dataclass(frozen=True, slots=True)
class Target:
    """Which configuration to read and under which name to save the run."""

    config: Path  # path of the config.toml
    name: str | None  # name given on the command line (takes priority over [output] name)
    fallback_name: str  # if there is no name on the command line or in [output] name


def resolve_target(arg: str | None) -> Target:
    """Interpret the command's only argument.

    - No argument: config.toml at the project root.
    - An existing folder (e.g. results/<ID>_<name>): its config.toml.
    - Any other text without path separators: the run's name, with the
      config.toml at the project root.
    """
    if arg is None:
        return Target(default_config_path(), None, DEFAULT_RUN_NAME)
    path = Path(arg).expanduser()
    if path.is_dir():
        return Target(path / CONFIG_NAME, None, path.resolve().name)
    if path.is_file():
        if path.name != CONFIG_NAME:
            raise ConfigError(f"the configuration file must be called {CONFIG_NAME}: {path}")
        return Target(path, None, path.resolve().parent.name)
    if "/" in arg or "\\" in arg or arg.endswith(".toml"):
        raise ConfigError(f"the folder {path} does not exist")
    return Target(default_config_path(), arg, DEFAULT_RUN_NAME)


def safe_name(name: str) -> str:
    """Name fit for a folder: letters, digits, '.', '_' and '-'."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or DEFAULT_RUN_NAME


def make_run_dir(base: Path, name: str, now: datetime) -> Path:
    """Create `base/<date-time>_<name>`; if it already exists, add a numeric suffix."""
    stem = f"{now:%Y%m%d-%H%M%S}_{safe_name(name)}"
    run_dir = base / stem
    k = 2
    while run_dir.exists():
        run_dir = base / f"{stem}-{k}"
        k += 1
    run_dir.mkdir(parents=True)
    return run_dir


def config_copy_text(settings: Settings, seed: int, name: str, run_dir: Path, now: datetime) -> str:
    """Copy of the configuration to replicate the run.

    It is the original text with a comment header, plus what is needed for the
    copy to reproduce the run: the seed, if it was random, and the name
    used in [output] name, so that the replica keeps the folder's suffix.
    """
    try:
        shown = run_dir.relative_to(Path.cwd())
    except ValueError:
        shown = run_dir
    header = (
        f"# Copia de la configuración usada en la corrida {run_dir.name}\n"
        f"# ({now:%Y-%m-%d %H:%M:%S}, traffic {__version__}). Para replicarla:\n"
        f"#   uv run traffic {shown}\n\n"
    )
    text = _COPY_HEADER.sub("", settings.text, count=1)  # when replicating a copy, do not stack headers
    if settings.run.seed is None:
        text = _set_key(text, "execution", "seed", seed, f"semilla aleatoria generada en la corrida {run_dir.name}")
    if settings.run.name != name:
        text = _set_key(text, "output", "name", name, f"nombre usado en la corrida {run_dir.name}")
    return header + text


_COPY_HEADER = re.compile(r"\A# Copia de .*?\n\n", flags=re.DOTALL)


def _set_key(text: str, section: str, key: str, value: int | str, comment: str) -> str:
    """Write `key = value` in [section], keeping the rest of the text as is.

    Replaces the key's line if it already exists, adds it at the start of the section
    if not, and creates the section at the end if it is missing. The result is checked by
    reading it back: it must equal the original except for that key. If the format does
    not allow it (e.g. an inline table), the text is left as is and the value is noted.
    """
    literal = json.dumps(value)  # an integer or a JSON string is also valid TOML
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
