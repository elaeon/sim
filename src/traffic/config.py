"""Simulation parameters: vehicle specifications, behaviour and run."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

from traffic.emission_sets import BUS_EMISSIONS, PETROL_CAR_EMISSIONS

DT = 0.1
"""Resolución temporal: segundos simulados por paso."""

MAX_TYPES = 8
"""Máximo de tipos de vehículo en una corrida (tamaño de la paleta categórica de la gráfica)."""

POLLUTANTS = ("co2", "nox", "voc", "pm")
"""Contaminantes del modelo de emisiones instantáneas de Int Panis et al. (2006)."""

POLLUTANT_LABELS = {"co2": "CO2", "nox": "NOx", "voc": "VOC", "pm": "PM"}

EMISSION_DECEL = -0.5
"""m/s²: por debajo de esta aceleración se usan los coeficientes de desaceleración (`<contaminante>_decel`)."""

FUELS = ("gasolina", "diesel", "glp")
"""Combustibles que puede usar un tipo de vehículo (`fuel`)."""

FUEL_LABELS = {"gasolina": "petrol", "diesel": "diesel", "glp": "LPG"}

FUEL_CO2_G_PER_L = {"gasolina": 8887 / 3.78541, "diesel": 10180 / 3.78541, "glp": 5720 / 3.78541}
"""g de CO2 por litro de combustible quemado (EPA-420-F-18-008; el gas LP como propano, EPA, Emission Factors for
Greenhouse Gas Inventories): el consumo se obtiene del CO2 del modelo de emisiones."""

FUEL_MJ_PER_L = {"gasolina": 32.3, "diesel": 35.8, "glp": 23.7}
"""Poder calorífico inferior (MJ/L): gasolina 43.4 MJ/kg × 0.745 kg/L, diésel 43.0 × 0.832, gas LP (propano)
46.4 × 0.51. Solo para la curva física de referencia del rendimiento."""

NO_LIGHT_WINDOW = 60.0
"""Ventana (s) del flujo en la línea final cuando no hay semáforo (con él, un ciclo)."""

RED, GREEN, YELLOW = 0, 1, 2
"""Fases del semáforo; el ciclo es verde → amarillo → rojo."""


@dataclass(frozen=True, slots=True)
class VehicleSpec:
    """Fixed characteristics of a vehicle type."""

    key: str  # key in the configuration file: [vehicles.<key>] and [demand] <key>_rate
    name: str  # label in plots, summary and CSV
    speed_kmh: float  # maximum speed the vehicle can reach (the mean, if it varies)
    length: float  # m
    gap_run: float  # m, distance to the leader when it is moving
    gap_stop: float  # m, distance to the stopped leader (compression)
    pax_min: int
    pax_max: int
    pax_mean: float
    pax_std: float
    can_change_lane: bool
    lane: int | None = None  # fixed lane (0 = right) of a type that does not change lanes; None = the free right lane
    exclusive: bool = False  # its fixed lane is reserved: only the types with that `lane` use it
    # Stop before the traffic light (passengers getting off and on): position of the front of the
    # vehicle, in m from the start of the segment (None = no stop), and duration normal(μ, σ) in s,
    # truncated to ≥ 0.
    stop_position: float | None = None
    stop_time_mean: float = 20.0
    stop_time_std: float = 5.0
    # When stopping, how many vehicles of this type fit side by side in a lane (e.g. 2 bikes).
    abreast: int = 1
    # Overtaking within the lane (e.g. bikes): while moving, a vehicle faster than the one ahead of the
    # same type passes it on one side without changing lanes. Requires abreast ≥ 2.
    pass_in_lane: bool = False
    # Aggressive overtaking (e.g. motorcycles): detects the slow leader by its actual speed and accepts a lane
    # with a lower limit if it advances more there than behind it.
    overtake: bool = False
    # The type's own lane change thresholds; None = those of [behavior].
    lookahead: float | None = None
    min_advantage: float | None = None
    lane_change_cooldown: float | None = None
    # Gradual acceleration and braking (m/s²). Without them (None), the vehicle reaches its speed
    # immediately and stops dead, as before. With `decel` it also anticipates: it does not exceed the speed
    # at which it can still stop braking at `decel` behind the one ahead or before a stop.
    accel: float | None = None
    decel: float | None = None
    # Car-following headway (s): behind a moving leader it keeps gap_stop plus what it covers in this time at its
    # speed (like IDM's s0 + v·T), instead of gap_run. None = gap_run (as before).
    time_headway: float | None = None
    # Maximum speed (km/h) at which it goes over the [speed_bump] bump; None = it goes over without braking.
    speed_bump_kmh: float | None = None
    # Probability that an arriving vehicle carries freight instead of passengers: it takes up the road,
    # but does not count in the passenger metrics (1 = the type only carries freight).
    cargo_prob: float = 0.0
    # Variable length: normal(length, length_std) truncated to [length_min, length_max] (by default,
    # length ∓ 3·length_std), drawn for each vehicle. With length_std = 0, all measure `length`.
    # In the TOML, the four go in [vehicles.<key>] length = {min, max, mean, std}.
    length_std: float = 0.0
    length_min: float | None = None
    length_max: float | None = None
    # Variable maximum speed: normal(speed_kmh, speed_std) truncated to [speed_min, speed_max] (by
    # default, speed_kmh ∓ 3·speed_std), drawn for each vehicle. With speed_std = 0, all go at
    # speed_kmh. In the TOML, the four go in [vehicles.<key>] speed_kmh = {min, max, mean, std}.
    speed_std: float = 0.0
    speed_min: float | None = None
    speed_max: float | None = None
    # Halts at [bottleneck]: probability that a vehicle of the type stops once in the
    # segment (0 = never) and duration normal(mean, σ) in s, truncated to ≥ 0. Where they happen (lanes and
    # zone) is defined in [bottleneck].
    bottleneck_prob: float = 0.0
    bottleneck_time_mean: float = 30.0
    bottleneck_time_std: float = 0.0
    # Instantaneous emissions (Int Panis et al., 2006): per pollutant, the coefficients [f1..f6] of
    # E = max(0, f1 + f2·v + f3·v² + f4·a + f5·a² + f6·v·a) in g/s (v in m/s, a in m/s²) for a ≥ −0.5 m/s² and
    # for a < −0.5 m/s². Empty = the type does not emit. Only computed with gradual acceleration and braking.
    emissions: tuple[tuple[str, tuple[float, ...], tuple[float, ...]], ...] = ()
    # Set of coefficients they come from (`EMISSION_SETS`); "" if they were written by hand.
    emission_source: str = ""
    # Fuel (one of FUELS; None = burns no fuel): consumption comes from the CO2 emitted. The mass (kg)
    # is only used in the physical reference curve of the fuel economy at constant speed.
    fuel: str | None = None
    mass_kg: float | None = None

    @property
    def burns(self) -> bool:
        """Its fuel consumption is computed: it has a fuel and emits CO2."""
        return self.fuel is not None and self.emits and self.emission_coefs("co2") is not None

    @property
    def emits(self) -> bool:
        """The type has emission coefficients and gradual dynamics (accel and decel), which the model needs."""
        return bool(self.emissions) and self.accel is not None and self.decel is not None

    def emission_coefs(self, pollutant: str) -> tuple[tuple[float, ...], tuple[float, ...]] | None:
        """(coefficients for a ≥ −0.5, for a < −0.5) of the pollutant; None if the type does not emit it."""
        return next(((acc, dec) for name, acc, dec in self.emissions if name == pollutant), None)

    @property
    def speed(self) -> float:
        """Mean maximum speed of the type (the `mean` parameter) in m/s."""
        return self.speed_kmh / 3.6

    @property
    def slowest_kmh(self) -> float:
        """Lowest maximum speed a vehicle of the type can have (km/h)."""
        if self.speed_std <= 0:
            return self.speed_kmh
        return self.speed_min if self.speed_min is not None else self.speed_kmh - 3 * self.speed_std

    @property
    def fastest_kmh(self) -> float:
        """Highest maximum speed a vehicle of the type can have (km/h)."""
        if self.speed_std <= 0:
            return self.speed_kmh
        return self.speed_max if self.speed_max is not None else self.speed_kmh + 3 * self.speed_std

    @property
    def shortest(self) -> float:
        """Minimum length a vehicle of the type can have (m)."""
        if self.length_std <= 0:
            return self.length
        return self.length_min if self.length_min is not None else self.length - 3 * self.length_std

    @property
    def longest(self) -> float:
        """Maximum length a vehicle of the type can have (m)."""
        if self.length_std <= 0:
            return self.length
        return self.length_max if self.length_max is not None else self.length + 3 * self.length_std

    @property
    def carries_passengers(self) -> bool:
        """Some vehicle of the type carries passengers (not all of them carry freight)."""
        return self.cargo_prob < 1

    @property
    def expected_stop_time(self) -> float:
        """Mean stop duration (s): that of the normal truncated to ≥ 0; 0 without a stop."""
        if self.stop_position is None:
            return 0.0
        mu, sigma = self.stop_time_mean, self.stop_time_std
        if sigma <= 0:
            return max(mu, 0.0)
        a = -mu / sigma  # E[X | X ≥ 0] = μ + σ·φ(a) / (1 − Φ(a))
        pdf = math.exp(-a * a / 2) / math.sqrt(2 * math.pi)
        tail = 0.5 * math.erfc(a / math.sqrt(2))
        return mu + sigma * pdf / tail


# Built-in types: they provide the defaults for [vehicles.car|bike|bus]. Any other type is
# defined in full in the configuration file, without touching the code. The car and the bus come with
# emission coefficients, which are only used if the type has accel and decel.
DEFAULT_SPECS: tuple[VehicleSpec, ...] = (
    VehicleSpec("car", "auto", 50.0, 4.5, 3.0, 1.0, 1, 6, 1.5, 1.0, True, emissions=PETROL_CAR_EMISSIONS,
                emission_source="int_panis_2006", fuel="gasolina", mass_kg=1250.0),
    VehicleSpec("bike", "bici", 15.0, 1.8, 1.0, 0.5, 1, 1, 1.0, 0.0, False),
    # The bus's moving gap is not specified: 4 m is assumed.
    VehicleSpec("bus", "autobús", 40.0, 12.0, 4.0, 1.5, 1, 80, 40.0, 10.0, False, emissions=BUS_EMISSIONS,
                emission_source="int_panis_2006", fuel="diesel", mass_kg=12000.0),
)
DEFAULT_RATES: tuple[float, ...] = (15.0, 4.0, 1.0)  # veh/min of the built-in types
CAR, BIKE, BUS = 0, 1, 2  # indices of the built-in types (they always come first)


@dataclass(frozen=True, slots=True)
class Behavior:
    """Behaviour parameters common to all vehicles."""

    reaction_min: float = 1.0  # s, reaction when it can move (green or the leader starting)
    reaction_max: float = 5.0
    # With reaction_std, the reaction is normal(reaction_mean, reaction_std) truncated to [min, max]
    # (default mean: the midpoint); without it, uniform in [min, max]. In the TOML, the four
    # go in [behavior] reaction = {min, max, mean, std}.
    reaction_mean: float | None = None
    reaction_std: float | None = None
    lane_change_min: float = 1.0  # s, duration of the lane change manoeuvre
    lane_change_max: float = 4.0
    lane_change_speed_factor: float = 0.5  # fraction of the speed during the manoeuvre
    lane_change_cooldown: float = 3.0  # s without changing again after a manoeuvre
    lane_change_interval: float = 0.5  # s between lane change decisions
    lookahead: float = 30.0  # m, distance at which a slow leader prompts a change
    min_advantage: float = 5.0  # m of extra free space for a change to be worth it
    release_space: float = 0.5  # m, space that releases a stopped vehicle
    # Lanes are only changed with a leader within `lookahead` going at least this fraction
    # below the lane's limit (without a limit, below the vehicle's maximum speed).
    leader_slowdown: float = 0.25
    # On yellow, a vehicle less than `yellow_approach` m from the stop line (without having
    # crossed it) advances at `yellow_speed_factor` of its speed.
    yellow_approach: float = 50.0
    yellow_speed_factor: float = 0.5
    # Stopped entry queue (the segment's queue reaches the entry and is stopped): each vehicle
    # waiting in it starts with its reaction time when the one ahead moves, instead of
    # entering already moving as soon as there is room.
    queue_reaction: bool = False


@dataclass(frozen=True, slots=True)
class Bottleneck:
    """Bottlenecks: where halts happen. Each vehicle entering the segment, with the
    probability of its type (`bottleneck_prob`), will stop once at a random point of
    `stop_zone`, if on reaching it it is in one of `stop_lanes`, for its type's duration.
    Those behind treat it like any stopped leader (they compress or change lanes)."""

    stop_lanes: tuple[int, ...] | None = None  # lanes (0 = right); None = all
    stop_zone: tuple[float, ...] | None = None  # (start, end) in m of the front; None = the whole segment


def per_lane(value: float | tuple[float, ...], lanes: int) -> tuple[float, ...]:
    """One value per lane from a number or a list. If there are more lanes than values,
    the extra ones use the last one; if there are fewer, the surplus values are ignored."""
    values = tuple(value) if isinstance(value, tuple) else (float(value),)
    return tuple(values[min(i, len(values) - 1)] for i in range(lanes))


@dataclass(frozen=True, slots=True)
class Rate:
    """Arrival rate (veh/min) of a type: normal(mean, std) truncated to [min, max], drawn again
    in each [demand] rate_interval. With std = 0 or min = max it is fixed."""

    min: float
    max: float
    mean: float
    std: float

    @classmethod
    def fixed(cls, value: float) -> Rate:
        return cls(value, value, value, 0.0)

    @property
    def variable(self) -> bool:
        return self.std > 0 and self.min < self.max

    @property
    def expected(self) -> float:
        """Actual mean rate: the mean of the normal truncated to [min, max]."""
        if not self.variable:
            return min(max(self.mean, self.min), self.max)
        mu, sigma = self.mean, self.std
        a, b = (self.min - mu) / sigma, (self.max - mu) / sigma
        # E[X | a ≤ Z ≤ b] = μ + σ·(φ(a) − φ(b)) / (Φ(b) − Φ(a)), with Z = (X − μ) / σ
        mass = 0.5 * (math.erf(b / math.sqrt(2)) - math.erf(a / math.sqrt(2)))
        if mass <= 0:  # mean very far from [min, max] (validation rejects it): the nearest bound
            return self.min if mu < self.min else self.max
        return mu + sigma * (math.exp(-a * a / 2) - math.exp(-b * b / 2)) / math.sqrt(2 * math.pi) / mass


PEDESTRIAN_RATE = Rate(0.5, 3.0, 1.5, 1.0)
"""Aparición de peatones (peatones/min) cuando `pedestrian = true` no trae `pedestrian_crossing`."""

PEDESTRIAN_TIME = 10.0
"""s que tarda un peatón en cruzar el tope (`[speed_bump] pedestrian_time`) si no se da."""

PEDESTRIAN_YIELD = 3.0
"""s que un peatón espera en la orilla del tope antes de cruzar: los vehículos lo ven (fase amarilla) y se detienen."""


@dataclass(frozen=True, slots=True)
class SpeedBump:
    """A bump (speed hump) at `position` m from the start of the segment, in the lanes `lanes`. While a
    vehicle is on it (from the front to the rear) it does not exceed its type's `speed_bump_kmh`. With `pedestrian`,
    the bump is also a pedestrian crossing: pedestrians appear (`pedestrian_crossing`, pedestrians/min) that cross in
    `pedestrian_time` s each, and vehicles stop completely while someone is crossing. A street can have
    several (`SimConfig.speed_bumps`), each with its own configuration."""

    position: float  # m from the start of the segment
    lanes: tuple[int, ...] | None = None  # lanes (0 = right); None = all
    pedestrian: bool = False
    pedestrian_crossing: Rate = PEDESTRIAN_RATE
    pedestrian_time: float = PEDESTRIAN_TIME


def as_positions(value: float | tuple[float, ...] | list[float] | None) -> tuple[float, ...]:
    """Positions (m) from a single position, a list of them or None, from the entry to the exit (empty if None)."""
    if value is None:
        return ()
    if isinstance(value, (tuple, list)):
        return tuple(sorted(float(p) for p in value))
    return (float(value),)


def phase_at(red: int, green: int, yellow: int, start_phase: str, tick: int) -> int:
    """Phase (RED, GREEN or YELLOW) at step `tick` of a traffic light with those phases (in steps): green → yellow →
    red, starting with red or with green."""
    cycle = red + green + yellow
    pos = tick % cycle
    if start_phase == "red":
        pos = (pos - red) % cycle  # the cycle starts at red
    if pos < green:
        return GREEN
    return YELLOW if pos < green + yellow else RED


@dataclass(frozen=True, slots=True)
class Light:
    """A traffic light on the segment at `position` m from the start, with its own phases and initial phase. The one at
    the end of the segment (x = length) is described by SimConfig's `red`, `green`, `yellow`, `start_phase`,
    `traffic_light` and `free_lanes` (`SimConfig.exit_light`); the intermediate ones go in `SimConfig.extra_lights`."""

    position: float
    red: float = 30.0  # s
    green: float = 30.0  # s
    yellow: float = 0.0  # s, between green and red; 0 = no yellow
    start_phase: str = "red"  # "red" | "green"
    enabled: bool = True
    # Lanes without this traffic light (0 = right): only applies to the types with an exclusive lane in them.
    free_lanes: tuple[int, ...] = ()
    # Pedestrian light: green for vehicles while no pedestrians are waiting (they appear at a rate of
    # `pedestrian_crossing`, pedestrians/min). With someone in the queue and once the minimum green (`green`) has
    # passed, and when the previous fixed-cycle light is red, it turns yellow (`yellow`) and red (`red`, the time they
    # take to cross). It has no cycle or `start_phase`: its schedule comes from the simulation (`pedestrians.light_schedule`).
    pedestrian: bool = False
    pedestrian_crossing: Rate = PEDESTRIAN_RATE
    # Offset (s): the whole cycle is delayed by `offset` s, e.g. to coordinate it with another light. 0 ≤ offset < cycle.
    offset: float = 0.0

    @property
    def active(self) -> bool:
        """There is a traffic light: it is enabled and one of its phases lasts more than 0 s (a pedestrian light only needs
        to be enabled)."""
        return self.enabled and (self.pedestrian or self.red + self.green + self.yellow > 0)

    @property
    def cycle(self) -> float:
        """Cycle duration (s); 0 if there is no traffic light or it is a pedestrian light (it has no cycle)."""
        return self.red + self.green + self.yellow if self.active and not self.pedestrian else 0.0

    @property
    def label(self) -> str:
        """Phases, e.g. «red 25 s / green 35 s / yellow 3 s»; a pedestrian light, «pedestrian: red 15 s with…»."""
        if self.pedestrian:
            label = f"pedestrian: red {self.red:g} s with pedestrians waiting, minimum green {self.green:g} s"
            return label + (f", yellow {self.yellow:g} s" if self.yellow > 0 else "")
        label = f"red {self.red:g} s / green {self.green:g} s"
        label += f" / yellow {self.yellow:g} s" if self.yellow > 0 else ""
        return label + (f", offset {self.offset:g} s" if self.offset > 0 else "")

    def phase(self, tick: int) -> int:
        """Phase (RED, GREEN or YELLOW) at step `tick`: green → yellow → red. Without a traffic light, always green; a
        pedestrian light too (its actual phase depends on each replica's pedestrians: the simulation provides it)."""
        if not self.active or self.pedestrian:
            return GREEN
        return phase_at(round(self.red / DT), round(self.green / DT), round(self.yellow / DT), self.start_phase,
                        tick - round(self.offset / DT))

    def intervals(self, offset: float, duration: float, horizon: float) -> list[tuple[float, float]]:
        """[start, end) intervals in simulated s, up to `horizon`, of the phase that starts `offset` s after the
        start of green and lasts `duration` s."""
        if duration <= 0 or not self.active or self.pedestrian:
            return []
        out = []
        # Two cycles before the first green (at t = red if it starts on red, plus the offset): covers the phase already
        # in progress at t = 0.
        start = (self.red if self.start_phase == "red" else 0.0) + self.offset + offset - 2 * self.cycle
        while start < horizon:
            if start + duration > 0:
                out.append((max(start, 0.0), min(start + duration, horizon)))
            start += self.cycle
        return out

    def red_intervals(self, horizon: float) -> list[tuple[float, float]]:
        return self.intervals(self.green + self.yellow, self.red, horizon)

    def yellow_intervals(self, horizon: float) -> list[tuple[float, float]]:
        return self.intervals(self.green, self.yellow, horizon)


@dataclass(frozen=True, slots=True)
class StopLine:
    """Intermediate stop line where vehicles may have to stop: an intermediate traffic light (`light`) or a
    bump's pedestrian crossing (`light` is None), at `position` m. `lanes` are the lanes where it applies (None = all)."""

    position: float
    light: Light | None = None
    lanes: tuple[int, ...] | None = None

    @property
    def free_lanes(self) -> tuple[int, ...]:
        return self.light.free_lanes if self.light is not None else ()

    @property
    def pedestrian(self) -> bool:
        """Its schedule depends on the pedestrians: a pedestrian crossing or a pedestrian light."""
        return self.light is None or self.light.pedestrian

    @property
    def kind(self) -> str:
        return "tope" if self.light is None else "semáforo"


@dataclass(frozen=True, slots=True)
class SimConfig:
    """Full configuration of a run."""

    length: float = 200.0  # m of the segment; the traffic light is at x = length
    lanes: int = 2
    lane_speed_limit: float | tuple[float, ...] | None = None  # km/h per lane (0 = right); None = no limit
    rates: tuple[float, ...] = DEFAULT_RATES  # mean veh/min, aligned with `specs`
    # Variable rates, one per type; if given, `rates` is computed from them (their actual mean). Empty = fixed.
    rate_dists: tuple[Rate, ...] = ()
    rate_interval: float = 60.0  # simulated s between draws of the variable rates
    red: float = 30.0  # s
    green: float = 30.0  # s
    yellow: float = 0.0  # s, between green and red; 0 = no yellow
    start_phase: str = "red"  # "red" | "green"
    # Traffic light enabled. Disabled (or with red, green and yellow at 0) there is no traffic light: the line at the end
    # of the segment is always open.
    traffic_light: bool = True
    # Lanes without a traffic light (0 = right), e.g. a free right turn. Only applies to the types with an
    # exclusive lane (`lane` and `exclusive`) in one of them: they cross the line even on red and do not slow
    # down on yellow.
    free_lanes: tuple[int, ...] = ()
    # The traffic light at the end of the segment is a pedestrian light (see `Light.pedestrian`); in that case `red` is
    # how long the red lasts, `green` the minimum green and `start_phase` is not used.
    light_pedestrian: bool = False
    light_pedestrian_crossing: Rate = PEDESTRIAN_RATE
    light_offset: float = 0.0  # s, offset of the traffic light at the end of the segment (see `Light.offset`)
    # Intermediate traffic lights (besides the one at the end of the segment), each at its `position` m with its own phases.
    extra_lights: tuple[Light, ...] = ()
    run: float = 10.0  # process s
    time_scale: float = 10.0  # simulated s per process s
    sample: float = 1.0  # simulated s between samples
    # Initial condition: fraction of each lane already occupied at the start by moving vehicles,
    # (length + gap_run) / length; one number for all or one per lane. 0 = empty segment.
    initial_occupancy: float | tuple[float, ...] = 0.0
    # Exit queue after the traffic light, per lane (one number for all or one per lane): how many
    # vehicles per minute it accepts (0 = no exit queue) and how many m it measures (each vehicle takes its length +
    # stopped gap). A vehicle that does not fit cannot cross the line even on green.
    exit_capacity: float | tuple[float, ...] = 0.0
    exit_storage: float | tuple[float, ...] = 60.0
    specs: tuple[VehicleSpec, ...] = DEFAULT_SPECS
    behavior: Behavior = field(default_factory=Behavior)
    bottleneck: Bottleneck = field(default_factory=Bottleneck)
    # Bumps, each with its position, lanes and pedestrians (see `bumps` to have them sorted).
    speed_bumps: tuple[SpeedBump, ...] = ()
    # Price of each fuel per litre, in `currency`; without a price, consumption is only given in litres.
    fuel_price: tuple[tuple[str, float], ...] = ()
    currency: str = "MXN"

    def __post_init__(self) -> None:
        if self.rate_dists:
            if len(self.rate_dists) != len(self.specs):
                raise ValueError(f"expected {len(self.specs)} arrival rates, one per type; there are {len(self.rate_dists)}")
            object.__setattr__(self, "rates", tuple(d.expected for d in self.rate_dists))
        if len(self.rates) != len(self.specs):
            raise ValueError(f"expected {len(self.specs)} arrival rates, one per type; there are {len(self.rates)}")

    def price(self, fuel: str | None) -> float | None:
        """Price per litre of the fuel (None if it has none)."""
        return next((p for f, p in self.fuel_price if f == fuel), None)

    def rate(self, k: int) -> Rate:
        """Arrival rate of type k (fixed if there are no `rate_dists`)."""
        return self.rate_dists[k] if self.rate_dists else Rate.fixed(self.rates[k])

    @property
    def rate_ticks(self) -> int:
        return max(1, round(self.rate_interval / DT))

    @property
    def n_types(self) -> int:
        return len(self.specs)

    def bottleneck_prob(self, k: int) -> float:
        """Probability that a vehicle of type k stops at a bottleneck."""
        return self.specs[k].bottleneck_prob

    def bottleneck_time(self, k: int) -> tuple[float, float]:
        """Duration (mean, σ) in s of the halt of type k."""
        return self.specs[k].bottleneck_time_mean, self.specs[k].bottleneck_time_std

    @property
    def bottleneck_active(self) -> bool:
        """Some participating type can stop at a bottleneck."""
        return any(rate > 0 and self.bottleneck_prob(k) > 0 for k, rate in enumerate(self.rates))

    def bottleneck_lanes(self) -> tuple[int, ...]:
        lanes = self.bottleneck.stop_lanes
        return tuple(range(self.lanes)) if lanes is None else lanes

    @property
    def bumps(self) -> tuple[SpeedBump, ...]:
        """The bumps, from the entry to the exit."""
        return tuple(sorted(self.speed_bumps, key=lambda b: b.position))

    def bump_lanes(self, bump: SpeedBump) -> tuple[int, ...]:
        """Lanes where the bump is."""
        return tuple(range(self.lanes)) if bump.lanes is None else bump.lanes

    @property
    def crossings(self) -> tuple[SpeedBump, ...]:
        """The bumps that are pedestrian crossings (and are on some lane), from the entry to the exit."""
        return tuple(b for b in self.bumps if b.pedestrian and self.bump_lanes(b))

    def bottleneck_zone(self) -> tuple[float, float]:
        zone = self.bottleneck.stop_zone
        return (0.0, self.length) if zone is None else (zone[0], zone[1])

    @property
    def lane_initial_occupancy(self) -> tuple[float, ...]:
        return per_lane(self.initial_occupancy, self.lanes)

    @property
    def lane_exit_capacity(self) -> tuple[float, ...]:
        """veh/min accepted by the exit queue of each lane (0 = no exit queue)."""
        return per_lane(self.exit_capacity, self.lanes)

    @property
    def lane_exit_storage(self) -> tuple[float, ...]:
        """m measured by the exit queue of each lane (each vehicle takes its length + stopped gap)."""
        return per_lane(self.exit_storage, self.lanes)

    @property
    def lane_max_kmh(self) -> tuple[float, ...]:
        """Speed limit of each lane (inf if there is no limit)."""
        if self.lane_speed_limit is None:
            return (math.inf,) * self.lanes
        return per_lane(self.lane_speed_limit, self.lanes)

    @property
    def reserved_lanes(self) -> frozenset[int]:
        """Exclusive lanes: those of the types with `exclusive` and a fixed lane."""
        return frozenset(s.lane for s in self.specs if s.exclusive and s.lane is not None)

    def ignores_light(self, k: int, light: Light | None = None) -> bool:
        """Type k does not obey the traffic light (by default, the one at the end of the segment): it has an exclusive lane
        and that lane is in its `free_lanes`."""
        spec = self.specs[k]
        lanes = self.free_lanes if light is None else light.free_lanes
        return spec.exclusive and spec.lane is not None and spec.lane in lanes

    def allowed_lanes(self, k: int) -> tuple[int, ...]:
        """Lanes type k can use: its fixed lane or, if it has none, the non-reserved ones."""
        spec = self.specs[k]
        if spec.lane is not None:
            return (spec.lane,)
        reserved = self.reserved_lanes
        return tuple(i for i in range(self.lanes) if i not in reserved)

    def entry_lanes(self, k: int) -> tuple[int, ...]:
        """Lanes through which type k enters. One that does not change lanes and has no `lane` goes in the
        rightmost free lane (without the exclusive lanes of other types)."""
        allowed = self.allowed_lanes(k)
        spec = self.specs[k]
        if spec.can_change_lane or spec.lane is not None:
            return allowed
        return allowed[:1]  # the rightmost free lane

    def free_flow_kmh(self, k: int) -> float:
        """Free-flow speed of type k: its maximum (the mean, if it varies), without exceeding the limit
        of the best lane it can use."""
        return min(self.specs[k].speed_kmh, self._best_limit_kmh(k))

    def free_flow_time(self, k: int) -> float:
        """Mean time (s) to cover the segment at free flow for type k, without stops. With a variable
        speed it is the mean of length / min(v, limit) over the truncated normal of v, not length divided by
        the mean speed."""
        spec = self.specs[k]
        limit = self._best_limit_kmh(k)
        if spec.speed_std <= 0:
            return self.length / (min(spec.speed_kmh, limit) / 3.6)
        # Midpoint rule over [slowest, fastest] with the normal density (the truncation normalizes).
        lo, hi, n = spec.slowest_kmh, spec.fastest_kmh, 2000
        step = (hi - lo) / n
        num = den = 0.0
        for i in range(n):
            v = lo + (i + 0.5) * step
            w = math.exp(-0.5 * ((v - spec.speed_kmh) / spec.speed_std) ** 2)
            num += w * self.length / (min(v, limit) / 3.6)
            den += w
        return num / den

    def _best_limit_kmh(self, k: int) -> float:
        """Limit of the fastest lane through which type k enters (inf if there are no limits)."""
        limits = self.lane_max_kmh
        return max((limits[i] for i in self.entry_lanes(k)), default=math.inf)

    @property
    def sim_seconds(self) -> float:
        return self.run * self.time_scale

    @property
    def n_ticks(self) -> int:
        return round(self.sim_seconds / DT)

    @property
    def red_ticks(self) -> int:
        return round(self.red / DT)

    @property
    def green_ticks(self) -> int:
        return round(self.green / DT)

    @property
    def yellow_ticks(self) -> int:
        return round(self.yellow / DT)

    @property
    def exit_light(self) -> Light:
        """The traffic light at the end of the segment (x = length), with the fields `red`, `green`, `yellow`, `start_phase`,
        `traffic_light` and `free_lanes`."""
        return Light(self.length, self.red, self.green, self.yellow, self.start_phase, self.traffic_light,
                     self.free_lanes, self.light_pedestrian, self.light_pedestrian_crossing, self.light_offset)  # fmt: skip

    @property
    def has_light(self) -> bool:
        """There is a traffic light at the end of the segment: it is enabled and one of its phases lasts more than 0 s."""
        return self.exit_light.active

    @property
    def inner_lights(self) -> tuple[Light, ...]:
        """The working intermediate traffic lights, from the entry to the exit."""
        return tuple(sorted((lt for lt in self.extra_lights if lt.active), key=lambda lt: lt.position))

    @property
    def lights(self) -> tuple[Light, ...]:
        """All the working traffic lights (the intermediate ones and the one at the end of the segment), from the entry to
        the exit."""
        return self.inner_lights + ((self.exit_light,) if self.has_light else ())

    @property
    def any_light(self) -> bool:
        return bool(self.lights)

    @property
    def stop_lines(self) -> tuple[StopLine, ...]:
        """Intermediate stop lines, from the entry to the exit: the intermediate traffic lights and the bumps' pedestrian
        crossings (with a light and a crossing at the same point, the light first). The engine and the video
        go through them in this order."""
        lines = [StopLine(lt.position, lt) for lt in self.inner_lights]
        lines += [StopLine(b.position, None, self.bump_lanes(b)) for b in self.crossings]
        return tuple(sorted(lines, key=lambda s: (s.position, s.light is None)))

    @property
    def pedestrian_spots(self) -> tuple[tuple[str, float], ...]:
        """(«semáforo» or «tope», position in m) of each point with pedestrians, from the entry to the exit. They are the
        rows of `summary["pedestrians"]`."""
        spots = [(s.kind, s.position) for s in self.stop_lines if s.pedestrian]
        if self.exit_light.active and self.exit_light.pedestrian:
            spots.append(("semáforo", self.length))
        return tuple(sorted(spots, key=lambda s: (s[1], s[0] == "tope")))

    def previous_light(self, light: Light) -> Light | None:
        """The closest fixed-cycle traffic light before `light` (towards the entry), with which a pedestrian light is
        coordinated: it waits for it to be red. None if there is none."""
        before = [lt for lt in self.lights if not lt.pedestrian and lt.position < light.position]
        return before[-1] if before else None

    @property
    def ref_light(self) -> Light | None:
        """The traffic light that sets the flow window and the shaded phases of the plots: the last fixed-cycle one
        (the one at the end of the segment or, if there is none, the last intermediate one). A pedestrian light has no cycle."""
        fixed = [lt for lt in self.lights if not lt.pedestrian]
        return fixed[-1] if fixed else None

    @property
    def light_label(self) -> str:
        """Phases of the light at the end of the segment, e.g. «red 25 s / green 35 s / yellow 3 s», or «no traffic
        light»."""
        return self.exit_light.label if self.has_light else "no traffic light"

    @property
    def lights_label(self) -> str:
        """The street's traffic lights: with only one, like `light_label`; with several, each with its position and phases."""
        if not self.inner_lights:
            return self.light_label
        return " · ".join(f"at {lt.position:g} m: {lt.label}"
                          + ("" if lt.pedestrian else f", starts {'red' if lt.start_phase == 'red' else 'green'}")
                          for lt in self.lights)  # fmt: skip

    @property
    def lights_short(self) -> str:
        """Like `lights_label`, but short for the plot subtitles: with several traffic lights, how many and where."""
        if not self.inner_lights:
            return self.light_label
        at = [f"{lt.position:g}" for lt in self.lights]
        if len(at) == 1:
            return f"1 traffic light (at {at[0]} m)"
        return f"{len(at)} traffic lights (at {', '.join(at[:-1])} and {at[-1]} m)"

    @property
    def line_name(self) -> str:
        """Name of the line at the end of the segment in texts: «the traffic light» or, without one, «the end of the
        segment»."""
        return "the traffic light" if self.has_light else "the end of the segment"

    @property
    def cycle(self) -> float:
        """Cycle duration of the reference traffic light (s); 0 without a traffic light."""
        ref = self.ref_light
        return ref.cycle if ref else 0.0

    @property
    def flow_window(self) -> float:
        """Window (s) of the flow at the line: one cycle of the reference traffic light or, without one, NO_LIGHT_WINDOW."""
        return self.cycle if self.ref_light else NO_LIGHT_WINDOW

    @property
    def flow_window_label(self) -> str:
        return f"one-cycle window, {self.cycle:g} s" if self.ref_light else f"{NO_LIGHT_WINDOW:g} s window"

    @property
    def sample_ticks(self) -> int:
        return max(1, round(self.sample / DT))

    @property
    def n_samples(self) -> int:
        return self.n_ticks // self.sample_ticks

    def with_lights(self, enabled: bool) -> SimConfig:
        """The same configuration with all the traffic lights (the one at the end of the segment and the intermediate ones)
        enabled or disabled."""
        return replace(self, traffic_light=enabled,
                       extra_lights=tuple(replace(lt, enabled=enabled) for lt in self.extra_lights))  # fmt: skip

    def phase(self, tick: int) -> int:
        """Phase of the traffic light at the end of the segment (RED, GREEN or YELLOW) at step `tick`: green → yellow → red.
        Without a traffic light, always green."""
        return self.exit_light.phase(tick)

    def inner_phases(self, tick: int) -> tuple[int, ...]:
        """Phase of each intermediate traffic light (`inner_lights`) at step `tick`."""
        return tuple(lt.phase(tick) for lt in self.inner_lights)

    def is_green(self, tick: int) -> bool:
        return self.phase(tick) == GREEN

    def is_red(self, tick: int) -> bool:
        return self.phase(tick) == RED

    def red_intervals(self) -> list[tuple[float, float]]:
        """[start, end) intervals in simulated s in which the reference traffic light is red."""
        ref = self.ref_light
        return ref.red_intervals(self.sim_seconds) if ref else []

    def yellow_intervals(self) -> list[tuple[float, float]]:
        """[start, end) intervals in simulated s in which the reference traffic light is yellow."""
        ref = self.ref_light
        return ref.yellow_intervals(self.sim_seconds) if ref else []
