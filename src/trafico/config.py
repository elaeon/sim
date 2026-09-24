"""Parámetros de la simulación: especificaciones de vehículos, comportamiento y corrida."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

DT = 0.1
"""Resolución temporal: segundos simulados por paso."""

MAX_TYPES = 8
"""Máximo de tipos de vehículo en una corrida (tamaño de la paleta categórica de la gráfica)."""


@dataclass(frozen=True, slots=True)
class VehicleSpec:
    """Características fijas de un tipo de vehículo."""

    key: str  # clave en el archivo de configuración: [vehicles.<key>] y [demand] <key>_rate
    name: str  # etiqueta en gráficas, resumen y CSV
    speed_kmh: float  # velocidad máxima que puede alcanzar el vehículo
    length: float  # m
    gap_run: float  # m, distancia al líder cuando éste está en marcha
    gap_stop: float  # m, distancia al líder detenido (compresión)
    pax_min: int
    pax_max: int
    pax_mean: float
    pax_std: float
    can_change_lane: bool
    lane: int | None = None  # carril fijo (0 = derecho) de un tipo que no cambia de carril; None = slow_lane
    exclusive: bool = False  # su carril fijo queda reservado: solo lo usan los tipos con ese `lane`
    # Parada antes del semáforo (descenso y ascenso de pasajeros): posición del frente del
    # vehículo, en m desde el inicio del tramo (None = sin parada), y duración normal(μ, σ) en s,
    # truncada a ≥ 0.
    stop_position: float | None = None
    stop_time_mean: float = 20.0
    stop_time_std: float = 5.0
    # Al detenerse, cuántos vehículos de este tipo caben lado a lado en un carril (p. ej. 2 bicis).
    abreast: int = 1

    @property
    def speed(self) -> float:
        """Velocidad constante en m/s."""
        return self.speed_kmh / 3.6

    @property
    def expected_stop_time(self) -> float:
        """Duración media de la parada (s): la de la normal truncada a ≥ 0; 0 sin parada."""
        if self.stop_position is None:
            return 0.0
        mu, sigma = self.stop_time_mean, self.stop_time_std
        if sigma <= 0:
            return max(mu, 0.0)
        a = -mu / sigma  # E[X | X ≥ 0] = μ + σ·φ(a) / (1 − Φ(a))
        pdf = math.exp(-a * a / 2) / math.sqrt(2 * math.pi)
        tail = 0.5 * math.erfc(a / math.sqrt(2))
        return mu + sigma * pdf / tail


# Tipos incorporados: dan valores por defecto a [vehicles.car|bike|bus]. Cualquier otro tipo se
# define completo en el archivo de configuración, sin tocar el código.
DEFAULT_SPECS: tuple[VehicleSpec, ...] = (
    VehicleSpec("car", "auto", 50.0, 4.5, 3.0, 1.0, 1, 6, 1.5, 1.0, True),
    VehicleSpec("bike", "bici", 15.0, 1.8, 1.0, 0.5, 1, 1, 1.0, 0.0, False),
    # El gap en marcha del autobús no está especificado: se asume 4 m.
    VehicleSpec("bus", "autobús", 40.0, 12.0, 4.0, 1.5, 1, 80, 40.0, 10.0, False),
)
DEFAULT_RATES: tuple[float, ...] = (15.0, 4.0, 1.0)  # veh/min de los tipos incorporados
CAR, BIKE, BUS = 0, 1, 2  # índices de los tipos incorporados (siempre van primero)


@dataclass(frozen=True, slots=True)
class Behavior:
    """Parámetros de conducta comunes a todos los vehículos."""

    reaction_min: float = 1.0  # s, reacción al poder avanzar (verde o arranque del líder)
    reaction_max: float = 5.0
    lane_change_min: float = 1.0  # s, duración de la maniobra de cambio de carril
    lane_change_max: float = 4.0
    lane_change_speed_factor: float = 0.5  # fracción de la velocidad durante la maniobra
    lane_change_cooldown: float = 3.0  # s sin volver a cambiar tras una maniobra
    lane_change_interval: float = 0.5  # s entre decisiones de cambio de carril
    lookahead: float = 30.0  # m, distancia a la que un líder lento motiva cambiar
    min_advantage: float = 5.0  # m de espacio libre extra para que valga la pena cambiar
    release_space: float = 0.5  # m, espacio que libera a un vehículo detenido
    # Reducción máx. de velocidad con el carril saturado de cola (0 = sin efecto): un número para
    # todos los carriles o un valor por carril, empezando por el derecho (carril 0).
    congestion_factor: float | tuple[float, ...] = 0.0

    def congestion_by_lane(self, lanes: int) -> tuple[float, ...]:
        return per_lane(self.congestion_factor, lanes)


def per_lane(value: float | tuple[float, ...], lanes: int) -> tuple[float, ...]:
    """Un valor por carril a partir de un número o una lista. Si hay más carriles que valores,
    los extra usan el último; si hay menos, los sobrantes se ignoran."""
    values = tuple(value) if isinstance(value, tuple) else (float(value),)
    return tuple(values[min(i, len(values) - 1)] for i in range(lanes))


@dataclass(frozen=True, slots=True)
class SimConfig:
    """Configuración completa de una corrida."""

    length: float = 200.0  # m del tramo; el semáforo está en x = length
    lanes: int = 2
    lane_speed_limit: float | tuple[float, ...] | None = None  # km/h por carril (0 = derecho); None = sin límite
    rates: tuple[float, ...] = DEFAULT_RATES  # veh/min, alineadas con `specs`
    red: float = 30.0  # s
    green: float = 30.0  # s
    start_phase: str = "red"  # "red" | "green"
    run: float = 10.0  # s de proceso
    time_scale: float = 10.0  # s simulados por cada s de proceso
    sample: float = 1.0  # s simulados entre muestras
    slow_lane: str = "right"  # carril de los tipos que no cambian de carril: "right" | "random"
    specs: tuple[VehicleSpec, ...] = DEFAULT_SPECS
    behavior: Behavior = field(default_factory=Behavior)

    def __post_init__(self) -> None:
        if len(self.rates) != len(self.specs):
            raise ValueError(f"se esperaban {len(self.specs)} tasas de llegada, una por tipo; hay {len(self.rates)}")

    @property
    def n_types(self) -> int:
        return len(self.specs)

    @property
    def lane_congestion(self) -> tuple[float, ...]:
        return self.behavior.congestion_by_lane(self.lanes)

    @property
    def lane_max_kmh(self) -> tuple[float, ...]:
        """Límite de velocidad de cada carril (inf si no hay límite)."""
        if self.lane_speed_limit is None:
            return (math.inf,) * self.lanes
        return per_lane(self.lane_speed_limit, self.lanes)

    @property
    def reserved_lanes(self) -> frozenset[int]:
        """Carriles exclusivos: los de los tipos con `exclusive` y carril fijo."""
        return frozenset(s.lane for s in self.specs if s.exclusive and s.lane is not None)

    def allowed_lanes(self, k: int) -> tuple[int, ...]:
        """Carriles que puede usar el tipo k: su carril fijo o, si no tiene, los no reservados."""
        spec = self.specs[k]
        if spec.lane is not None:
            return (spec.lane,)
        reserved = self.reserved_lanes
        return tuple(i for i in range(self.lanes) if i not in reserved)

    def entry_lanes(self, k: int) -> tuple[int, ...]:
        """Carriles por los que entra el tipo k (uno fijo si no cambia de carril y slow_lane = "right")."""
        allowed = self.allowed_lanes(k)
        spec = self.specs[k]
        if spec.can_change_lane or spec.lane is not None or self.slow_lane == "random":
            return allowed
        return allowed[:1]  # el carril libre más a la derecha

    def free_flow_kmh(self, k: int) -> float:
        """Velocidad a flujo libre del tipo k: su máxima, sin rebasar el límite del mejor carril que puede usar."""
        limits = self.lane_max_kmh
        return min(self.specs[k].speed_kmh, max((limits[i] for i in self.entry_lanes(k)), default=math.inf))

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
    def cycle(self) -> float:
        return self.red + self.green

    @property
    def sample_ticks(self) -> int:
        return max(1, round(self.sample / DT))

    @property
    def n_samples(self) -> int:
        return self.n_ticks // self.sample_ticks

    def is_green(self, tick: int) -> bool:
        pos = tick % (self.red_ticks + self.green_ticks)
        if self.start_phase == "red":
            return pos >= self.red_ticks
        return pos < self.green_ticks

    def red_intervals(self) -> list[tuple[float, float]]:
        """Intervalos [inicio, fin) en s simulados en los que el semáforo está en rojo."""
        if self.red <= 0:
            return []
        out = []
        start = 0.0 if self.start_phase == "red" else self.green
        while start < self.sim_seconds:
            out.append((start, min(start + self.red, self.sim_seconds)))
            start += self.cycle
        return out
