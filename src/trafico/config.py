"""Parámetros de la simulación: especificaciones de vehículos, comportamiento y corrida."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

DT = 0.1
"""Resolución temporal: segundos simulados por paso."""

MAX_TYPES = 8
"""Máximo de tipos de vehículo en una corrida (tamaño de la paleta categórica de la gráfica)."""

RED, GREEN, YELLOW = 0, 1, 2
"""Fases del semáforo; el ciclo es verde → amarillo → rojo."""


@dataclass(frozen=True, slots=True)
class VehicleSpec:
    """Características fijas de un tipo de vehículo."""

    key: str  # clave en el archivo de configuración: [vehicles.<key>] y [demand] <key>_rate
    name: str  # etiqueta en gráficas, resumen y CSV
    speed_kmh: float  # velocidad máxima que puede alcanzar el vehículo (media, si es variable)
    length: float  # m
    gap_run: float  # m, distancia al líder cuando éste está en marcha
    gap_stop: float  # m, distancia al líder detenido (compresión)
    pax_min: int
    pax_max: int
    pax_mean: float
    pax_std: float
    can_change_lane: bool
    lane: int | None = None  # carril fijo (0 = derecho) de un tipo que no cambia de carril; None = el derecho libre
    exclusive: bool = False  # su carril fijo queda reservado: solo lo usan los tipos con ese `lane`
    # Parada antes del semáforo (descenso y ascenso de pasajeros): posición del frente del
    # vehículo, en m desde el inicio del tramo (None = sin parada), y duración normal(μ, σ) en s,
    # truncada a ≥ 0.
    stop_position: float | None = None
    stop_time_mean: float = 20.0
    stop_time_std: float = 5.0
    # Al detenerse, cuántos vehículos de este tipo caben lado a lado en un carril (p. ej. 2 bicis).
    abreast: int = 1
    # Rebase dentro del carril (p. ej. bicis): en marcha, quien es más rápido que el de adelante de su
    # mismo tipo lo rebasa por un lado sin cambiar de carril. Requiere abreast ≥ 2.
    pass_in_lane: bool = False
    # Rebase agresivo (p. ej. motos): detecta al líder lento por su velocidad real y acepta un carril
    # de menor límite si ahí avanza más que detrás de él.
    overtake: bool = False
    # Umbrales de cambio de carril propios del tipo; None = los de [behavior].
    lookahead: float | None = None
    min_advantage: float | None = None
    lane_change_cooldown: float | None = None
    # Aceleración y frenado graduales (m/s²). Sin ellos (None), el vehículo pasa de inmediato a su
    # velocidad y se detiene en seco, como antes. Con `decel` además anticipa: no rebasa la velocidad a
    # la que aún puede detenerse frenando a `decel` detrás del de adelante o antes de un alto.
    accel: float | None = None
    decel: float | None = None
    # Probabilidad de que un vehículo que llega lleve mercancía en vez de pasajeros: ocupa la vía,
    # pero no cuenta en las métricas de pasajeros (1 = el tipo solo lleva mercancía).
    cargo_prob: float = 0.0
    # Largo variable: normal(length, length_std) truncada a [length_min, length_max] (por defecto,
    # length ∓ 3·length_std), sorteada para cada vehículo. Con length_std = 0, todos miden `length`.
    # En el TOML, las cuatro van en [vehicles.<clave>] length = {min, max, mean, std}.
    length_std: float = 0.0
    length_min: float | None = None
    length_max: float | None = None
    # Velocidad máxima variable: normal(speed_kmh, speed_std) truncada a [speed_min, speed_max] (por
    # defecto, speed_kmh ∓ 3·speed_std), sorteada para cada vehículo. Con speed_std = 0, todos van a
    # speed_kmh. En el TOML, las cuatro van en [vehicles.<clave>] speed_kmh = {min, max, mean, std}.
    speed_std: float = 0.0
    speed_min: float | None = None
    speed_max: float | None = None
    # Detenciones de [bottleneck]: probabilidad de que un vehículo del tipo se detenga una vez en el
    # tramo (0 = nunca) y duración normal(media, σ) en s, truncada a ≥ 0. Dónde ocurren (carriles y
    # zona) se define en [bottleneck].
    bottleneck_prob: float = 0.0
    bottleneck_time_mean: float = 30.0
    bottleneck_time_std: float = 0.0

    @property
    def speed(self) -> float:
        """Velocidad máxima media del tipo (el parámetro `mean`) en m/s."""
        return self.speed_kmh / 3.6

    @property
    def slowest_kmh(self) -> float:
        """Velocidad máxima más baja que puede tener un vehículo del tipo (km/h)."""
        if self.speed_std <= 0:
            return self.speed_kmh
        return self.speed_min if self.speed_min is not None else self.speed_kmh - 3 * self.speed_std

    @property
    def fastest_kmh(self) -> float:
        """Velocidad máxima más alta que puede tener un vehículo del tipo (km/h)."""
        if self.speed_std <= 0:
            return self.speed_kmh
        return self.speed_max if self.speed_max is not None else self.speed_kmh + 3 * self.speed_std

    @property
    def shortest(self) -> float:
        """Largo mínimo que puede tener un vehículo del tipo (m)."""
        if self.length_std <= 0:
            return self.length
        return self.length_min if self.length_min is not None else self.length - 3 * self.length_std

    @property
    def longest(self) -> float:
        """Largo máximo que puede tener un vehículo del tipo (m)."""
        if self.length_std <= 0:
            return self.length
        return self.length_max if self.length_max is not None else self.length + 3 * self.length_std

    @property
    def carries_passengers(self) -> bool:
        """Algún vehículo del tipo lleva pasajeros (no todos llevan mercancía)."""
        return self.cargo_prob < 1

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
    # Con reaction_std, la reacción es normal(reaction_mean, reaction_std) truncada a [min, max]
    # (media por defecto: el punto medio); sin ella, uniforme en [min, max]. En el TOML, las cuatro
    # van en [behavior] reaction = {min, max, mean, std}.
    reaction_mean: float | None = None
    reaction_std: float | None = None
    lane_change_min: float = 1.0  # s, duración de la maniobra de cambio de carril
    lane_change_max: float = 4.0
    lane_change_speed_factor: float = 0.5  # fracción de la velocidad durante la maniobra
    lane_change_cooldown: float = 3.0  # s sin volver a cambiar tras una maniobra
    lane_change_interval: float = 0.5  # s entre decisiones de cambio de carril
    lookahead: float = 30.0  # m, distancia a la que un líder lento motiva cambiar
    min_advantage: float = 5.0  # m de espacio libre extra para que valga la pena cambiar
    release_space: float = 0.5  # m, espacio que libera a un vehículo detenido
    # Solo se cambia de carril con un líder dentro de `lookahead` que va al menos esta fracción por
    # debajo del límite del carril (sin límite, de la velocidad máxima del vehículo).
    leader_slowdown: float = 0.25
    # En amarillo, quien está a menos de `yellow_approach` m de la línea de alto (sin haberla
    # cruzado) avanza a `yellow_speed_factor` de su velocidad.
    yellow_approach: float = 50.0
    yellow_speed_factor: float = 0.5
    # Cola de entrada detenida (la cola del tramo llega hasta la entrada y está detenida): cada vehículo
    # que espera en ella arranca con su tiempo de reacción cuando se mueve el de adelante, en vez de
    # entrar ya en marcha en cuanto hay lugar.
    queue_reaction: bool = False


@dataclass(frozen=True, slots=True)
class Bottleneck:
    """Cuellos de botella: dónde ocurren las detenciones. Cada vehículo que entra al tramo, con la
    probabilidad de su tipo (`bottleneck_prob`), se detendrá una vez en un punto al azar de
    `stop_zone`, si al llegar a él va por uno de `stop_lanes`, durante la duración de su tipo.
    Quien viene detrás lo trata como cualquier líder detenido (se comprime o se cambia de carril)."""

    stop_lanes: tuple[int, ...] | None = None  # carriles (0 = derecho); None = todos
    stop_zone: tuple[float, ...] | None = None  # (inicio, fin) en m del frente; None = todo el tramo


def per_lane(value: float | tuple[float, ...], lanes: int) -> tuple[float, ...]:
    """Un valor por carril a partir de un número o una lista. Si hay más carriles que valores,
    los extra usan el último; si hay menos, los sobrantes se ignoran."""
    values = tuple(value) if isinstance(value, tuple) else (float(value),)
    return tuple(values[min(i, len(values) - 1)] for i in range(lanes))


@dataclass(frozen=True, slots=True)
class Rate:
    """Tasa de llegada (veh/min) de un tipo: normal(mean, std) truncada a [min, max], que se vuelve a
    sortear en cada intervalo de [demand] rate_interval. Con std = 0 o min = max es fija."""

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
        """Tasa media real: la media de la normal truncada a [min, max]."""
        if not self.variable:
            return min(max(self.mean, self.min), self.max)
        mu, sigma = self.mean, self.std
        a, b = (self.min - mu) / sigma, (self.max - mu) / sigma
        # E[X | a ≤ Z ≤ b] = μ + σ·(φ(a) − φ(b)) / (Φ(b) − Φ(a)), con Z = (X − μ) / σ
        mass = 0.5 * (math.erf(b / math.sqrt(2)) - math.erf(a / math.sqrt(2)))
        if mass <= 0:  # media muy lejos de [min, max] (la validación lo rechaza): el extremo más cercano
            return self.min if mu < self.min else self.max
        return mu + sigma * (math.exp(-a * a / 2) - math.exp(-b * b / 2)) / math.sqrt(2 * math.pi) / mass


@dataclass(frozen=True, slots=True)
class SimConfig:
    """Configuración completa de una corrida."""

    length: float = 200.0  # m del tramo; el semáforo está en x = length
    lanes: int = 2
    lane_speed_limit: float | tuple[float, ...] | None = None  # km/h por carril (0 = derecho); None = sin límite
    rates: tuple[float, ...] = DEFAULT_RATES  # veh/min medias, alineadas con `specs`
    # Tasas variables, una por tipo; si se dan, `rates` se calcula de ellas (su media real). Vacía = fijas.
    rate_dists: tuple[Rate, ...] = ()
    rate_interval: float = 60.0  # s simulados entre sorteos de las tasas variables
    red: float = 30.0  # s
    green: float = 30.0  # s
    yellow: float = 0.0  # s, entre el verde y el rojo; 0 = sin amarillo
    start_phase: str = "red"  # "red" | "green"
    # Carriles sin semáforo (0 = derecho), p. ej. una vuelta continua. Solo aplica a los tipos con carril
    # exclusivo (`lane` y `exclusive`) en uno de ellos: cruzan la línea aunque esté en rojo y no bajan la
    # velocidad en amarillo.
    free_lanes: tuple[int, ...] = ()
    run: float = 10.0  # s de proceso
    time_scale: float = 10.0  # s simulados por cada s de proceso
    sample: float = 1.0  # s simulados entre muestras
    # Condición inicial: fracción de cada carril ya ocupada al empezar por vehículos en marcha,
    # (largo + gap_run) / length; un número para todos o uno por carril. 0 = tramo vacío.
    initial_occupancy: float | tuple[float, ...] = 0.0
    # Cola de salida después del semáforo, por carril (un número para todos o uno por carril): cuántos
    # vehículos por minuto acepta (0 = sin cola de salida) y cuántos m mide (cada vehículo ocupa su largo +
    # gap detenido). Quien no cabe no puede cruzar la línea aunque esté en verde.
    exit_capacity: float | tuple[float, ...] = 0.0
    exit_storage: float | tuple[float, ...] = 60.0
    specs: tuple[VehicleSpec, ...] = DEFAULT_SPECS
    behavior: Behavior = field(default_factory=Behavior)
    bottleneck: Bottleneck = field(default_factory=Bottleneck)

    def __post_init__(self) -> None:
        if self.rate_dists:
            if len(self.rate_dists) != len(self.specs):
                raise ValueError(f"se esperaban {len(self.specs)} tasas de llegada, una por tipo; hay {len(self.rate_dists)}")
            object.__setattr__(self, "rates", tuple(d.expected for d in self.rate_dists))
        if len(self.rates) != len(self.specs):
            raise ValueError(f"se esperaban {len(self.specs)} tasas de llegada, una por tipo; hay {len(self.rates)}")

    def rate(self, k: int) -> Rate:
        """Tasa de llegada del tipo k (fija si no hay `rate_dists`)."""
        return self.rate_dists[k] if self.rate_dists else Rate.fixed(self.rates[k])

    @property
    def rate_ticks(self) -> int:
        return max(1, round(self.rate_interval / DT))

    @property
    def n_types(self) -> int:
        return len(self.specs)

    def bottleneck_prob(self, k: int) -> float:
        """Probabilidad de que un vehículo del tipo k se detenga en un cuello de botella."""
        return self.specs[k].bottleneck_prob

    def bottleneck_time(self, k: int) -> tuple[float, float]:
        """Duración (media, σ) en s de la detención del tipo k."""
        return self.specs[k].bottleneck_time_mean, self.specs[k].bottleneck_time_std

    @property
    def bottleneck_active(self) -> bool:
        """Algún tipo que participa puede detenerse en un cuello de botella."""
        return any(rate > 0 and self.bottleneck_prob(k) > 0 for k, rate in enumerate(self.rates))

    def bottleneck_lanes(self) -> tuple[int, ...]:
        lanes = self.bottleneck.stop_lanes
        return tuple(range(self.lanes)) if lanes is None else lanes

    def bottleneck_zone(self) -> tuple[float, float]:
        zone = self.bottleneck.stop_zone
        return (0.0, self.length) if zone is None else (zone[0], zone[1])

    @property
    def lane_initial_occupancy(self) -> tuple[float, ...]:
        return per_lane(self.initial_occupancy, self.lanes)

    @property
    def lane_exit_capacity(self) -> tuple[float, ...]:
        """veh/min que acepta la cola de salida de cada carril (0 = sin cola de salida)."""
        return per_lane(self.exit_capacity, self.lanes)

    @property
    def lane_exit_storage(self) -> tuple[float, ...]:
        """m que mide la cola de salida de cada carril (cada vehículo ocupa su largo + gap detenido)."""
        return per_lane(self.exit_storage, self.lanes)

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

    def ignores_light(self, k: int) -> bool:
        """El tipo k no obedece el semáforo: tiene carril exclusivo y ese carril está en `free_lanes`."""
        spec = self.specs[k]
        return spec.exclusive and spec.lane is not None and spec.lane in self.free_lanes

    def allowed_lanes(self, k: int) -> tuple[int, ...]:
        """Carriles que puede usar el tipo k: su carril fijo o, si no tiene, los no reservados."""
        spec = self.specs[k]
        if spec.lane is not None:
            return (spec.lane,)
        reserved = self.reserved_lanes
        return tuple(i for i in range(self.lanes) if i not in reserved)

    def entry_lanes(self, k: int) -> tuple[int, ...]:
        """Carriles por los que entra el tipo k. Uno que no cambia de carril y no tiene `lane` va por el
        carril libre más a la derecha (sin los exclusivos de otros tipos)."""
        allowed = self.allowed_lanes(k)
        spec = self.specs[k]
        if spec.can_change_lane or spec.lane is not None:
            return allowed
        return allowed[:1]  # el carril libre más a la derecha

    def free_flow_kmh(self, k: int) -> float:
        """Velocidad a flujo libre del tipo k: su máxima (la media, si es variable), sin rebasar el límite
        del mejor carril que puede usar."""
        return min(self.specs[k].speed_kmh, self._best_limit_kmh(k))

    def free_flow_time(self, k: int) -> float:
        """Tiempo medio (s) en recorrer el tramo a flujo libre para el tipo k, sin paradas. Con velocidad
        variable es la media de length / min(v, límite) sobre la normal truncada de v, no length entre
        la velocidad media."""
        spec = self.specs[k]
        limit = self._best_limit_kmh(k)
        if spec.speed_std <= 0:
            return self.length / (min(spec.speed_kmh, limit) / 3.6)
        # Regla del punto medio sobre [slowest, fastest] con la densidad normal (la truncadura normaliza).
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
        """Límite del carril más rápido por el que entra el tipo k (inf si no hay límites)."""
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
    def light_label(self) -> str:
        """Duración de las fases del semáforo, p. ej. «rojo 25 s / verde 35 s / amarillo 3 s»."""
        label = f"rojo {self.red:g} s / verde {self.green:g} s"
        return label + (f" / amarillo {self.yellow:g} s" if self.yellow > 0 else "")

    @property
    def cycle(self) -> float:
        return self.red + self.green + self.yellow

    @property
    def sample_ticks(self) -> int:
        return max(1, round(self.sample / DT))

    @property
    def n_samples(self) -> int:
        return self.n_ticks // self.sample_ticks

    def phase(self, tick: int) -> int:
        """Fase del semáforo (RED, GREEN o YELLOW) en el paso `tick`: verde → amarillo → rojo."""
        red, green, yellow = self.red_ticks, self.green_ticks, self.yellow_ticks
        pos = tick % (red + green + yellow)
        if self.start_phase == "red":
            pos = (pos - red) % (red + green + yellow)  # el ciclo empieza en el rojo
        if pos < green:
            return GREEN
        return YELLOW if pos < green + yellow else RED

    def is_green(self, tick: int) -> bool:
        return self.phase(tick) == GREEN

    def is_red(self, tick: int) -> bool:
        return self.phase(tick) == RED

    def red_intervals(self) -> list[tuple[float, float]]:
        """Intervalos [inicio, fin) en s simulados en los que el semáforo está en rojo."""
        return self._intervals(self.green + self.yellow, self.red)

    def yellow_intervals(self) -> list[tuple[float, float]]:
        """Intervalos [inicio, fin) en s simulados en los que el semáforo está en amarillo."""
        return self._intervals(self.green, self.yellow)

    def _intervals(self, offset: float, duration: float) -> list[tuple[float, float]]:
        """Intervalos de la fase que empieza `offset` s después del inicio del verde y dura `duration` s."""
        if duration <= 0:
            return []
        out = []
        # Un ciclo antes del primer verde (en t = red si empieza en rojo): cubre la fase ya en curso en t = 0.
        start = (self.red if self.start_phase == "red" else 0.0) + offset - self.cycle
        while start < self.sim_seconds:
            if start + duration > 0:
                out.append((max(start, 0.0), min(start + duration, self.sim_seconds)))
            start += self.cycle
        return out
