"""Parámetros de la simulación: especificaciones de vehículos, comportamiento y corrida."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

from trafico.emission_sets import BUS_EMISSIONS, PETROL_CAR_EMISSIONS

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

FUEL_LABELS = {"gasolina": "gasolina", "diesel": "diésel", "glp": "gas LP"}

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
    # Velocidad máxima (km/h) a la que pasa el tope de [speed_bump]; None = pasa sin frenar.
    speed_bump_kmh: float | None = None
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
    # Emisiones instantáneas (Int Panis et al., 2006): por contaminante, los coeficientes [f1..f6] de
    # E = max(0, f1 + f2·v + f3·v² + f4·a + f5·a² + f6·v·a) en g/s (v en m/s, a en m/s²) para a ≥ −0.5 m/s² y
    # para a < −0.5 m/s². Vacío = el tipo no emite. Solo se calculan con aceleración y frenado graduales.
    emissions: tuple[tuple[str, tuple[float, ...], tuple[float, ...]], ...] = ()
    # Conjunto de coeficientes del que salen (`EMISSION_SETS`); "" si se escribieron a mano.
    emission_source: str = ""
    # Combustible (uno de FUELS; None = no quema combustible): el consumo sale del CO2 emitido. La masa (kg)
    # solo se usa en la curva física de referencia del rendimiento a velocidad constante.
    fuel: str | None = None
    mass_kg: float | None = None

    @property
    def burns(self) -> bool:
        """Se calcula su consumo: tiene combustible y emite CO2."""
        return self.fuel is not None and self.emits and self.emission_coefs("co2") is not None

    @property
    def emits(self) -> bool:
        """El tipo tiene coeficientes de emisión y dinámica gradual (accel y decel), que el modelo necesita."""
        return bool(self.emissions) and self.accel is not None and self.decel is not None

    def emission_coefs(self, pollutant: str) -> tuple[tuple[float, ...], tuple[float, ...]] | None:
        """(coeficientes para a ≥ −0.5, para a < −0.5) del contaminante; None si el tipo no lo emite."""
        return next(((acc, dec) for name, acc, dec in self.emissions if name == pollutant), None)

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
# define completo en el archivo de configuración, sin tocar el código. El auto y el autobús traen
# coeficientes de emisión, que solo se usan si el tipo tiene accel y decel.
DEFAULT_SPECS: tuple[VehicleSpec, ...] = (
    VehicleSpec("car", "auto", 50.0, 4.5, 3.0, 1.0, 1, 6, 1.5, 1.0, True, emissions=PETROL_CAR_EMISSIONS,
                emission_source="int_panis_2006", fuel="gasolina", mass_kg=1250.0),
    VehicleSpec("bike", "bici", 15.0, 1.8, 1.0, 0.5, 1, 1, 1.0, 0.0, False),
    # El gap en marcha del autobús no está especificado: se asume 4 m.
    VehicleSpec("bus", "autobús", 40.0, 12.0, 4.0, 1.5, 1, 80, 40.0, 10.0, False, emissions=BUS_EMISSIONS,
                emission_source="int_panis_2006", fuel="diesel", mass_kg=12000.0),
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


PEDESTRIAN_RATE = Rate(0.5, 3.0, 1.5, 1.0)
"""Aparición de peatones (peatones/min) cuando `pedestrian = true` no trae `pedestrian_crossing`."""

PEDESTRIAN_TIME = 10.0
"""s que tarda un peatón en cruzar el tope (`[speed_bump] pedestrian_time`) si no se da."""

PEDESTRIAN_YIELD = 3.0
"""s que un peatón espera en la orilla del tope antes de cruzar: los vehículos lo ven (fase amarilla) y se detienen."""


@dataclass(frozen=True, slots=True)
class SpeedBump:
    """Un tope (reductor de velocidad) a `position` m del inicio del tramo, en los carriles `lanes`. Mientras un
    vehículo lo pisa (del frente a la parte trasera) no rebasa el `speed_bump_kmh` de su tipo. Con `pedestrian`, el
    tope es además un paso peatonal: aparecen peatones (`pedestrian_crossing`, peatones/min) que cruzan en
    `pedestrian_time` s cada uno, y los vehículos se detienen del todo mientras cruza alguno. Una calle puede tener
    varios (`SimConfig.speed_bumps`), cada uno con su configuración."""

    position: float  # m desde el inicio del tramo
    lanes: tuple[int, ...] | None = None  # carriles (0 = derecho); None = todos
    pedestrian: bool = False
    pedestrian_crossing: Rate = PEDESTRIAN_RATE
    pedestrian_time: float = PEDESTRIAN_TIME


def as_positions(value: float | tuple[float, ...] | list[float] | None) -> tuple[float, ...]:
    """Posiciones (m) de una posición suelta, una lista de ellas o None, de la entrada a la salida (vacía si None)."""
    if value is None:
        return ()
    if isinstance(value, (tuple, list)):
        return tuple(sorted(float(p) for p in value))
    return (float(value),)


def phase_at(red: int, green: int, yellow: int, start_phase: str, tick: int) -> int:
    """Fase (RED, GREEN o YELLOW) en el paso `tick` de un semáforo con esas fases (en pasos): verde → amarillo →
    rojo, empezando por el rojo o por el verde."""
    cycle = red + green + yellow
    pos = tick % cycle
    if start_phase == "red":
        pos = (pos - red) % cycle  # el ciclo empieza en el rojo
    if pos < green:
        return GREEN
    return YELLOW if pos < green + yellow else RED


@dataclass(frozen=True, slots=True)
class Light:
    """Un semáforo del tramo a `position` m del inicio, con sus propias fases y fase inicial. El del final del
    tramo (x = length) lo describen `red`, `green`, `yellow`, `start_phase`, `traffic_light` y `free_lanes` de
    SimConfig (`SimConfig.exit_light`); los intermedios van en `SimConfig.extra_lights`."""

    position: float
    red: float = 30.0  # s
    green: float = 30.0  # s
    yellow: float = 0.0  # s, entre el verde y el rojo; 0 = sin amarillo
    start_phase: str = "red"  # "red" | "green"
    enabled: bool = True
    # Carriles sin este semáforo (0 = derecho): solo aplica a los tipos con carril exclusivo en ellos.
    free_lanes: tuple[int, ...] = ()
    # Semáforo peatonal: verde para los vehículos mientras no haya peatones esperando (aparecen a razón de
    # `pedestrian_crossing`, peatones/min). Con alguno en la cola y pasado el verde mínimo (`green`), y cuando el
    # semáforo de ciclo fijo anterior está en rojo, pasa a amarillo (`yellow`) y a rojo (`red`, lo que tardan en
    # cruzar). No tiene ciclo ni `start_phase`: su calendario sale de la simulación (`pedestrians.light_schedule`).
    pedestrian: bool = False
    pedestrian_crossing: Rate = PEDESTRIAN_RATE

    @property
    def active(self) -> bool:
        """Hay semáforo: está activado y alguna de sus fases dura más de 0 s (el peatonal basta con que esté activado)."""
        return self.enabled and (self.pedestrian or self.red + self.green + self.yellow > 0)

    @property
    def cycle(self) -> float:
        """Duración del ciclo (s); 0 si no hay semáforo o es peatonal (no tiene ciclo)."""
        return self.red + self.green + self.yellow if self.active and not self.pedestrian else 0.0

    @property
    def label(self) -> str:
        """Fases, p. ej. «rojo 25 s / verde 35 s / amarillo 3 s»; el peatonal, «peatonal: rojo 15 s con peatones…»."""
        if self.pedestrian:
            label = f"peatonal: rojo {self.red:g} s con peatones en espera, verde mínimo {self.green:g} s"
            return label + (f", amarillo {self.yellow:g} s" if self.yellow > 0 else "")
        label = f"rojo {self.red:g} s / verde {self.green:g} s"
        return label + (f" / amarillo {self.yellow:g} s" if self.yellow > 0 else "")

    def phase(self, tick: int) -> int:
        """Fase (RED, GREEN o YELLOW) en el paso `tick`: verde → amarillo → rojo. Sin semáforo, siempre verde; el
        peatonal también (su fase real depende de los peatones de cada réplica: la da la simulación)."""
        if not self.active or self.pedestrian:
            return GREEN
        return phase_at(round(self.red / DT), round(self.green / DT), round(self.yellow / DT), self.start_phase, tick)

    def intervals(self, offset: float, duration: float, horizon: float) -> list[tuple[float, float]]:
        """Intervalos [inicio, fin) en s simulados, hasta `horizon`, de la fase que empieza `offset` s después del
        inicio del verde y dura `duration` s."""
        if duration <= 0 or not self.active or self.pedestrian:
            return []
        out = []
        # Un ciclo antes del primer verde (en t = red si empieza en rojo): cubre la fase ya en curso en t = 0.
        start = (self.red if self.start_phase == "red" else 0.0) + offset - self.cycle
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
    """Línea de alto intermedia donde los vehículos pueden tener que detenerse: un semáforo intermedio (`light`) o el
    paso peatonal de un tope (`light` es None), a `position` m. `lanes` son los carriles donde rige (None = todos)."""

    position: float
    light: Light | None = None
    lanes: tuple[int, ...] | None = None

    @property
    def free_lanes(self) -> tuple[int, ...]:
        return self.light.free_lanes if self.light is not None else ()

    @property
    def pedestrian(self) -> bool:
        """Su calendario depende de los peatones: un paso peatonal o un semáforo peatonal."""
        return self.light is None or self.light.pedestrian

    @property
    def kind(self) -> str:
        return "tope" if self.light is None else "semáforo"


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
    # Semáforo activado. Desactivado (o con red, green y yellow en 0) no hay semáforo: la línea al final
    # del tramo siempre está abierta.
    traffic_light: bool = True
    # Carriles sin semáforo (0 = derecho), p. ej. una vuelta continua. Solo aplica a los tipos con carril
    # exclusivo (`lane` y `exclusive`) en uno de ellos: cruzan la línea aunque esté en rojo y no bajan la
    # velocidad en amarillo.
    free_lanes: tuple[int, ...] = ()
    # El semáforo del final del tramo es peatonal (ver `Light.pedestrian`); en ese caso `red` es lo que dura el
    # rojo, `green` el verde mínimo y `start_phase` no se usa.
    light_pedestrian: bool = False
    light_pedestrian_crossing: Rate = PEDESTRIAN_RATE
    # Semáforos intermedios (además del del final del tramo), cada uno a su `position` m con sus propias fases.
    extra_lights: tuple[Light, ...] = ()
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
    # Topes, cada uno con su posición, carriles y peatones (ver `bumps` para tenerlos ordenados).
    speed_bumps: tuple[SpeedBump, ...] = ()
    # Precio de cada combustible por litro, en `currency`; sin precio, el consumo solo se da en litros.
    fuel_price: tuple[tuple[str, float], ...] = ()
    currency: str = "MXN"

    def __post_init__(self) -> None:
        if self.rate_dists:
            if len(self.rate_dists) != len(self.specs):
                raise ValueError(f"se esperaban {len(self.specs)} tasas de llegada, una por tipo; hay {len(self.rate_dists)}")
            object.__setattr__(self, "rates", tuple(d.expected for d in self.rate_dists))
        if len(self.rates) != len(self.specs):
            raise ValueError(f"se esperaban {len(self.specs)} tasas de llegada, una por tipo; hay {len(self.rates)}")

    def price(self, fuel: str | None) -> float | None:
        """Precio por litro del combustible (None si no tiene)."""
        return next((p for f, p in self.fuel_price if f == fuel), None)

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

    @property
    def bumps(self) -> tuple[SpeedBump, ...]:
        """Los topes, de la entrada a la salida."""
        return tuple(sorted(self.speed_bumps, key=lambda b: b.position))

    def bump_lanes(self, bump: SpeedBump) -> tuple[int, ...]:
        """Carriles donde está el tope."""
        return tuple(range(self.lanes)) if bump.lanes is None else bump.lanes

    @property
    def crossings(self) -> tuple[SpeedBump, ...]:
        """Los topes que son paso peatonal (y están en algún carril), de la entrada a la salida."""
        return tuple(b for b in self.bumps if b.pedestrian and self.bump_lanes(b))

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

    def ignores_light(self, k: int, light: Light | None = None) -> bool:
        """El tipo k no obedece el semáforo (por defecto, el del final del tramo): tiene carril exclusivo y ese
        carril está en sus `free_lanes`."""
        spec = self.specs[k]
        lanes = self.free_lanes if light is None else light.free_lanes
        return spec.exclusive and spec.lane is not None and spec.lane in lanes

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
    def exit_light(self) -> Light:
        """El semáforo del final del tramo (x = length), con los campos `red`, `green`, `yellow`, `start_phase`,
        `traffic_light` y `free_lanes`."""
        return Light(self.length, self.red, self.green, self.yellow, self.start_phase, self.traffic_light,
                     self.free_lanes, self.light_pedestrian, self.light_pedestrian_crossing)  # fmt: skip

    @property
    def has_light(self) -> bool:
        """Hay semáforo al final del tramo: está activado y alguna de sus fases dura más de 0 s."""
        return self.exit_light.active

    @property
    def inner_lights(self) -> tuple[Light, ...]:
        """Los semáforos intermedios que funcionan, de la entrada a la salida."""
        return tuple(sorted((lt for lt in self.extra_lights if lt.active), key=lambda lt: lt.position))

    @property
    def lights(self) -> tuple[Light, ...]:
        """Todos los semáforos que funcionan (intermedios y el del final del tramo), de la entrada a la salida."""
        return self.inner_lights + ((self.exit_light,) if self.has_light else ())

    @property
    def any_light(self) -> bool:
        return bool(self.lights)

    @property
    def stop_lines(self) -> tuple[StopLine, ...]:
        """Líneas de alto intermedias, de la entrada a la salida: los semáforos intermedios y los pasos peatonales de
        los topes (con un semáforo y un paso en el mismo punto, primero el semáforo). El motor y el video las
        recorren en este orden."""
        lines = [StopLine(lt.position, lt) for lt in self.inner_lights]
        lines += [StopLine(b.position, None, self.bump_lanes(b)) for b in self.crossings]
        return tuple(sorted(lines, key=lambda s: (s.position, s.light is None)))

    @property
    def pedestrian_spots(self) -> tuple[tuple[str, float], ...]:
        """(«semáforo» o «tope», posición en m) de cada punto con peatones, de la entrada a la salida. Son las filas de
        `summary["pedestrians"]`."""
        spots = [(s.kind, s.position) for s in self.stop_lines if s.pedestrian]
        if self.exit_light.active and self.exit_light.pedestrian:
            spots.append(("semáforo", self.length))
        return tuple(sorted(spots, key=lambda s: (s[1], s[0] == "tope")))

    def previous_light(self, light: Light) -> Light | None:
        """El semáforo de ciclo fijo más cercano antes de `light` (hacia la entrada), con el que se coordina un
        semáforo peatonal: espera a que esté en rojo. None si no hay."""
        before = [lt for lt in self.lights if not lt.pedestrian and lt.position < light.position]
        return before[-1] if before else None

    @property
    def ref_light(self) -> Light | None:
        """El semáforo que marca la ventana del flujo y las fases sombreadas de las gráficas: el último de ciclo fijo
        (el del final del tramo o, si no hay, el último intermedio). Un semáforo peatonal no tiene ciclo."""
        fixed = [lt for lt in self.lights if not lt.pedestrian]
        return fixed[-1] if fixed else None

    @property
    def light_label(self) -> str:
        """Fases del semáforo del final del tramo, p. ej. «rojo 25 s / verde 35 s / amarillo 3 s», o «sin
        semáforo»."""
        return self.exit_light.label if self.has_light else "sin semáforo"

    @property
    def lights_label(self) -> str:
        """Los semáforos de la calle: con uno solo, como `light_label`; con varios, cada uno con su posición y fases."""
        if not self.inner_lights:
            return self.light_label
        return " · ".join(f"a {lt.position:g} m: {lt.label}"
                          + ("" if lt.pedestrian else f", empieza en {'rojo' if lt.start_phase == 'red' else 'verde'}")
                          for lt in self.lights)  # fmt: skip

    @property
    def lights_short(self) -> str:
        """Como `lights_label`, pero corto para los subtítulos de las gráficas: con varios semáforos, cuántos y dónde."""
        if not self.inner_lights:
            return self.light_label
        at = [f"{lt.position:g}" for lt in self.lights]
        return f"{len(at)} semáforos (a {', '.join(at[:-1])} y {at[-1]} m)"

    @property
    def line_name(self) -> str:
        """Cómo se llama la línea al final del tramo en textos: «el semáforo» o, sin él, «el final del tramo»."""
        return "el semáforo" if self.has_light else "el final del tramo"

    @property
    def cycle(self) -> float:
        """Duración del ciclo del semáforo de referencia (s); 0 sin semáforo."""
        ref = self.ref_light
        return ref.cycle if ref else 0.0

    @property
    def flow_window(self) -> float:
        """Ventana (s) del flujo en la línea: un ciclo del semáforo de referencia o, sin él, NO_LIGHT_WINDOW."""
        return self.cycle if self.ref_light else NO_LIGHT_WINDOW

    @property
    def flow_window_label(self) -> str:
        return f"ventana de un ciclo, {self.cycle:g} s" if self.ref_light else f"ventana de {NO_LIGHT_WINDOW:g} s"

    @property
    def sample_ticks(self) -> int:
        return max(1, round(self.sample / DT))

    @property
    def n_samples(self) -> int:
        return self.n_ticks // self.sample_ticks

    def with_lights(self, enabled: bool) -> SimConfig:
        """La misma configuración con todos los semáforos (el del final del tramo y los intermedios) activados o
        desactivados."""
        return replace(self, traffic_light=enabled,
                       extra_lights=tuple(replace(lt, enabled=enabled) for lt in self.extra_lights))  # fmt: skip

    def phase(self, tick: int) -> int:
        """Fase del semáforo del final del tramo (RED, GREEN o YELLOW) en el paso `tick`: verde → amarillo → rojo.
        Sin semáforo, siempre verde."""
        return self.exit_light.phase(tick)

    def inner_phases(self, tick: int) -> tuple[int, ...]:
        """Fase de cada semáforo intermedio (`inner_lights`) en el paso `tick`."""
        return tuple(lt.phase(tick) for lt in self.inner_lights)

    def is_green(self, tick: int) -> bool:
        return self.phase(tick) == GREEN

    def is_red(self, tick: int) -> bool:
        return self.phase(tick) == RED

    def red_intervals(self) -> list[tuple[float, float]]:
        """Intervalos [inicio, fin) en s simulados en los que el semáforo de referencia está en rojo."""
        ref = self.ref_light
        return ref.red_intervals(self.sim_seconds) if ref else []

    def yellow_intervals(self) -> list[tuple[float, float]]:
        """Intervalos [inicio, fin) en s simulados en los que el semáforo de referencia está en amarillo."""
        ref = self.ref_light
        return ref.yellow_intervals(self.sim_seconds) if ref else []
