"""Motor microscópico: tramo de `lanes` carriles con un semáforo en x = length.

Estado en estructura de arreglos (SoA) compacta: los vehículos activos ocupan
las posiciones [0, n) de cada arreglo; los que salen se eliminan compactando.
La capacidad inicial es la cota física de vehículos que caben en el tramo, así
que la memoria no crece con la duración de la corrida.

Modelo por paso de DT segundos (actualización paralela con posiciones previas):
  * Cada vehículo avanza a su velocidad máxima (sorteada al llegar si la del tipo
    es variable) sin rebasar el límite del carril; el avance está limitado por el
    espacio al líder del carril y, en rojo, por la línea de alto.
  * Semáforo: ciclo verde → amarillo → rojo. En amarillo la línea de alto sigue
    abierta, pero quien está a menos de `yellow_approach` m de ella avanza a
    `yellow_speed_factor` de su velocidad; en rojo, se detiene. Los tipos con
    carril exclusivo en uno de `free_lanes` no obedecen ni el rojo ni el
    amarillo (no cambian de carril, así que siempre van por él). Además del
    semáforo del final del tramo puede haber otros a `position` m (`Light`),
    cada uno con sus fases, su fase inicial y sus `free_lanes`: se detienen en
    su línea como en la del final, pero el flujo y las colas se miden en la
    del final del tramo.
  * Peatones (opcionales, `pedestrian`): un semáforo peatonal es verde para los vehículos hasta que hay peatones
    esperando; un tope puede ser paso peatonal. Su calendario de fases (verde/amarillo/rojo) se calcula al crear la
    réplica (pedestrians.py) y los vehículos lo obedecen como a un semáforo intermedio, en los carriles del tope.
  * Condición inicial (opcional): los carriles empiezan con vehículos en marcha
    que ocupan una fracción del tramo, para no esperar a que se llene.
  * Compresión: detrás de un líder detenido basta `gap_stop` (< `gap_run`).
  * Estiramiento: un vehículo detenido que queda libre (se puso verde o arrancó
    su líder) espera un tiempo de reacción U[1,5] s antes de avanzar, y además
    necesita reabrir `gap_run` detrás del líder en marcha.
  * Aceleración y frenado graduales (tipos con `accel`/`decel`): la velocidad
    sube a lo más `accel`·DT por paso hacia la deseada y baja a lo más `decel`·DT
    si la deseada baja; el gap detrás de un líder en marcha va de `gap_stop`
    (casi detenido) a `gap_run` (a su velocidad deseada). Además no rebasa la velocidad a la que aún puede
    detenerse, frenando a `decel` (con un paso de reacción), detrás del de
    adelante (contando lo que éste necesita para detenerse), antes de la línea en
    rojo o de una parada; en amarillo se detiene si todavía puede hacerlo y, si
    no, cruza sin bajar la velocidad (no aplica `yellow_speed_factor`). El
    espacio al de adelante sigue siendo un límite duro (frenado de emergencia).
    Al arrancar tras la reacción parte de 0.
  * Cola de salida (opcional, [exit]): quien cruza la línea entra a la cola de
    salida de su carril, que acepta a lo más `capacity` veh/min y mide
    `storage` m; cada vehículo ocupa en ella su largo + su gap detenido. Si el
    siguiente no cabe, la línea de ese carril se cierra para él como en rojo
    (aunque esté en verde): se detiene en ella y arranca con su reacción cuando
    hay lugar; si llegan más de los que acepta, el carril se satura. Una cola
    vacía acepta a cualquiera.
  * Topes (opcionales, una sección [[speed_bump]] cada uno): en sus carriles `lanes`, a `position` m del
    inicio, quien lo pisa (del frente a la parte trasera) no rebasa el
    `speed_bump_kmh` de su tipo; sin `decel` baja de golpe, con `decel` frena
    antes para llegar a esa velocidad. Los tipos sin `speed_bump_kmh` no frenan.
  * Cola de entrada: los vehículos que no caben esperan fuera del tramo y entran
    en marcha en cuanto hay lugar. Con `queue_reaction`, si la cola del tramo
    llega detenida hasta la entrada, la cola de entrada también está detenida:
    cada vehículo arranca con su tiempo de reacción cuando se mueve el de
    adelante, como en el resto de la cola.
  * Cambio de carril (solo tipos que pueden): la maniobra dura U[1,4] s, el
    vehículo ocupa ambos carriles a velocidad reducida y los seguidores de los
    dos carriles se comprimen detrás de él. Se cambia para rebasar a un líder
    lento o para subir a un carril adyacente de mayor límite cuando hay lugar.
    Los tipos con `overtake` (p. ej. motos) juzgan al líder por su velocidad
    real y bajan incluso a un carril de menor límite si ahí avanzan más; cada
    tipo puede fijar sus umbrales (`lookahead`, `min_advantage`, enfriamiento).
  * Lado a lado (tipos con `abreast` > 1, p. ej. bicis): quien llega detrás de
    un vehículo detenido de su mismo tipo se detiene a su lado, con los frentes
    alineados, mientras la fila tenga lugar; el siguiente hace fila detrás. Al
    arrancar salen uno tras otro (reacción y gap en marcha habituales).
  * Rebase dentro del carril (tipos con `pass_in_lane`, p. ej. bicis): en marcha,
    quien tiene una velocidad máxima mayor (acotada por el límite del carril) que
    el de adelante de su mismo tipo lo rebasa por un lado sin cambiar de carril;
    solo respeta al siguiente vehículo. El rebasado no frena por quien se le
    adelanta hasta que éste le saca `gap_run`. Nunca van más de dos lado a lado.
  * Parada (tipos con `stop_position`, p. ej. el autobús): el vehículo se detiene
    con el frente en esa posición durante un tiempo normal(μ, σ) de descenso y
    ascenso de pasajeros, y luego arranca sin tiempo de reacción adicional. Quien
    viene detrás en su carril espera y se comprime.
"""

from __future__ import annotations

import math
from collections import deque
from typing import NamedTuple

import numpy as np

from trafico.config import DT, EMISSION_DECEL, GREEN, PEDESTRIAN_YIELD, RED, YELLOW, SimConfig
from trafico.distributions import (
    normal_ticks, reaction_ticks, sample_lengths, sample_passengers, sample_rates, sample_speeds, stop_ticks,
    uniform_ticks,
)  # fmt: skip
from trafico.emissions import EMIS_BIN, coefficient_table
from trafico.metrics import Recorder
from trafico.pedestrians import Schedule, arrivals, bump_schedule, fixed_phases, light_schedule

EPS = 1e-6
INF = np.inf
SIDE_EPS = 0.01  # m que queda atrás el frente de quien se detiene al lado (conserva el orden)
ARRIVAL_CHUNK = 1024  # pasos por bloque de sorteo de llegadas
PER_TYPE_STREAMS = 5  # generadores aleatorios por tipo de vehículo
MAX_PAX = 255  # cota de pasajeros por vehículo (pax es uint8)
_TRUE = np.ones(1, np.bool_)  # para concatenar (np.r_ es varias veces más lento en arreglos chicos)
INITIAL_MISSES = 30  # sorteos seguidos que no caben antes de dar por lleno un carril en la condición inicial


def _safe_speed(space, v_lead, dec, dec_lead) -> np.ndarray:
    """Velocidad máxima (m/paso) con la que aún se detiene frenando a `dec` (m/paso²), con un paso de
    reacción, sin rebasar `space` m más lo que el de adelante (a `v_lead`, frenando a `dec_lead`)
    recorre al detenerse: v + v²/(2·dec) ≤ space + v_lead²/(2·dec_lead) (regla de Gipps). inf donde
    dec es infinito (frenado instantáneo, sin anticipación) o no hay nada adelante; un líder de frenado
    instantáneo se detiene en seco.

    Se supone que el de adelante frena al menos tan fuerte como él (dec_lead ≥ dec): así, siguiendo a
    un líder a velocidad constante, el equilibrio es space = v_lead, el mismo que impone el límite duro
    de espacio (actualización en paralelo), y se acerca sin frenazos. Con un líder que frena menos
    (p. ej. un auto detrás de una bici), la regla lo dejaría acercarse más y el límite duro lo frenaría
    de golpe."""
    with np.errstate(invalid="ignore"):  # inf − inf con dec infinito: se vuelve inf abajo
        lead = np.where(np.isfinite(dec_lead), v_lead * v_lead * dec / np.maximum(dec_lead, dec), 0.0)
        out = -dec + np.sqrt(dec * dec + 2 * dec * np.maximum(space, 0.0) + lead)
    return np.where(np.isnan(out), INF, out)


def _gipps(space, lead, dec, finite) -> np.ndarray:
    """La velocidad de `_safe_speed` sin `errstate` ni NaN, para el paso: `lead` es el término del de adelante
    (v_lead²·dec/max(dec_lead, dec), o 0 si no cuenta), `dec` no tiene infinitos (donde el frenado del tipo es
    infinito trae un valor positivo cualquiera) y `finite` marca dónde es finito; None = en todos. Da los mismos
    números que `_safe_speed`: sumar el término 0 no cambia nada y −dec + raíz = raíz − dec."""
    out = np.sqrt(dec * dec + 2 * dec * np.maximum(space, 0.0) + lead) - dec
    return out if finite is None else np.where(finite, out, INF)


def _cruise_rate(c: np.ndarray, v) -> np.ndarray:
    """emission_rate(coef, v, 0.0) con `c` = los coeficientes del juego de a ≥ −0.5, sin los términos de la aceleración
    (con a = 0 suman ±0, que no cambia el resultado): la tasa a velocidad constante de `_emit`."""
    return np.maximum(c[..., 0] + c[..., 1] * v + c[..., 2] * v * v, 0.0)


def _safe_speed_1(space: float, v_lead: float, dec: float, dec_lead: float) -> float:
    """_safe_speed para un solo vehículo (sin arreglos)."""
    if not math.isfinite(dec):
        return INF
    lead = v_lead * v_lead * dec / max(dec_lead, dec) if math.isfinite(dec_lead) else 0.0
    return -dec + math.sqrt(dec * dec + 2 * dec * max(space, 0.0) + lead)


def _next(a: np.ndarray) -> np.ndarray:
    """Valor de la entrada siguiente en el orden (la del líder); False en la última."""
    out = np.zeros_like(a)
    out[:-1] = a[1:]
    return out


def _prev(a: np.ndarray) -> np.ndarray:
    """Valor de la entrada anterior en el orden (la del seguidor); False en la primera."""
    out = np.zeros_like(a)
    out[1:] = a[:-1]
    return out


class Occupancy(NamedTuple):
    """Entradas carril-vehículo ordenadas por (carril, x). Quien cambia de carril aparece dos veces."""

    order: np.ndarray  # posición ordenada -> índice de entrada (primarias [0,n), secundarias [n,m))
    veh: np.ndarray  # índice de vehículo de cada entrada ordenada
    lane: np.ndarray  # carril de cada entrada ordenada
    x: np.ndarray  # posición del frente de cada entrada ordenada
    has_leader: np.ndarray  # la siguiente entrada está en el mismo carril
    leader: np.ndarray  # índice de vehículo del líder (válido si has_leader)
    changing: np.ndarray  # vehículos con cambio de carril en curso (orden de las secundarias)


class Simulation:
    _FIELDS = (
        "vtype", "pax", "lane", "lc_target", "lc_timer", "cooldown",
        "react", "x", "stopped", "crossed", "entry_tick", "vid", "stop_state", "dwell", "v_last", "vcap_last", "vlen",
        "stop_pos", "stop_kind", "vmax",
    )  # fmt: skip
    # Estados de parada: sin parada pendiente, antes de la parada, detenido en ella.
    NO_STOP, STOP_AHEAD, AT_STOP = 0, 1, 2
    # Clase de parada: la propia del tipo (p. ej. el autobús) o una detención de [bottleneck].
    TYPE_STOP, BOTTLENECK_STOP = 0, 1

    def __init__(self, cfg: SimConfig, rng: np.random.Generator):
        self.cfg = cfg
        # Un generador por proceso aleatorio y por tipo, hijos de la semilla de la réplica (rng.spawn
        # no consume sorteos). Así, cambiar un proceso —la parada, la tasa o los pasajeros de un
        # tipo— no desplaza los sorteos de los demás: dos escenarios con la misma semilla comparten
        # los números aleatorios donde el cambio no interviene. Los de cada tipo van después de los
        # comunes y en el orden de los tipos, así que agregar un tipo nuevo no altera los anteriores.
        streams = rng.spawn(1 + PER_TYPE_STREAMS * cfg.n_types)
        self.rng_lane_change = streams[0]  # orden de decisión y duración de los cambios de carril
        per_type = streams[1:]
        self.rng_arrivals = per_type[0::PER_TYPE_STREAMS]  # llegadas Poisson
        self.rng_pax = per_type[1::PER_TYPE_STREAMS]  # pasajeros por vehículo
        self.rng_react = per_type[2::PER_TYPE_STREAMS]  # tiempos de reacción
        self.rng_entry = per_type[3::PER_TYPE_STREAMS]  # carril de entrada (desempates)
        self.rng_stop = per_type[4::PER_TYPE_STREAMS]  # duración de la parada
        # Hijos del generador de pasajeros (spawn no consume sus sorteos): si lleva mercancía y su
        # largo. Agregarlos no altera los sorteos anteriores ni dependen del número de tipos.
        self.rng_cargo, self.rng_length = zip(*(g.spawn(2) for g in self.rng_pax))
        # Cuellos de botella: si se detiene y dónde (al llegar) y cuánto tiempo (al detenerse).
        self.rng_bottleneck, self.rng_bottleneck_time = zip(*(g.spawn(2) for g in self.rng_pax))
        # Hijo del generador de llegadas: la tasa de cada intervalo de los tipos con tasa variable.
        self.rng_rate = [g.spawn(1)[0] for g in self.rng_arrivals]
        # Hijo del generador de pasajeros: la velocidad máxima de cada vehículo (si es variable).
        self.rng_speed = [g.spawn(1)[0] for g in self.rng_pax]
        # Hijo del generador de reacción: la reacción de quien arranca en la cola de entrada detenida.
        self.rng_queue_react = [g.spawn(1)[0] for g in self.rng_react]
        specs = cfg.specs
        b = cfg.behavior
        self.L = float(cfg.length)
        self.n_lanes = cfg.lanes
        self.n_types = cfg.n_types

        # Tablas por tipo de vehículo (indexadas por vtype).
        self.v_step = np.array([s.speed * DT for s in specs])  # velocidad media; la de cada vehículo está en vmax
        self.length_t = np.array([s.length for s in specs])  # largo medio; el de cada vehículo está en vlen
        self.gap_run_t = np.array([s.gap_run for s in specs])
        self.gap_stop_t = np.array([s.gap_stop for s in specs])
        # Intervalo de seguimiento (s) de los tipos con time_headway: en marcha, gap_stop + time_headway · v.
        self.has_headway_t = np.array([s.time_headway is not None for s in specs])
        self.headway_t = np.array([s.time_headway or 0.0 for s in specs])
        self.any_headway = bool(self.has_headway_t.any())
        self.can_change = np.array([s.can_change_lane for s in specs])
        self.stop_x = np.array([np.inf if s.stop_position is None else s.stop_position for s in specs])
        self.has_stop = np.isfinite(self.stop_x)
        # Probabilidad y duración (media, σ) de la detención de cada tipo: la propia o la de [bottleneck].
        self.bn_prob = np.array([cfg.bottleneck_prob(k) for k in range(self.n_types)])
        self.bn_time = [cfg.bottleneck_time(k) for k in range(self.n_types)]
        self.bn_type = self.bn_prob > 0  # tipos que pueden detenerse en un cuello de botella
        self.free_type = np.array([cfg.ignores_light(k) for k in range(self.n_types)], np.bool_)  # sin semáforo
        # Líneas de alto intermedias (semáforos intermedios y pasos peatonales de los topes): posición, tipos que no
        # las obedecen (carril exclusivo en los free_lanes del semáforo) y carriles donde rigen (None = todos).
        self.lines = cfg.stop_lines
        self.inner_x = [float(s.position) for s in self.lines]
        self.inner_free = [np.array([s.light is not None and cfg.ignores_light(k, s.light) for k in range(self.n_types)],
                                    np.bool_) for s in self.lines]  # fmt: skip
        self.inner_lane = [None if s.lanes is None else np.isin(np.arange(cfg.lanes), s.lanes) for s in self.lines]
        self.bn_lane = np.zeros(cfg.lanes, np.bool_)
        self.bn_lane[list(cfg.bottleneck_lanes())] = True
        self.bn_zone = cfg.bottleneck_zone()
        self.any_stop = bool(self.has_stop.any() or self.bn_type.any())
        self.abreast = np.array([s.abreast for s in specs])
        self.any_abreast = bool((self.abreast > 1).any())
        self.pass_in_lane = np.array([s.pass_in_lane for s in specs])  # rebasan dentro del carril
        self.any_pass_in_lane = bool(self.pass_in_lane.any())
        self.allowed = np.zeros((self.n_types, cfg.lanes), np.bool_)  # carriles que puede usar cada tipo
        for k in range(self.n_types):
            self.allowed[k, list(cfg.allowed_lanes(k))] = True
        self._entry_lanes = [np.array(cfg.entry_lanes(k)) for k in range(self.n_types)]
        self.rate_tick = np.array(cfg.rates, dtype=np.float64) / 60.0 * DT
        # Tasas variables: llegadas por paso en cada intervalo de rate_interval, sorteadas al inicio.
        n_intervals = -(-cfg.n_ticks // cfg.rate_ticks)
        self.rate_by_interval = {
            k: sample_rates(self.rng_rate[k], cfg.rate(k), n_intervals) / 60.0 * DT
            for k in range(self.n_types) if cfg.rate(k).variable
        }  # fmt: skip

        self.lane_vmax = np.array(cfg.lane_max_kmh) / 3.6 * DT  # límite de cada carril, m/paso
        self.lc_every = max(1, round(b.lane_change_interval / DT))
        # Umbrales de cambio de carril por tipo: los del tipo o, si no los fija, los de [behavior].
        def own(field: str) -> np.ndarray:
            return np.array([getattr(b, field) if getattr(s, field) is None else getattr(s, field) for s in specs])

        self.lookahead_t = own("lookahead")
        self.min_adv_t = own("min_advantage")
        self.cooldown_ticks_t = np.round(own("lane_change_cooldown") / DT).astype(np.int16)
        self.overtake = np.array([s.overtake for s in specs])
        # Aceleración y frenado por tipo en m/paso² (inf = instantáneos, sin anticipación).
        self.acc_t = np.array([INF if s.accel is None else s.accel * DT * DT for s in specs])
        self.dec_t = np.array([INF if s.decel is None else s.decel * DT * DT for s in specs])
        self.dyn_t = np.isfinite(self.acc_t) | np.isfinite(self.dec_t)  # tipos con dinámica gradual
        self.dec_fin_t = np.isfinite(self.dec_t)
        self.dec_safe_t = np.where(self.dec_fin_t, self.dec_t, 1.0)  # sin infinitos, para _gipps
        self.any_dynamics = bool(self.dyn_t.any())
        # Tope ([speed_bump]): posición, carriles donde está y velocidad máxima de cada tipo al pisarlo (m/paso).
        self.bump_x = np.array([b.position for b in cfg.bumps], dtype=np.float64)  # uno o varios topes
        self.bump_lane = np.zeros((len(cfg.bumps), cfg.lanes), np.bool_)  # carriles de cada tope
        for i, b in enumerate(cfg.bumps):
            self.bump_lane[i, list(cfg.bump_lanes(b))] = True
        self.bump_v_t = np.array([INF if s.speed_bump_kmh is None else s.speed_bump_kmh / 3.6 * DT for s in specs])
        self.any_bump = bool(self.bump_x.size and self.bump_lane.any() and np.isfinite(self.bump_v_t).any())
        # Emisiones (Int Panis et al., 2006): coeficientes por tipo y contaminante, límites de la aceleración que
        # entra al modelo (m/s²) y gramos acumulados: en la corrida, a flujo libre (mismo recorrido a velocidad
        # constante), en el intervalo de muestreo y por posición (contaminante × carril × intervalo de EMIS_BIN m).
        self.emis_coef, self.emis_on = coefficient_table(specs)
        self.any_emissions = bool(self.emis_on.any())
        self.emits_t = self.emis_on.any(axis=1)  # el tipo emite algún contaminante
        # Coeficientes por (tipo, juego): la fila 2·tipo es la de a ≥ −0.5 y la 2·tipo + 1 la de a < −0.5.
        self._coef_sets = np.ascontiguousarray(self.emis_coef.transpose(0, 2, 1, 3)).reshape(-1, *self.emis_coef.shape[1::2])
        self.emis_amax = np.array([s.accel or 0.0 for s in specs])
        self.emis_dmax = np.array([s.decel or 0.0 for s in specs])
        n_pol = self.emis_on.shape[1]
        self.emis_g = np.zeros((self.n_types, n_pol))
        self.emis_free_g = np.zeros((self.n_types, n_pol))
        self.emis_interval = np.zeros((self.n_types, n_pol))
        self.emis_pos = np.zeros((n_pol, cfg.lanes, max(1, math.ceil(cfg.length / EMIS_BIN))))
        self._pol_idx = np.arange(n_pol)  # índice de cada contaminante, para aplanar (tipo o carril, contaminante)
        self.veh_m = np.zeros(self.n_types)  # m recorridos dentro del tramo, por tipo
        # Sin límites distintos entre carriles no hay a dónde subir.
        self.any_rise = len(set(cfg.lane_max_kmh)) > 1
        self._key_stride = self.L + 1e4  # separa los carriles en la clave de orden

        min_slot = min(s.shortest + s.gap_stop for s in specs)
        per_lane = math.ceil((self.L + max(s.longest for s in specs)) / min_slot) + 2
        self._alloc(cfg.lanes * per_lane)
        self.n = 0
        # (tipo, pax, largo, posición de su detención de [bottleneck], velocidad máxima km/h, paso de llegada a la cola)
        self.queues: list[deque[tuple[int, int, float, float, float, int]]] = [deque() for _ in range(cfg.lanes)]
        # Cola de entrada detenida por carril y pasos de reacción que le faltan a su primero (-1 = sin empezar).
        self.queue_stopped = np.zeros(cfg.lanes, np.bool_)
        self.queue_react = np.full(cfg.lanes, -1, np.int32)
        self.tick = 0
        self._pending_arrivals: dict[int, np.ndarray] = {}

        self.arrived_veh = np.zeros(self.n_types)
        self.arrived_pax = np.zeros(self.n_types)
        self.arrived_cargo = np.zeros(self.n_types)  # llegadas con mercancía (pax = 0)
        self.entered_veh = np.zeros(self.n_types)
        self.cum_veh = np.zeros(self.n_types)
        self.cum_pax = np.zeros(self.n_types)
        self.travel_ticks = np.zeros(self.n_types)
        self.pax_m = np.zeros(self.n_types)  # pasajeros·m dentro del tramo en el intervalo de muestreo
        self.lane_dist = np.zeros(cfg.lanes)  # m recorridos en cada carril en el intervalo de muestreo
        self.lane_time = np.zeros(cfg.lanes)  # vehículo·s presentes en cada carril en el intervalo
        self.type_dist = np.zeros(self.n_types)  # m recorridos dentro del tramo por tipo, en toda la corrida
        self.type_time = np.zeros(self.n_types)  # vehículo·s dentro del tramo por tipo (con los detenidos)
        self.lane_changes = 0
        self.lane_changes_t = np.zeros(self.n_types)  # cambios de carril por tipo
        self.in_lane_passes = np.zeros(self.n_types)  # rebases dentro del carril, por tipo de quien rebasa
        self.stops = np.zeros(self.n_types)  # paradas completadas por tipo
        self.stop_time_ticks = np.zeros(self.n_types)  # pasos detenidos en la parada, por tipo
        self.queue_wait_ticks = np.zeros(self.n_types)  # pasos en la cola de entrada de los que ya entraron
        self.bn_stops = np.zeros(self.n_types)  # detenciones de [bottleneck] por tipo
        self.bn_ticks = np.zeros(self.n_types)  # pasos detenidos en ellas, por tipo
        self.pax_hist = np.zeros((self.n_types, MAX_PAX + 1), np.int64)  # vehículos llegados por nº de pasajeros
        # Cola de salida por carril: vehículos que acepta por paso (inf = sin cola de salida).
        exit_cap = np.array(cfg.lane_exit_capacity, dtype=np.float64)
        self.exit_rate = np.where(exit_cap > 0, exit_cap / 60.0 * DT, INF)
        self._exit_limited = np.isfinite(self.exit_rate)  # carriles con cola de salida
        self._exit_add = np.where(self._exit_limited, self.exit_rate, 0.0)  # salidas que acepta por paso
        # m de cada cola de salida; sin cola de salida, sin límite.
        self.exit_storage = np.where(exit_cap > 0, np.array(cfg.lane_exit_storage, dtype=np.float64), INF)
        self.any_exit = bool((exit_cap > 0).any())
        self.exit_q = np.zeros(cfg.lanes)  # m ocupados en la cola de salida (largo + gap detenido de cada vehículo)
        # Lo que ocupa cada vehículo en la cola de salida, en orden de llegada: sale el primero que entró.
        self.exit_items: list[deque[float]] = [deque() for _ in range(cfg.lanes)]
        self.exit_credit = np.zeros(cfg.lanes)  # fracción acumulada de la siguiente salida
        self.exit_closed = np.zeros(cfg.lanes, np.bool_)  # el siguiente vehículo del carril no cabe en la salida
        self.exit_blocked_ticks = np.zeros(cfg.lanes)  # pasos con la línea cerrada por la cola de salida
        self.lane_cross = np.zeros(cfg.lanes)  # cruces de la línea por carril en el intervalo de muestreo
        self.lane_crossed = np.zeros(cfg.lanes)  # cruces de la línea por carril en toda la corrida
        self.recorder = Recorder(cfg.n_samples, cfg.n_types, cfg.lanes)
        self.initial_veh = np.zeros(self.n_types)  # vehículos de la condición inicial
        self.initial_pax = np.zeros(self.n_types)
        self.timed_veh = np.zeros(self.n_types)  # cruces con tiempo de recorrido (sin los iniciales)
        # Generadores propios (hijos nuevos: no alteran los anteriores): el relleno inicial y los peatones. Se crean
        # siempre, para que el índice de cada uno no dependa de la configuración.
        fill_rng, ped_rng = rng.spawn(1)[0], rng.spawn(1)[0]
        self._init_pedestrians(ped_rng)
        if any(v > 0 for v in cfg.lane_initial_occupancy):
            self._initial_fill(fill_rng)

    # ------------------------------------------------------------------ estado

    def _alloc(self, cap: int) -> None:
        self.vtype = np.zeros(cap, np.uint8)
        self.pax = np.zeros(cap, np.uint8)
        self.lane = np.zeros(cap, np.int8)
        self.lc_target = np.full(cap, -1, np.int8)
        self.lc_timer = np.zeros(cap, np.int16)
        self.cooldown = np.zeros(cap, np.int16)
        self.react = np.full(cap, -1, np.int16)
        self.x = np.zeros(cap, np.float64)
        self.stopped = np.zeros(cap, np.bool_)
        self.crossed = np.zeros(cap, np.bool_)
        self.entry_tick = np.zeros(cap, np.int32)
        self.vid = np.zeros(cap, np.int32)  # identificador del vehículo (orden de entrada al tramo)
        self.stop_state = np.zeros(cap, np.int8)  # NO_STOP | STOP_AHEAD | AT_STOP
        self.dwell = np.zeros(cap, np.int32)  # pasos que le quedan en la parada
        self.v_last = np.zeros(cap, np.float64)  # m avanzados en el último paso (velocidad real)
        self.vcap_last = np.zeros(cap, np.float64)  # m que podía avanzar en él sin líder ni alto
        self.vlen = np.zeros(cap, np.float64)  # largo del vehículo (m)
        self.vmax = np.zeros(cap, np.float64)  # velocidad máxima del vehículo (m/paso)
        self.stop_pos = np.full(cap, np.inf)  # posición de su parada o detención (frente); inf = ninguna
        self.stop_kind = np.zeros(cap, np.int8)  # TYPE_STOP | BOTTLENECK_STOP

    def _grow(self) -> None:
        old = {f: getattr(self, f) for f in self._FIELDS}
        self._alloc(2 * old["x"].size)
        for f, arr in old.items():
            getattr(self, f)[: arr.size] = arr

    def _compact(self, keep: np.ndarray) -> None:
        n = self.n
        k = int(np.count_nonzero(keep))
        for f in self._FIELDS:
            arr = getattr(self, f)
            arr[:k] = arr[:n][keep]
        self.n = k

    def _add(
        self, vt: int, pax: int, lane: int, length: float | None = None, x: float | None = None,
        bottleneck_at: float = INF, speed_kmh: float | None = None, v0: float | None = None,
    ) -> None:  # fmt: skip
        """Agrega un vehículo al inicio del tramo; pax = 0 si lleva mercancía. Sin `length`, mide
        el largo medio de su tipo; sin `speed_kmh`, su velocidad máxima es la media del tipo. Con `x`
        (frente), es un vehículo de la condición inicial: ya está en el tramo, no cuenta como llegada
        y no tiene tiempo de recorrido. `bottleneck_at` es la posición de su detención de [bottleneck]
        (inf = no se detiene). `v0` es su velocidad al entrar (m/paso); sin ella, entra a su máxima."""
        if self.n == self.x.size:
            self._grow()
        i = self.n
        self.vtype[i] = vt
        self.pax[i] = pax
        self.lane[i] = lane
        self.lc_target[i] = -1
        self.lc_timer[i] = 0
        self.cooldown[i] = 0
        self.react[i] = -1
        self.vlen[i] = self.length_t[vt] if length is None else length
        self.x[i] = self.vlen[i] if x is None else x  # entra con la parte trasera en x = 0
        self.vmax[i] = self.v_step[vt] if speed_kmh is None else speed_kmh / 3.6 * DT
        self.stopped[i] = False
        self.crossed[i] = False
        self.entry_tick[i] = self.tick if x is None else -1
        self.vid[i] = self.entered_veh.sum()
        if self.has_stop[vt]:
            self.stop_pos[i], self.stop_kind[i] = self.stop_x[vt], self.TYPE_STOP
        else:
            self.stop_pos[i], self.stop_kind[i] = bottleneck_at, self.BOTTLENECK_STOP
        ahead = self.x[i] < self.stop_pos[i] - EPS  # si empieza pasada su parada, no para
        self.stop_state[i] = self.STOP_AHEAD if ahead else self.NO_STOP
        self.dwell[i] = 0
        self.v_last[i] = self.vcap_last[i] = min(self.vmax[i], self.lane_vmax[lane])  # entra en marcha
        if v0 is not None:
            self.v_last[i] = min(v0, self.vcap_last[i])
        self.n += 1
        self.entered_veh[vt] += 1
        if x is None:
            self.pax_m[vt] += pax * self.vlen[i]
        else:
            self.initial_veh[vt] += 1
            self.initial_pax[vt] += pax

    def _initial_fill(self, rng: np.random.Generator) -> None:
        """Condición inicial: cada carril empieza con vehículos que ocupan la fracción `occupancy`
        de su largo, midiendo cada uno como largo + gap_stop (1 = cola compacta, sin más espacio que
        el gap detenido). Los tipos se eligen en proporción a su tasa de llegada entre los que pueden
        circular por ese carril; pasajeros, mercancía y largo se sortean como en las llegadas. Si un
        vehículo no cabe se sortea otro, hasta INITIAL_MISSES intentos seguidos.

        El espacio libre se reparte primero para que cada uno tenga su gap_run (en marcha) y el resto
        al azar. Si no alcanza, todos se comprimen por igual y empiezan detenidos, como un
        embotellamiento: arrancan con su tiempo de reacción cuando se abre espacio."""
        cfg = self.cfg
        for lane, occupancy in enumerate(cfg.lane_initial_occupancy):
            types = [k for k in range(self.n_types) if cfg.rates[k] > 0 and lane in cfg.entry_lanes(k)]
            if occupancy <= 0 or not types:
                continue
            weights = np.array([cfg.rates[k] for k in types])
            weights = weights / weights.sum()
            chosen, used, misses = [], 0.0, 0
            while misses < INITIAL_MISSES:
                k = types[rng.choice(len(types), p=weights)]
                spec = cfg.specs[k]
                length = float(sample_lengths(rng, spec, 1)[0])
                if used + length + self.gap_stop_t[k] > occupancy * self.L + EPS:
                    misses += 1
                    continue
                misses = 0
                pax = int(sample_passengers(rng, spec, 1)[0])
                if spec.cargo_prob > 0 and rng.random() < spec.cargo_prob:
                    pax = 0
                at = rng.uniform(*self.bn_zone) if self.bn_type[k] and rng.random() < self.bn_prob[k] else INF
                speed = float(sample_speeds(rng, spec, 1)[0])  # sin desviación no consume sorteos
                chosen.append((k, pax, length, at, speed))
                used += length + self.gap_stop_t[k]
            if not chosen:
                continue
            # Espacio sobre el mínimo (gap_stop detrás de cada líder; el primero no tiene líder).
            kinds = np.array([k for k, *_ in chosen])
            pool = self.L - sum(length for _, _, length, *_ in chosen) - self.gap_stop_t[kinds[1:]].sum()
            # hasta el gap en marcha (gap_run, o el de time_headway a su velocidad deseada)
            run_gap = [self._run_gap(k, min(speed / 3.6 * DT, self.lane_vmax[lane])) for k, *_, speed in chosen]
            need = np.r_[0.0, np.array(run_gap[1:]) - self.gap_stop_t[kinds[1:]]]
            if pool >= need.sum():
                spare = np.diff(np.sort(rng.uniform(0.0, pool - need.sum(), len(chosen))), prepend=0.0)
                extra = need + spare
            else:  # no alcanza para ir en marcha: se comprimen por igual
                extra = need * pool / need.sum()
            # Se coloca del semáforo hacia atrás; quien queda a menos de su gap_run empieza detenido.
            rear = self.L
            ahead = -1  # el colocado justo delante en este carril
            for idx, ((k, pax, length, at, speed), free) in enumerate(zip(chosen, extra)):
                gap = (self.gap_stop_t[k] if idx else 0.0) + free
                self._add(k, pax, lane, length, x=rear - gap, bottleneck_at=at, speed_kmh=speed)
                i = self.n - 1
                if idx and gap < run_gap[idx] - EPS:
                    self.stopped[i] = True
                    self.v_last[i] = 0.0
                elif self.dyn_t[k]:
                    # Con frenado gradual empieza a la velocidad con la que aún frena detrás del de adelante
                    # (o antes de la línea si la corrida empieza en rojo), no de golpe en el primer paso.
                    if ahead >= 0:
                        room = self.x[ahead] - self.vlen[ahead] - run_gap[idx] - self.x[i]
                        v_ahead, dec_ahead = float(self.v_last[ahead]), self.dec_t[self.vtype[ahead]]
                    else:
                        room = self.L - self.x[i] if self.exit_phase(0) == RED and not self.free_type[k] else INF
                        v_ahead, dec_ahead = 0.0, INF
                    if np.isfinite(room):
                        self.v_last[i] = min(self.v_last[i], _safe_speed_1(room, v_ahead, self.dec_t[k], dec_ahead))
                    light = self._line_room(k, self.x[i], lane)
                    if np.isfinite(light):  # línea de alto intermedia en rojo al empezar
                        self.v_last[i] = min(self.v_last[i], _safe_speed_1(light, 0.0, self.dec_t[k], INF))
                    self.v_last[i] = min(self.v_last[i], self._bump_speed_1(k, lane, self.x[i], self.x[i] - length))
                ahead = i
                rear = self.x[i] - length

    def _run_gap(self, k: int, v_step: float) -> float:
        """Gap (m) que guarda un vehículo del tipo k detrás de un líder en marcha yendo a `v_step` m/paso: gap_run o, con
        time_headway, gap_stop más lo que recorre en ese tiempo."""
        if not self.has_headway_t[k]:
            return float(self.gap_run_t[k])
        return float(self.gap_stop_t[k] + self.headway_t[k] * v_step / DT)

    def _line_room(self, k: int, x: float, lane: int) -> float:
        """Distancia desde `x` hasta la línea de alto intermedia (semáforo o paso peatonal) en rojo al empezar la
        corrida más cercana adelante que obedece el tipo k en ese carril; INF si no hay."""
        return min((pos - x for pos, free, ok, ph in zip(self.inner_x, self.inner_free, self.inner_lane, self.line_phases(0))
                    if ph == RED and x <= pos and not free[k] and (ok is None or ok[lane])), default=INF)  # fmt: skip

    # ---------------------------------------------------------------- peatones

    def _init_pedestrians(self, rng: np.random.Generator) -> None:
        """Calendario de peatones de la réplica (ver pedestrians.py): uno por semáforo peatonal y por tope con
        peatones. Los generadores van por índice fijo —el semáforo del final del tramo, cada semáforo intermedio y
        cada tope— así que quitar uno de los demás elementos no mueve los sorteos del resto."""
        cfg = self.cfg
        n = cfg.n_ticks
        lights_rng, bumps_rng = rng.spawn(2)
        light_gens = lights_rng.spawn(1 + len(cfg.extra_lights))
        bump_gens = bumps_rng.spawn(len(cfg.bumps))

        def light_schedule_of(light, gen) -> Schedule:
            before = cfg.previous_light(light)
            prev_red = fixed_phases(before, n + 1) == RED if before else None  # espera al rojo del anterior
            counts = arrivals(*gen.spawn(2), light.pedestrian_crossing, cfg)
            return light_schedule(counts, light, prev_red, n)

        self.ped_schedules: dict[tuple[str, float], Schedule] = {}
        self._exit_sched: Schedule | None = None
        if cfg.exit_light.active and cfg.exit_light.pedestrian:
            self._exit_sched = light_schedule_of(cfg.exit_light, light_gens[0])
            self.ped_schedules[("semáforo", cfg.length)] = self._exit_sched
        for i, lt in enumerate(cfg.extra_lights):
            if lt.active and lt.pedestrian:
                self.ped_schedules[("semáforo", lt.position)] = light_schedule_of(lt, light_gens[1 + i])
        wait = round(PEDESTRIAN_YIELD / DT)
        for bump, gen in zip(cfg.bumps, bump_gens):  # cada tope con peatones, con sus propios generadores
            if bump.pedestrian:
                counts = arrivals(*gen.spawn(2), bump.pedestrian_crossing, cfg)
                cross = round(bump.pedestrian_time / DT)
                self.ped_schedules[("tope", bump.position)] = bump_schedule(counts, cross, wait, n)
        # Calendario de cada línea de alto intermedia (None = semáforo de ciclo fijo: sale de Light.phase).
        self._line_sched = [self.ped_schedules.get((s.kind, s.position)) if s.pedestrian else None for s in self.lines]

    def exit_phase(self, tick: int) -> int:
        """Fase (RED, GREEN o YELLOW) del semáforo del final del tramo en el paso `tick`."""
        if self._exit_sched is None:
            return self.cfg.phase(tick)
        return int(self._exit_sched.phases[min(tick, self.cfg.n_ticks)])

    def line_phases(self, tick: int) -> tuple[int, ...]:
        """Fase de cada línea de alto intermedia (`cfg.stop_lines`) en el paso `tick`."""
        n = self.cfg.n_ticks
        return tuple(int(sched.phases[min(tick, n)]) if sched is not None else line.light.phase(tick)
                     for line, sched in zip(self.lines, self._line_sched))  # fmt: skip

    def occupancy(self) -> Occupancy:
        n = self.n
        changing = (self.lc_target[:n] >= 0).nonzero()[0]
        veh = np.concatenate((np.arange(n), changing))
        lane = np.concatenate((self.lane[:n], self.lc_target[changing])).astype(np.int64)
        x = self.x[veh]
        order = np.argsort(lane * self._key_stride + x)
        s_veh, s_lane, s_x = veh[order], lane[order], x[order]
        has_leader = np.zeros(order.size, np.bool_)
        has_leader[:-1] = s_lane[1:] == s_lane[:-1]
        leader = np.zeros(order.size, np.int64)
        leader[:-1] = s_veh[1:]
        return Occupancy(order, s_veh, s_lane, s_x, has_leader, leader, changing)

    # ------------------------------------------------------------------- paso

    def run(self) -> None:
        for _ in range(self.cfg.n_ticks):
            self.step()

    def step(self) -> None:
        self._arrivals()
        self._spawn()
        if self.any_exit:
            self._drain_exit()
        if self.n:
            self._move(self.exit_phase(self.tick), self.line_phases(self.tick) if self.inner_x else ())
        self.tick += 1
        if self.n_lanes > 1 and self.n and self.tick % self.lc_every == 0:
            self._lane_changes()
        if self.tick % self.cfg.sample_ticks == 0:
            self._sample()

    def _arrivals(self) -> None:
        offset = self.tick % ARRIVAL_CHUNK
        if offset == 0:
            # Sorteo de llegadas Poisson por bloques; solo se guardan los pasos con llegadas.
            chunk = np.column_stack([self._arrival_counts(k) for k in range(self.n_types)])
            self._pending_arrivals = {int(t): chunk[t] for t in np.flatnonzero(chunk.any(axis=1))}
        counts = self._pending_arrivals.get(offset)
        if counts is None:
            return
        for vt in np.flatnonzero(counts):
            k = int(counts[vt])
            spec = self.cfg.specs[vt]
            paxs = sample_passengers(self.rng_pax[vt], spec, k)
            if spec.cargo_prob > 0:  # los que llevan mercancía no llevan pasajeros
                paxs[self.rng_cargo[vt].random(k) < spec.cargo_prob] = 0
            lengths = sample_lengths(self.rng_length[vt], spec, k)
            speeds = sample_speeds(self.rng_speed[vt], spec, k)
            at = np.full(k, INF)  # dónde se detendrá (cuello de botella); inf = no se detiene
            if self.bn_type[vt]:
                g = self.rng_bottleneck[vt]
                stops = g.random(k) < self.bn_prob[vt]
                at = np.where(stops, np.maximum(g.uniform(*self.bn_zone, k), lengths + 0.01), INF)
            self.arrived_veh[vt] += k
            self.arrived_cargo[vt] += int(np.count_nonzero(paxs == 0))
            self.arrived_pax[vt] += int(paxs.sum())
            self.pax_hist[vt] += np.bincount(paxs[paxs > 0], minlength=MAX_PAX + 1)
            for p, length, a, v in zip(paxs, lengths, at, speeds):
                entry = (int(vt), int(p), float(length), float(a), float(v), self.tick)
                self.queues[self._entry_lane(vt)].append(entry)

    def _arrival_counts(self, k: int) -> np.ndarray:
        """Llegadas del tipo k en cada paso del bloque que empieza en el paso actual."""
        g = self.rng_arrivals[k]
        by_interval = self.rate_by_interval.get(k)
        if by_interval is None:
            return g.poisson(self.rate_tick[k], ARRIVAL_CHUNK)
        idx = (self.tick + np.arange(ARRIVAL_CHUNK)) // self.cfg.rate_ticks
        return g.poisson(by_interval[np.minimum(idx, by_interval.size - 1)])

    def _entry_lane(self, vt: int) -> int:
        if self.n_lanes == 1:
            return 0
        lanes = self._entry_lanes[vt]  # sin los carriles exclusivos de otros tipos
        if lanes.size == 1:  # carril fijo: los que no cambian de carril
            return int(lanes[0])
        # Los que cambian de carril: cola de entrada más corta; desempate por más espacio libre a la entrada.
        qlen = np.fromiter((len(self.queues[ln]) for ln in lanes), np.int64, lanes.size)
        best = lanes[qlen == qlen.min()]
        if best.size > 1:
            tail = self._lane_tails()[0][best]
            best = best[tail == tail.max()]
        return int(best[0] if best.size == 1 else self.rng_entry[vt].choice(best))

    def _lane_tails(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Parte trasera del último vehículo de cada carril, si está detenido y su índice (-1 = carril vacío)."""
        rear_min = np.full(self.n_lanes, INF)
        tail_stopped = np.zeros(self.n_lanes, np.bool_)
        tail = np.full(self.n_lanes, -1)
        n = self.n
        if n == 0:
            return rear_min, tail_stopped, tail
        rear = self.x[:n] - self.vlen[:n]
        lane, tgt = self.lane[:n], self.lc_target[:n]
        for ln in range(self.n_lanes):
            occ = ((lane == ln) | (tgt == ln)).nonzero()[0]
            if occ.size:
                j = occ[np.argmin(rear[occ])]
                rear_min[ln] = rear[j]
                tail_stopped[ln] = self.stopped[j]
                tail[ln] = j
        return rear_min, tail_stopped, tail

    def _spawn(self) -> None:
        if not any(self.queues):
            return
        rear_min, tail_stopped, tail = self._lane_tails()
        queue_reaction = self.cfg.behavior.queue_reaction
        for ln, q in enumerate(self.queues):
            if not q:
                continue
            vt, pax, length, at, speed, arrived = q[0]
            if tail_stopped[ln]:
                gap = self.gap_stop_t[vt]
            elif self.has_headway_t[vt]:  # entra a su velocidad deseada: guarda el gap de time_headway a ella
                gap = self._run_gap(vt, min(speed / 3.6 * DT, self.lane_vmax[ln]))
            else:
                gap = self.gap_run_t[vt]
            fits = rear_min[ln] >= length + gap
            from_stop = queue_reaction and self.queue_stopped[ln]
            if from_stop and not self._queue_ready(ln, vt, fits, tail_stopped[ln]):
                continue
            if fits:
                q.popleft()
                self.queue_wait_ticks[vt] += self.tick - arrived
                v0 = None
                if self.any_dynamics:
                    v0 = self._entry_speed(vt, speed, from_stop, rear_min[ln] - length - gap, tail[ln])
                    bump = self._bump_speed_1(vt, ln, length, 0.0) if self.dyn_t[vt] else INF
                    if np.isfinite(bump):  # entra a lo más a la velocidad con la que aún llega al tope
                        v0 = min(speed / 3.6 * DT if v0 is None else v0, bump)
                self._add(vt, pax, ln, length, bottleneck_at=at, speed_kmh=speed, v0=v0)
                self.queue_react[ln] = -1
                # Quien sigue esperaba detrás de un detenido: también arranca con reacción.
                self.queue_stopped[ln] &= bool(q)
            elif queue_reaction and tail_stopped[ln]:
                self.queue_stopped[ln] = True  # la cola del tramo llega detenida hasta la entrada

    def _entry_speed(self, vt: int, speed_kmh: float, from_stop: bool, space: float, tail: int) -> float | None:
        """Velocidad (m/paso) con la que entra un vehículo con aceleración o frenado graduales: 0 si sale
        de la cola de entrada detenida y acelera gradualmente; si no, a lo más la que le permite frenar
        detrás del último del carril. None = a su máxima, como sin dinámica."""
        if from_stop and np.isfinite(self.acc_t[vt]):
            return 0.0
        if tail < 0 or not np.isfinite(self.dec_t[vt]):
            return None
        v_lead = 0.0 if self.stopped[tail] else float(self.v_last[tail])
        safe = _safe_speed_1(space, v_lead, self.dec_t[vt], self.dec_t[self.vtype[tail]])
        return min(speed_kmh / 3.6 * DT, safe)

    def _bump_speed_1(self, k: int, lane: int, front: float, rear: float) -> float:
        """Velocidad máxima (m/paso) de un vehículo del tipo k en `lane` por los topes: la del tipo si pisa uno;
        antes de él, con frenado gradual, la que aún le permite llegar a ella frenando a `decel`; INF si no
        aplica (sin tope en el carril, ya los pasó o el tipo pasa sin frenar). Con varios, la menor."""
        vb = self.bump_v_t[k]
        if not np.isfinite(vb):
            return INF
        dec = self.dec_t[k]
        speed = INF
        for bump, lanes in zip(self.bump_x, self.bump_lane):
            if rear >= bump or not lanes[lane]:
                continue
            if front >= bump:
                speed = min(speed, float(vb))
            elif np.isfinite(dec):
                speed = min(speed, _safe_speed_1(bump - front, vb, dec, dec))
        return speed

    def _queue_ready(self, ln: int, vt: int, fits: bool, tail_stopped: bool) -> bool:
        """El primero de la cola de entrada detenida del carril `ln` ya reaccionó y puede entrar.

        Empieza a reaccionar cuando el de adelante se mueve (o ya hay lugar para él) y entra cuando
        termina su reacción, en cuanto quepa."""
        left = self.queue_react[ln]
        if left < 0:
            if tail_stopped and not fits:
                return False
            self.queue_react[ln] = int(reaction_ticks(self.rng_queue_react[vt], self.cfg.behavior, 1)[0])
            return False
        if left > 0:
            self.queue_react[ln] = left = left - 1
        return left == 0

    def _move(self, phase: int, inner_phases: tuple[int, ...] = ()) -> None:
        n = self.n
        b = self.cfg.behavior
        vt = self.vtype[:n]
        x = self.x[:n]
        occ = self.occupancy()

        # Espacio disponible de cada entrada respecto a su líder.
        lead = occ.leader
        has_lead = occ.has_leader
        side = None
        if self.any_abreast:
            rows = self.rows(occ)
            side = self._side_slot(occ, rows)
        passing = yielding = None
        if self.any_pass_in_lane:
            # Rebase dentro del carril: el líder que se rebasa (o que se adelanta) no cuenta; manda el siguiente.
            skip, passing = self._in_lane_pass(occ, rows[0], side)
            m = occ.order.size
            after = np.minimum(np.arange(m) + 2, m - 1)
            lead = np.where(skip, occ.veh[after], lead)
            has_lead = np.where(skip, has_lead & _next(has_lead), has_lead)
        lead_stop = has_lead & self.stopped[lead]
        svt = self.vtype[occ.veh]
        gap_need = np.where(lead_stop, self.gap_stop_t[svt], self.gap_run_t[svt])
        if self.any_dynamics:
            # Con dinámica gradual, el gap detrás de un líder en marcha crece con la velocidad propia: gap_stop
            # casi detenido, gap_run a su velocidad deseada. Así, cuando la cola arranca, no le exige de golpe
            # gap_run a quien estaba a gap_stop.
            own = self.v_last[occ.veh] / np.minimum(self.vmax[occ.veh], self.lane_vmax[occ.lane])
            gradual = self.dyn_t[svt] & ~lead_stop
            ramp = self.gap_stop_t[svt] + (self.gap_run_t[svt] - self.gap_stop_t[svt]) * np.minimum(np.maximum(own, 0.0), 1.0)
            gap_need = np.where(gradual, ramp, gap_need)
        if self.any_headway:
            # Con time_headway, detrás de un líder en marcha guarda gap_stop más lo que recorre en ese tiempo a su
            # velocidad: la real del paso anterior con dinámica gradual; sin ella, la deseada (que alcanza de golpe).
            hw = self.has_headway_t[svt] & ~lead_stop
            if hw.any():
                v_own = np.where(self.dyn_t[svt], self.v_last[occ.veh],
                                 np.minimum(self.vmax[occ.veh], self.lane_vmax[occ.lane]))  # fmt: skip
                gap_need = np.where(hw, self.gap_stop_t[svt] + self.headway_t[svt] * v_own / DT, gap_need)
        s = self.x[lead] - self.vlen[lead] - gap_need - occ.x
        if passing is not None:
            # Quien deja pasar puede quedar al lado del que se le adelanta, pero no volver a rebasarlo; si
            # ambos se detienen detrás del mismo vehículo, queda SIDE_EPS atrás (como en una fila lado a lado).
            yielding = skip & ~passing
            front = self.x[occ.leader] - SIDE_EPS - occ.x
            if self.any_dynamics:
                # Con dinámica gradual cuenta lo que el que se adelanta avanzó en el último paso: si apenas lo
                # rebasó, no lo obliga a frenar en seco.
                front = np.where(self.dyn_t[svt], front + self.v_last[occ.leader], front)
            s = np.where(yielding, np.minimum(s - SIDE_EPS, front), s)
            # Si el siguiente vehículo frena a los dos, quien rebasa queda SIDE_EPS detrás del rebasado: van
            # lado a lado en vez de empatados en la misma posición.
            s = np.where(passing, s - SIDE_EPS, s)
            passing = (occ.veh[passing], occ.leader[passing])  # (quien rebasa, a quién)
        if side is not None:
            # Lugar libre al lado del líder detenido: se detiene con el frente alineado al suyo.
            s = np.where(side, self.x[occ.leader] - SIDE_EPS - occ.x, s)
        stop_e = np.empty(occ.order.size)
        move_e = np.empty(occ.order.size)
        stop_e[occ.order] = np.where(lead_stop, s, INF)
        move_e[occ.order] = np.where(has_lead & ~lead_stop, s, INF)

        # Por vehículo: mínimo entre su carril y, si está cambiando, el carril destino.
        stop_space = stop_e[:n]
        move_space = move_e[:n]
        ch = occ.changing
        if ch.size:
            stop_space[ch] = np.minimum(stop_space[ch], stop_e[n:])
            move_space[ch] = np.minimum(move_space[ch], move_e[n:])
        crossed = self.crossed[:n]
        # Fuera del semáforo: quien ya cruzó y los tipos con carril exclusivo sin semáforo.
        ruled_out = crossed | self.free_type[vt] if self.free_type.any() else crossed
        dynamics = self.any_dynamics
        if dynamics:
            # Velocidad segura respecto al de adelante (el que manda en cada entrada), por vehículo.
            v_lead = np.where(lead_stop, 0.0, self.v_last[lead])
            if yielding is not None:
                v_lead = np.where(yielding, np.minimum(v_lead, self.v_last[occ.leader]), v_lead)
            safe_e = np.empty(occ.order.size)
            fin_e = self.dec_fin_t[svt]
            dec_e = self.dec_safe_t[svt]
            vt_lead = self.vtype[lead]
            lead_term = np.where(
                self.dec_fin_t[vt_lead], v_lead * v_lead * dec_e / np.maximum(self.dec_t[vt_lead], dec_e), 0.0
            )
            safe_e[occ.order] = np.where(has_lead, _gipps(s, lead_term, dec_e, None if fin_e.all() else fin_e), INF)
            vsafe = safe_e[:n].copy()
            if ch.size:
                vsafe[ch] = np.minimum(vsafe[ch], safe_e[n:])
            dec_v = self.dec_t[vt]
            fin_v = self.dec_fin_t[vt]
            fin_v = None if fin_v.all() else fin_v  # para _gipps
            dec_vf = self.dec_safe_t[vt]
            v_prev = self.v_last[:n]
            to_line = np.where(crossed, INF, self.L - x)
            safe_line = _gipps(to_line, 0.0, dec_vf, fin_v)
            if phase == YELLOW:
                # Amarillo: quien todavía puede detenerse antes de la línea (frenando a decel) se detiene.
                halt = np.isfinite(dec_v) & ~ruled_out & (v_prev - dec_v <= safe_line + EPS)
                stop_space = np.minimum(stop_space, np.where(halt, to_line, INF))
                vsafe = np.minimum(vsafe, np.where(halt, safe_line, INF))
            elif phase == RED:
                vsafe = np.minimum(vsafe, np.where(ruled_out, INF, safe_line))
        if phase == RED:
            stop_space = np.minimum(stop_space, np.where(ruled_out, INF, self.L - x))
        # Líneas de alto intermedias (semáforos y pasos peatonales): en rojo, quien no las ha pasado (su frente) se
        # detiene en la línea; en amarillo, se detiene si aún puede (con frenado gradual) o baja la velocidad cerca de
        # ella, como en el semáforo del final. Un paso peatonal solo rige en los carriles de su tope.
        inner_yellow = []
        for pos, free, lane_ok, ph in zip(self.inner_x, self.inner_free, self.inner_lane, inner_phases):
            if ph == GREEN:
                continue
            before = (x <= pos) & ~free[vt] if free.any() else x <= pos
            if lane_ok is not None:
                target = self.lc_target[:n]
                before = before & (lane_ok[self.lane[:n]] | ((target >= 0) & lane_ok[np.maximum(target, 0)]))
            to_inner = np.where(before, pos - x, INF)
            if dynamics:
                safe_inner = _gipps(to_inner, 0.0, dec_vf, fin_v)
                if ph == YELLOW:
                    halt = np.isfinite(dec_v) & before & (v_prev - dec_v <= safe_inner + EPS)
                    stop_space = np.minimum(stop_space, np.where(halt, to_inner, INF))
                    vsafe = np.minimum(vsafe, np.where(halt, safe_inner, INF))
                else:
                    vsafe = np.minimum(vsafe, safe_inner)
            if ph == RED:
                stop_space = np.minimum(stop_space, to_inner)
            else:
                near = before & (pos - x <= b.yellow_approach)
                inner_yellow.append(near & ~np.isfinite(dec_v) if dynamics else near)
        if self.any_exit:
            # Cola de salida sin lugar: quien no cabe en los m libres de la cola de su carril (o del destino, si
            # está cambiando) no cruza la línea. Una cola vacía acepta a cualquiera, aunque no quepa entero.
            need = self.vlen[:n] + self.gap_stop_t[vt]
            used, room = self.exit_q, self.exit_storage

            def no_room(lane):
                return (used[lane] > 0) & (used[lane] + need > room[lane] + EPS)

            tgt = self.lc_target[:n]
            closed = ~crossed & (no_room(self.lane[:n]) | ((tgt >= 0) & no_room(np.maximum(tgt, 0))))
            # Línea cerrada en el carril si no cabe el siguiente en cruzar (el de adelante que no ha cruzado).
            self.exit_closed[:] = False
            waiting = (~crossed).nonzero()[0]
            if waiting.size:
                lane_w = self.lane[:n][waiting]
                order = np.lexsort((x[waiting], lane_w))
                lane_o = lane_w[order]
                last = np.concatenate((lane_o[1:] != lane_o[:-1], _TRUE))
                front = waiting[order][last]
                self.exit_closed[self.lane[:n][front]] = closed[front]
            self.exit_blocked_ticks += self.exit_closed
            if closed.any():
                stop_space = np.minimum(stop_space, np.where(closed, self.L - x, INF))
                if dynamics:
                    vsafe = np.minimum(vsafe, np.where(closed, safe_line, INF))
        stop_state = self.stop_state[:n]
        stop_pos = self.stop_pos[:n]
        if self.any_stop:
            # Detenciones de [bottleneck] que no aplican ahora: fuera de sus carriles o cambiando de carril.
            bn_lane_ok = self.bn_lane[self.lane[:n]] & (self.lc_target[:n] < 0)
            bn_skip = (self.stop_kind[:n] == self.BOTTLENECK_STOP) & ~bn_lane_ok
            # Parada: cuenta regresiva de quien está en ella; al terminar arranca (ya estaba listo).
            at_stop = stop_state == self.AT_STOP
            if at_stop.any():
                dwell = self.dwell[:n]
                dwell[at_stop] -= 1
                leave = at_stop & (dwell <= 0)
                stop_state[leave] = self.NO_STOP
                self.stopped[:n][leave] = False
                self.react[:n][leave] = -1
            # Mientras la parada esté pendiente o en curso, es un alto en su posición.
            pending = (stop_state == self.AT_STOP) | ((stop_state == self.STOP_AHEAD) & ~bn_skip)
            stop_space = np.minimum(stop_space, np.where(pending, stop_pos - x, INF))
            if dynamics:
                vsafe = np.minimum(vsafe, np.where(pending, _gipps(stop_pos - x, 0.0, dec_vf, fin_v), INF))
        space = np.minimum(stop_space, move_space)
        binding_stop = stop_space <= move_space

        # Máquina de estados: reacción de arranque de los detenidos.
        stopped = self.stopped[:n]
        react = self.react[:n]
        counting = stopped & (react > 0)
        react[counting] -= 1
        done = counting & (react == 0)
        stopped[done] = False
        react[done] = -1
        released = stopped & (react < 0) & (~binding_stop | (space > b.release_space))
        if released.any():
            for t in np.unique(vt[released]):  # cada tipo con su generador, en orden de llegada
                sel = released & (vt == t)
                react[sel] = reaction_ticks(self.rng_react[t], b, int(sel.sum()))

        # Avance: velocidad máxima dentro del límite del carril, reducida al cambiar de carril y en amarillo cerca
        # de la línea de alto.
        moving = ~stopped
        lane_changing = self.lc_target[:n] >= 0
        # Velocidad máxima del vehículo sin rebasar el límite del carril (el menor de sus dos
        # carriles si está cambiando), reducida durante la maniobra.
        vcap = np.minimum(self.vmax[:n], self.lane_vmax[self.lane[:n]])
        if lane_changing.any():
            tgt_lane = self.lc_target[:n][lane_changing]
            vcap[lane_changing] = np.minimum(vcap[lane_changing], self.lane_vmax[tgt_lane])
        vcap = vcap * np.where(lane_changing, b.lane_change_speed_factor, 1.0)
        if self.any_bump:
            # Tope: quien lo pisa, o lo alcanzaría en este paso, no rebasa la velocidad de su tipo. Con frenado
            # gradual, además frena antes a decel: con la distancia al tope, la velocidad segura (Gipps, un paso de
            # reacción) solo deja alcanzarlo en un paso a la velocidad del tope o menos. Cuenta el carril de origen
            # y el destino si está cambiando.
            vb = self.bump_v_t[vt]
            tgt = self.lc_target[:n]
            for bump_x, lane_ok in zip(self.bump_x, self.bump_lane):  # cada tope por separado; queda la menor velocidad
                in_lane = lane_ok[self.lane[:n]] | ((tgt >= 0) & lane_ok[np.maximum(tgt, 0)])
                before = in_lane & np.isfinite(vb) & (x - self.vlen[:n] < bump_x)  # aún no lo pasa entero
                vcap = np.where(before & (x + vcap > bump_x), np.minimum(vcap, vb), vcap)
                if dynamics:
                    ahead = before & (x < bump_x)
                    lead_term = vb * vb * dec_vf / dec_vf  # el de adelante es el tope: frena como él
                    vsafe = np.minimum(vsafe, np.where(ahead, _gipps(bump_x - x, lead_term, dec_vf, fin_v), INF))
        if phase == YELLOW:  # quien se aproxima a la línea de alto baja la velocidad
            near = ~ruled_out & (self.L - x <= b.yellow_approach)
            if dynamics:
                # Con frenado gradual decide: se detiene si aún puede (halt, arriba) o sigue sin frenar.
                near &= ~np.isfinite(dec_v)
            vcap = np.where(near, vcap * b.yellow_speed_factor, vcap)
        for near in inner_yellow:
            vcap = np.where(near, vcap * b.yellow_speed_factor, vcap)
        if dynamics:
            # Hacia la velocidad deseada a lo más accel·DT por paso (y bajando a lo más decel·DT), sin
            # rebasar la velocidad segura; el espacio al de adelante sigue siendo un límite duro.
            v_new = np.maximum(np.minimum(v_prev + self.acc_t[vt], vcap), v_prev - dec_v)
            v_new = np.minimum(v_new, vsafe)
        else:
            v_new = vcap
        adv = np.where(moving, np.maximum(np.minimum(v_new, space), 0.0), 0.0)
        stopped[moving & (adv < EPS) & binding_stop] = True
        if self.any_emissions:
            self._emit(vt, x, adv, crossed)  # antes de actualizar v_last: la aceleración usa la del paso anterior
        self.v_last[:n] = adv
        self.vcap_last[:n] = vcap

        pax = self.pax[:n]
        inside = np.minimum(adv, np.maximum(self.L - x, 0.0))
        self.pax_m += np.bincount(vt, weights=pax * inside, minlength=self.n_types)
        self.veh_m += np.bincount(vt, weights=inside, minlength=self.n_types)
        # Velocidad media por carril: distancia y tiempo de los vehículos dentro del tramo (con los
        # detenidos), asignados a su carril actual.
        present = ~crossed
        lane_now = self.lane[:n][present]
        self.lane_dist += np.bincount(lane_now, weights=inside[present], minlength=self.n_lanes)
        self.lane_time += np.bincount(lane_now, minlength=self.n_lanes) * DT
        vt_now = vt[present]
        self.type_dist += np.bincount(vt_now, weights=inside[present], minlength=self.n_types)
        # En el paso en que cruza la línea solo cuenta la fracción del paso que pasó dentro del tramo.
        frac = np.where(adv > EPS, inside / np.maximum(adv, EPS), 1.0)[present]
        self.type_time += np.bincount(vt_now, weights=frac, minlength=self.n_types) * DT
        x += adv
        if passing is not None and passing[0].size:
            me, other = passing
            done = (self.x[me] > self.x[other]) & (self.x[me] - adv[me] <= self.x[other] - adv[other])
            self.in_lane_passes += np.bincount(self.vtype[me[done]], minlength=self.n_types)

        # Llegada a la parada: queda detenido el tiempo de descenso y ascenso.
        if self.any_stop:
            arrive = (stop_state == self.STOP_AHEAD) & (x >= stop_pos - EPS)
            for i in arrive.nonzero()[0]:
                if self.stop_kind[i] == self.BOTTLENECK_STOP:
                    if bn_skip[i]:  # pasó el punto fuera de los carriles de la detención: no se detiene
                        stop_state[i] = self.NO_STOP
                        continue
                    mean, std = self.bn_time[vt[i]]
                    ticks = int(normal_ticks(self.rng_bottleneck_time[vt[i]], mean, std, 1)[0])
                    self.bn_stops[vt[i]] += 1
                    self.bn_ticks[vt[i]] += ticks
                else:
                    ticks = int(stop_ticks(self.rng_stop[vt[i]], self.cfg.specs[vt[i]], 1)[0])
                    self.stops[vt[i]] += 1
                    self.stop_time_ticks[vt[i]] += ticks
                if ticks == 0:
                    stop_state[i] = self.NO_STOP
                    continue
                stop_state[i] = self.AT_STOP
                self.dwell[i] = ticks
                stopped[i] = True
                react[i] = -1

        # Temporizadores de cambio de carril.
        cd = self.cooldown[:n]
        np.subtract(cd, 1, out=cd, where=cd > 0)
        if lane_changing.any():
            timer = self.lc_timer[:n]
            timer[lane_changing] -= 1
            fin = lane_changing & (timer <= 0)
            tgt = self.lc_target[:n]
            self.lane[:n][fin] = tgt[fin]
            tgt[fin] = -1
            cd[fin] = self.cooldown_ticks_t[vt[fin]]

        # Cruce de la línea de alto.
        newly = ~crossed & (x > self.L)
        if newly.any():
            t = vt[newly]
            self.cum_veh += np.bincount(t, minlength=self.n_types)
            self.cum_pax += np.bincount(t, weights=pax[newly], minlength=self.n_types)
            # Tiempo de recorrido solo de quien recorrió todo el tramo (no los de la condición inicial).
            timed = self.entry_tick[:n][newly] >= 0
            travel = self.tick + 1 - self.entry_tick[:n][newly][timed]
            self.timed_veh += np.bincount(t[timed], minlength=self.n_types)
            self.travel_ticks += np.bincount(t[timed], weights=travel, minlength=self.n_types)
            crossed[newly] = True
            by_lane = np.bincount(self.lane[:n][newly], minlength=self.n_lanes).astype(np.float64)
            self.lane_cross += by_lane
            self.lane_crossed += by_lane
            if self.any_exit:
                for i in newly.nonzero()[0]:
                    ln = self.lane[i]
                    if np.isfinite(self.exit_rate[ln]):
                        size = float(self.vlen[i] + self.gap_stop_t[vt[i]])
                        self.exit_items[ln].append(size)
                        self.exit_q[ln] += size

        # Sale del sistema cuando su parte trasera rebasa la línea.
        gone = x - self.vlen[:n] > self.L
        if gone.any():
            self._compact(~gone)

    def _emit(self, vt: np.ndarray, x: np.ndarray, adv: np.ndarray, crossed: np.ndarray) -> None:
        """Gramos que emite en el paso cada vehículo que aún no cruza y cuyo tipo emite, con su velocidad (adv) y
        su aceleración respecto al paso anterior, acotada a [−decel, accel] de su tipo (un frenado de emergencia
        no dispara el término a²). También lo que emitiría recorriendo lo mismo a flujo libre (a velocidad
        constante: su máxima dentro del límite del carril) y dónde lo emite."""
        idx = (~crossed & self.emits_t[vt]).nonzero()[0]
        if idx.size == 0:
            return
        t = vt[idx]
        step = adv[idx]
        v = step / DT
        a = np.minimum(np.maximum((step - self.v_last[idx]) / (DT * DT), -self.emis_dmax[t]), self.emis_amax[t])
        on = self.emis_on[t]
        # La misma cuenta que emission_rate, eligiendo el juego de cada vehículo con un índice en vez de np.where.
        c = self._coef_sets[2 * t + ~(a >= EMISSION_DECEL)]
        v1, a1 = v[:, None], a[:, None]
        e = c[..., 0] + c[..., 1] * v1 + c[..., 2] * v1 * v1 + c[..., 3] * a1 + c[..., 4] * a1 * a1 + c[..., 5] * v1 * a1
        grams = np.maximum(e, 0.0) * DT * on
        vf = np.minimum(self.vmax[idx], self.lane_vmax[self.lane[idx]])  # m/paso
        free = _cruise_rate(self._coef_sets[2 * t], (vf / DT)[:, None]) * DT * (step / vf)[:, None] * on
        # Todos los contaminantes de una vez, con índices aplanados: cada celda suma a los vehículos en el mismo
        # orden que un bincount (o un add.at) por contaminante, así que los totales son idénticos.
        n_pol = grams.shape[1]
        cell = (t[:, None] * n_pol + self._pol_idx).ravel()
        size = self.n_types * n_pol
        by_type = np.bincount(cell, weights=grams.ravel(), minlength=size).reshape(self.n_types, n_pol)
        self.emis_g += by_type
        self.emis_interval += by_type
        self.emis_free_g += np.bincount(cell, weights=free.ravel(), minlength=size).reshape(self.n_types, n_pol)
        _, n_lanes, n_bins = self.emis_pos.shape
        bins = np.minimum(np.maximum((x[idx] / EMIS_BIN).astype(np.int64), 0), n_bins - 1)
        where = (self._pol_idx * n_lanes + self.lane[idx].astype(np.int64)[:, None]) * n_bins + bins[:, None]
        np.add.at(self.emis_pos.reshape(-1), where.ravel(), grams.ravel())

    def _drain_exit(self) -> None:
        """Salen de cada cola de salida los vehículos que acepta su capacidad en este paso. Sin cola, la
        fracción acumulada no pasa de una salida (no se ahorran salidas para después)."""
        limited = self._exit_limited
        self.exit_credit += self._exit_add
        for ln in (limited & (self.exit_credit >= 1)).nonzero()[0]:
            items = self.exit_items[ln]
            while items and self.exit_credit[ln] >= 1:
                self.exit_q[ln] -= items.popleft()
                self.exit_credit[ln] -= 1
            if not items:
                self.exit_q[ln] = 0.0  # sin residuos de redondeo
                self.exit_credit[ln] = min(self.exit_credit[ln], 1.0)

    def rows(self, occ: Occupancy) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Filas lado a lado en el orden de `occ`: (junto, id de fila, índice del último de su fila).

        `junto` marca a quien se traslapa con su líder, es decir, está a su lado en la misma fila.
        Una fila es un tramo consecutivo de entradas unidas así; sin `abreast`, cada vehículo es
        su propia fila."""
        m = occ.order.size
        beside = occ.has_leader & (self.x[occ.leader] - self.vlen[occ.leader] - occ.x < 0)
        new_row = np.ones(m, np.bool_)
        new_row[1:] = ~beside[:-1]  # la entrada i+1 sigue en la fila de i si i está junto a ella
        rid = np.cumsum(new_row) - 1
        last = np.concatenate((new_row[1:], _TRUE)).nonzero()[0]  # último índice de cada fila
        return beside, rid, last[rid]

    def row_positions(self) -> tuple[np.ndarray, np.ndarray]:
        """Por vehículo [0, n): (lugar en su fila lado a lado, 0 = el de adelante; tamaño de la fila)."""
        n = self.n
        rank = np.zeros(n, np.int8)
        size = np.ones(n, np.int8)
        if n < 2 or not self.any_abreast:
            return rank, size
        occ = self.occupancy()
        _, rid, row_last = self.rows(occ)
        idx = np.arange(occ.order.size)
        first = np.concatenate((_TRUE, rid[1:] != rid[:-1])).nonzero()[0][rid]
        primary = occ.order < n  # quien cambia de carril aparece dos veces; se usa su carril actual
        rank[occ.veh[primary]] = (row_last - idx)[primary]
        size[occ.veh[primary]] = (row_last - first + 1)[primary]
        return rank, size

    def _side_slot(self, occ: Occupancy, rows: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None) -> np.ndarray:
        """Entradas que pueden detenerse al lado de su líder: su tipo admite `abreast` > 1, el líder
        está detenido, es de su mismo tipo (como toda su fila) y la fila tiene lugar. Solo quien
        llega en marcha toma el lugar (o quien ya lo ocupa): un vehículo ya detenido en fila, o que
        tiene a otro a su lado, no se adelanta a otra fila. No aplica a quien cambia de carril ni a
        quien ya cruzó. `rows` evita recalcular self.rows(occ)."""
        m = occ.order.size
        if m < 2:
            return np.zeros(m, np.bool_)
        n = self.n
        svt = self.vtype[occ.veh]
        lead = occ.leader
        beside, rid, row_last = self.rows(occ) if rows is None else rows
        nxt = np.minimum(np.arange(m) + 1, m - 1)  # entrada del líder (la siguiente en el orden)
        ahead = row_last[nxt] - np.arange(m)  # vehículos de la fila del líder que van delante
        starts = np.concatenate((_TRUE, rid[1:] != rid[:-1])).nonzero()[0]
        row_lo = np.minimum.reduceat(svt, starts)[rid[nxt]]
        row_hi = np.maximum.reduceat(svt, starts)[rid[nxt]]
        changing = self.lc_target[:n] >= 0
        companion = np.zeros(m, np.bool_)  # alguien se detuvo a su lado (la entrada anterior lo tiene de líder)
        companion[1:] = beside[:-1]
        return (
            occ.has_leader
            & (self.abreast[svt] > 1)
            & self.stopped[lead]
            & (row_lo == svt) & (row_hi == svt)
            & (ahead < self.abreast[svt])
            & (~self.stopped[occ.veh] | beside)
            & ~companion
            & ~changing[occ.veh] & ~changing[lead]
            & ~self.crossed[occ.veh] & ~self.crossed[lead]
        )  # fmt: skip

    def _in_lane_pass(self, occ: Occupancy, beside: np.ndarray, side: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Rebase dentro del carril entre vehículos en marcha del mismo tipo con `pass_in_lane`.

        Devuelve, por entrada, (no cuenta a su líder, lo está rebasando). Rebasa quien tiene una
        velocidad máxima mayor que su líder (ambas acotadas por el límite del carril), siempre que el
        líder no tenga ya a otro a su lado ni a él lo esté alcanzando alguien por un lado: así nunca
        van más de dos lado a lado. El líder no cuenta tampoco para quien es rebasado mientras el que
        se le adelanta, más rápido, va a su lado o se aleja sin sacarle aún `gap_run` más un paso.

        `beside` (se traslapa con su líder) y `side` (puede detenerse al lado de su líder) son los de
        self.rows(occ) y self._side_slot(occ), ya calculados en el paso."""
        me, lead = occ.veh, occ.leader
        svt = self.vtype[me]
        changing = self.lc_target[: self.n] >= 0
        base = (
            occ.has_leader
            & self.pass_in_lane[svt]
            & (self.vtype[lead] == svt)
            & ~self.stopped[me] & ~self.stopped[lead]
            & ~changing[me] & ~changing[lead]
        )  # fmt: skip
        if not base.any():
            return base, base
        lane_v = self.lane_vmax[occ.lane]
        v_me = np.minimum(self.vmax[me], lane_v)
        v_lead = np.minimum(self.vmax[lead], lane_v)
        # El líder ya tiene a otro a su lado o va a detenerse al lado del suyo.
        lead_beside = _next(beside | side)
        behind_beside = _prev(beside)  # alguien lo alcanza por un lado
        faster = v_me > v_lead + EPS
        if self.any_dynamics:
            # Con dinámica gradual además debe ir ya al menos tan rápido como él (no rebasa mientras acelera).
            faster &= ~self.dyn_t[svt] | (self.v_last[me] >= self.v_last[lead] - EPS)
        passing = base & faster & ~lead_beside & ~behind_beside
        passing &= ~_next(passing)  # no rebasa a quien está rebasando: tres quedarían lado a lado
        gap = self.x[lead] - self.vlen[lead] - occ.x
        # Mientras están lado a lado y, después, mientras el que se adelanta de verdad se aleja (avanzó en el
        # último paso al menos lo que él puede), hasta que el espacio le permita avanzar sin frenar.
        # Con dinámica gradual se compara con su velocidad real (no puede acelerar de golpe a la deseada).
        mine = np.where(self.dyn_t[svt], self.v_last[me], self.vcap_last[me]) if self.any_dynamics else self.vcap_last[me]
        pulling_away = (self.v_last[lead] >= mine - EPS) & (gap < self.gap_run_t[svt] + v_me)
        yielding = base & (v_lead > v_me + EPS) & ((gap < 0) | pulling_away)
        return passing | yielding, passing

    def _lane_saturation(self, occ: Occupancy) -> np.ndarray:
        """Saturación de cada carril: largo de su cola de detenidos dentro del tramo
        (largo + gap detenido de cada uno) entre la longitud del tramo, acotada a 1.
        Quien está cambiando de carril cuenta en sus dos carriles; una fila lado a lado
        cuenta una vez."""
        waiting = self.stopped[occ.veh] & ~self.crossed[occ.veh]
        if self.any_abreast:
            waiting &= ~self.rows(occ)[0]
        vt = self.vtype[occ.veh[waiting]]
        queue_m = np.bincount(
            occ.lane[waiting], weights=self.vlen[occ.veh[waiting]] + self.gap_stop_t[vt], minlength=self.n_lanes
        )
        return np.minimum(queue_m / self.L, 1.0)

    def _lane_changes(self) -> None:
        """Decisiones de cambio de carril de quienes circulan (no detenidos, sin maniobra ni
        enfriamiento), en orden aleatorio y con a lo más una entrada a cada carril por ronda.

        Condición para cualquier cambio: un líder dentro de su `lookahead` cuya velocidad real en el
        último paso esté al menos `leader_slowdown` (p. ej. 25 %) por debajo del límite del carril
        (en un carril sin límite, de la velocidad máxima del propio vehículo). Sin líder, no cambia.

        * Rebase: con un líder lento dentro de su `lookahead`, busca un carril adyacente donde
          avance más rápido y tenga `min_advantage` m más de espacio libre. Los tipos con
          `overtake` juzgan la lentitud por la velocidad real del último paso (líder en maniobra,
          frenado por su cola) y aceptan un carril de menor límite si ahí avanzan
          más que detrás del líder.
        * Subida: si no puede rebasar, pasa a un carril adyacente de mayor límite si hay lugar sin
          comprimir a nadie (`gap_run` adelante y atrás) y ahí no lo frena un líder dentro de su
          `lookahead`.
        """
        n = self.n
        vt = self.vtype[:n]
        cand = (
            self.can_change[vt]
            & (self.lc_target[:n] < 0)
            & (self.cooldown[:n] == 0)
            & ~self.stopped[:n]
            & ~self.crossed[:n]
        )
        if not cand.any():
            return
        occ = self.occupancy()
        svt = self.vtype[occ.veh]
        gap = self.x[occ.leader] - self.vlen[occ.leader] - occ.x
        # Velocidades efectivas en el carril de cada entrada (máxima del vehículo, sin rebasar el límite).
        lane_v = self.lane_vmax[occ.lane]
        v_self = self.vmax[occ.veh]
        eff_self = np.minimum(v_self, lane_v)
        eff_lead = np.minimum(self.vmax[occ.leader], lane_v)
        lead_stopped = self.stopped[occ.leader]
        slow = occ.has_leader & ((eff_lead < eff_self) | lead_stopped)
        stuck_speed = np.where(lead_stopped, 0.0, eff_lead)  # a la que lo obliga su líder
        over = self.overtake[svt]
        if over.any():
            # Rebase agresivo: el líder es lento si en el último paso avanzó menos de lo que podía él.
            v_lead = self.v_last[occ.leader]
            slow |= occ.has_leader & over & (v_lead < self.vcap_last[occ.veh] - EPS)
            stuck_speed = np.where(over, v_lead, stuck_speed)
        # Condición común: líder cercano que va al menos `leader_slowdown` por debajo del límite del carril.
        b = self.cfg.behavior
        lane_max = np.where(np.isfinite(lane_v), lane_v, v_self)
        v_ahead = np.where(lead_stopped, 0.0, self.v_last[occ.leader])
        blocked = (
            cand[occ.veh]
            & occ.has_leader
            & (gap < self.lookahead_t[svt])
            & (v_ahead < (1.0 - b.leader_slowdown) * lane_max - EPS)
        )
        passing = blocked & slow
        rising = blocked & self._can_rise(occ, eff_self) if self.any_rise else np.zeros_like(blocked)
        idx = np.flatnonzero(passing | rising)
        if idx.size == 0:
            return

        bounds = np.searchsorted(occ.lane, np.arange(self.n_lanes + 1))
        used: set[int] = set()
        for k in self.rng_lane_change.permutation(idx):
            best = self._pass_lane(occ, bounds, used, k, gap[k], stuck_speed[k]) if passing[k] else -1
            if best < 0 and rising[k]:
                best = self._rise_lane(occ, bounds, used, k)
            if best >= 0:
                i = int(occ.veh[k])
                used.add(best)
                self.lc_target[i] = best
                self.lc_timer[i] = uniform_ticks(self.rng_lane_change, b.lane_change_min, b.lane_change_max)
                self.lane_changes += 1
                self.lane_changes_t[self.vtype[i]] += 1

    def _can_rise(self, occ: Occupancy, eff_self: np.ndarray) -> np.ndarray:
        """Entradas con un carril adyacente permitido donde su velocidad máxima (acotada por el límite)
        es mayor que en el propio: candidatos a subir."""
        svt = self.vtype[occ.veh]
        out = np.zeros(occ.order.size, np.bool_)
        for d in (1, -1):
            adj = occ.lane + d
            ok = (adj >= 0) & (adj < self.n_lanes)
            adj = np.clip(adj, 0, self.n_lanes - 1)
            faster = np.minimum(self.vmax[occ.veh], self.lane_vmax[adj]) > eff_self + EPS
            out |= ok & self.allowed[svt, adj] & faster
        return out & self.can_change[svt]

    def _neighbors(self, occ: Occupancy, bounds: np.ndarray, lane: int, x: float) -> tuple[int, int]:
        """(líder, seguidor) que tendría en `lane` un vehículo con el frente en x; -1 si no hay."""
        lo, hi = bounds[lane], bounds[lane + 1]
        p = lo + int(np.searchsorted(occ.x[lo:hi], x))
        return (int(occ.veh[p]) if p < hi else -1), (int(occ.veh[p - 1]) if p > lo else -1)

    def _pass_lane(
        self, occ: Occupancy, bounds: np.ndarray, used: set[int], k: int, gap: float, stuck_speed: float
    ) -> int:  # fmt: skip
        """Carril adyacente para rebasar al líder lento de la entrada k; -1 si ninguno conviene."""
        i, xi, li = int(occ.veh[k]), occ.x[k], int(occ.lane[k])
        vi = self.vtype[i]
        over = self.overtake[vi]
        look = self.lookahead_t[vi]
        best, best_score = -1, min(gap, look) + self.min_adv_t[vi]
        for tl in (li + 1, li - 1):  # prefiere rebasar por la izquierda
            if tl < 0 or tl >= self.n_lanes or tl in used or not self.allowed[vi, tl]:
                continue
            v_there = min(self.vmax[i], self.lane_vmax[tl])
            if v_there <= stuck_speed:  # ese carril no mejora la velocidad actual
                continue
            j, f = self._neighbors(occ, bounds, tl, xi)
            score = look
            if j >= 0:  # líder en el carril destino
                front = self.x[j] - self.vlen[j] - xi
                if front < self.gap_stop_t[vi] or not self._can_brake_behind(i, j, front):
                    continue
                v_j = self.v_last[j] + EPS if over else min(self.vmax[j], self.lane_vmax[tl])
                if v_j < v_there or self.stopped[j]:
                    score = min(front, look)
            if f >= 0:  # seguidor en el carril destino: se comprime, pero sin traslape
                back = xi - self.vlen[i] - self.x[f]
                if back < self.gap_stop_t[self.vtype[f]] or not self._can_brake_behind(f, i, back):
                    continue
            if score >= best_score:
                best, best_score = tl, score
        return best

    def _rise_lane(self, occ: Occupancy, bounds: np.ndarray, used: set[int], k: int) -> int:
        """Carril adyacente de mayor límite al que la entrada k puede subir; -1 si no hay lugar."""
        i, xi, li = int(occ.veh[k]), occ.x[k], int(occ.lane[k])
        vi = self.vtype[i]
        best, best_v = -1, min(self.vmax[i], self.lane_vmax[li]) + EPS
        for tl in (li + 1, li - 1):
            if tl < 0 or tl >= self.n_lanes or tl in used or not self.allowed[vi, tl]:
                continue
            v_there = min(self.vmax[i], self.lane_vmax[tl])
            if v_there <= best_v:
                continue
            j, f = self._neighbors(occ, bounds, tl, xi)
            if j >= 0:
                front = self.x[j] - self.vlen[j] - xi
                if front < self.gap_run_t[vi] or not self._can_brake_behind(i, j, front):
                    continue
                # Un líder cercano detenido o no más rápido que él le quita la ventaja.
                if front < self.lookahead_t[vi] and (self.stopped[j] or self.v_last[j] <= self.v_last[i] + EPS):
                    continue
            if f >= 0:
                back = xi - self.vlen[i] - self.x[f]
                if back < self.gap_run_t[self.vtype[f]] or not self._can_brake_behind(f, i, back):
                    continue
            best, best_v = tl, v_there
        return best

    def _can_brake_behind(self, i: int, j: int, gap: float) -> bool:
        """Con frenado gradual, `i` puede quedar `gap` m detrás de `j` sin frenar más que su `decel`
        (con un cambio de carril, sea quien cambia o el seguidor del carril destino). Sin frenado
        gradual, siempre."""
        dec = self.dec_t[self.vtype[i]]
        if not math.isfinite(dec):
            return True
        stopped = self.stopped[j]
        room = gap - (self.gap_stop_t if stopped else self.gap_run_t)[self.vtype[i]]
        safe = _safe_speed_1(room, 0.0 if stopped else float(self.v_last[j]), dec, self.dec_t[self.vtype[j]])
        # También el límite duro: en un paso no avanza más que `room` (medido desde donde está j ahora).
        return self.v_last[i] - dec <= min(safe, room) + EPS

    def _sample(self) -> None:
        n = self.n
        pax_on = np.zeros(self.n_types)
        footprint = np.zeros(self.n_types)
        lane_sat = np.zeros(self.n_lanes)
        if n:
            vt = self.vtype[:n]
            inside = ~self.crossed[:n]
            pax_on = np.bincount(vt[inside], weights=self.pax[:n][inside], minlength=self.n_types)
            occ = self.occupancy()
            svt = self.vtype[occ.veh]
            gap = np.where(
                occ.has_leader,
                self.x[occ.leader] - self.vlen[occ.leader] - occ.x,
                INF,
            )
            # Detenido ocupa su largo + gap comprimido; en marcha, a lo más largo + gap en marcha.
            cap = np.where(self.stopped[occ.veh], self.gap_stop_t[svt], self.gap_run_t[svt])
            fp = self.vlen[occ.veh] + np.minimum(gap, cap)
            # Espacio de los que llevan pasajeros: los de mercancía no cuentan en pax/m.
            ins = ~self.crossed[occ.veh] & (self.pax[occ.veh] > 0)
            footprint = np.bincount(svt[ins], weights=fp[ins], minlength=self.n_types)
            lane_sat = self._lane_saturation(occ)
        self.recorder.record(self.cum_pax, self.pax_m, pax_on, footprint, lane_sat, self.lane_dist, self.lane_time,
                             self.lane_cross, self.exit_q, [len(q) for q in self.queues], self.emis_interval)  # fmt: skip
        self.emis_interval[:] = 0.0
        self.lane_cross[:] = 0.0
        self.pax_m[:] = 0.0
        self.lane_dist[:] = 0.0
        self.lane_time[:] = 0.0

    # --------------------------------------------------------------- resumen

    def summary(self) -> dict[str, np.ndarray]:
        n = self.n
        queued = np.bincount(
            np.fromiter((vt for q in self.queues for vt, *_ in q), np.int64),
            minlength=self.n_types,
        ).astype(np.float64)
        on_road = np.bincount(
            self.vtype[:n][~self.crossed[:n]], minlength=self.n_types
        ).astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            travel_time = self.travel_ticks * DT / self.timed_veh
            # Espera media en la cola de entrada de los que llegaron por la demanda y ya entraron al tramo.
            queue_wait = self.queue_wait_ticks * DT / (self.entered_veh - self.initial_veh)
            pax_per_veh = self.arrived_pax / (self.arrived_veh - self.arrived_cargo)  # solo los de pasajeros
            # Velocidad media dentro del tramo: distancia / tiempo de sus vehículos, con los detenidos.
            mean_speed = self.type_dist / self.type_time * 3.6
        return {
            "arrived_veh": self.arrived_veh.copy(),
            "arrived_pax": self.arrived_pax.copy(),
            "arrived_cargo": self.arrived_cargo.copy(),
            "entered_veh": self.entered_veh.copy(),
            "initial_veh": self.initial_veh.copy(),
            "initial_pax": self.initial_pax.copy(),
            "crossed_veh": self.cum_veh.copy(),
            "crossed_pax": self.cum_pax.copy(),
            "travel_time": travel_time,
            "mean_speed": mean_speed,
            "queue_wait": queue_wait,
            "pax_per_veh": pax_per_veh,
            "queued": queued,
            "on_road": on_road,
            "lane_changes": np.array([float(self.lane_changes)]),
            "lane_changes_type": self.lane_changes_t.copy(),
            "in_lane_passes": self.in_lane_passes.copy(),
            "lane_crossed": self.lane_crossed.copy(),  # cruces de la línea por carril
            "exit_blocked": self.exit_blocked_ticks / max(self.tick, 1),  # fracción del tiempo con la línea cerrada
            "stop_time": self.stop_time_ticks * DT / np.where(self.stops > 0, self.stops, np.nan),
            "bottleneck_stops": self.bn_stops.copy(),
            "bottleneck_time": self.bn_ticks * DT / np.where(self.bn_stops > 0, self.bn_stops, np.nan),
            "veh_km": self.veh_m / 1000.0,  # km recorridos dentro del tramo, por tipo
            "emissions": self.emis_g.copy(),  # g por tipo y contaminante (POLLUTANTS)
            "emissions_free": self.emis_free_g.copy(),  # g del mismo recorrido a velocidad constante
            "emissions_pos": self.emis_pos.copy(),  # g por contaminante, carril e intervalo de EMIS_BIN m
            **self._pedestrian_summary(),
        }

    def _pedestrian_summary(self) -> dict[str, np.ndarray]:
        """Una fila por punto con peatones (`cfg.pedestrian_spots`): peatones que cruzaron, cierres del paso, s de
        paso cerrado y espera total de los peatones (s). Vacío si no hay peatones."""
        spots = self.cfg.pedestrian_spots
        if not spots:
            return {}
        rows = [(s.crossed, s.crossings, s.closed_ticks * DT, s.wait_ticks * DT)
                for s in (self.ped_schedules[spot] for spot in spots)]  # fmt: skip
        return {"pedestrians": np.array(rows, dtype=np.float64)}
