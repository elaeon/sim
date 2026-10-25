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

El cálculo de cada paso está en kernel.py, compilado con numba. Aquí se arman las tablas y el estado de la réplica,
la condición inicial, los calendarios de peatones, las llegadas por bloques y el resumen.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from trafico import kernel
from trafico.config import DT, MAX_TYPES, PEDESTRIAN_YIELD, RED, SimConfig
from trafico.distributions import sample_lengths, sample_passengers, sample_rates, sample_speeds
from trafico.emissions import EMIS_BIN, coefficient_table
from trafico.kernel import ARRIVAL_CHUNK, EPS, INF, NAN, Occupancy
from trafico.metrics import Recorder
from trafico.pedestrians import Schedule, arrivals, bump_schedule, fixed_phases, light_schedule

PER_TYPE_STREAMS = 5  # generadores aleatorios por tipo de vehículo
MAX_PAX = 255  # cota de pasajeros por vehículo (pax es uint8)
INITIAL_MISSES = 30  # sorteos seguidos que no caben antes de dar por lleno un carril en la condición inicial
# Series del Recorder que escribe el paso (State.r_<nombre>).
RECORDER_FIELDS = ("cum_pax", "pax_m", "pax_on", "footprint", "lane_sat", "lane_dist", "lane_time", "lane_cross",
                   "exit_q", "entry_q", "emis")  # fmt: skip


def compile_kernel(cfg: SimConfig) -> None:
    """Compila el kernel en este proceso, o lo carga de la caché de numba, con una réplica corta de `cfg`. Lleva
    condición inicial para compilar también lo que la arma."""
    Simulation(replace(cfg, run=1.0, time_scale=1.0, initial_occupancy=0.5), np.random.default_rng(0)).advance(1)


class Simulation:
    _FIELDS = (
        "vtype", "pax", "lane", "lc_target", "lc_timer", "cooldown",
        "react", "x", "stopped", "crossed", "entry_tick", "vid", "stop_state", "dwell", "v_last", "vcap_last", "vlen",
        "stop_pos", "stop_kind", "vmax",
    )  # fmt: skip
    _QUEUE_FIELDS = ("q_vt", "q_pax", "q_len", "q_at", "q_speed", "q_arrived")
    NO_STOP, STOP_AHEAD, AT_STOP = kernel.NO_STOP, kernel.STOP_AHEAD, kernel.AT_STOP
    TYPE_STOP, BOTTLENECK_STOP = kernel.TYPE_STOP, kernel.BOTTLENECK_STOP

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
        # las obedecen (carril exclusivo en los free_lanes del semáforo) y carriles donde rigen (todos si no dice).
        self.lines = cfg.stop_lines
        self.inner_x = np.array([s.position for s in self.lines], np.float64)
        self.inner_free = np.zeros((len(self.lines), self.n_types), np.bool_)
        self.inner_lane = np.ones((len(self.lines), cfg.lanes), np.bool_)
        for j, s in enumerate(self.lines):
            self.inner_free[j] = [s.light is not None and cfg.ignores_light(k, s.light) for k in range(self.n_types)]
            if s.lanes is not None:
                self.inner_lane[j] = np.isin(np.arange(cfg.lanes), s.lanes)
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
        # Carriles de entrada de cada tipo (los primeros entry_count de su fila).
        self.entry_count = np.array([len(cfg.entry_lanes(k)) for k in range(self.n_types)], np.int64)
        self.entry_lanes = np.zeros((self.n_types, cfg.lanes), np.int64)
        for k in range(self.n_types):
            self.entry_lanes[k, : self.entry_count[k]] = cfg.entry_lanes(k)
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
        self.dec_safe_t = np.where(self.dec_fin_t, self.dec_t, 1.0)  # sin infinitos, para kernel.gipps
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
        self.veh_m = np.zeros(self.n_types)  # m recorridos dentro del tramo, por tipo
        # Sin límites distintos entre carriles no hay a dónde subir.
        self.any_rise = len(set(cfg.lane_max_kmh)) > 1
        self._key_stride = self.L + 1e4  # separa los carriles en la clave de orden

        self._ints = np.zeros(5, np.int64)  # n, paso, cambios de carril, muestras, posición en el bloque de llegadas
        min_slot = min(s.shortest + s.gap_stop for s in specs)
        per_lane = math.ceil((self.L + max(s.longest for s in specs)) / min_slot) + 2
        self._alloc(cfg.lanes * per_lane)
        # Cola de entrada de cada carril (anillo): (tipo, pax, largo, posición de su detención de [bottleneck],
        # velocidad máxima km/h, paso de llegada a la cola).
        queue_cap = 64
        self.q_vt = np.zeros((cfg.lanes, queue_cap), np.int64)
        self.q_pax = np.zeros((cfg.lanes, queue_cap), np.int64)
        self.q_len = np.zeros((cfg.lanes, queue_cap))
        self.q_at = np.zeros((cfg.lanes, queue_cap))
        self.q_speed = np.zeros((cfg.lanes, queue_cap))
        self.q_arrived = np.zeros((cfg.lanes, queue_cap), np.int64)
        self.q_head = np.zeros(cfg.lanes, np.int64)
        self.q_count = np.zeros(cfg.lanes, np.int64)
        # Cola de entrada detenida por carril y pasos de reacción que le faltan a su primero (-1 = sin empezar).
        self.queue_stopped = np.zeros(cfg.lanes, np.bool_)
        self.queue_react = np.full(cfg.lanes, -1, np.int32)
        # Bloque de llegadas en curso (ninguno: se sortea al empezar).
        self._arr = kernel.Arrivals(-1, *(np.zeros(0, d) for d in (np.int64,) * 3 + (np.float64,) * 3))

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
        # Lo que ocupa cada vehículo en la cola de salida, en orden de llegada (anillo): sale el primero que entró.
        self.exit_ring = np.zeros((cfg.lanes, self.x.size))
        self.exit_head = np.zeros(cfg.lanes, np.int64)
        self.exit_count = np.zeros(cfg.lanes, np.int64)
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
        self._p = kernel.as_ref(self._params())
        self._st = kernel.as_ref(self._state())
        self._rngs = kernel.as_ref(self._kernel_rngs())
        if any(v > 0 for v in cfg.lane_initial_occupancy):
            self._initial_fill(fill_rng)

    # ------------------------------------------------------------------ estado

    @property
    def n(self) -> int:
        """Vehículos en el tramo: ocupan las posiciones [0, n) de cada arreglo."""
        return int(self._ints[kernel.N])

    @property
    def tick(self) -> int:
        return int(self._ints[kernel.TICK])

    @property
    def lane_changes(self) -> int:
        return int(self._ints[kernel.LANE_CHANGES])

    @property
    def queues(self) -> list[list[tuple[int, int, float, float, float, int]]]:
        """Cola de entrada de cada carril, del primero al último: (tipo, pax, largo, posición de su detención de
        [bottleneck], velocidad máxima km/h, paso de llegada). Es una copia."""
        cap = self.q_vt.shape[1]
        out = []
        for ln in range(self.n_lanes):
            idx = (self.q_head[ln] + np.arange(self.q_count[ln])) % cap
            out.append([(int(self.q_vt[ln, j]), int(self.q_pax[ln, j]), float(self.q_len[ln, j]),
                         float(self.q_at[ln, j]), float(self.q_speed[ln, j]), int(self.q_arrived[ln, j]))
                        for j in idx])  # fmt: skip
        return out

    @property
    def queue_len(self) -> np.ndarray:
        """Vehículos en la cola de entrada de cada carril."""
        return self.q_count.copy()

    def queued_types(self) -> np.ndarray:
        """Vehículos de cada tipo en las colas de entrada."""
        cap = self.q_vt.shape[1]
        types = [self.q_vt[ln, (self.q_head[ln] + np.arange(self.q_count[ln])) % cap] for ln in range(self.n_lanes)]
        return np.bincount(np.concatenate(types), minlength=self.n_types).astype(np.float64)

    @property
    def exit_items(self) -> list[list[float]]:
        """Lo que ocupa cada vehículo en la cola de salida de cada carril, del primero al último. Es una copia."""
        cap = self.exit_ring.shape[1]
        return [self.exit_ring[ln, (self.exit_head[ln] + np.arange(self.exit_count[ln])) % cap].tolist()
                for ln in range(self.n_lanes)]  # fmt: skip

    def _queue_exit(self, ln: int, size: float) -> None:
        """Pone en la cola de salida del carril `ln` un vehículo que ocupa `size` m."""
        cap = self.exit_ring.shape[1]
        self.exit_ring[ln, (self.exit_head[ln] + self.exit_count[ln]) % cap] = size
        self.exit_count[ln] += 1
        self.exit_q[ln] += size

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

    def _make_room(self) -> None:
        """Agranda al doble los arreglos que no alcanzan para el paso actual (ver kernel.has_room)."""
        n = self.n
        if n + self.n_lanes > self.x.size:
            old = {f: getattr(self, f) for f in self._FIELDS}
            self._alloc(2 * old["x"].size)
            for f, arr in old.items():
                getattr(self, f)[: arr.size] = arr
        k = self._ints[kernel.ARRIVAL_POS]
        arriving = int(np.count_nonzero(self._arr.tick[k:] == self.tick))
        while (self.q_count + arriving > self.q_vt.shape[1]).any():
            self._grow_ring(self._QUEUE_FIELDS, self.q_head, self.q_count)
        while self.any_exit and (self.exit_count + n > self.exit_ring.shape[1]).any():
            self._grow_ring(("exit_ring",), self.exit_head, self.exit_count)
        self._st = kernel.as_ref(self._state())

    def _grow_ring(self, fields: tuple[str, ...], head: np.ndarray, count: np.ndarray) -> None:
        """Duplica los anillos `fields` (carriles × capacidad); cada carril queda en orden desde la posición 0."""
        cap = getattr(self, fields[0]).shape[1]
        for f in fields:
            old = getattr(self, f)
            new = np.zeros((old.shape[0], 2 * cap), old.dtype)
            for ln in range(old.shape[0]):
                new[ln, : count[ln]] = old[ln, (head[ln] + np.arange(count[ln])) % cap]
            setattr(self, f, new)
        head[:] = 0

    def _params(self) -> kernel.Params:
        cfg, b = self.cfg, self.cfg.behavior
        specs = cfg.specs
        if b.reaction_std is None:
            react_mode, react_mean = kernel.REACT_UNIFORM, 0.0
        else:
            react_mean = b.reaction_mean if b.reaction_mean is not None else (b.reaction_min + b.reaction_max) / 2
            react_mode = kernel.REACT_FIXED if b.reaction_std <= 0 else kernel.REACT_NORMAL
        # Fases de las líneas de alto: la del final del tramo y las intermedias.
        n_lines = 1 + len(self.lines)
        kind = np.zeros(n_lines, np.int64)
        red, green, yellow, offset = (np.zeros(n_lines, np.int64) for _ in range(4))
        start_red = np.zeros(n_lines, np.bool_)
        sched = np.zeros((n_lines, cfg.n_ticks + 1), np.int8)
        lights = [cfg.exit_light] + [s.light for s in self.lines]
        for j, (light, schedule) in enumerate(zip(lights, [self._exit_sched] + self._line_sched)):
            if schedule is not None:
                kind[j] = kernel.LINE_SCHEDULE
                sched[j] = schedule.phases
            elif light is not None and light.active and not light.pedestrian:
                kind[j] = kernel.LINE_CYCLE
                red[j], green[j], yellow[j] = round(light.red / DT), round(light.green / DT), round(light.yellow / DT)
                offset[j] = round(light.offset / DT)
                start_red[j] = light.start_phase == "red"
        stop_mean = np.array([s.stop_time_mean for s in specs], np.float64)
        stop_std = np.array([s.stop_time_std for s in specs], np.float64)
        bn_mean = np.array([m for m, _ in self.bn_time], np.float64)
        bn_std = np.array([sd for _, sd in self.bn_time], np.float64)
        return kernel.Params(
            L=self.L, n_lanes=self.n_lanes, n_types=self.n_types, n_ticks=cfg.n_ticks, lc_every=self.lc_every,
            sample_ticks=cfg.sample_ticks, key_stride=self._key_stride,
            any_headway=self.any_headway, any_stop=self.any_stop, any_abreast=self.any_abreast,
            any_pass_in_lane=self.any_pass_in_lane, any_dynamics=self.any_dynamics, any_bump=self.any_bump,
            any_emissions=self.any_emissions, any_exit=self.any_exit, any_rise=self.any_rise,
            yellow_approach=float(b.yellow_approach), yellow_speed_factor=float(b.yellow_speed_factor),
            release_space=float(b.release_space), lane_change_speed_factor=float(b.lane_change_speed_factor),
            leader_slowdown=float(b.leader_slowdown), queue_reaction=bool(b.queue_reaction),
            react_mode=react_mode, react_lo=round(b.reaction_min / DT), react_hi=round(b.reaction_max / DT),
            react_mean=float(react_mean), react_std=float(b.reaction_std or 0.0),
            react_min=float(b.reaction_min), react_max=float(b.reaction_max),
            lc_min=round(b.lane_change_min / DT), lc_max=round(b.lane_change_max / DT),
            v_step=self.v_step, length_t=self.length_t, gap_run_t=self.gap_run_t, gap_stop_t=self.gap_stop_t,
            has_headway_t=self.has_headway_t, headway_t=self.headway_t, can_change=self.can_change,
            stop_x=self.stop_x, has_stop=self.has_stop, stop_mean=stop_mean, stop_std=stop_std,
            stop_fixed=np.array([round(max(m, 0.0) / DT) for m in stop_mean], np.int64),
            bn_mean=bn_mean, bn_std=bn_std, bn_fixed=np.array([round(max(m, 0.0) / DT) for m in bn_mean], np.int64),
            free_type=self.free_type, abreast=self.abreast, pass_in_lane=self.pass_in_lane, allowed=self.allowed,
            entry_lanes=self.entry_lanes, entry_count=self.entry_count, lookahead_t=self.lookahead_t,
            min_adv_t=self.min_adv_t, cooldown_ticks_t=self.cooldown_ticks_t, overtake=self.overtake,
            acc_t=self.acc_t, dec_t=self.dec_t, dyn_t=self.dyn_t, dec_fin_t=self.dec_fin_t, dec_safe_t=self.dec_safe_t,
            bump_v_t=self.bump_v_t, emits_t=self.emits_t, emis_on=self.emis_on, coef_sets=self._coef_sets,
            emis_amax=self.emis_amax, emis_dmax=self.emis_dmax,
            lane_vmax=self.lane_vmax, bn_lane=self.bn_lane, exit_rate=self.exit_rate, exit_limited=self._exit_limited,
            exit_add=self._exit_add, exit_storage=self.exit_storage, bump_x=self.bump_x, bump_lane=self.bump_lane,
            inner_x=self.inner_x, inner_free=self.inner_free, inner_lane=self.inner_lane,
            line_kind=kind, line_red=red, line_green=green, line_yellow=yellow, line_start_red=start_red,
            line_offset=offset, line_sched=sched,
        )  # fmt: skip

    def _state(self) -> kernel.State:
        """El estado para el kernel: los mismos arreglos de la simulación (sin copias)."""
        rec = {f"r_{name}": getattr(self.recorder, name) for name in RECORDER_FIELDS}
        return kernel.State(ints=self._ints, **{f: rec[f] if f in rec else getattr(self, f)
                                                for f in kernel.State._fields if f != "ints"})  # fmt: skip

    def _kernel_rngs(self) -> kernel.Rngs:
        """Los generadores del paso, MAX_TYPES por proceso (numba compila una sola vez para cualquier número de
        tipos); los de relleno no se usan."""
        spare = [np.random.default_rng(0) for _ in range(MAX_TYPES - self.n_types)]

        def full(gens) -> tuple:
            return tuple(gens) + tuple(spare)

        return kernel.Rngs(full(self.rng_react), full(self.rng_stop), full(self.rng_bottleneck_time),
                           full(self.rng_queue_react), full(self.rng_entry), self.rng_lane_change)  # fmt: skip

    def _add(
        self, vt: int, pax: int, lane: int, length: float | None = None, x: float | None = None,
        bottleneck_at: float = INF, speed_kmh: float | None = None, v0: float | None = None,
    ) -> None:  # fmt: skip
        """Agrega un vehículo (ver kernel.add_vehicle); sin `x`, entra al inicio del tramo."""
        if self.n == self.x.size:
            self._make_room()
        kernel.py_add_vehicle(self._p, self._st, int(vt), int(pax), int(lane), NAN if length is None else float(length),
                           NAN if x is None else float(x), float(bottleneck_at),
                           NAN if speed_kmh is None else float(speed_kmh), NAN if v0 is None else float(v0))  # fmt: skip

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
            run_gap = [kernel.py_run_gap(self._p, k, min(speed / 3.6 * DT, self.lane_vmax[lane]))
                       for k, *_, speed in chosen]  # fmt: skip
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
                        self.v_last[i] = min(self.v_last[i],
                                             kernel.py_safe_speed_1(room, v_ahead, self.dec_t[k], dec_ahead))  # fmt: skip
                    light = self._line_room(k, self.x[i], lane)
                    if np.isfinite(light):  # línea de alto intermedia en rojo al empezar
                        self.v_last[i] = min(self.v_last[i], kernel.py_safe_speed_1(light, 0.0, self.dec_t[k], INF))
                    bump = kernel.py_bump_speed_1(self._p, k, lane, self.x[i], self.x[i] - length)
                    self.v_last[i] = min(self.v_last[i], bump)
                ahead = i
                rear = self.x[i] - length

    def _line_room(self, k: int, x: float, lane: int) -> float:
        """Distancia desde `x` hasta la línea de alto intermedia (semáforo o paso peatonal) en rojo al empezar la
        corrida más cercana adelante que obedece el tipo k en ese carril; INF si no hay."""
        return min((pos - x for pos, free, ok, ph in zip(self.inner_x, self.inner_free, self.inner_lane, self.line_phases(0))
                    if ph == RED and x <= pos and not free[k] and ok[lane]), default=INF)  # fmt: skip

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
        # Calendario de cada línea de alto intermedia (None = semáforo de ciclo fijo: su fase sale de su ciclo).
        self._line_sched = [self.ped_schedules.get((s.kind, s.position)) if s.pedestrian else None for s in self.lines]

    def exit_phase(self, tick: int) -> int:
        """Fase (RED, GREEN o YELLOW) del semáforo del final del tramo en el paso `tick`."""
        return int(kernel.py_line_phase(self._p, 0, tick))

    def line_phases(self, tick: int) -> tuple[int, ...]:
        """Fase de cada línea de alto intermedia (`cfg.stop_lines`) en el paso `tick`."""
        return tuple(int(kernel.py_line_phase(self._p, 1 + j, tick)) for j in range(len(self.lines)))

    # ------------------------------------------------------------------- paso

    def advance(self, until: int) -> None:
        """Corre los pasos hasta el paso `until` (el kernel; aquí solo las llegadas de cada bloque y el lugar)."""
        while (code := kernel.advance(self._p, self._st, self._rngs, self._arr, until)) != kernel.DONE:
            if code == kernel.NEED_ARRIVALS:
                self._load_arrivals()
            else:
                self._make_room()
        self.recorder.count = int(self._ints[kernel.SAMPLES])

    def run(self) -> None:
        self.advance(self.tick + self.cfg.n_ticks)

    def step(self) -> None:
        self.advance(self.tick + 1)

    def _load_arrivals(self) -> None:
        """Sorteo de llegadas Poisson del bloque de ARRIVAL_CHUNK pasos que empieza en el paso actual, y de lo de cada
        llegada (pasajeros, mercancía, largo, velocidad y detención), con los generadores de cada tipo y en orden de
        paso: los mismos números que sorteando al llegar."""
        start = self.tick
        chunk = np.column_stack([self._arrival_counts(k) for k in range(self.n_types)])
        cols: list[list[np.ndarray]] = [[], [], [], [], [], []]
        for offset in np.flatnonzero(chunk.any(axis=1)):
            counts = chunk[offset]
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
                for col, values in zip(cols, (np.full(k, start + offset), np.full(k, vt), paxs, lengths, at, speeds)):
                    col.append(values)
        dtypes = (np.int64,) * 3 + (np.float64,) * 3
        self._arr = kernel.Arrivals(start, *(np.concatenate(c).astype(d) if c else np.zeros(0, d)
                                             for c, d in zip(cols, dtypes)))  # fmt: skip
        self._ints[kernel.ARRIVAL_POS] = 0

    def _arrival_counts(self, k: int) -> np.ndarray:
        """Llegadas del tipo k en cada paso del bloque que empieza en el paso actual."""
        g = self.rng_arrivals[k]
        by_interval = self.rate_by_interval.get(k)
        if by_interval is None:
            return g.poisson(self.rate_tick[k], ARRIVAL_CHUNK)
        idx = (self.tick + np.arange(ARRIVAL_CHUNK)) // self.cfg.rate_ticks
        return g.poisson(by_interval[np.minimum(idx, by_interval.size - 1)])

    # ------------------------------------------------------- consultas del estado

    def occupancy(self) -> Occupancy:
        return kernel.py_occupancy(self._p, self._st)

    def rows(self, occ: Occupancy) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Filas lado a lado en el orden de `occ` (ver kernel.rows)."""
        return kernel.py_rows(self._st, occ)

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
        first = np.concatenate(([True], rid[1:] != rid[:-1])).nonzero()[0][rid]
        primary = occ.order < n  # quien cambia de carril aparece dos veces; se usa su carril actual
        rank[occ.veh[primary]] = (row_last - idx)[primary]
        size[occ.veh[primary]] = (row_last - first + 1)[primary]
        return rank, size

    def _side_slot(self, occ: Occupancy, rows: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None) -> np.ndarray:
        """Entradas que pueden detenerse al lado de su líder (ver kernel.side_slot)."""
        return kernel.py_side_slot(self._p, self._st, occ, *(self.rows(occ) if rows is None else rows))

    def _lane_saturation(self, occ: Occupancy) -> np.ndarray:
        return kernel.py_lane_saturation(self._p, self._st, occ)

    def _lane_changes(self) -> None:
        """Una ronda de decisiones de cambio de carril (ver kernel.lane_changes)."""
        kernel.py_lane_changes(self._p, self._st, self._rngs)

    # --------------------------------------------------------------- resumen

    def summary(self) -> dict[str, np.ndarray]:
        n = self.n
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
            "queued": self.queued_types(),
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
