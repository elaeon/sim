"""Motor microscópico: tramo de `lanes` carriles con un semáforo en x = length.

Estado en estructura de arreglos (SoA) compacta: los vehículos activos ocupan
las posiciones [0, n) de cada arreglo; los que salen se eliminan compactando.
La capacidad inicial es la cota física de vehículos que caben en el tramo, así
que la memoria no crece con la duración de la corrida.

Modelo por paso de DT segundos (actualización paralela con posiciones previas):
  * Cada tipo avanza a su velocidad máxima sin rebasar el límite del carril; el
    avance está limitado por el espacio al líder del carril y, en rojo, por la
    línea de alto.
  * Semáforo: ciclo verde → amarillo → rojo. En amarillo la línea de alto sigue
    abierta, pero quien está a menos de `yellow_approach` m de ella avanza a
    `yellow_speed_factor` de su velocidad; en rojo, se detiene.
  * Condición inicial (opcional): los carriles empiezan con vehículos en marcha
    que ocupan una fracción del tramo, para no esperar a que se llene.
  * Compresión: detrás de un líder detenido basta `gap_stop` (< `gap_run`).
  * Estiramiento: un vehículo detenido que queda libre (se puso verde o arrancó
    su líder) espera un tiempo de reacción U[1,5] s antes de avanzar, y además
    necesita reabrir `gap_run` detrás del líder en marcha.
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

from trafico.config import DT, RED, YELLOW, SimConfig
from trafico.distributions import (
    normal_ticks, reaction_ticks, sample_lengths, sample_passengers, stop_ticks, uniform_ticks,
)  # fmt: skip
from trafico.metrics import Recorder

EPS = 1e-6
INF = np.inf
SIDE_EPS = 0.01  # m que queda atrás el frente de quien se detiene al lado (conserva el orden)
ARRIVAL_CHUNK = 1024  # pasos por bloque de sorteo de llegadas
PER_TYPE_STREAMS = 5  # generadores aleatorios por tipo de vehículo
MAX_PAX = 255  # cota de pasajeros por vehículo (pax es uint8)
INITIAL_MISSES = 30  # sorteos seguidos que no caben antes de dar por lleno un carril en la condición inicial


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
        "stop_pos", "stop_kind",
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
        self.rng_entry = per_type[3::PER_TYPE_STREAMS]  # carril de entrada (desempates, slow_lane = "random")
        self.rng_stop = per_type[4::PER_TYPE_STREAMS]  # duración de la parada
        # Hijos del generador de pasajeros (spawn no consume sus sorteos): si lleva mercancía y su
        # largo. Agregarlos no altera los sorteos anteriores ni dependen del número de tipos.
        self.rng_cargo, self.rng_length = zip(*(g.spawn(2) for g in self.rng_pax))
        # Cuellos de botella: si se detiene y dónde (al llegar) y cuánto tiempo (al detenerse).
        self.rng_bottleneck, self.rng_bottleneck_time = zip(*(g.spawn(2) for g in self.rng_pax))
        specs = cfg.specs
        b = cfg.behavior
        self.L = float(cfg.length)
        self.n_lanes = cfg.lanes
        self.n_types = cfg.n_types

        # Tablas por tipo de vehículo (indexadas por vtype).
        self.v_step = np.array([s.speed * DT for s in specs])
        self.length_t = np.array([s.length for s in specs])  # largo medio; el de cada vehículo está en vlen
        self.gap_run_t = np.array([s.gap_run for s in specs])
        self.gap_stop_t = np.array([s.gap_stop for s in specs])
        self.can_change = np.array([s.can_change_lane for s in specs])
        self.stop_x = np.array([np.inf if s.stop_position is None else s.stop_position for s in specs])
        self.has_stop = np.isfinite(self.stop_x)
        # Probabilidad y duración (media, σ) de la detención de cada tipo: la propia o la de [bottleneck].
        self.bn_prob = np.array([cfg.bottleneck_prob(k) for k in range(self.n_types)])
        self.bn_time = [cfg.bottleneck_time(k) for k in range(self.n_types)]
        self.bn_type = self.bn_prob > 0  # tipos que pueden detenerse en un cuello de botella
        self.bn_lane = np.zeros(cfg.lanes, np.bool_)
        self.bn_lane[list(cfg.bottleneck_lanes())] = True
        self.bn_zone = cfg.bottleneck_zone()
        self.any_stop = bool(self.has_stop.any() or self.bn_type.any())
        self.abreast = np.array([s.abreast for s in specs])
        self.any_abreast = bool((self.abreast > 1).any())
        self.allowed = np.zeros((self.n_types, cfg.lanes), np.bool_)  # carriles que puede usar cada tipo
        for k in range(self.n_types):
            self.allowed[k, list(cfg.allowed_lanes(k))] = True
        self._entry_lanes = [np.array(cfg.entry_lanes(k)) for k in range(self.n_types)]
        self.rate_tick = np.array(cfg.rates, dtype=np.float64) / 60.0 * DT

        self.lane_cf = np.array(cfg.lane_congestion)  # factor de congestión de cada carril
        self.lane_vmax = np.array(cfg.lane_max_kmh) / 3.6 * DT  # límite de cada carril, m/paso
        self.lc_every = max(1, round(b.lane_change_interval / DT))
        # Umbrales de cambio de carril por tipo: los del tipo o, si no los fija, los de [behavior].
        def own(field: str) -> np.ndarray:
            return np.array([getattr(b, field) if getattr(s, field) is None else getattr(s, field) for s in specs])

        self.lookahead_t = own("lookahead")
        self.min_adv_t = own("min_advantage")
        self.cooldown_ticks_t = np.round(own("lane_change_cooldown") / DT).astype(np.int16)
        self.overtake = np.array([s.overtake for s in specs])
        # Tipo y carril desde los que un carril adyacente permitido es más rápido (subida).
        eff = np.minimum(self.v_step[:, None], self.lane_vmax[None, :])
        self.can_rise = np.zeros_like(self.allowed)
        self.can_rise[:, :-1] |= self.allowed[:, 1:] & (eff[:, 1:] > eff[:, :-1] + EPS)
        self.can_rise[:, 1:] |= self.allowed[:, :-1] & (eff[:, :-1] > eff[:, 1:] + EPS)
        self.can_rise &= self.can_change[:, None]
        self._key_stride = self.L + 1e4  # separa los carriles en la clave de orden

        min_slot = min(s.shortest + s.gap_stop for s in specs)
        per_lane = math.ceil((self.L + max(s.longest for s in specs)) / min_slot) + 2
        self._alloc(cfg.lanes * per_lane)
        self.n = 0
        # (tipo, pax, largo, posición de su detención de [bottleneck], paso de llegada a la cola)
        self.queues: list[deque[tuple[int, int, float, float, int]]] = [deque() for _ in range(cfg.lanes)]
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
        self.lane_changes = 0
        self.lane_changes_t = np.zeros(self.n_types)  # cambios de carril por tipo
        self.stops = np.zeros(self.n_types)  # paradas completadas por tipo
        self.stop_time_ticks = np.zeros(self.n_types)  # pasos detenidos en la parada, por tipo
        self.queue_wait_ticks = np.zeros(self.n_types)  # pasos en la cola de entrada de los que ya entraron
        self.bn_stops = np.zeros(self.n_types)  # detenciones de [bottleneck] por tipo
        self.bn_ticks = np.zeros(self.n_types)  # pasos detenidos en ellas, por tipo
        self.pax_hist = np.zeros((self.n_types, MAX_PAX + 1), np.int64)  # vehículos llegados por nº de pasajeros
        self.recorder = Recorder(cfg.n_samples, cfg.n_types, cfg.lanes)
        self.initial_veh = np.zeros(self.n_types)  # vehículos de la condición inicial
        self.initial_pax = np.zeros(self.n_types)
        self.timed_veh = np.zeros(self.n_types)  # cruces con tiempo de recorrido (sin los iniciales)
        if any(v > 0 for v in cfg.lane_initial_occupancy):
            # Generador propio (hijo nuevo: no altera los anteriores): con occupancy = 0 todo sigue igual.
            self._initial_fill(rng.spawn(1)[0])

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
        bottleneck_at: float = INF,
    ) -> None:  # fmt: skip
        """Agrega un vehículo al inicio del tramo; pax = 0 si lleva mercancía. Sin `length`, mide
        el largo medio de su tipo. Con `x` (frente), es un vehículo de la condición inicial: ya está
        en el tramo, no cuenta como llegada y no tiene tiempo de recorrido. `bottleneck_at` es la
        posición de su detención de [bottleneck] (inf = no se detiene)."""
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
        self.v_last[i] = self.vcap_last[i] = min(self.v_step[vt], self.lane_vmax[lane])  # entra en marcha
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
                chosen.append((k, pax, length, at))
                used += length + self.gap_stop_t[k]
            if not chosen:
                continue
            # Espacio sobre el mínimo (gap_stop detrás de cada líder; el primero no tiene líder).
            kinds = np.array([k for k, *_ in chosen])
            pool = self.L - sum(length for _, _, length, _ in chosen) - self.gap_stop_t[kinds[1:]].sum()
            need = np.r_[0.0, self.gap_run_t[kinds[1:]] - self.gap_stop_t[kinds[1:]]]  # hasta el gap_run
            if pool >= need.sum():
                spare = np.diff(np.sort(rng.uniform(0.0, pool - need.sum(), len(chosen))), prepend=0.0)
                extra = need + spare
            else:  # no alcanza para ir en marcha: se comprimen por igual
                extra = need * pool / need.sum()
            # Se coloca del semáforo hacia atrás; quien queda a menos de su gap_run empieza detenido.
            rear = self.L
            for idx, ((k, pax, length, at), free) in enumerate(zip(chosen, extra)):
                gap = (self.gap_stop_t[k] if idx else 0.0) + free
                self._add(k, pax, lane, length, x=rear - gap, bottleneck_at=at)
                if idx and gap < self.gap_run_t[k] - EPS:
                    i = self.n - 1
                    self.stopped[i] = True
                    self.v_last[i] = 0.0
                rear = self.x[self.n - 1] - length

    def occupancy(self) -> Occupancy:
        n = self.n
        changing = np.flatnonzero(self.lc_target[:n] >= 0)
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
        if self.n:
            self._move(self.cfg.phase(self.tick))
        self.tick += 1
        if self.n_lanes > 1 and self.n and self.tick % self.lc_every == 0:
            self._lane_changes()
        if self.tick % self.cfg.sample_ticks == 0:
            self._sample()

    def _arrivals(self) -> None:
        offset = self.tick % ARRIVAL_CHUNK
        if offset == 0:
            # Sorteo de llegadas Poisson por bloques; solo se guardan los pasos con llegadas.
            chunk = np.column_stack(
                [g.poisson(rate, ARRIVAL_CHUNK) for g, rate in zip(self.rng_arrivals, self.rate_tick)]
            )
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
            at = np.full(k, INF)  # dónde se detendrá (cuello de botella); inf = no se detiene
            if self.bn_type[vt]:
                g = self.rng_bottleneck[vt]
                stops = g.random(k) < self.bn_prob[vt]
                at = np.where(stops, np.maximum(g.uniform(*self.bn_zone, k), lengths + 0.01), INF)
            self.arrived_veh[vt] += k
            self.arrived_cargo[vt] += int(np.count_nonzero(paxs == 0))
            self.arrived_pax[vt] += int(paxs.sum())
            self.pax_hist[vt] += np.bincount(paxs[paxs > 0], minlength=MAX_PAX + 1)
            for p, length, a in zip(paxs, lengths, at):
                self.queues[self._entry_lane(vt)].append((int(vt), int(p), float(length), float(a), self.tick))

    def _entry_lane(self, vt: int) -> int:
        if self.n_lanes == 1:
            return 0
        lanes = self._entry_lanes[vt]  # sin los carriles exclusivos de otros tipos
        if lanes.size == 1:
            return int(lanes[0])
        if not self.can_change[vt]:
            return int(self.rng_entry[vt].choice(lanes))  # slow_lane = "random"
        # Los que cambian de carril: cola de entrada más corta; desempate por más espacio libre a la entrada.
        qlen = np.fromiter((len(self.queues[ln]) for ln in lanes), np.int64, lanes.size)
        best = lanes[qlen == qlen.min()]
        if best.size > 1:
            tail = self._lane_tails()[0][best]
            best = best[tail == tail.max()]
        return int(best[0] if best.size == 1 else self.rng_entry[vt].choice(best))

    def _lane_tails(self) -> tuple[np.ndarray, np.ndarray]:
        """Parte trasera del último vehículo de cada carril y si está detenido."""
        rear_min = np.full(self.n_lanes, INF)
        tail_stopped = np.zeros(self.n_lanes, np.bool_)
        n = self.n
        if n == 0:
            return rear_min, tail_stopped
        rear = self.x[:n] - self.vlen[:n]
        lane, tgt = self.lane[:n], self.lc_target[:n]
        for ln in range(self.n_lanes):
            occ = np.flatnonzero((lane == ln) | (tgt == ln))
            if occ.size:
                j = occ[np.argmin(rear[occ])]
                rear_min[ln] = rear[j]
                tail_stopped[ln] = self.stopped[j]
        return rear_min, tail_stopped

    def _spawn(self) -> None:
        if not any(self.queues):
            return
        rear_min, tail_stopped = self._lane_tails()
        for ln, q in enumerate(self.queues):
            if not q:
                continue
            vt, pax, length, at, arrived = q[0]
            gap = self.gap_stop_t[vt] if tail_stopped[ln] else self.gap_run_t[vt]
            if rear_min[ln] >= length + gap:
                q.popleft()
                self.queue_wait_ticks[vt] += self.tick - arrived
                self._add(vt, pax, ln, length, bottleneck_at=at)

    def _move(self, phase: int) -> None:
        n = self.n
        b = self.cfg.behavior
        vt = self.vtype[:n]
        x = self.x[:n]
        occ = self.occupancy()

        # Espacio disponible de cada entrada respecto a su líder.
        lead = occ.leader
        lead_stop = occ.has_leader & self.stopped[lead]
        svt = self.vtype[occ.veh]
        gap_need = np.where(lead_stop, self.gap_stop_t[svt], self.gap_run_t[svt])
        s = self.x[lead] - self.vlen[lead] - gap_need - occ.x
        if self.any_abreast:
            # Lugar libre al lado del líder detenido: se detiene con el frente alineado al suyo.
            s = np.where(self._side_slot(occ), self.x[lead] - SIDE_EPS - occ.x, s)
        stop_e = np.empty(occ.order.size)
        move_e = np.empty(occ.order.size)
        stop_e[occ.order] = np.where(lead_stop, s, INF)
        move_e[occ.order] = np.where(occ.has_leader & ~lead_stop, s, INF)
        congestion = self._congestion(occ) if self.lane_cf.any() else None

        # Por vehículo: mínimo entre su carril y, si está cambiando, el carril destino.
        stop_space = stop_e[:n]
        move_space = move_e[:n]
        ch = occ.changing
        if ch.size:
            stop_space[ch] = np.minimum(stop_space[ch], stop_e[n:])
            move_space[ch] = np.minimum(move_space[ch], move_e[n:])
        crossed = self.crossed[:n]
        if phase == RED:
            stop_space = np.minimum(stop_space, np.where(crossed, INF, self.L - x))
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

        # Avance: velocidad máxima dentro del límite del carril, reducida al cambiar de carril, por congestión
        # y en amarillo cerca de la línea de alto.
        moving = ~stopped
        lane_changing = self.lc_target[:n] >= 0
        # Velocidad máxima del vehículo sin rebasar el límite del carril (el menor de sus dos
        # carriles si está cambiando), reducida durante la maniobra y por congestión.
        vcap = np.minimum(self.v_step[vt], self.lane_vmax[self.lane[:n]])
        if lane_changing.any():
            tgt_lane = self.lc_target[:n][lane_changing]
            vcap[lane_changing] = np.minimum(vcap[lane_changing], self.lane_vmax[tgt_lane])
        vcap = vcap * np.where(lane_changing, b.lane_change_speed_factor, 1.0)
        if congestion is not None:
            vcap = vcap * congestion
        if phase == YELLOW:  # quien se aproxima a la línea de alto baja la velocidad
            near = ~crossed & (self.L - x <= b.yellow_approach)
            vcap = np.where(near, vcap * b.yellow_speed_factor, vcap)
        adv = np.where(moving, np.maximum(np.minimum(vcap, space), 0.0), 0.0)
        stopped[moving & (adv < EPS) & binding_stop] = True
        self.v_last[:n] = adv
        self.vcap_last[:n] = vcap

        pax = self.pax[:n]
        inside = np.minimum(adv, np.maximum(self.L - x, 0.0))
        self.pax_m += np.bincount(vt, weights=pax * inside, minlength=self.n_types)
        # Velocidad media por carril: distancia y tiempo de los vehículos dentro del tramo (con los
        # detenidos), asignados a su carril actual.
        present = ~crossed
        lane_now = self.lane[:n][present]
        self.lane_dist += np.bincount(lane_now, weights=inside[present], minlength=self.n_lanes)
        self.lane_time += np.bincount(lane_now, minlength=self.n_lanes) * DT
        x += adv

        # Llegada a la parada: queda detenido el tiempo de descenso y ascenso.
        if self.any_stop:
            arrive = (stop_state == self.STOP_AHEAD) & (x >= stop_pos - EPS)
            for i in np.flatnonzero(arrive):
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

        # Sale del sistema cuando su parte trasera rebasa la línea.
        gone = x - self.vlen[:n] > self.L
        if gone.any():
            self._compact(~gone)

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
        last = np.flatnonzero(np.append(new_row[1:], True))  # último índice de cada fila
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
        first = np.flatnonzero(np.r_[True, rid[1:] != rid[:-1]])[rid]
        primary = occ.order < n  # quien cambia de carril aparece dos veces; se usa su carril actual
        rank[occ.veh[primary]] = (row_last - idx)[primary]
        size[occ.veh[primary]] = (row_last - first + 1)[primary]
        return rank, size

    def _side_slot(self, occ: Occupancy) -> np.ndarray:
        """Entradas que pueden detenerse al lado de su líder: su tipo admite `abreast` > 1, el líder
        está detenido, es de su mismo tipo (como toda su fila) y la fila tiene lugar. Solo quien
        llega en marcha toma el lugar (o quien ya lo ocupa): un vehículo ya detenido en fila, o que
        tiene a otro a su lado, no se adelanta a otra fila. No aplica a quien cambia de carril ni a
        quien ya cruzó."""
        m = occ.order.size
        if m < 2:
            return np.zeros(m, np.bool_)
        n = self.n
        svt = self.vtype[occ.veh]
        lead = occ.leader
        beside, rid, row_last = self.rows(occ)
        nxt = np.minimum(np.arange(m) + 1, m - 1)  # entrada del líder (la siguiente en el orden)
        ahead = row_last[nxt] - np.arange(m)  # vehículos de la fila del líder que van delante
        starts = np.flatnonzero(np.r_[True, rid[1:] != rid[:-1]])
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

    def _congestion(self, occ: Occupancy) -> np.ndarray:
        """Factor de velocidad por vehículo según la saturación de su carril.

        La saturación de un carril es el largo de su cola de vehículos detenidos
        (largo + gap detenido, dentro del tramo) entre la longitud del tramo. En un
        carril con saturación s la velocidad se multiplica por 1 − factor del carril × s;
        sin detenidos no hay reducción. Quien cambia de carril toma el menor factor
        de sus dos carriles, y quien ya cruzó el semáforo no se ve afectado.
        """
        n = self.n
        lane_factor = 1.0 - self.lane_cf * self._lane_saturation(occ)
        factor = np.empty(occ.order.size)
        factor[occ.order] = lane_factor[occ.lane]
        per_vehicle = factor[:n].copy()
        if occ.changing.size:
            per_vehicle[occ.changing] = np.minimum(per_vehicle[occ.changing], factor[n:])
        return np.where(self.crossed[:n], 1.0, per_vehicle)

    def _lane_changes(self) -> None:
        """Decisiones de cambio de carril de quienes circulan (no detenidos, sin maniobra ni
        enfriamiento), en orden aleatorio y con a lo más una entrada a cada carril por ronda.

        Condición para cualquier cambio: un líder dentro de su `lookahead` cuya velocidad real en el
        último paso esté al menos `leader_slowdown` (p. ej. 25 %) por debajo del límite del carril
        (en un carril sin límite, de la velocidad máxima del propio vehículo). Sin líder, no cambia.

        * Rebase: con un líder lento dentro de su `lookahead`, busca un carril adyacente donde
          avance más rápido y tenga `min_advantage` m más de espacio libre. Los tipos con
          `overtake` juzgan la lentitud por la velocidad real del último paso (líder en maniobra,
          frenado por su cola o por congestión) y aceptan un carril de menor límite si ahí avanzan
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
        lvt = self.vtype[occ.leader]
        gap = self.x[occ.leader] - self.vlen[occ.leader] - occ.x
        # Velocidades efectivas en el carril de cada entrada (máxima del tipo, sin rebasar el límite).
        lane_v = self.lane_vmax[occ.lane]
        eff_self = np.minimum(self.v_step[svt], lane_v)
        eff_lead = np.minimum(self.v_step[lvt], lane_v)
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
        lane_max = np.where(np.isfinite(lane_v), lane_v, self.v_step[svt])
        v_ahead = np.where(lead_stopped, 0.0, self.v_last[occ.leader])
        blocked = (
            cand[occ.veh]
            & occ.has_leader
            & (gap < self.lookahead_t[svt])
            & (v_ahead < (1.0 - b.leader_slowdown) * lane_max - EPS)
        )
        passing = blocked & slow
        rising = blocked & self.can_rise[svt, occ.lane]
        idx = np.flatnonzero(passing | rising)
        if idx.size == 0:
            return

        # Factor de congestión de cada carril, para comparar con la velocidad real del líder.
        lane_factor = 1.0 - self.lane_cf * self._lane_saturation(occ) if over.any() and self.lane_cf.any() else None
        bounds = np.searchsorted(occ.lane, np.arange(self.n_lanes + 1))
        used: set[int] = set()
        for k in self.rng_lane_change.permutation(idx):
            best = self._pass_lane(occ, bounds, used, k, gap[k], stuck_speed[k], lane_factor) if passing[k] else -1
            if best < 0 and rising[k]:
                best = self._rise_lane(occ, bounds, used, k)
            if best >= 0:
                i = int(occ.veh[k])
                used.add(best)
                self.lc_target[i] = best
                self.lc_timer[i] = uniform_ticks(self.rng_lane_change, b.lane_change_min, b.lane_change_max)
                self.lane_changes += 1
                self.lane_changes_t[self.vtype[i]] += 1

    def _neighbors(self, occ: Occupancy, bounds: np.ndarray, lane: int, x: float) -> tuple[int, int]:
        """(líder, seguidor) que tendría en `lane` un vehículo con el frente en x; -1 si no hay."""
        lo, hi = bounds[lane], bounds[lane + 1]
        p = lo + int(np.searchsorted(occ.x[lo:hi], x))
        return (int(occ.veh[p]) if p < hi else -1), (int(occ.veh[p - 1]) if p > lo else -1)

    def _pass_lane(
        self, occ: Occupancy, bounds: np.ndarray, used: set[int], k: int, gap: float, stuck_speed: float,
        lane_factor: np.ndarray | None,
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
            v_there = min(self.v_step[vi], self.lane_vmax[tl])
            if over and lane_factor is not None:
                v_there *= lane_factor[tl]
            if v_there <= stuck_speed:  # ese carril no mejora la velocidad actual
                continue
            j, f = self._neighbors(occ, bounds, tl, xi)
            score = look
            if j >= 0:  # líder en el carril destino
                front = self.x[j] - self.vlen[j] - xi
                if front < self.gap_stop_t[vi]:
                    continue
                v_j = self.v_last[j] + EPS if over else min(self.v_step[self.vtype[j]], self.lane_vmax[tl])
                if v_j < v_there or self.stopped[j]:
                    score = min(front, look)
            if f >= 0:  # seguidor en el carril destino: se comprime, pero sin traslape
                back = xi - self.vlen[i] - self.x[f]
                if back < self.gap_stop_t[self.vtype[f]]:
                    continue
            if score >= best_score:
                best, best_score = tl, score
        return best

    def _rise_lane(self, occ: Occupancy, bounds: np.ndarray, used: set[int], k: int) -> int:
        """Carril adyacente de mayor límite al que la entrada k puede subir; -1 si no hay lugar."""
        i, xi, li = int(occ.veh[k]), occ.x[k], int(occ.lane[k])
        vi = self.vtype[i]
        best, best_v = -1, min(self.v_step[vi], self.lane_vmax[li]) + EPS
        for tl in (li + 1, li - 1):
            if tl < 0 or tl >= self.n_lanes or tl in used or not self.allowed[vi, tl]:
                continue
            v_there = min(self.v_step[vi], self.lane_vmax[tl])
            if v_there <= best_v:
                continue
            j, f = self._neighbors(occ, bounds, tl, xi)
            if j >= 0:
                front = self.x[j] - self.vlen[j] - xi
                if front < self.gap_run_t[vi]:
                    continue
                # Un líder cercano detenido o no más rápido que él le quita la ventaja.
                if front < self.lookahead_t[vi] and (self.stopped[j] or self.v_last[j] <= self.v_last[i] + EPS):
                    continue
            if f >= 0 and xi - self.vlen[i] - self.x[f] < self.gap_run_t[self.vtype[f]]:
                continue
            best, best_v = tl, v_there
        return best

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
        self.recorder.record(self.cum_pax, self.pax_m, pax_on, footprint, lane_sat, self.lane_dist, self.lane_time)
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
            "queue_wait": queue_wait,
            "pax_per_veh": pax_per_veh,
            "queued": queued,
            "on_road": on_road,
            "lane_changes": np.array([float(self.lane_changes)]),
            "lane_changes_type": self.lane_changes_t.copy(),
            "stop_time": self.stop_time_ticks * DT / np.where(self.stops > 0, self.stops, np.nan),
            "bottleneck_stops": self.bn_stops.copy(),
            "bottleneck_time": self.bn_ticks * DT / np.where(self.bn_stops > 0, self.bn_stops, np.nan),
        }
