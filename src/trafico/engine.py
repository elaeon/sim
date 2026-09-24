"""Motor microscópico: tramo de `lanes` carriles con un semáforo en x = length.

Estado en estructura de arreglos (SoA) compacta: los vehículos activos ocupan
las posiciones [0, n) de cada arreglo; los que salen se eliminan compactando.
La capacidad inicial es la cota física de vehículos que caben en el tramo, así
que la memoria no crece con la duración de la corrida.

Modelo por paso de DT segundos (actualización paralela con posiciones previas):
  * Cada tipo avanza a su velocidad máxima sin rebasar el límite del carril; el
    avance está limitado por el espacio al líder del carril y, en rojo, por la
    línea de alto.
  * Compresión: detrás de un líder detenido basta `gap_stop` (< `gap_run`).
  * Estiramiento: un vehículo detenido que queda libre (se puso verde o arrancó
    su líder) espera un tiempo de reacción U[1,5] s antes de avanzar, y además
    necesita reabrir `gap_run` detrás del líder en marcha.
  * Cambio de carril (solo tipos que pueden): la maniobra dura U[1,4] s, el
    vehículo ocupa ambos carriles a velocidad reducida y los seguidores de los
    dos carriles se comprimen detrás de él.
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

from trafico.config import DT, SimConfig
from trafico.distributions import sample_passengers, stop_ticks, uniform_ticks
from trafico.metrics import Recorder

EPS = 1e-6
INF = np.inf
SIDE_EPS = 0.01  # m que queda atrás el frente de quien se detiene al lado (conserva el orden)
ARRIVAL_CHUNK = 1024  # pasos por bloque de sorteo de llegadas
PER_TYPE_STREAMS = 5  # generadores aleatorios por tipo de vehículo
MAX_PAX = 255  # cota de pasajeros por vehículo (pax es uint8)


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
        "react", "x", "stopped", "crossed", "entry_tick", "vid", "stop_state", "dwell",
    )  # fmt: skip
    # Estados de parada: sin parada pendiente, antes de la parada, detenido en ella.
    NO_STOP, STOP_AHEAD, AT_STOP = 0, 1, 2

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
        specs = cfg.specs
        b = cfg.behavior
        self.L = float(cfg.length)
        self.n_lanes = cfg.lanes
        self.n_types = cfg.n_types

        # Tablas por tipo de vehículo (indexadas por vtype).
        self.v_step = np.array([s.speed * DT for s in specs])
        self.length_t = np.array([s.length for s in specs])
        self.gap_run_t = np.array([s.gap_run for s in specs])
        self.gap_stop_t = np.array([s.gap_stop for s in specs])
        self.can_change = np.array([s.can_change_lane for s in specs])
        self.stop_x = np.array([np.inf if s.stop_position is None else s.stop_position for s in specs])
        self.has_stop = np.isfinite(self.stop_x)
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
        self.cooldown_ticks = round(b.lane_change_cooldown / DT)
        self._key_stride = self.L + 1e4  # separa los carriles en la clave de orden

        min_slot = float(np.min(self.length_t + self.gap_stop_t))
        per_lane = math.ceil((self.L + self.length_t.max()) / min_slot) + 2
        self._alloc(cfg.lanes * per_lane)
        self.n = 0
        self.queues: list[deque[tuple[int, int]]] = [deque() for _ in range(cfg.lanes)]
        self.tick = 0
        self._pending_arrivals: dict[int, np.ndarray] = {}

        self.arrived_veh = np.zeros(self.n_types)
        self.arrived_pax = np.zeros(self.n_types)
        self.entered_veh = np.zeros(self.n_types)
        self.cum_veh = np.zeros(self.n_types)
        self.cum_pax = np.zeros(self.n_types)
        self.travel_ticks = np.zeros(self.n_types)
        self.pax_m = np.zeros(self.n_types)  # pasajeros·m dentro del tramo en el intervalo de muestreo
        self.lane_dist = np.zeros(cfg.lanes)  # m recorridos en cada carril en el intervalo de muestreo
        self.lane_time = np.zeros(cfg.lanes)  # vehículo·s presentes en cada carril en el intervalo
        self.lane_changes = 0
        self.stops = np.zeros(self.n_types)  # paradas completadas por tipo
        self.stop_time_ticks = np.zeros(self.n_types)  # pasos detenidos en la parada, por tipo
        self.pax_hist = np.zeros((self.n_types, MAX_PAX + 1), np.int64)  # vehículos llegados por nº de pasajeros
        self.recorder = Recorder(cfg.n_samples, cfg.n_types, cfg.lanes)

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

    def _add(self, vt: int, pax: int, lane: int) -> None:
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
        self.x[i] = self.length_t[vt]  # entra con la parte trasera en x = 0
        self.stopped[i] = False
        self.crossed[i] = False
        self.entry_tick[i] = self.tick
        self.vid[i] = self.entered_veh.sum()
        self.stop_state[i] = self.STOP_AHEAD if self.has_stop[vt] else self.NO_STOP
        self.dwell[i] = 0
        self.n += 1
        self.entered_veh[vt] += 1
        self.pax_m[vt] += pax * self.length_t[vt]

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
            self._move(self.cfg.is_green(self.tick))
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
            paxs = sample_passengers(self.rng_pax[vt], self.cfg.specs[vt], k)
            self.arrived_veh[vt] += k
            self.arrived_pax[vt] += int(paxs.sum())
            self.pax_hist[vt] += np.bincount(paxs, minlength=MAX_PAX + 1)
            for p in paxs:
                self.queues[self._entry_lane(vt)].append((int(vt), int(p)))

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
        rear = self.x[:n] - self.length_t[self.vtype[:n]]
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
            vt, pax = q[0]
            gap = self.gap_stop_t[vt] if tail_stopped[ln] else self.gap_run_t[vt]
            if rear_min[ln] >= self.length_t[vt] + gap:
                q.popleft()
                self._add(vt, pax, ln)

    def _move(self, green: bool) -> None:
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
        s = self.x[lead] - self.length_t[self.vtype[lead]] - gap_need - occ.x
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
        if not green:
            stop_space = np.minimum(stop_space, np.where(crossed, INF, self.L - x))
        stop_state = self.stop_state[:n]
        if self.has_stop.any():
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
            pending = stop_state != self.NO_STOP
            stop_space = np.minimum(stop_space, np.where(pending, self.stop_x[vt] - x, INF))
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
                react[sel] = uniform_ticks(self.rng_react[t], b.reaction_min, b.reaction_max, int(sel.sum()))

        # Avance: velocidad máxima dentro del límite del carril, reducida al cambiar de carril y por congestión.
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
        adv = np.where(moving, np.maximum(np.minimum(vcap, space), 0.0), 0.0)
        stopped[moving & (adv < EPS) & binding_stop] = True

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
        if self.has_stop.any():
            arrive = (stop_state == self.STOP_AHEAD) & (x >= self.stop_x[vt] - EPS)
            for i in np.flatnonzero(arrive):
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
            cd[fin] = self.cooldown_ticks

        # Cruce de la línea de alto.
        newly = ~crossed & (x > self.L)
        if newly.any():
            t = vt[newly]
            self.cum_veh += np.bincount(t, minlength=self.n_types)
            self.cum_pax += np.bincount(t, weights=pax[newly], minlength=self.n_types)
            travel = self.tick + 1 - self.entry_tick[:n][newly]
            self.travel_ticks += np.bincount(t, weights=travel, minlength=self.n_types)
            crossed[newly] = True

        # Sale del sistema cuando su parte trasera rebasa la línea.
        gone = x - self.length_t[vt] > self.L
        if gone.any():
            self._compact(~gone)

    def rows(self, occ: Occupancy) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Filas lado a lado en el orden de `occ`: (junto, id de fila, índice del último de su fila).

        `junto` marca a quien se traslapa con su líder, es decir, está a su lado en la misma fila.
        Una fila es un tramo consecutivo de entradas unidas así; sin `abreast`, cada vehículo es
        su propia fila."""
        m = occ.order.size
        lvt = self.vtype[occ.leader]
        beside = occ.has_leader & (self.x[occ.leader] - self.length_t[lvt] - occ.x < 0)
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
            occ.lane[waiting], weights=self.length_t[vt] + self.gap_stop_t[vt], minlength=self.n_lanes
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
        n = self.n
        b = self.cfg.behavior
        cand = (
            self.can_change[self.vtype[:n]]
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
        gap = self.x[occ.leader] - self.length_t[lvt] - occ.x
        # Velocidades efectivas en el carril de cada entrada (máxima del tipo, sin rebasar el límite).
        lane_v = self.lane_vmax[occ.lane]
        eff_self = np.minimum(self.v_step[svt], lane_v)
        eff_lead = np.minimum(self.v_step[lvt], lane_v)
        slow = occ.has_leader & ((eff_lead < eff_self) | self.stopped[occ.leader])
        stuck_speed = np.where(self.stopped[occ.leader], 0.0, eff_lead)  # a la que lo obliga su líder
        idx = np.flatnonzero(slow & (gap < b.lookahead) & cand[occ.veh])
        if idx.size == 0:
            return

        bounds = np.searchsorted(occ.lane, np.arange(self.n_lanes + 1))
        used: set[int] = set()
        for k in self.rng_lane_change.permutation(idx):
            i = int(occ.veh[k])
            xi = occ.x[k]
            li = int(occ.lane[k])
            vi = self.vtype[i]
            best, best_score = -1, min(gap[k], b.lookahead) + b.min_advantage
            for tl in (li + 1, li - 1):  # prefiere rebasar por la izquierda
                if tl < 0 or tl >= self.n_lanes or tl in used or not self.allowed[vi, tl]:
                    continue
                v_there = min(self.v_step[vi], self.lane_vmax[tl])
                if v_there <= stuck_speed[k]:  # el límite de ese carril no mejora la velocidad actual
                    continue
                lo, hi = bounds[tl], bounds[tl + 1]
                p = lo + int(np.searchsorted(occ.x[lo:hi], xi))
                score = b.lookahead
                if p < hi:  # líder en el carril destino
                    j = occ.veh[p]
                    front = self.x[j] - self.length_t[self.vtype[j]] - xi
                    if front < self.gap_stop_t[vi]:
                        continue
                    if min(self.v_step[self.vtype[j]], self.lane_vmax[tl]) < v_there or self.stopped[j]:
                        score = min(front, b.lookahead)
                if p > lo:  # seguidor en el carril destino: se comprime, pero sin traslape
                    f = occ.veh[p - 1]
                    back = xi - self.length_t[vi] - self.x[f]
                    if back < self.gap_stop_t[self.vtype[f]]:
                        continue
                if score >= best_score:
                    best, best_score = tl, score
            if best >= 0:
                used.add(best)
                self.lc_target[i] = best
                self.lc_timer[i] = uniform_ticks(self.rng_lane_change, b.lane_change_min, b.lane_change_max)
                self.lane_changes += 1

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
                self.x[occ.leader] - self.length_t[self.vtype[occ.leader]] - occ.x,
                INF,
            )
            # Detenido ocupa su largo + gap comprimido; en marcha, a lo más largo + gap en marcha.
            cap = np.where(self.stopped[occ.veh], self.gap_stop_t[svt], self.gap_run_t[svt])
            fp = self.length_t[svt] + np.minimum(gap, cap)
            ins = ~self.crossed[occ.veh]
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
            np.fromiter((vt for q in self.queues for vt, _ in q), np.int64),
            minlength=self.n_types,
        ).astype(np.float64)
        on_road = np.bincount(
            self.vtype[:n][~self.crossed[:n]], minlength=self.n_types
        ).astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            travel_time = self.travel_ticks * DT / self.cum_veh
            pax_per_veh = self.arrived_pax / self.arrived_veh
        return {
            "arrived_veh": self.arrived_veh.copy(),
            "arrived_pax": self.arrived_pax.copy(),
            "entered_veh": self.entered_veh.copy(),
            "crossed_veh": self.cum_veh.copy(),
            "crossed_pax": self.cum_pax.copy(),
            "travel_time": travel_time,
            "pax_per_veh": pax_per_veh,
            "queued": queued,
            "on_road": on_road,
            "lane_changes": np.array([float(self.lane_changes)]),
            "stop_time": self.stop_time_ticks * DT / np.where(self.stops > 0, self.stops, np.nan),
        }
