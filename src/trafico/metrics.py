"""Registro de series por réplica y agregación estadística en línea."""

from __future__ import annotations

import numpy as np

from trafico.config import DT, SimConfig

SERIES = ("cum_pax", "pax_flow", "paxkm_h", "pax_per_m")
LANE_SATURATION = "lane_saturation"
"""Serie con una columna por carril: fracción del tramo ocupada por la cola de detenidos."""
LANE_SPEED = "lane_speed"
"""Serie con una columna por carril: velocidad media (km/h) de sus vehículos, con los detenidos."""
LANE_SERIES = (LANE_SATURATION, LANE_SPEED)
SMOOTH_S = 10.0  # s de la media móvil de pax·km/h
"""Series derivadas que se agregan y grafican (una columna por tipo de vehículo)."""


class Recorder:
    """Series crudas muestreadas cada `sample` segundos, preasignadas en float32."""

    def __init__(self, n_samples: int, n_types: int, n_lanes: int):
        shape = (n_samples, n_types)
        self.cum_pax = np.zeros(shape, np.float32)  # pasajeros acumulados que cruzaron el semáforo
        self.pax_m = np.zeros(shape, np.float32)  # pasajeros·m recorridos en el intervalo
        self.pax_on = np.zeros(shape, np.float32)  # pasajeros dentro del tramo
        self.footprint = np.zeros(shape, np.float32)  # m de carril ocupados (largo + gap real)
        self.lane_sat = np.zeros((n_samples, n_lanes), np.float32)  # saturación de cada carril
        self.lane_dist = np.zeros((n_samples, n_lanes), np.float32)  # m recorridos en el intervalo
        self.lane_time = np.zeros((n_samples, n_lanes), np.float32)  # vehículo·s en el intervalo
        self.count = 0

    def record(self, cum_pax, pax_m, pax_on, footprint, lane_sat, lane_dist, lane_time) -> None:
        i = self.count
        if i >= self.cum_pax.shape[0]:
            return
        self.cum_pax[i] = cum_pax
        self.pax_m[i] = pax_m
        self.pax_on[i] = pax_on
        self.footprint[i] = footprint
        self.lane_sat[i] = lane_sat
        self.lane_dist[i] = lane_dist
        self.lane_time[i] = lane_time
        self.count += 1


def _window_rate(cum: np.ndarray, w: int, dt: float) -> np.ndarray:
    """Tasa por segundo de una serie acumulada en una ventana de `w` muestras.

    Al inicio la ventana se acota al historial disponible.
    """
    n = cum.shape[0]
    padded = np.vstack((np.zeros((1, cum.shape[1])), cum))
    idx = np.arange(1, n + 1)
    lag = np.maximum(idx - max(1, w), 0)
    return (padded[idx] - padded[lag]) / ((idx - lag) * dt)[:, None]


def derive_series(rec: Recorder, cfg: SimConfig) -> dict[str, np.ndarray]:
    """Convierte las series crudas en las métricas de movilidad de pasajeros."""
    dt = cfg.sample_ticks * DT
    cum = rec.cum_pax.astype(np.float64)

    # Flujo en ventana móvil de un ciclo de semáforo.
    flow = _window_rate(cum, round(cfg.cycle / dt), dt) * 60.0
    # pax·m del intervalo -> pax·km/h, suavizado con media móvil.
    pax_m_cum = np.cumsum(rec.pax_m, axis=0, dtype=np.float64)
    paxkm_h = _window_rate(pax_m_cum, round(SMOOTH_S / dt), dt) * 3600.0 / 1000.0
    # Velocidad media por carril en la misma media móvil: distancia / tiempo de sus vehículos.
    w = round(SMOOTH_S / dt)
    dist = _window_rate(np.cumsum(rec.lane_dist, axis=0, dtype=np.float64), w, dt)
    veh_time = _window_rate(np.cumsum(rec.lane_time, axis=0, dtype=np.float64), w, dt)
    with np.errstate(invalid="ignore", divide="ignore"):
        per_m = np.where(rec.footprint > 0, rec.pax_on / rec.footprint, np.nan)
        lane_speed = np.where(veh_time > 0, dist / veh_time * 3.6, np.nan)

    return {
        "cum_pax": cum.astype(np.float32),
        "pax_flow": flow.astype(np.float32),
        "paxkm_h": paxkm_h.astype(np.float32),
        "pax_per_m": per_m.astype(np.float32),
        LANE_SATURATION: rec.lane_sat.copy(),
        LANE_SPEED: lane_speed.astype(np.float32),
    }


class RunningStats:
    """Media y desviación estándar elemento a elemento (Welford), ignorando NaN.

    Ocupa O(tamaño de una réplica) sin importar cuántas réplicas se agreguen.
    """

    def __init__(self, shape):
        self.count = np.zeros(shape, np.int64)
        self._mean = np.zeros(shape, np.float64)
        self._m2 = np.zeros(shape, np.float64)

    def push(self, value) -> None:
        value = np.asarray(value, dtype=np.float64)
        valid = ~np.isnan(value)
        self.count += valid
        delta = np.where(valid, value - self._mean, 0.0)
        self._mean += np.where(valid, delta / np.maximum(self.count, 1), 0.0)
        self._m2 += np.where(valid, delta * (np.where(valid, value, 0.0) - self._mean), 0.0)

    @property
    def mean(self) -> np.ndarray:
        return np.where(self.count > 0, self._mean, np.nan)

    @property
    def std(self) -> np.ndarray:
        var = np.where(self.count > 1, self._m2 / np.maximum(self.count - 1, 1), 0.0)
        return np.where(self.count > 0, np.sqrt(var), np.nan)
