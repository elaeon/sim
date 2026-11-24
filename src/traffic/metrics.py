"""Recording of per-replica series and online statistical aggregation."""

from __future__ import annotations

import numpy as np

from traffic.config import DT, POLLUTANTS, SimConfig

SERIES = ("cum_pax", "pax_flow", "paxkm_h", "pax_per_m")
LANE_SATURATION = "lane_saturation"
"""Serie con una columna por carril: fracción del tramo ocupada por la cola de detenidos."""
LANE_SPEED = "lane_speed"
"""Serie con una columna por carril: velocidad media (km/h) de sus vehículos, con los detenidos."""
LANE_EXIT_FLOW = "lane_exit_flow"
"""Serie con una columna por carril: vehículos por minuto que cruzan el semáforo (ventana de un ciclo; sin
semáforo, de NO_LIGHT_WINDOW s)."""
EXIT_QUEUE = "exit_queue"
"""Serie con una columna por carril: m ocupados en la cola de salida después del semáforo."""
ENTRY_QUEUE = "entry_queue"
"""Serie con una columna por carril: vehículos en la cola de entrada (ya llegaron, aún no caben en el tramo)."""
LANE_STATS = (LANE_SATURATION, LANE_EXIT_FLOW, LANE_SPEED, ENTRY_QUEUE, EXIT_QUEUE)
EMIS_SERIES = tuple(f"emis_{pol}" for pol in POLLUTANTS)
"""Series con una columna por tipo: g/min de cada contaminante (ventana de un ciclo; sin semáforo, de
NO_LIGHT_WINDOW s)."""
"""Series por carril que se agregan entre réplicas."""
SMOOTH_S = 10.0  # s of the moving average of pax·km/h
SPEED_BIN = 0.5  # km/h, width of the bins of the per-lane speed histogram
"""Series derivadas que se agregan y grafican (una columna por tipo de vehículo)."""


class Recorder:
    """Raw series sampled every `sample` seconds, preallocated as float32."""

    def __init__(self, n_samples: int, n_types: int, n_lanes: int):
        shape = (n_samples, n_types)
        self.cum_pax = np.zeros(shape, np.float32)  # cumulative passengers that crossed the traffic light
        self.pax_m = np.zeros(shape, np.float32)  # passenger·m travelled in the interval
        self.pax_on = np.zeros(shape, np.float32)  # passengers inside the segment
        self.footprint = np.zeros(shape, np.float32)  # m of lane occupied (length + actual gap)
        self.lane_sat = np.zeros((n_samples, n_lanes), np.float32)  # saturation of each lane
        self.lane_dist = np.zeros((n_samples, n_lanes), np.float32)  # m travelled in the interval
        self.lane_time = np.zeros((n_samples, n_lanes), np.float32)  # vehicle·s in the interval
        self.lane_cross = np.zeros((n_samples, n_lanes), np.float32)  # line crossings in the interval
        self.exit_q = np.zeros((n_samples, n_lanes), np.float32)  # m occupied in the exit queue
        self.entry_q = np.zeros((n_samples, n_lanes), np.float32)  # vehicles in the entry queue
        self.emis = np.zeros((n_samples, n_types, len(POLLUTANTS)), np.float32)  # g emitted in the interval
        self.count = 0

    def record(self, cum_pax, pax_m, pax_on, footprint, lane_sat, lane_dist, lane_time, lane_cross, exit_q, entry_q,
               emis=None) -> None:
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
        self.lane_cross[i] = lane_cross
        self.exit_q[i] = exit_q
        self.entry_q[i] = entry_q
        if emis is not None:
            self.emis[i] = emis
        self.count += 1


def _window_rate(cum: np.ndarray, w: int, dt: float) -> np.ndarray:
    """Per-second rate of a cumulative series over a window of `w` samples.

    At the start the window is limited to the available history.
    """
    n = cum.shape[0]
    padded = np.vstack((np.zeros((1, cum.shape[1])), cum))
    idx = np.arange(1, n + 1)
    lag = np.maximum(idx - max(1, w), 0)
    return (padded[idx] - padded[lag]) / ((idx - lag) * dt)[:, None]


def derive_series(rec: Recorder, cfg: SimConfig) -> dict[str, np.ndarray]:
    """Turn the raw series into the passenger mobility metrics."""
    dt = cfg.sample_ticks * DT
    cum = rec.cum_pax.astype(np.float64)

    # Flow over a moving window of one traffic light cycle (without a light, NO_LIGHT_WINDOW s).
    window = round(cfg.flow_window / dt)
    flow = _window_rate(cum, window, dt) * 60.0
    lane_flow = _window_rate(np.cumsum(rec.lane_cross, axis=0, dtype=np.float64), window, dt) * 60.0
    # pax·m of the interval -> pax·km/h, smoothed with a moving average.
    pax_m_cum = np.cumsum(rec.pax_m, axis=0, dtype=np.float64)
    paxkm_h = _window_rate(pax_m_cum, round(SMOOTH_S / dt), dt) * 3600.0 / 1000.0
    # Mean speed per lane with the same moving average: distance / time of its vehicles.
    w = round(SMOOTH_S / dt)
    dist = _window_rate(np.cumsum(rec.lane_dist, axis=0, dtype=np.float64), w, dt)
    veh_time = _window_rate(np.cumsum(rec.lane_time, axis=0, dtype=np.float64), w, dt)
    with np.errstate(invalid="ignore", divide="ignore"):
        per_m = np.where(rec.footprint > 0, rec.pax_on / rec.footprint, np.nan)
        lane_speed = np.where(veh_time > 0, dist / veh_time * 3.6, np.nan)

    emis_cum = np.cumsum(rec.emis, axis=0, dtype=np.float64)
    emis = {
        name: (_window_rate(emis_cum[:, :, p], window, dt) * 60.0).astype(np.float32)
        for p, name in enumerate(EMIS_SERIES)
    }
    return {
        **emis,
        "cum_pax": cum.astype(np.float32),
        "pax_flow": flow.astype(np.float32),
        "paxkm_h": paxkm_h.astype(np.float32),
        "pax_per_m": per_m.astype(np.float32),
        LANE_SATURATION: rec.lane_sat.copy(),
        LANE_SPEED: lane_speed.astype(np.float32),
        LANE_EXIT_FLOW: lane_flow.astype(np.float32),
        EXIT_QUEUE: rec.exit_q.copy(),
        ENTRY_QUEUE: rec.entry_q.copy(),
    }


def speed_bin_count(cfg: SimConfig) -> int:
    """SPEED_BIN km/h bins from 0 up to the maximum speed of any type."""
    return int(np.ceil(max(s.fastest_kmh for s in cfg.specs) / SPEED_BIN)) + 1


def lane_speed_histogram(rec: Recorder, cfg: SimConfig) -> np.ndarray:
    """(lanes, bins): how many samples of the replica each lane had in each mean speed
    bin (distance / time of its vehicles in the sampling interval, including stopped
    ones). Only samples with vehicles in the lane count."""
    n_bins = speed_bin_count(cfg)
    dist, time = rec.lane_dist[: rec.count], rec.lane_time[: rec.count]
    out = np.zeros((cfg.lanes, n_bins), np.int64)
    for lane in range(cfg.lanes):
        present = time[:, lane] > 0
        speed = dist[present, lane] / time[present, lane] * 3.6
        out[lane] = np.bincount(np.minimum((speed / SPEED_BIN).astype(np.int64), n_bins - 1), minlength=n_bins)
    return out


class RunningStats:
    """Element-wise mean and standard deviation (Welford), ignoring NaN.

    Uses O(size of one replica) no matter how many replicas are aggregated.
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
