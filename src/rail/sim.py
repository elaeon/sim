"""1D kinematic train run: accelerate to the segment limit, brake for limits and stations, dwell."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rail.route import Route

MIN_SPEED_MS = 1.0  # creep speed so the braking curve still reaches the platform


@dataclass(frozen=True, slots=True)
class Stop:
    name: str
    km: float
    elevation_m: float
    arrival_s: float
    departure_s: float


@dataclass(frozen=True, slots=True)
class Journey:
    t_s: np.ndarray  # sample times (dwells are two samples: arrival and departure)
    km: np.ndarray
    speed_kmh: np.ndarray
    stops: tuple[Stop, ...]

    @property
    def duration_s(self) -> float:
        return float(self.t_s[-1])

    def at(self, t_s):
        """(km, speed_kmh) at time t_s, linearly interpolated (scalar or array)."""
        return np.interp(t_s, self.t_s, self.km), np.interp(t_s, self.t_s, self.speed_kmh)

    def arrival(self, name: str) -> float:
        return next(s.arrival_s for s in self.stops if s.name == name)


def simulate(route: Route) -> Journey:
    """Run the train from the first to the last station of the route."""
    spec = route.train
    dt, a, b = spec.dt_s, spec.accel_ms2, spec.brake_ms2
    # Every point may start a lower limit; the next station itself is handled by the stopping curve.
    marks_m = np.array([p.km * 1000 for p in route.points])
    limits_ms = np.array([p.max_speed_kmh / 3.6 for p in route.points])
    stations = route.stations
    x, v, t = 0.0, 0.0, 0.0
    ts, xs, vs = [0.0], [0.0], [0.0]
    stops = [Stop(stations[0].name, stations[0].km, stations[0].elevation_m, 0.0, 0.0)]
    for station in stations[1:]:
        target_m = station.km * 1000
        while True:
            ahead = (marks_m > x) & (marks_m < target_m)
            # Braking curve: the speed from which the train can still slow to each upcoming limit / the stop.
            # Distances discount the longest possible step, so the train never enters a slower segment too fast.
            step_m = (v + a * dt) * dt
            caps = np.sqrt(limits_ms[ahead] ** 2 + 2 * b * np.maximum(marks_m[ahead] - x - step_m, 0.0))
            v_max = min(route.speed_limit(x / 1000) / 3.6, np.sqrt(2 * b * (target_m - x)), *caps)
            v_max = max(v_max, MIN_SPEED_MS)
            v = min(v + a * dt, v_max) if v < v_max else v_max
            if x + v * dt >= target_m:  # reaches the platform within this step
                t += (target_m - x) / v
                x, v = target_m, 0.0
                break
            x += v * dt
            t += dt
            ts.append(t), xs.append(x), vs.append(v)
        ts.append(t), xs.append(x), vs.append(0.0)
        arrival = t
        if station is not stations[-1]:
            t += station.dwell_min * 60
            ts.append(t), xs.append(x), vs.append(0.0)
        stops.append(Stop(station.name, station.km, station.elevation_m, arrival, t))
    return Journey(t_s=np.array(ts), km=np.array(xs) / 1000, speed_kmh=np.array(vs) * 3.6, stops=tuple(stops))


def timetable(journey: Journey, depart_s: float = 0.0) -> list[dict]:
    """Rows for timetable.csv: one per station, clock times from the departure time of day."""
    return [{"station": s.name, "km": round(s.km, 1), "elevation_m": round(s.elevation_m),
             "arrival": clock(depart_s + s.arrival_s), "departure": clock(depart_s + s.departure_s),
             "dwell_min": round(float(s.departure_s - s.arrival_s) / 60, 1)} for s in journey.stops]  # fmt: skip


def clock(seconds: float) -> str:
    """Time of day as HH:MM, with a +N day suffix past midnight."""
    days, rest = divmod(int(round(seconds)), 86400)
    hours, rest = divmod(rest, 3600)
    text = f"{hours:02d}:{rest // 60:02d}"
    return f"{text} (+{days}d)" if days else text


def duration(seconds: float) -> str:
    """Elapsed time as 'Xh YYmin'."""
    minutes = int(round(seconds / 60))
    return f"{minutes // 60}h {minutes % 60:02d}min"
