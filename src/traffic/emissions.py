"""Instantaneous emissions from speed and acceleration (Int Panis, Broekx and Liu, 2006).

E = max(0, f1 + f2·v + f3·v² + f4·a + f5·a² + f6·v·a) in g/s, with v in m/s and a in m/s². Each vehicle
type and pollutant has two sets of coefficients: one for a ≥ −0.5 m/s² and another for a < −0.5 m/s².
The model was fitted to a European fleet of the early 2000s: it is for comparing scenarios (with and
without a bump, with and without a traffic light), not for absolute values.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from traffic.config import EMISSION_DECEL, POLLUTANT_LABELS, POLLUTANTS, SimConfig, VehicleSpec
from traffic.emission_sets import SET_LABELS

EMIS_BIN = 5.0
"""m: ancho de los intervalos de posición en que se acumulan las emisiones a lo largo del tramo."""

ACCEL_STEPS = ((0.0, 20.0), (20.0, 40.0), (40.0, 60.0), (60.0, 80.0))
"""Escalones (km/h) de la tabla de referencia de emisiones al acelerar."""


def coefficient_table(specs: tuple[VehicleSpec, ...]) -> tuple[np.ndarray, np.ndarray]:
    """(coefficients of shape (types, pollutants, 2, 6), which type emits which pollutant). Only types that
    emit count (coefficients and gradual dynamics); set 0 is for a ≥ −0.5 and set 1 for a < −0.5."""
    coef = np.zeros((len(specs), len(POLLUTANTS), 2, 6))
    on = np.zeros((len(specs), len(POLLUTANTS)), np.bool_)
    for k, spec in enumerate(specs):
        if not spec.emits:
            continue
        for p, pol in enumerate(POLLUTANTS):
            pair = spec.emission_coefs(pol)
            if pair is not None:
                coef[k, p] = pair
                on[k, p] = True
    return coef, on


def emitted(cfg: SimConfig) -> tuple[list[int], list[int]]:
    """(indices in POLLUTANTS of the pollutants emitted by some participating type, participating types that
    emit)."""
    types = [k for k, sp in enumerate(cfg.specs) if cfg.rates[k] > 0 and sp.emits]
    pols = [p for p, pol in enumerate(POLLUTANTS) if any(cfg.specs[k].emission_coefs(pol) for k in types)]
    return pols, types


def unit(pol: str) -> tuple[str, float]:
    """Unit the pollutant is shown in and its factor from grams: CO2 in g, the rest in mg."""
    return ("g", 1.0) if pol == "co2" else ("mg", 1000.0)


def emissions_label(cfg: SimConfig) -> str | None:
    """Header line: which pollutants each participating type emits and which types emit nothing."""
    active = [k for k, rate in enumerate(cfg.rates) if rate > 0]
    if not any(cfg.specs[k].emits for k in active):
        return None
    def source(spec) -> str:
        return f" ({SET_LABELS.get(spec.emission_source, spec.emission_source)})" if spec.emission_source else ""

    parts = [
        f"{cfg.specs[k].name}{source(cfg.specs[k])} "
        + ", ".join(POLLUTANT_LABELS[p] for p in POLLUTANTS if cfg.specs[k].emission_coefs(p))
        for k in active if cfg.specs[k].emits
    ]  # fmt: skip
    silent = [cfg.specs[k].name for k in active if not cfg.specs[k].emits]
    return "Emissions (Int Panis et al., 2006 model): " + " · ".join(parts) + (
        f" · no emissions (no coefficients or no accel/decel): {', '.join(silent)}" if silent else "")


def emission_rate(coef: np.ndarray, v, a) -> np.ndarray:
    """Emission rate (g/s) with coefficients of shape (..., 2, 6) at speed v (m/s) and acceleration a (m/s²),
    broadcast to the shape (...) of the coefficients without their last two axes."""
    v = np.asarray(v, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    c = np.where((a >= EMISSION_DECEL)[..., None], coef[..., 0, :], coef[..., 1, :])
    e = c[..., 0] + c[..., 1] * v + c[..., 2] * v * v + c[..., 3] * a + c[..., 4] * a * a + c[..., 5] * v * a
    return np.maximum(e, 0.0)


@dataclass(frozen=True, slots=True)
class AccelStep:
    """One step of the table: from v1 to v2 km/h at the type's acceleration `accel` (m/s²)."""

    v1: float
    v2: float
    accel: float
    seconds: float
    meters: float
    grams: dict[str, float]  # emitted while accelerating
    cruise: dict[str, float]  # at constant speed (the step's mean) over the same time and distance


def acceleration_table(spec: VehicleSpec, steps=ACCEL_STEPS, n: int = 2000) -> list[AccelStep]:
    """Grams a vehicle of the type emits when accelerating through each step at its constant `accel` (midpoint
    integration over `n` substeps), and what it would emit covering the same distance in the same time at constant
    speed. Empty if the type emits nothing."""
    if not spec.emits:
        return []
    coef = np.array([pair for pol in POLLUTANTS if (pair := spec.emission_coefs(pol)) is not None])
    names = [pol for pol in POLLUTANTS if spec.emission_coefs(pol) is not None]
    rows = []
    for v1, v2 in steps:
        a = spec.accel
        seconds = (v2 - v1) / 3.6 / a
        t = (np.arange(n) + 0.5) * seconds / n
        v = v1 / 3.6 + a * t
        grams = emission_rate(coef[:, None], v[None, :], np.full((1, n), a)).sum(axis=1) * seconds / n
        mean_v = (v1 + v2) / 2 / 3.6
        cruise = emission_rate(coef, np.full(len(names), mean_v), np.zeros(len(names))) * seconds
        rows.append(AccelStep(
            v1, v2, a, seconds, mean_v * seconds,
            dict(zip(names, grams.tolist())), dict(zip(names, cruise.tolist())),
        ))  # fmt: skip
    return rows
