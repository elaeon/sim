"""Muestreo de variables aleatorias de la simulación."""

from __future__ import annotations

import math

import numpy as np

from trafico.config import DT, VehicleSpec


def sample_passengers(rng: np.random.Generator, spec: VehicleSpec, size: int) -> np.ndarray:
    """Pasajeros por vehículo: normal(μ, σ) redondeada y truncada a [min, max] por remuestreo.

    La truncadura desplaza la media real cuando μ está cerca de un extremo
    (autos: μ=1.5 con mínimo 1 da una media observada de ~1.8).
    """
    if spec.pax_std <= 0 or spec.pax_min == spec.pax_max:
        value = int(np.clip(round(spec.pax_mean), spec.pax_min, spec.pax_max))
        return np.full(size, value, dtype=np.uint8)
    out = np.empty(size, dtype=np.uint8)
    filled = 0
    while filled < size:
        draw = np.rint(rng.normal(spec.pax_mean, spec.pax_std, size=2 * (size - filled) + 8))
        draw = draw[(draw >= spec.pax_min) & (draw <= spec.pax_max)]
        take = min(draw.size, size - filled)
        out[filled : filled + take] = draw[:take]
        filled += take
    return out


def stop_ticks(rng: np.random.Generator, spec: VehicleSpec, size: int) -> np.ndarray:
    """Duración de la parada en pasos: normal(μ, σ) truncada a ≥ 0 por remuestreo, cuantizada a DT."""
    if spec.stop_time_std <= 0:
        return np.full(size, round(max(spec.stop_time_mean, 0.0) / DT), dtype=np.int32)
    out = np.empty(size, dtype=np.float64)
    filled = 0
    while filled < size:
        draw = rng.normal(spec.stop_time_mean, spec.stop_time_std, size=2 * (size - filled) + 8)
        draw = draw[draw >= 0]
        take = min(draw.size, size - filled)
        out[filled : filled + take] = draw[:take]
        filled += take
    return np.rint(out / DT).astype(np.int32)


def passenger_pmf(spec: VehicleSpec) -> tuple[np.ndarray, np.ndarray]:
    """Distribución exacta que produce `sample_passengers`: (valores pax_min..pax_max, probabilidades).

    Normal(μ, σ) redondeada al entero más cercano y truncada al rango: P(v) ∝ Φ(v + ½) − Φ(v − ½).
    """
    values = np.arange(spec.pax_min, spec.pax_max + 1)
    if spec.pax_std <= 0 or spec.pax_min == spec.pax_max:
        probs = (values == int(np.clip(round(spec.pax_mean), spec.pax_min, spec.pax_max))).astype(float)
        return values, probs

    def cdf(x: float) -> float:
        return 0.5 * (1 + math.erf((x - spec.pax_mean) / (spec.pax_std * math.sqrt(2))))

    probs = np.array([cdf(v + 0.5) - cdf(v - 0.5) for v in values])
    return values, probs / probs.sum()


def uniform_ticks(rng: np.random.Generator, low: float, high: float, size: int | None = None):
    """Duración uniforme en [low, high] segundos, cuantizada a pasos de DT."""
    return rng.integers(round(low / DT), round(high / DT), size=size, endpoint=True)
