"""Sampling of the simulation's random variables."""

from __future__ import annotations

import math

import numpy as np

from traffic.config import DT, Behavior, Rate, VehicleSpec


def sample_passengers(rng: np.random.Generator, spec: VehicleSpec, size: int) -> np.ndarray:
    """Passengers per vehicle: normal(μ, σ) rounded and truncated to [min, max] by resampling.

    The truncation shifts the actual mean when μ is close to a bound
    (cars: μ=1.5 with minimum 1 gives an observed mean of ~1.8).
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


def sample_lengths(rng: np.random.Generator, spec: VehicleSpec, size: int) -> np.ndarray:
    """Length of each vehicle (m): normal(length, length_std) truncated to [shortest, longest] by
    resampling. Without a deviation, all measure `length` and the generator is not used."""
    if spec.length_std <= 0:
        return np.full(size, spec.length)
    return truncated_normal(rng, spec.length, spec.length_std, spec.shortest, spec.longest, size)


def sample_speeds(rng: np.random.Generator, spec: VehicleSpec, size: int) -> np.ndarray:
    """Maximum speed of each vehicle (km/h): normal(speed_kmh, speed_std) truncated to
    [slowest_kmh, fastest_kmh] by resampling. Without a deviation, all go at `speed_kmh` and the
    generator is not used."""
    if spec.speed_std <= 0:
        return np.full(size, spec.speed_kmh)
    return truncated_normal(rng, spec.speed_kmh, spec.speed_std, spec.slowest_kmh, spec.fastest_kmh, size)


def sample_rates(rng: np.random.Generator, rate: Rate, size: int) -> np.ndarray:
    """Arrival rate (veh/min) of each interval: normal(mean, std) truncated to [min, max]."""
    if not rate.variable:
        return np.full(size, rate.expected)
    return truncated_normal(rng, rate.mean, rate.std, rate.min, rate.max, size)


def truncated_normal(rng: np.random.Generator, mean: float, std: float, lo: float, hi: float, size: int) -> np.ndarray:
    """normal(mean, std) truncated to [lo, hi] by resampling."""
    out = np.empty(size, dtype=np.float64)
    filled = 0
    while filled < size:
        draw = rng.normal(mean, std, size=2 * (size - filled) + 8)
        draw = draw[(draw >= lo) & (draw <= hi)]
        take = min(draw.size, size - filled)
        out[filled : filled + take] = draw[:take]
        filled += take
    return out


def stop_ticks(rng: np.random.Generator, spec: VehicleSpec, size: int) -> np.ndarray:
    """Stop duration in steps: normal(μ, σ) truncated to ≥ 0 by resampling, quantized to DT."""
    return normal_ticks(rng, spec.stop_time_mean, spec.stop_time_std, size)


def normal_ticks(rng: np.random.Generator, mean: float, std: float, size: int) -> np.ndarray:
    """Duration in steps: normal(mean, std) truncated to ≥ 0 by resampling, quantized to DT."""
    if std <= 0:
        return np.full(size, round(max(mean, 0.0) / DT), dtype=np.int32)
    out = np.empty(size, dtype=np.float64)
    filled = 0
    while filled < size:
        draw = rng.normal(mean, std, size=2 * (size - filled) + 8)
        draw = draw[draw >= 0]
        take = min(draw.size, size - filled)
        out[filled : filled + take] = draw[:take]
        filled += take
    return np.rint(out / DT).astype(np.int32)


def passenger_pmf(spec: VehicleSpec) -> tuple[np.ndarray, np.ndarray]:
    """Exact distribution produced by `sample_passengers`: (values pax_min..pax_max, probabilities).

    Normal(μ, σ) rounded to the nearest integer and truncated to the range: P(v) ∝ Φ(v + ½) − Φ(v − ½).
    """
    values = np.arange(spec.pax_min, spec.pax_max + 1)
    if spec.pax_std <= 0 or spec.pax_min == spec.pax_max:
        probs = (values == int(np.clip(round(spec.pax_mean), spec.pax_min, spec.pax_max))).astype(float)
        return values, probs

    def cdf(x: float) -> float:
        return 0.5 * (1 + math.erf((x - spec.pax_mean) / (spec.pax_std * math.sqrt(2))))

    probs = np.array([cdf(v + 0.5) - cdf(v - 0.5) for v in values])
    return values, probs / probs.sum()


def reaction_ticks(rng: np.random.Generator, b: Behavior, size: int) -> np.ndarray:
    """Reaction time in steps (at least one). Without `reaction_std`, uniform in
    [reaction_min, reaction_max]; with it, normal(reaction_mean, reaction_std) truncated to that
    range by resampling (default mean: the midpoint), quantized to DT."""
    if b.reaction_std is None:
        ticks = uniform_ticks(rng, b.reaction_min, b.reaction_max, size)
    else:
        mean = b.reaction_mean if b.reaction_mean is not None else (b.reaction_min + b.reaction_max) / 2
        if b.reaction_std <= 0:
            out = np.full(size, mean)
        else:
            out = np.empty(size, dtype=np.float64)
            filled = 0
            while filled < size:
                draw = rng.normal(mean, b.reaction_std, size=2 * (size - filled) + 8)
                draw = draw[(draw >= b.reaction_min) & (draw <= b.reaction_max)]
                take = min(draw.size, size - filled)
                out[filled : filled + take] = draw[:take]
                filled += take
        ticks = np.rint(out / DT).astype(np.int64)
    return np.maximum(ticks, 1)  # with 0 steps it would never start


def uniform_ticks(rng: np.random.Generator, low: float, high: float, size: int | None = None):
    """Uniform duration in [low, high] seconds, quantized to DT steps."""
    return rng.integers(round(low / DT), round(high / DT), size=size, endpoint=True)
