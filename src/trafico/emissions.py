"""Emisiones instantáneas por velocidad y aceleración (Int Panis, Broekx y Liu, 2006).

E = max(0, f1 + f2·v + f3·v² + f4·a + f5·a² + f6·v·a) en g/s, con v en m/s y a en m/s². Cada tipo de
vehículo y contaminante tiene dos juegos de coeficientes: uno para a ≥ −0.5 m/s² y otro para a < −0.5 m/s².
El modelo se ajustó con una flota europea de principios de los 2000: sirve para comparar escenarios (con y
sin tope, con y sin semáforo), no para dar valores absolutos.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trafico.config import EMISSION_DECEL, POLLUTANT_LABELS, POLLUTANTS, SimConfig, VehicleSpec
from trafico.emission_sets import SET_LABELS

EMIS_BIN = 5.0
"""m: ancho de los intervalos de posición en que se acumulan las emisiones a lo largo del tramo."""

ACCEL_STEPS = ((0.0, 20.0), (20.0, 40.0), (40.0, 60.0), (60.0, 80.0))
"""Escalones (km/h) de la tabla de referencia de emisiones al acelerar."""


def coefficient_table(specs: tuple[VehicleSpec, ...]) -> tuple[np.ndarray, np.ndarray]:
    """(coeficientes de forma (tipos, contaminantes, 2, 6), qué tipo emite qué contaminante). Solo cuentan los
    tipos que emiten (coeficientes y dinámica gradual); el juego 0 es para a ≥ −0.5 y el 1 para a < −0.5."""
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
    """(índices en POLLUTANTS de los contaminantes que emite algún tipo que participa, tipos que participan y
    emiten)."""
    types = [k for k, sp in enumerate(cfg.specs) if cfg.rates[k] > 0 and sp.emits]
    pols = [p for p, pol in enumerate(POLLUTANTS) if any(cfg.specs[k].emission_coefs(pol) for k in types)]
    return pols, types


def unit(pol: str) -> tuple[str, float]:
    """Unidad con que se muestra el contaminante y su factor desde gramos: CO2 en g, los demás en mg."""
    return ("g", 1.0) if pol == "co2" else ("mg", 1000.0)


def emissions_label(cfg: SimConfig) -> str | None:
    """Línea de la cabecera: qué contaminantes emite cada tipo que participa y cuáles no emiten."""
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
    return "Emisiones (modelo de Int Panis et al., 2006): " + " · ".join(parts) + (
        f" · sin emisiones (sin coeficientes o sin accel/decel): {', '.join(silent)}" if silent else "")


def emission_rate(coef: np.ndarray, v, a) -> np.ndarray:
    """Tasa de emisión (g/s) con coeficientes de forma (..., 2, 6) a velocidad v (m/s) y aceleración a (m/s²),
    que se difunden con la forma (...) de los coeficientes sin sus dos últimos ejes."""
    v = np.asarray(v, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    c = np.where((a >= EMISSION_DECEL)[..., None], coef[..., 0, :], coef[..., 1, :])
    e = c[..., 0] + c[..., 1] * v + c[..., 2] * v * v + c[..., 3] * a + c[..., 4] * a * a + c[..., 5] * v * a
    return np.maximum(e, 0.0)


@dataclass(frozen=True, slots=True)
class AccelStep:
    """Un escalón de la tabla: de v1 a v2 km/h a la aceleración `accel` (m/s²) del tipo."""

    v1: float
    v2: float
    accel: float
    seconds: float
    meters: float
    grams: dict[str, float]  # emitidos al acelerar
    cruise: dict[str, float]  # a velocidad constante (la media del escalón) en el mismo tiempo y distancia


def acceleration_table(spec: VehicleSpec, steps=ACCEL_STEPS, n: int = 2000) -> list[AccelStep]:
    """Gramos que emite un vehículo del tipo al acelerar en cada escalón a su `accel` constante (integración por
    punto medio en `n` subpasos), y lo que emitiría recorriendo la misma distancia en el mismo tiempo a velocidad
    constante. Vacía si el tipo no emite."""
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
