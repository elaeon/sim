"""Fuel consumption and cost from the CO2 of the emission model.

All the carbon in the fuel leaves as CO2, so the litres are the CO2 emitted divided by the g of CO2 per litre
of its fuel (`FUEL_CO2_G_PER_L`); the carbon in CO and HC (under 1 %) is not counted. So consumption inherits
the shape of the Int Panis CO2 (fitted to urban driving: at constant speed its consumption per km falls as
speed rises and drops below what overcoming air drag requires above ≈ 70–80 km/h).

For contrast, the constant-speed fuel economy table also gives a physical curve (Willans line): the model's
idling plus the power at the wheel to overcome rolling resistance and drag, VSP(v, 0) = 0.132·v +
0.000302·v³ kW/t, times the type's mass, divided by the engine's marginal efficiency and the fuel's heating
value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from traffic.config import FUEL_CO2_G_PER_L, FUEL_MJ_PER_L, SimConfig, VehicleSpec
from traffic.emissions import emission_rate

WILLANS_EFFICIENCY = 0.246
"""Eficiencia marginal del motor (energía en la rueda / energía del combustible) de la curva física: con ella, el
auto de 1250 kg a 77.7 km/h constantes (la velocidad media del ciclo de carretera de la EPA) rinde 18.6 km/L, la
mediana en carretera de los autos a gasolina de 4 cilindros y 1.4–2.0 L del catálogo CONUEE 2026."""

CRUISE_KMH = (20.0, 40.0, 60.0, 80.0)
"""Velocidades de la tabla de rendimiento a velocidad constante."""


def liters(spec: VehicleSpec, co2_g):
    """Litres of fuel that emit `co2_g` grams of CO2 (NaN if the type burns no fuel)."""
    if not spec.burns:
        return np.full_like(np.asarray(co2_g, dtype=np.float64), np.nan)
    return np.asarray(co2_g, dtype=np.float64) / FUEL_CO2_G_PER_L[spec.fuel]


def cost(cfg: SimConfig, spec: VehicleSpec, l):
    """Cost of `l` litres at the price of the type's fuel (NaN if it has no price)."""
    price = cfg.price(spec.fuel)
    return np.asarray(l, dtype=np.float64) * (np.nan if price is None else price)


def vsp(v, a=0.0):
    """Specific power (kW/t) at v m/s and a m/s² on flat ground."""
    return v * (1.1 * a + 0.132) + 0.000302 * v**3


@dataclass(frozen=True, slots=True)
class CruiseRow:
    """Fuel economy at a constant `kmh`: L/100 km from the model (from CO2) and from the physical curve (None without mass)."""

    kmh: float
    l_100km: float
    physical_l_100km: float | None


def constant_speed_table(spec: VehicleSpec, speeds=CRUISE_KMH) -> list[CruiseRow]:
    """L/100 km at each constant speed in `speeds` that the type reaches. Empty if it burns no fuel."""
    if not spec.burns:
        return []
    acc, dec = spec.emission_coefs("co2")
    coef = np.array([acc, dec])
    g_per_l = FUEL_CO2_G_PER_L[spec.fuel]
    rows = []
    for kmh in speeds:
        if kmh > spec.fastest_kmh + 1e-9:
            continue
        v = kmh / 3.6
        l_s = float(emission_rate(coef, v, 0.0)) / g_per_l
        physical = None
        if spec.mass_kg is not None:
            idle = float(emission_rate(coef, 0.0, 0.0)) / g_per_l
            wheel_kw = spec.mass_kg / 1000.0 * vsp(v)
            physical = (idle + wheel_kw / (WILLANS_EFFICIENCY * FUEL_MJ_PER_L[spec.fuel] * 1000.0)) / v * 1e5
        rows.append(CruiseRow(kmh, l_s / v * 1e5, physical))
    return rows
