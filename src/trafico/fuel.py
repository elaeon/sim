"""Consumo y costo de combustible a partir del CO2 del modelo de emisiones.

Todo el carbono del combustible sale como CO2, así que los litros son el CO2 emitido entre los g de CO2 por litro
de su combustible (`FUEL_CO2_G_PER_L`); el carbono del CO y del HC (menos de 1 %) no se cuenta. Por eso el
consumo hereda la forma del CO2 de Int Panis (ajustado con manejo urbano: a velocidad constante, su consumo por
km baja al subir la velocidad y pasa por debajo de lo que exige vencer el aire arriba de ≈ 70–80 km/h).

Como contraste, la tabla de rendimiento a velocidad constante da también una curva física (línea de Willans):
el ralentí del modelo más la potencia en la rueda para vencer la rodadura y el aire, VSP(v, 0) = 0.132·v +
0.000302·v³ kW/t, por la masa del tipo, entre la eficiencia marginal del motor y el poder calorífico del
combustible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trafico.config import FUEL_CO2_G_PER_L, FUEL_MJ_PER_L, SimConfig, VehicleSpec
from trafico.emissions import emission_rate

WILLANS_EFFICIENCY = 0.246
"""Eficiencia marginal del motor (energía en la rueda / energía del combustible) de la curva física: con ella, el
auto de 1250 kg a 77.7 km/h constantes (la velocidad media del ciclo de carretera de la EPA) rinde 18.6 km/L, la
mediana en carretera de los autos a gasolina de 4 cilindros y 1.4–2.0 L del catálogo CONUEE 2026."""

CRUISE_KMH = (20.0, 40.0, 60.0, 80.0)
"""Velocidades de la tabla de rendimiento a velocidad constante."""


def liters(spec: VehicleSpec, co2_g):
    """Litros de combustible que emiten `co2_g` gramos de CO2 (NaN si el tipo no quema combustible)."""
    if not spec.burns:
        return np.full_like(np.asarray(co2_g, dtype=np.float64), np.nan)
    return np.asarray(co2_g, dtype=np.float64) / FUEL_CO2_G_PER_L[spec.fuel]


def cost(cfg: SimConfig, spec: VehicleSpec, l):
    """Costo de `l` litros al precio del combustible del tipo (NaN si no tiene precio)."""
    price = cfg.price(spec.fuel)
    return np.asarray(l, dtype=np.float64) * (np.nan if price is None else price)


def vsp(v, a=0.0):
    """Potencia específica (kW/t) a v m/s y a m/s² en plano."""
    return v * (1.1 * a + 0.132) + 0.000302 * v**3


@dataclass(frozen=True, slots=True)
class CruiseRow:
    """Rendimiento a `kmh` constantes: L/100 km del modelo (del CO2) y de la curva física (None sin masa)."""

    kmh: float
    l_100km: float
    physical_l_100km: float | None


def constant_speed_table(spec: VehicleSpec, speeds=CRUISE_KMH) -> list[CruiseRow]:
    """L/100 km a cada velocidad constante de `speeds` que el tipo alcanza. Vacía si no quema combustible."""
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
