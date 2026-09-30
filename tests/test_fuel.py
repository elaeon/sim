from dataclasses import replace

import numpy as np
import pytest

from trafico.config import DEFAULT_SPECS, FUEL_CO2_G_PER_L, SimConfig
from trafico.emission_sets import EMISSION_SETS
from trafico.emissions import emission_rate
from trafico.fuel import constant_speed_table, cost, liters

CAR = replace(DEFAULT_SPECS[0], accel=2.5, decel=4.5, speed_kmh=80.0)
SEDEMA_CAR = replace(CAR, emissions=EMISSION_SETS["sedema_cdmx_2018"]["car"], emission_source="sedema_cdmx_2018")


def test_liters_and_cost():
    assert liters(CAR, FUEL_CO2_G_PER_L["gasolina"]) == pytest.approx(1.0)
    assert FUEL_CO2_G_PER_L["gasolina"] == pytest.approx(2347.7, abs=0.1)
    assert np.isnan(liters(DEFAULT_SPECS[1], 100.0))  # la bici no quema combustible
    cfg = SimConfig(fuel_price=(("gasolina", 24.0),))
    assert cost(cfg, CAR, 2.0) == pytest.approx(48.0)
    assert np.isnan(cost(SimConfig(), CAR, 2.0))  # sin precio


def test_burns_requires_fuel_co2_and_dynamics():
    assert CAR.burns and not DEFAULT_SPECS[0].burns  # sin accel/decel no emite
    bus = replace(DEFAULT_SPECS[2], accel=1.2, decel=3.5)
    assert bus.fuel == "diesel" and not bus.burns  # solo PM: sin CO2 no hay consumo
    assert not replace(CAR, fuel=None).burns


def test_constant_speed_table_matches_the_reference_scripts():
    """Int Panis a 40 km/h: 7.6 L/100 km; la curva física del auto de 1250 kg rinde 18.6 km/L a 77.7 km/h y
    gasta menos alrededor de 50 km/h (consumo_velocidad.py de multi-dashboard)."""
    rows = {r.kmh: r for r in constant_speed_table(CAR)}
    assert list(rows) == [20.0, 40.0, 60.0, 80.0]
    assert rows[40.0].l_100km == pytest.approx(7.61, abs=0.01)
    assert 100 / constant_speed_table(CAR, (77.7,))[0].physical_l_100km == pytest.approx(18.6, abs=0.1)
    speeds = np.arange(10.0, 121.0, 5.0)
    physical = [r.physical_l_100km for r in constant_speed_table(replace(CAR, speed_kmh=120.0), speeds)]
    assert 45 <= speeds[int(np.argmin(physical))] <= 55
    assert constant_speed_table(replace(CAR, mass_kg=None))[0].physical_l_100km is None
    assert [r.kmh for r in constant_speed_table(replace(CAR, speed_kmh=50.0))] == [20.0, 40.0]
    assert constant_speed_table(DEFAULT_SPECS[1]) == []


def test_sedema_set_matches_the_calibration_events():
    """Contra eventos_sim.csv: el auto gasta 14.1 mL por minuto en ralentí y emite 0.88 mg/s de NOx a 40 km/h
    constantes; al frenar, el CO2 es el del ralentí (no el término a²)."""
    co2 = np.array(SEDEMA_CAR.emission_coefs("co2"))
    nox = np.array(SEDEMA_CAR.emission_coefs("nox"))
    assert float(emission_rate(co2, 0.0, 0.0)) * 60 / FUEL_CO2_G_PER_L["gasolina"] * 1000 == pytest.approx(14.13, abs=0.01)
    assert float(emission_rate(nox, 40 / 3.6, 0.0)) * 1000 == pytest.approx(0.8769, abs=1e-3)
    assert float(emission_rate(co2, 5.0, -4.5)) == pytest.approx(0.553)
    int_panis = np.array(CAR.emission_coefs("co2"))
    assert float(emission_rate(int_panis, 5.0, -4.5)) > 5 * 0.553
    assert set(EMISSION_SETS["sedema_cdmx_2018"]) == {"car", "taxi", "carga_ligera", "colectivo"}
