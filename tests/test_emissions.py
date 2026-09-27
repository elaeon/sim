from dataclasses import replace

import numpy as np
import pytest

from trafico.config import BUS, CAR, DEFAULT_SPECS, POLLUTANTS
from trafico.emissions import acceleration_table, coefficient_table, emission_rate

CAR_GRADUAL = replace(DEFAULT_SPECS[CAR], accel=2.5, decel=4.5)


def _coef(pol: str) -> np.ndarray:
    return np.array(CAR_GRADUAL.emission_coefs(pol))


def test_emission_rate_matches_int_panis_petrol_car():
    """CO2 a 10 m/s sin acelerar: 0.553 + 0.161·10 − 0.00289·100 = 1.874 g/s; con a = −1 el NOx usa el juego de
    desaceleración (constante 2.17e-4 g/s)."""
    assert emission_rate(_coef("co2"), 10.0, 0.0) == pytest.approx(1.874)
    assert emission_rate(_coef("nox"), 10.0, -1.0) == pytest.approx(2.17e-4)
    assert emission_rate(_coef("nox"), 10.0, 0.0) == pytest.approx(6.19e-4 + 8e-4 - 4.03e-4)


def test_emission_rate_is_never_negative():
    v, a = np.meshgrid(np.linspace(0, 40, 41), np.linspace(-5, 3, 33))
    for pol in ("co2", "nox", "voc"):
        assert (emission_rate(_coef(pol), v, a) >= 0).all()
    bus_pm = np.array(DEFAULT_SPECS[BUS].emission_coefs("pm"))
    assert emission_rate(bus_pm, 25.0, 0.0) == 0.0  # el polinomio del PM de autobús cae bajo cero a 90 km/h


def test_coefficient_table_only_for_types_that_emit():
    """El auto sin accel/decel no emite (el modelo necesita la aceleración real); con ellos, CO2, NOx y VOC."""
    coef, on = coefficient_table((DEFAULT_SPECS[CAR], CAR_GRADUAL, DEFAULT_SPECS[1]))
    assert not on[0].any() and not on[2].any()
    assert on[1].tolist() == [pol in ("co2", "nox", "voc") for pol in POLLUTANTS]
    assert coef.shape == (3, len(POLLUTANTS), 2, 6)


def test_acceleration_table():
    """0→20 km/h a 2.5 m/s² dura 2.22 s y recorre 6.2 m; sus gramos son la integral del modelo y superan a los
    de recorrer lo mismo a velocidad constante. Sin accel/decel, no hay tabla."""
    rows = acceleration_table(CAR_GRADUAL)
    assert [(r.v1, r.v2) for r in rows] == [(0, 20), (20, 40), (40, 60), (60, 80)]
    first = rows[0]
    assert first.seconds == pytest.approx(20 / 3.6 / 2.5) and first.meters == pytest.approx(10 / 3.6 * first.seconds)
    t = np.linspace(0, first.seconds, 20001)
    manual = np.trapezoid(emission_rate(_coef("co2"), 2.5 * t, 2.5), t)
    assert first.grams["co2"] == pytest.approx(manual, rel=1e-4)
    assert all(r.grams[p] > r.cruise[p] for r in rows for p in ("co2", "nox"))
    assert acceleration_table(DEFAULT_SPECS[CAR]) == []
