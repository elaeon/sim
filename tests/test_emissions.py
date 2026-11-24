from dataclasses import replace

import numpy as np
import pytest

from traffic.config import BUS, CAR, DEFAULT_SPECS, POLLUTANTS
from traffic.emissions import acceleration_table, coefficient_table, emission_rate

CAR_GRADUAL = replace(DEFAULT_SPECS[CAR], accel=2.5, decel=4.5)


def _coef(pol: str) -> np.ndarray:
    return np.array(CAR_GRADUAL.emission_coefs(pol))


def test_emission_rate_matches_int_panis_petrol_car():
    """CO2 at 10 m/s without accelerating: 0.553 + 0.161·10 − 0.00289·100 = 1.874 g/s; with a = −1 NOx uses the
    deceleration set (constant 2.17e-4 g/s)."""
    assert emission_rate(_coef("co2"), 10.0, 0.0) == pytest.approx(1.874)
    assert emission_rate(_coef("nox"), 10.0, -1.0) == pytest.approx(2.17e-4)
    assert emission_rate(_coef("nox"), 10.0, 0.0) == pytest.approx(6.19e-4 + 8e-4 - 4.03e-4)


def test_emission_rate_is_never_negative():
    v, a = np.meshgrid(np.linspace(0, 40, 41), np.linspace(-5, 3, 33))
    for pol in ("co2", "nox", "voc"):
        assert (emission_rate(_coef(pol), v, a) >= 0).all()
    bus_pm = np.array(DEFAULT_SPECS[BUS].emission_coefs("pm"))
    assert emission_rate(bus_pm, 25.0, 0.0) == 0.0  # the bus PM polynomial falls below zero at 90 km/h


def test_coefficient_table_only_for_types_that_emit():
    """The car without accel/decel does not emit (the model needs the actual acceleration); with them, CO2, NOx and VOC."""
    coef, on = coefficient_table((DEFAULT_SPECS[CAR], CAR_GRADUAL, DEFAULT_SPECS[1]))
    assert not on[0].any() and not on[2].any()
    assert on[1].tolist() == [pol in ("co2", "nox", "voc") for pol in POLLUTANTS]
    assert coef.shape == (3, len(POLLUTANTS), 2, 6)


def test_acceleration_table():
    """0→20 km/h at 2.5 m/s² takes 2.22 s and covers 6.2 m; its grams are the integral of the model and exceed those
    of covering the same distance at constant speed. Without accel/decel, there is no table."""
    rows = acceleration_table(CAR_GRADUAL)
    assert [(r.v1, r.v2) for r in rows] == [(0, 20), (20, 40), (40, 60), (60, 80)]
    first = rows[0]
    assert first.seconds == pytest.approx(20 / 3.6 / 2.5) and first.meters == pytest.approx(10 / 3.6 * first.seconds)
    t = np.linspace(0, first.seconds, 20001)
    manual = np.trapezoid(emission_rate(_coef("co2"), 2.5 * t, 2.5), t)
    assert first.grams["co2"] == pytest.approx(manual, rel=1e-4)
    assert all(r.grams[p] > r.cruise[p] for r in rows for p in ("co2", "nox"))
    assert acceleration_table(DEFAULT_SPECS[CAR]) == []
