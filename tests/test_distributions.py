import numpy as np

from trafico.config import BIKE, BUS, CAR, DEFAULT_SPECS, DT
from trafico.distributions import passenger_pmf, sample_passengers, uniform_ticks


def test_passengers_within_bounds_and_means():
    rng = np.random.default_rng(0)
    car = sample_passengers(rng, DEFAULT_SPECS[CAR], 100_000)
    bus = sample_passengers(rng, DEFAULT_SPECS[BUS], 100_000)
    bike = sample_passengers(rng, DEFAULT_SPECS[BIKE], 1_000)

    assert car.min() >= 1 and car.max() <= 6
    assert bus.min() >= 1 and bus.max() <= 80
    assert np.all(bike == 1)
    # La truncadura en 1 sube la media de los autos respecto a μ = 1.5.
    assert 1.7 < car.mean() < 1.9
    assert abs(bus.mean() - 40.0) < 0.2
    assert abs(bus.std() - 10.0) < 0.3


def test_uniform_ticks_quantized_range():
    rng = np.random.default_rng(1)
    ticks = uniform_ticks(rng, 1.0, 5.0, 50_000)
    assert ticks.min() == round(1.0 / DT)
    assert ticks.max() == round(5.0 / DT)


def test_passenger_pmf_matches_sampler():
    """La distribución esperada que dibuja la gráfica es la que produce el muestreo."""
    import dataclasses

    rng = np.random.default_rng(11)
    wide_bus = dataclasses.replace(DEFAULT_SPECS[BUS], pax_max=120, pax_mean=80.0, pax_std=15.0)
    for spec in (*DEFAULT_SPECS, wide_bus):
        values, probs = passenger_pmf(spec)
        assert values[0] == spec.pax_min and values[-1] == spec.pax_max
        assert abs(probs.sum() - 1) < 1e-12
        draws = sample_passengers(rng, spec, 200_000)
        freq = np.bincount(draws, minlength=spec.pax_max + 1)[values] / draws.size
        np.testing.assert_allclose(freq, probs, atol=0.005)
    # Autos: μ = 1.5 con mínimo 1 → la media esperada sube a ~1.8 por la truncadura.
    values, probs = passenger_pmf(DEFAULT_SPECS[CAR])
    assert 1.75 < (values * probs).sum() < 1.85
