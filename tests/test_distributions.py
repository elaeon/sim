import numpy as np

from trafico.config import BIKE, BUS, CAR, DEFAULT_SPECS, DT
from trafico.config import VehicleSpec
from trafico.config import Behavior
from trafico.distributions import passenger_pmf, reaction_ticks, sample_lengths, sample_passengers, uniform_ticks


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


def test_lengths_truncated_normal():
    """Carga: normal(10, 2) truncada a [8, 16]; la media teórica sube a ≈ 10.57 m por el corte en 8."""
    spec = VehicleSpec("carga", "carga", 40.0, 10.0, 4.0, 1.5, 1, 1, 1.0, 0.0, True,
                       cargo_prob=1.0, length_std=2.0, length_min=8.0)  # fmt: skip
    assert (spec.shortest, spec.longest) == (8.0, 16.0)
    lengths = sample_lengths(np.random.default_rng(0), spec, 200_000)
    assert lengths.min() >= 8.0 and lengths.max() <= 16.0
    assert abs(lengths.mean() - 10.57) < 0.02
    fixed = sample_lengths(np.random.default_rng(0), DEFAULT_SPECS[CAR], 10)
    np.testing.assert_array_equal(fixed, 4.5)


def test_reaction_uniform_without_std():
    """Sin reaction_std, la reacción es la uniforme de antes (mismos sorteos)."""
    b = Behavior(reaction_min=0.7, reaction_max=3.0)
    np.testing.assert_array_equal(
        reaction_ticks(np.random.default_rng(1), b, 1000), uniform_ticks(np.random.default_rng(1), 0.7, 3.0, 1000)
    )


def test_reaction_truncated_normal():
    b = Behavior(reaction_min=0.7, reaction_max=3.0, reaction_mean=1.5, reaction_std=0.5)
    ticks = reaction_ticks(np.random.default_rng(0), b, 200_000)
    assert ticks.min() >= round(0.7 / DT) and ticks.max() <= round(3.0 / DT)
    # La truncadura en [0.7, 3.0] alrededor de 1.5 ± 0.5 sube un poco la media (≈ 1.56 s).
    assert abs(ticks.mean() * DT - 1.56) < 0.01
    assert abs(ticks.std() * DT - 0.45) < 0.02
    mid = reaction_ticks(np.random.default_rng(0), Behavior(reaction_min=1, reaction_max=3, reaction_std=0.2), 10_000)
    assert abs(mid.mean() * DT - 2.0) < 0.01  # sin media: el punto medio
    fixed = reaction_ticks(np.random.default_rng(0), Behavior(reaction_mean=2.0, reaction_std=0.0), 5)
    np.testing.assert_array_equal(fixed, round(2.0 / DT))
    tiny = reaction_ticks(np.random.default_rng(0), Behavior(reaction_min=0.01, reaction_max=0.02), 5)
    assert tiny.min() >= 1  # nunca 0 pasos
