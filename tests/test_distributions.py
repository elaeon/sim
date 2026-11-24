import numpy as np

from traffic.config import BIKE, BUS, CAR, DEFAULT_SPECS, DT
from traffic.config import VehicleSpec
from traffic.config import Behavior
from traffic.distributions import (
    passenger_pmf, reaction_ticks, sample_lengths, sample_passengers, sample_speeds, uniform_ticks,
)  # fmt: skip


def test_passengers_within_bounds_and_means():
    rng = np.random.default_rng(0)
    car = sample_passengers(rng, DEFAULT_SPECS[CAR], 100_000)
    bus = sample_passengers(rng, DEFAULT_SPECS[BUS], 100_000)
    bike = sample_passengers(rng, DEFAULT_SPECS[BIKE], 1_000)

    assert car.min() >= 1 and car.max() <= 6
    assert bus.min() >= 1 and bus.max() <= 80
    assert np.all(bike == 1)
    # Truncation at 1 raises the cars' mean with respect to μ = 1.5.
    assert 1.7 < car.mean() < 1.9
    assert abs(bus.mean() - 40.0) < 0.2
    assert abs(bus.std() - 10.0) < 0.3


def test_uniform_ticks_quantized_range():
    rng = np.random.default_rng(1)
    ticks = uniform_ticks(rng, 1.0, 5.0, 50_000)
    assert ticks.min() == round(1.0 / DT)
    assert ticks.max() == round(5.0 / DT)


def test_passenger_pmf_matches_sampler():
    """The expected distribution the plot draws is the one the sampling produces."""
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
    # Cars: μ = 1.5 with minimum 1 → the expected mean rises to ~1.8 because of the truncation.
    values, probs = passenger_pmf(DEFAULT_SPECS[CAR])
    assert 1.75 < (values * probs).sum() < 1.85


def test_lengths_truncated_normal():
    """Freight: normal(10, 2) truncated to [8, 16]; the theoretical mean rises to ≈ 10.57 m because of the cut at 8."""
    spec = VehicleSpec("carga", "carga", 40.0, 10.0, 4.0, 1.5, 1, 1, 1.0, 0.0, True,
                       cargo_prob=1.0, length_std=2.0, length_min=8.0)  # fmt: skip
    assert (spec.shortest, spec.longest) == (8.0, 16.0)
    lengths = sample_lengths(np.random.default_rng(0), spec, 200_000)
    assert lengths.min() >= 8.0 and lengths.max() <= 16.0
    assert abs(lengths.mean() - 10.57) < 0.02
    fixed = sample_lengths(np.random.default_rng(0), DEFAULT_SPECS[CAR], 10)
    np.testing.assert_array_equal(fixed, 4.5)


def test_speeds_truncated_normal():
    """Maximum speed normal(70, 10) truncated to [60, 90]: the mean rises to ≈ 72.3 km/h because of the cut at 60."""
    spec = VehicleSpec("car", "auto", 70.0, 4.5, 4.0, 1.0, 1, 6, 1.5, 0.8, True,
                       speed_std=10.0, speed_min=60.0, speed_max=90.0)  # fmt: skip
    assert (spec.slowest_kmh, spec.fastest_kmh) == (60.0, 90.0)
    speeds = sample_speeds(np.random.default_rng(0), spec, 200_000)
    assert speeds.min() >= 60.0 and speeds.max() <= 90.0
    assert abs(speeds.mean() - 72.3) < 0.1
    fixed = sample_speeds(np.random.default_rng(0), DEFAULT_SPECS[CAR], 10)
    np.testing.assert_array_equal(fixed, DEFAULT_SPECS[CAR].speed_kmh)


def test_reaction_uniform_without_std():
    """Without reaction_std, the reaction is the earlier uniform one (the same draws)."""
    b = Behavior(reaction_min=0.7, reaction_max=3.0)
    np.testing.assert_array_equal(
        reaction_ticks(np.random.default_rng(1), b, 1000), uniform_ticks(np.random.default_rng(1), 0.7, 3.0, 1000)
    )


def test_reaction_truncated_normal():
    b = Behavior(reaction_min=0.7, reaction_max=3.0, reaction_mean=1.5, reaction_std=0.5)
    ticks = reaction_ticks(np.random.default_rng(0), b, 200_000)
    assert ticks.min() >= round(0.7 / DT) and ticks.max() <= round(3.0 / DT)
    # Truncation to [0.7, 3.0] around 1.5 ± 0.5 raises the mean a little (≈ 1.56 s).
    assert abs(ticks.mean() * DT - 1.56) < 0.01
    assert abs(ticks.std() * DT - 0.45) < 0.02
    mid = reaction_ticks(np.random.default_rng(0), Behavior(reaction_min=1, reaction_max=3, reaction_std=0.2), 10_000)
    assert abs(mid.mean() * DT - 2.0) < 0.01  # no mean: the midpoint
    fixed = reaction_ticks(np.random.default_rng(0), Behavior(reaction_mean=2.0, reaction_std=0.0), 5)
    np.testing.assert_array_equal(fixed, round(2.0 / DT))
    tiny = reaction_ticks(np.random.default_rng(0), Behavior(reaction_min=0.01, reaction_max=0.02), 5)
    assert tiny.min() >= 1  # never 0 steps
