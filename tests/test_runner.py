import numpy as np

from trafico.config import SimConfig
from trafico.metrics import SERIES, RunningStats
from trafico.runner import run_parallel


def test_running_stats_matches_numpy_with_nan():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(9, 4, 3))
    data[2, 1, 0] = np.nan
    data[5, 3, 2] = np.nan
    stats = RunningStats((4, 3))
    for row in data:
        stats.push(row)
    np.testing.assert_allclose(stats.mean, np.nanmean(data, axis=0))
    np.testing.assert_allclose(stats.std, np.nanstd(data, axis=0, ddof=1))


def test_results_do_not_depend_on_worker_count():
    cfg = SimConfig(run=3)
    serial = run_parallel(cfg, replicas=4, workers=1, seed=123)
    parallel = run_parallel(cfg, replicas=4, workers=3, seed=123)
    assert serial.replicas == parallel.replicas == 4
    for key in SERIES:
        np.testing.assert_array_equal(serial.series[key].mean, parallel.series[key].mean)
        np.testing.assert_array_equal(serial.series[key].std, parallel.series[key].std)
    for key, stats in serial.summary.items():
        np.testing.assert_array_equal(stats.mean, parallel.summary[key].mean)


def test_simulated_duration_label():
    from trafico.plotting import _duration

    assert _duration(1000) == "16 min 40 s"
    assert _duration(5000) == "1 h 23 min"
    assert _duration(7200) == "2 h"
    assert _duration(3599.9) == "1 h"  # redondea antes de elegir la unidad
    assert _duration(120) == "2 min"
    assert _duration(20) == "20 s"
    assert _duration(360_000) == "100 h"


def test_rate_label():
    from trafico.plotting import _rate_label

    assert _rate_label(20) == "20/min"
    assert _rate_label(1) == "1/min"
    assert _rate_label(0.15) == "1 cada 7 min"
    assert _rate_label(0.2) == "1 cada 5 min"
    assert _rate_label(0.9) == "1 cada 1 min"


def test_lane_label_shows_fixed_lanes():
    import dataclasses

    from trafico.config import DEFAULT_SPECS, SimConfig
    from trafico.plotting import _lane_assignments, _lane_label

    car, bike, bus = DEFAULT_SPECS
    cfg = SimConfig(lanes=3, specs=(car, dataclasses.replace(bike, lane=0), dataclasses.replace(bus, lane=1)))
    assigned = _lane_assignments(cfg, [0, 1, 2])
    assert assigned == [["bici"], ["autobús"], []]
    assert _lane_label(0, 3, assigned[0]) == "carril 0 (derecho, bici)"
    assert _lane_label(1, 3, assigned[1]) == "carril 1 (autobús)"
    assert _lane_label(2, 3, assigned[2]) == "carril 2 (izquierdo)"
    assert _lane_assignments(cfg, [0, 2]) == [[], ["autobús"], []]  # la bici no participa
    cfg = dataclasses.replace(cfg, specs=(car, dataclasses.replace(bike, lane=0, exclusive=True), bus))
    assert _lane_label(0, 3, _lane_assignments(cfg, [0, 1, 2])[0]) == "carril 0 (derecho, solo bici)"


def test_lane_colors_are_fixed():
    from trafico.plotting import TYPE_COLORS, _lane_colors

    assert _lane_colors(4) == list(TYPE_COLORS[4:])  # rosa, verde, violeta, rojo
    assert _lane_colors(2) == _lane_colors(4)[:2]  # el color de un carril no cambia con el total
    assert len(set(_lane_colors(8))) == 8
    assert len(set(_lane_colors(12))) == 12


def test_passenger_histogram_matches_arrivals():
    from trafico.config import DEFAULT_SPECS

    cfg = SimConfig(run=20)
    agg = run_parallel(cfg, 3, 1, 5)
    np.testing.assert_array_equal(agg.pax_hist.sum(axis=1), agg.summary["arrived_veh"].mean * 3)
    for k, spec in enumerate(DEFAULT_SPECS):  # solo valores dentro del rango configurado
        outside = np.ones(agg.pax_hist.shape[1], bool)
        outside[spec.pax_min : spec.pax_max + 1] = False
        assert agg.pax_hist[k, outside].sum() == 0


def test_passenger_histogram_bins():
    from trafico.plotting import MAX_PAX_BARS, _pax_bin_width, _pax_bins

    assert _pax_bin_width(6) == 1 and _pax_bin_width(30) == 1
    assert _pax_bin_width(120) == 5  # autobús [1, 120] → 24 intervalos de 5
    assert all(-(-n // _pax_bin_width(n)) <= MAX_PAX_BARS for n in range(1, 256))
    values = np.arange(1, 13)
    lo, hi, total = _pax_bins(values, np.ones(12), 5)
    assert lo.tolist() == [1, 6, 11] and hi.tolist() == [5, 10, 12] and total.tolist() == [5, 5, 2]
