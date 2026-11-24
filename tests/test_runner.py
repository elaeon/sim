import numpy as np

from traffic.config import SimConfig
from traffic.metrics import SERIES, RunningStats
from traffic.runner import run_parallel


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
    from traffic.plotting import _duration

    assert _duration(1000) == "16 min 40 s"
    assert _duration(5000) == "1 h 23 min"
    assert _duration(7200) == "2 h"
    assert _duration(3599.9) == "1 h"  # rounds before choosing the unit
    assert _duration(120) == "2 min"
    assert _duration(20) == "20 s"
    assert _duration(360_000) == "100 h"


def test_rate_label():
    from traffic.plotting import _rate_label

    assert _rate_label(20) == "20/min"
    assert _rate_label(1) == "1/min"
    assert _rate_label(21.1482) == "21.1/min"  # actual mean of a variable rate
    assert _rate_label(0.15) == "1 every 7 min"
    assert _rate_label(0.2) == "1 every 5 min"
    assert _rate_label(0.9) == "1 every 1 min"


def test_lane_label_shows_fixed_lanes():
    import dataclasses

    from traffic.config import DEFAULT_SPECS, SimConfig
    from traffic.plotting import _lane_assignments, _lane_label

    car, bike, bus = DEFAULT_SPECS
    cfg = SimConfig(lanes=3, specs=(car, dataclasses.replace(bike, lane=0), dataclasses.replace(bus, lane=1)))
    assigned = _lane_assignments(cfg, [0, 1, 2])
    assert assigned == [["bici"], ["autobús"], []]
    assert _lane_label(0, 3, assigned[0]) == "lane 0 (right, bici)"
    assert _lane_label(1, 3, assigned[1]) == "lane 1 (autobús)"
    assert _lane_label(2, 3, assigned[2]) == "lane 2 (left)"
    assert _lane_assignments(cfg, [0, 2]) == [[], ["autobús"], []]  # the bike does not take part
    cfg = dataclasses.replace(cfg, specs=(car, dataclasses.replace(bike, lane=0, exclusive=True), bus))
    assert _lane_label(0, 3, _lane_assignments(cfg, [0, 1, 2])[0]) == "lane 0 (right, only bici)"


def test_lane_colors_are_fixed():
    from traffic.plotting import TYPE_COLORS, _lane_colors

    assert _lane_colors(4) == list(TYPE_COLORS[4:])  # pink, green, violet, red
    assert _lane_colors(2) == _lane_colors(4)[:2]  # a lane's colour does not change with the total
    assert len(set(_lane_colors(8))) == 8
    assert len(set(_lane_colors(12))) == 12


def test_passenger_histogram_matches_arrivals():
    from traffic.config import DEFAULT_SPECS

    cfg = SimConfig(run=20)
    agg = run_parallel(cfg, 3, 1, 5)
    np.testing.assert_array_equal(agg.pax_hist.sum(axis=1), agg.summary["arrived_veh"].mean * 3)
    for k, spec in enumerate(DEFAULT_SPECS):  # only values within the configured range
        outside = np.ones(agg.pax_hist.shape[1], bool)
        outside[spec.pax_min : spec.pax_max + 1] = False
        assert agg.pax_hist[k, outside].sum() == 0


def test_passenger_histogram_bins():
    from traffic.plotting import MAX_PAX_BARS, _pax_bin_width, _pax_bins

    assert _pax_bin_width(6) == 1 and _pax_bin_width(30) == 1
    assert _pax_bin_width(120) == 5  # bus [1, 120] → 24 bins of 5
    assert all(-(-n // _pax_bin_width(n)) <= MAX_PAX_BARS for n in range(1, 256))
    values = np.arange(1, 13)
    lo, hi, total = _pax_bins(values, np.ones(12), 5)
    assert lo.tolist() == [1, 6, 11] and hi.tolist() == [5, 10, 12] and total.tolist() == [5, 5, 2]


def test_lane_speed_histogram_counts_only_samples_with_vehicles():
    from traffic.metrics import SPEED_BIN, Recorder, lane_speed_histogram

    cfg = SimConfig(lanes=2)
    rec = Recorder(4, cfg.n_types, cfg.lanes)
    zeros = np.zeros(cfg.n_types)
    # Lane 0: 10 m in 1 vehicle·s = 36 km/h, then stopped (0 km/h). Lane 1: empty except for one sample.
    for dist, time in (([10, 0], [1, 0]), ([0, 0], [2, 0]), ([0, 5], [0, 1]), ([0, 0], [0, 0])):
        rec.record(zeros, zeros, zeros, zeros, np.zeros(2), np.array(dist, float), np.array(time, float), np.zeros(2), np.zeros(2), np.zeros(2))
    hist = lane_speed_histogram(rec, cfg)
    assert hist[0].sum() == 2 and hist[0, 0] == 1 and hist[0, int(36 / SPEED_BIN)] == 1
    assert hist[1].sum() == 1 and hist[1, int(18 / SPEED_BIN)] == 1


def test_outputs_without_traffic_light(tmp_path):
    """Without a traffic light the plot, the CSV and the space-time diagram are written, without phases or traffic light texts."""
    from traffic.cli import write_csv
    from traffic.movement import plot_space_time, record
    from traffic.plotting import _titles, phase_handles, plot_mobility

    cfg = SimConfig(length=150, lanes=2, rates=(20, 4, 1), run=10, traffic_light=False)
    agg = run_parallel(cfg, 2, 1, 5)
    plot_mobility(agg, tmp_path / "movilidad.png")
    write_csv(agg, tmp_path / "series.csv")
    plot_space_time(record(cfg, 5, 1, 0, 30), tmp_path / "espacio.png", "prueba")
    assert (tmp_path / "movilidad.png").stat().st_size > 10_000 and (tmp_path / "espacio.png").exists()
    assert phase_handles(cfg) == [] and "60 s window" in _titles(cfg)["pax_flow"]
    assert "traffic light" not in " ".join(_titles(cfg).values())


def test_emission_outputs(tmp_path):
    """With cars that emit: summary rows, acceleration table, CSV columns, per-position CSV and figure, and the
    emissions row in the plot. Without types that emit, none of that."""
    from dataclasses import replace

    from traffic.cli import format_summary, write_csv, write_emissions_csv
    from traffic.config import DEFAULT_SPECS
    from traffic.plotting import plot_emissions_by_position, plot_mobility

    car = replace(DEFAULT_SPECS[0], accel=2.5, decel=4.5)
    cfg = SimConfig(length=150, lanes=2, rates=(10, 2, 0), run=10, specs=(car, *DEFAULT_SPECS[1:]),
                    fuel_price=(("gasolina", 24.0),))  # fmt: skip
    agg = run_parallel(cfg, 2, 1, 5)
    text = format_summary(agg)
    assert "CO2 (g/km)" in text and "NOx excess vs free flow (%)" in text and "PM" not in text
    assert "0→20 km/h in 2.2 s" in text
    # Fuel from CO2: L/100 km = g/km of CO2 ÷ g/L × 100, and its cost at the petrol price.
    assert "Fuel (L/100 km)" in text and "Cost per trip (MXN)" in text
    assert "40 km/h: 7.6 L/100 km (13.1 km/L, 1.83 MXN/km) · physical 4.8 L/100 km" in text
    assert "60 km/h:" not in text  # the built-in car does not go above 50 km/h
    write_csv(agg, tmp_path / "series.csv")
    header = (tmp_path / "series.csv").read_text().splitlines()[0]
    assert "emis_co2_g_min_auto_mean" in header and "fuel_l_min_auto_mean" in header
    assert "fuel_l_min_bici" not in header
    write_emissions_csv(agg, tmp_path / "pos.csv")
    header = (tmp_path / "pos.csv").read_text().splitlines()[0]
    assert header.startswith("x_start_m,x_end_m,co2_g_m_h_lane0")
    plot_emissions_by_position(agg, tmp_path / "pos.png")
    plot_mobility(agg, tmp_path / "mov.png")
    assert (tmp_path / "pos.png").stat().st_size > 10_000
    quiet = run_parallel(SimConfig(length=150, lanes=2, rates=(10, 2, 0), run=10), 1, 1, 5)
    assert "CO2" not in format_summary(quiet) and "Combustible" not in format_summary(quiet)


def test_run_many_matches_separate_runs_in_any_worker_count():
    """A single process pool for several configurations gives the same as running each one separately (the same
    seeds), serially or in parallel: serially, each configuration gets new seeds (a SeedSequence counts
    the children it derives and the replica derives its own)."""
    from traffic.config import SpeedBump
    from traffic.runner import run_many

    cfgs = [SimConfig(length=150, lanes=2, run=3), SimConfig(length=150, lanes=2, run=3, speed_bumps=(SpeedBump(60.0),))]
    alone = [run_parallel(cfg, 2, 1, 9) for cfg in cfgs]
    calls = []
    for workers in (1, 3):
        together = run_many(cfgs, 2, workers, 9, progress=lambda i, d, n: calls.append((i, d, n)))
        assert [a.replicas for a in together] == [2, 2] and together[0].workers == workers
        for a, b in zip(alone, together):
            for key, stats in a.summary.items():
                np.testing.assert_array_equal(stats.mean, b.summary[key].mean)
    assert calls[:4] == [(0, 1, 2), (0, 2, 2), (1, 1, 2), (1, 2, 2)]  # progress in order
