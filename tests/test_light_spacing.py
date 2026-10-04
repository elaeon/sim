"""`trafico-emisiones --separacion-semaforos` y el desfase de los semáforos (`offset`)."""

import json
from pathlib import Path

import numpy as np
import pytest

from trafico.bump_spacing import (
    DATA_NAME, LIGHT_CYCLE, META_NAME, PROFILE_CSV, SUMMARY_NAME, Spacing, analyze, build_configs, first_position,
    travel_time,
)  # fmt: skip
from trafico.cli import CONFIG_NAME, emissions
from trafico.config import DT, GREEN, POLLUTANTS, RED, YELLOW, Light, SimConfig, SpeedBump
from trafico.emissions import EMIS_BIN
from trafico.pedestrians import fixed_phases
from trafico.results import read_results
from trafico.settings import ConfigError, parse_settings

# ---------------------------------------------------------------- desfase


def test_offset_delays_the_whole_cycle():
    plain = Light(100.0, red=20, green=25, yellow=3)
    late = Light(100.0, red=20, green=25, yellow=3, offset=12.5)
    shift = round(12.5 / DT)
    ticks = np.arange(0, 2000)
    assert [late.phase(int(t)) for t in ticks] == [plain.phase(int(t) - shift) for t in ticks]
    assert np.array_equal(fixed_phases(late, 2000), np.array([late.phase(int(t)) for t in ticks], np.int8))
    assert {late.phase(int(t)) for t in ticks} == {RED, GREEN, YELLOW}
    # Los intervalos en rojo coinciden con las fases, también con el desfase mayor que el rojo y en t = 0.
    for light in (late, Light(100.0, red=20, green=25, yellow=3, offset=40.0, start_phase="green")):
        red = np.array([light.phase(t) == RED for t in range(3000)])
        edges = np.flatnonzero(np.diff(np.r_[False, red, False].astype(int)))
        expected = [(a * DT, b * DT) for a, b in zip(edges[::2], edges[1::2])]
        assert light.red_intervals(300.0) == pytest.approx(expected)
    assert "desfase 12.5 s" in late.label and "desfase" not in plain.label


def _settings(light: str) -> str:
    return f"[road]\nlength = 200.0\n[[traffic_light]]\nposition = 100.0\n{light}\n"


def test_offset_in_the_configuration():
    sim = parse_settings(_settings("red = 20.0\ngreen = 25.0\noffset = 10.0"), Path("x.toml")).sim
    assert sim.extra_lights[0].offset == 10.0 and sim.exit_light.offset == 0.0
    sim = parse_settings("[road]\nlength = 200.0\n[traffic_light]\nred = 20.0\ngreen = 25.0\noffset = 4.0\n", Path("x.toml")).sim
    assert sim.light_offset == 4.0 and sim.exit_light.offset == 4.0
    for text, match in [
        ("red = 20.0\ngreen = 25.0\noffset = 45.0", "offset debe estar entre 0 y el ciclo"),
        ("red = 20.0\ngreen = 25.0\noffset = 0.05", "offset debe ser múltiplo"),
        ("pedestrian = true\noffset = 3.0", "offset no aplica"),
    ]:
        with pytest.raises(ConfigError, match=match):
            parse_settings(_settings(text), Path("x.toml"))


# ---------------------------------------------------------------- escenarios


def _base(**kw) -> SimConfig:
    bump = SpeedBump(40.0, None)
    exit_and_inner = dict(traffic_light=True, extra_lights=(Light(60.0, pedestrian=True),), speed_bumps=(bump,))
    return SimConfig(length=400, lanes=2, rates=(10, 0, 0), run=3, **{**exit_and_inner, **kw})


def _spacing(**kw) -> Spacing:
    args = dict(first=100.0, distances=(50.0, 150.0), bump_lanes=None, light=None, length=None, tolerance=0.05,
                threshold=0.1, replicas=1, run=3.0, kind="semaforo", cycle=Light(0.0, red=30, green=30, yellow=3))  # fmt: skip
    return Spacing(**{**args, **kw})


def test_build_configs_puts_only_the_sweep_lights_with_red_on_arrival():
    configs, skipped = build_configs(_base(), _spacing(distances=(50.0, 150.0, 300.0)))
    assert skipped == [300.0]
    assert [n for n, _ in configs][:3] == ["sin semáforo", "un semáforo a 100 m", "semáforos a 100 y 150 m (d = 50 m)"]
    for _, cfg in configs:  # sin el semáforo del final ni los de la configuración, y sin topes
        assert not cfg.has_light and cfg.speed_bumps == ()
    assert configs[0][1].inner_lights == ()
    pair = configs[3][1].inner_lights
    assert [lt.position for lt in pair] == [100.0, 250.0] and all(lt.start_phase == "red" for lt in pair)
    # El segundo pasa a rojo cuando llega la cabeza del pelotón que salió del primero al ponerse en verde (t = 30 s).
    car = configs[0][1].specs[0]
    v, a = min(car.speed_kmh, 1e9) / 3.6, car.accel
    trip = travel_time(configs[0][1], 150.0)
    assert trip == pytest.approx(150 / v + v / (2 * a)) if a else trip == pytest.approx(150 / v)
    assert pair[1].offset == pytest.approx(round((30.0 + trip) / DT) * DT)
    t = round((30.0 + trip) / DT)
    assert pair[0].phase(round(30.0 / DT) - 1) == RED and pair[0].phase(round(30.0 / DT)) == GREEN
    assert pair[1].phase(t - 2) != RED and pair[1].phase(t + 1) == RED
    # Con d corta la cabeza no llega a su velocidad: viaja acelerando.
    if a:
        assert travel_time(configs[0][1], 5.0) == pytest.approx(np.sqrt(2 * 5.0 / a))


def test_build_configs_chain_offsets_same_phase_and_fixed_bumps():
    configs, _ = build_configs(_base(), _spacing(distances=(50.0,), count=3, bumps=(40.0,)))
    chain = configs[2][1].inner_lights
    step = 30.0 + travel_time(configs[0][1], 50.0)
    assert [lt.offset for lt in chain] == pytest.approx([0.0, round(step, 1), round((2 * step) % 63.0, 1)], abs=0.051)
    assert [b.position for b in configs[2][1].bumps] == [40.0]
    same, _ = build_configs(_base(), _spacing(distances=(50.0,), offset_mode="igual"))
    assert all(lt.offset == 0.0 for lt in same[2][1].inner_lights)
    with pytest.raises(ConfigError, match="--primer-semaforo debe estar entre 0 y 400 m"):
        build_configs(_base(), _spacing(first=400.0))
    with pytest.raises(ConfigError, match="al menos 2 semáforos"):
        build_configs(_base(), _spacing(count=1))


def test_green_wave_and_mixed_chain():
    configs, _ = build_configs(_base(), _spacing(distances=(50.0,), offset_mode="verde"))
    a, b = configs[2][1].inner_lights
    trip = travel_time(configs[0][1], 50.0)
    assert b.offset == pytest.approx(round(trip / DT) * DT)
    # B pasa a verde cuando llega la cabeza del pelotón que salió de A al ponerse en verde (t = 30 s).
    t = round((30.0 + trip) / DT)
    assert b.phase(t - 2) == RED and b.phase(t + 1) == GREEN
    # Cadena mixta: A y B (d) en onda verde, C 150 m después de B en rojo al llegar.
    s = _spacing(distances=(40.0, 60.0), count=3, offset_mode="verde,rojo", fixed_gaps=(150.0,))
    configs, skipped = build_configs(_base(), s)
    assert skipped == [] and configs[2][0] == "semáforos a 100, 140 y 290 m (d = 40 m)"
    a, b, c = configs[2][1].inner_lights
    base = configs[0][1]
    expected_b = travel_time(base, 40.0)
    expected_c = expected_b + 30.0 + travel_time(base, 150.0)
    assert b.offset == pytest.approx(expected_b, abs=0.051) and c.offset == pytest.approx(expected_c % 63.0, abs=0.051)
    assert [lt.position for lt in configs[3][1].inner_lights] == [100.0, 160.0, 310.0]
    for bad in (dict(fixed_gaps=(150.0, 20.0)), dict(offset_mode="verde,rojo,rojo"), dict(offset_mode="azul")):
        with pytest.raises(ConfigError, match="--separaciones-fijas|--desfase"):
            build_configs(_base(), _spacing(distances=(40.0,), count=3, **{"offset_mode": "rojo", **bad}))


# ---------------------------------------------------------------- análisis

BINS = 120  # 600 m
X = (np.arange(BINS) + 0.5) * EMIS_BIN
FIRST = 100.0
BEFORE, AFTER = 30.0, 40.0  # m de cola y de estela


def _footprint(at: float, height: float = 600.0) -> np.ndarray:
    """Exceso de un semáforo: meseta de la cola (BEFORE m antes) y estela triangular de AFTER m después."""
    queue = np.where((X >= at - BEFORE) & (X < at), 0.5 * height, 0.0)
    wake = height * np.clip(1 - (X - at) / AFTER, 0, 1) * (X >= at)
    return queue + wake


def _synthetic(distances, far_factor: float = 1.1, near_extra: dict | None = None, entry: tuple = ()) -> tuple[dict, dict]:
    """El segundo semáforo emite `far_factor` × uno aislado (todo el pelotón se detiene); con d en `near_extra`, la
    pareja emite además ese exceso uniforme; con d en `entry`, la cola llega a la entrada."""
    base = np.full(BINS, 100.0)
    profs = [base, base + _footprint(FIRST)]
    for d in distances:
        p = base + _footprint(FIRST) + far_factor * _footprint(FIRST + d)
        p += (near_extra or {}).get(d, 0.0) * (X < FIRST + d)
        if d in entry:
            p[0] += 1000.0
        profs.append(p)
    pos = np.zeros((len(profs), len(POLLUTANTS), 1, BINS))
    for i, prof in enumerate(profs):
        pos[i, POLLUTANTS.index("co2"), 0] = prof * EMIS_BIN
    meta = {"contaminantes": ["co2"], "primer_semaforo_m": FIRST, "distancias_m": list(distances), "tolerancia": 0.05,
            "umbral": 0.1, "s_simulados": 3600.0, "topes_por_cadena": 2, "elemento": "semaforo"}  # fmt: skip
    tt = np.array([[20.0], [30.0], *([[40.0 + 0 * d] for d in distances])])
    return meta, {"emissions_pos": pos, "travel_time": tt, "crossed_veh": np.ones_like(tt)}


def test_two_sided_footprint_and_stable_separation():
    dist = [50.0, 80.0, 120.0, 200.0]
    meta, data = _synthetic(dist, near_extra={50.0: 150.0})
    assert first_position(meta) == FIRST
    r = analyze(meta, data)
    co2 = r["pollutants"]["co2"]
    assert co2["huella_antes_m"] == pytest.approx(BEFORE + EMIS_BIN / 2)  # centro del intervalo libre anterior
    assert co2["huella_m"] == pytest.approx(37.5)
    # Fuera de la huella: d ≥ antes + después; la pareja lejana emite 1.05 × dos aislados (no 1): se compara con ella.
    assert co2["d_huella"] == 80.0
    assert co2["aditividad"][-1] == pytest.approx(1.05, abs=0.01)
    assert co2["estabilidad"][0] > 1.05 and co2["estabilidad"][1:] == pytest.approx(1.0, abs=0.01)
    assert co2["d_aditiva"] == 80.0 and co2["d_minima"] == 80.0 and r["recomendada_m"] == 80.0
    # Demora: (40 − 20) / (2 × (30 − 20)) = 1.
    assert np.allclose(r["demora"], 1.0) and not r["cola_en_entrada"]


def test_entry_blocking_and_no_plateau():
    meta, data = _synthetic([80.0, 120.0, 200.0], entry=(80.0,))
    r = analyze(meta, data)
    assert list(r["cola_entrada_d"]) == [True, False, False]
    assert r["pollutants"]["co2"]["d_aditiva"] == 120.0  # la de la cola en la entrada no cuenta
    # Si solo la más separada cumple (no hay meseta), no hay separación estable.
    meta, data = _synthetic([80.0, 120.0], near_extra={80.0: 150.0, 120.0: 150.0})
    assert analyze(meta, data)["pollutants"]["co2"]["d_aditiva"] is None


# ---------------------------------------------------------------- comando


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr("trafico.settings.project_root", lambda: tmp_path)
    (tmp_path / CONFIG_NAME).write_text(
        "[road]\nlength = 150.0\nmax_line_speed = [50, 50]\n[traffic_light]\nenabled = false\n"
        "[[speed_bump]]\nposition = 40.0\n"
        "[demand]\ncar_rate = {min = 12, max = 12, mean = 12, std = 0}\nbike_rate = {min = 0, max = 0, mean = 0, std = 0}\n"
        "bus_rate = {min = 0, max = 0, mean = 0, std = 0}\n"
        "[vehicles.car]\naccel = 2.5\ndecel = 4.5\nspeed_bump_kmh = 10.0\n"
        "[execution]\nrun = 6\nreplicas = 2\nworkers = 2\nseed = 7\n[output]\nprogress = false\n",
        encoding="utf-8",
    )
    return tmp_path


def test_command_runs_the_light_sweep_and_redraws(root, capsys):
    folder = emissions(["--separacion-semaforos", "40", "100", "--replicas", "1", "--ciclo", "20/20/2"])
    names = {p.name for p in folder.iterdir()}
    assert {"separacion_semaforos.png", "separacion_semaforos.csv", PROFILE_CSV, DATA_NAME, META_NAME, SUMMARY_NAME,
            "config_base.toml", "resultados.json"} <= names  # fmt: skip
    meta = json.loads((folder / META_NAME).read_text())
    assert meta["elemento"] == "semaforo" and meta["primer_semaforo_m"] == 100.0 and meta["desfase"] == "rojo"
    assert meta["ciclo"] == {"rojo_s": 20.0, "verde_s": 20.0, "amarillo_s": 2.0} and meta["topes_fijos_m"] == []
    assert meta["tipo_peloton"] and len(meta["viaje_s"]) == 2 and meta["viaje_s"][0] < meta["viaje_s"][1]
    assert meta["largo_m"] == 350.0  # 100 + 100 + 150, ampliado desde 150 m
    assert meta["escenarios"][:2] == ["sin semáforo", "un semáforo a 100 m"]
    out = capsys.readouterr().out
    assert "el tramo se amplía de 150 a 350 m" in out and "Separación mínima entre semáforos" in out
    header = (folder / "separacion_semaforos.csv").read_text().splitlines()[0]
    assert "exceso_entre_semaforos_pct_pico" in header and header.endswith("demora_x_aislados")
    doc = read_results(folder)
    assert doc["modo"] == "separacion" and doc["metricas"]["elemento"] == "semaforo"
    assert "huella_antes_m" in doc["metricas"] and len(doc["metricas"]["cola_hasta_la_entrada"]) == 2
    emissions(["--redibujar", str(folder)])
    assert (folder / "separacion_semaforos-2.png").stat().st_size > 10_000


def test_command_mixed_chain(root, capsys):
    folder = emissions(["--separacion-semaforos", "40", "--cadena", "3", "--separaciones-fijas", "120", "--desfase",
                        "verde,rojo", "--replicas", "1", "--ciclo", "20/20/2"])  # fmt: skip
    meta = json.loads((folder / META_NAME).read_text())
    assert meta["desfase"] == "verde,rojo" and meta["separaciones_fijas_m"] == [120.0]
    assert meta["escenarios"][2] == "semáforos a 100, 140 y 260 m (d = 40 m)" and meta["largo_m"] == 410.0
    summary = (folder / SUMMARY_NAME).read_text()
    assert "2.º en verde al llegar el pelotón (onda verde), 3.º en rojo al llegar el pelotón" in summary
    assert "≡ aislados" in summary and "La cadena de 3 emite como" in summary and "Separación mínima" not in summary


def test_command_light_sweep_with_fixed_bumps_and_given_length(root):
    folder = emissions(["--separacion-semaforos", "40", "200", "--replicas", "1", "--topes", "40", "--largo", "250",
                        "--desfase", "igual"])  # fmt: skip
    meta = json.loads((folder / META_NAME).read_text())
    assert meta["topes_fijos_m"] == [40.0] and meta["largo_m"] == 250.0 and meta["omitidas_m"] == [200.0]
    assert meta["desfase"] == "igual"
    assert meta["ciclo"]["rojo_s"] == LIGHT_CYCLE.red


@pytest.mark.parametrize(
    "args",
    [
        ["--separacion", "20", "--separacion-semaforos", "50"],
        ["--ciclo", "30/30"],
        ["--desfase", "rojo"],
        ["--primer-semaforo", "50"],
        ["--separacion-semaforos", "50", "--semaforo", "si"],
        ["--separacion-semaforos", "50", "--ciclo", "30"],
        ["--separacion-semaforos", "50", "--topes", "40", "sin"],
        ["--separacion-semaforos", "0"],
        ["--separacion-semaforos", "50", "--desfase", "azul"],
        ["--separacion-semaforos", "50", "--separaciones-fijas", "100"],
        ["--separacion-semaforos", "50", "--cadena", "3", "--desfase", "verde,rojo,rojo"],
        ["--separaciones-fijas", "100"],
    ],
)
def test_command_rejects_invalid_light_options(root, args):
    with pytest.raises(SystemExit):
        emissions(args)
