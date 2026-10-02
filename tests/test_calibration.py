import json
from dataclasses import replace

import numpy as np
import pytest

from trafico import api, calibration
from trafico.calibration import REFERENCES, compare, measure
from trafico.cli import CONFIG_NAME
from trafico.config import DEFAULT_SPECS, Behavior, SimConfig
from trafico.settings import ConfigError

CAR = DEFAULT_SPECS[0]
FAST = dict(queues=4, rates=(20, 120), flow_seconds=240.0, workers=1)


def _cfg(spec=CAR, reaction=1.5, **kw) -> SimConfig:
    b = Behavior(reaction_min=reaction, reaction_max=reaction)
    return SimConfig(specs=(spec, *DEFAULT_SPECS[1:]), rates=(10, 2, 1), lane_speed_limit=50.0, behavior=b, **kw)


def test_measurements_are_consistent():
    """Sin aceleración ni frenado graduales: con poca demanda todos van a 50 km/h (flujo / densidad), la densidad de
    embotellamiento es 1000 / (largo + gap_stop) y, en marcha a gap_run fijo, la capacidad queda muy por encima de la
    de referencia (más de 5,000 veh/h)."""
    spec = replace(CAR, speed_kmh=50.0, speed_std=0.0, length_std=0.0, accel=None, decel=None)
    m = measure(_cfg(spec), "car", **FAST)
    assert m["velocidad_kmh"][0] == pytest.approx(50.0, rel=0.03)
    assert m["densidad_embotellamiento_veh_km"] == pytest.approx(1000 / (spec.length + spec.gap_stop))
    assert m["capacidad_marcha_veh_h"] > 5000 and m["flujo_veh_h"][1] == m["capacidad_marcha_veh_h"]
    assert m["intervalo_marcha_s"] == pytest.approx(3600 / m["capacidad_marcha_veh_h"])
    # Con reacción fija, cada auto arranca su reacción después de que se mueve el de adelante: el intervalo de
    # saturación es al menos la reacción.
    assert 1.5 <= m["intervalo_saturacion_s"] < 2.5
    assert m["flujo_saturacion_veh_h"] == pytest.approx(3600 / m["intervalo_saturacion_s"])
    assert len(m["intervalo_por_posicion_s"]) == calibration.QUEUE and m["colas"] == 4


def test_slower_reaction_lowers_saturation_flow():
    spec = replace(CAR, speed_std=0.0, length_std=0.0)
    fast = measure(_cfg(spec, reaction=1.0), "car", **FAST)
    slow = measure(_cfg(spec, reaction=2.5), "car", **FAST)
    assert slow["flujo_saturacion_veh_h"] < fast["flujo_saturacion_veh_h"] - 300


def test_compare_flags_values_out_of_range_only_for_cars():
    m = {"tipo": "car", **{k: (r.low + r.high) / 2 for k, r in REFERENCES.items()}}
    m["capacidad_marcha_veh_h"] = 4000.0
    m["flujo_saturacion_veh_h"] = 1000.0
    states = {r["medida"]: r["estado"] for r in compare(m)}
    assert states["capacidad_marcha_veh_h"] == "arriba" and states["flujo_saturacion_veh_h"] == "abajo"
    assert states["tiempo_perdido_s"] == "dentro"
    assert all(r["estado"] is None for r in compare({**m, "tipo": "bike"}))


def test_measure_errors():
    with pytest.raises(ConfigError, match="no existe el tipo"):
        measure(_cfg(), "avion", **FAST)
    with pytest.raises(ConfigError, match="al menos 2 colas"):
        measure(_cfg(), "car", **{**FAST, "queues": 1})


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr("trafico.settings.project_root", lambda: tmp_path)
    monkeypatch.setattr(calibration, "FLOW_RATES", (20, 120))
    monkeypatch.setattr(calibration, "FLOW_SECONDS", 240.0)
    (tmp_path / CONFIG_NAME).write_text(
        "[road]\nlength = 150.0\nmax_line_speed = [50, 50]\n[behavior]\nreaction = {min = 1, max = 2}\n"
        "[execution]\nreplicas = 2\nworkers = 1\nseed = 7\n[output]\nprogress = false\n",
        encoding="utf-8",
    )
    return tmp_path


def test_command_and_api(root, capsys):
    r = api.calibrate(queues=3)
    assert r.mode == "calibracion" and r.folder.parent == root / "resultados" and r.folder.name.endswith("_calibracion")
    names = set(r.files)
    assert {calibration.PLOT_NAME, calibration.TABLE_CSV, calibration.QUEUE_CSV, calibration.FLOW_CSV,
            calibration.SUMMARY_NAME, "config_base.toml", "resultados.json"} <= names  # fmt: skip
    m = r.results["metricas"]
    assert r.results["semilla"] == 7 and m["tipo"] == "car" and m["colas"] == 3
    assert [x["medida"] for x in m["medidas"]] == list(REFERENCES)
    assert all(x["estado"] in ("dentro", "abajo", "arriba") for x in m["medidas"])
    assert m["demanda_veh_min"] == [20, 120] and len(m["flujo_veh_h"]) == 2
    json.dumps(r.results, allow_nan=False)
    assert (r.folder / calibration.PLOT_NAME).stat().st_size > 10_000
    assert "Capacidad en flujo continuo" in (r.folder / calibration.SUMMARY_NAME).read_text()
    assert api.list_runs(mode="calibracion")[0]["id"] == r.folder.name
    with pytest.raises(api.ConfigError, match="no existe el tipo"):
        api.calibrate(vehicle="avion", queues=3)
    from trafico.cli import calibrate

    capsys.readouterr()
    calibrate(["--colas", "3", "--tipo", "bike", "--json"])
    doc = json.loads(capsys.readouterr().out)
    assert doc["modo"] == "calibracion" and doc["metricas"]["tipo"] == "bike"
    assert all(x["estado"] is None for x in doc["metricas"]["medidas"])  # referencias solo de autos
    assert np.isfinite(doc["metricas"]["capacidad_marcha_veh_h"])
