import json

import numpy as np
import pytest

from trafico import api
from trafico.cli import CONFIG_NAME, emissions, run, variants
from trafico.results import RESULTS_NAME, SCHEMA, list_runs, plain, read_results

CONFIG = (
    "[road]\nlength = 150.0\nmax_line_speed = [50, 50]\n[traffic_light]\nenabled = false\n"
    "[[speed_bump]]\nposition = 40.0\n"
    "[demand]\ncar_rate = {min = 12, max = 12, mean = 12, std = 0}\nbike_rate = {min = 0, max = 0, mean = 0, std = 0}\n"
    "bus_rate = {min = 0, max = 0, mean = 0, std = 0}\n"
    "[vehicles.car]\naccel = 2.5\ndecel = 4.5\nspeed_bump_kmh = 10.0\n"
    "[execution]\nrun = 6\nreplicas = 2\nworkers = 1\nseed = 7\n[output]\nprogress = false\n"
)


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr("trafico.settings.project_root", lambda: tmp_path)
    (tmp_path / CONFIG_NAME).write_text(CONFIG, encoding="utf-8")
    return tmp_path


def test_plain_makes_json_safe():
    doc = plain({"a": np.float64("nan"), "b": np.array([1.0, np.inf, 2.5]), "c": (np.int64(3), True), 7: None})
    assert doc == {"a": None, "b": [1.0, None, 2.5], "c": [3, True], "7": None}
    json.dumps(doc, allow_nan=False)


def test_simulate_returns_the_results_document(root):
    r = api.simulate()
    assert r.mode == "corrida" and r.folder.parent == root / "resultados" and "Resultados en" in r.log
    doc = r.results
    assert doc["schema"] == SCHEMA and doc["modo"] == "corrida" and doc["semilla"] == 7 and doc["replicas"] == 2
    assert doc["id"] == r.folder.name and doc["comando"] == "uv run trafico"
    assert {"resumen.txt", RESULTS_NAME, CONFIG_NAME} <= set(r.files) and r.path("resumen.txt").is_file()
    car = doc["metricas"]["tipos"]["car"]
    assert car["nombre"] and car["tiempo_recorrido_s"] > 0 and car["vehiculos_cruzan"] > 0
    assert "co2_g_km" in car and car["co2_g_km"] > 0 and car["velocidad_media_kmh"] > 0
    assert len(doc["metricas"]["carriles"]["cruzan_veh_min"]) == 2 and doc["metricas"]["ejecucion"]["tiempo_real_s"] > 0
    text = (r.folder / RESULTS_NAME).read_text(encoding="utf-8")
    assert "NaN" not in text and json.loads(text) == doc  # JSON estricto
    assert read_results(r.folder) == doc


def test_compare_emissions_and_spacing(root):
    r = api.compare_emissions(bumps=["sin", 40, (40, 90)], replicas=1)
    assert r.mode == "emisiones" and r.results["metricas"]["referencia"] == "sin tope"
    names = [s["nombre"] for s in r.results["metricas"]["escenarios"]]
    assert names[0] == "sin tope" and len(names) == 3
    first, second = (s["tipos"]["car"]["co2"] for s in r.results["metricas"]["escenarios"][:2])
    assert first["cambio_pct"] == 0 and second["cambio_pct"] > 0 and second["unidad"] == "g/km"
    assert r.results["metricas"]["escenarios"][2]["topes_m"] == [40.0, 90.0]

    s = api.bump_spacing(distances=[20, 50], chain=3, replicas=1)
    m = s.results["metricas"]
    assert s.mode == "separacion" and m["topes_por_cadena"] == 3 and m["distancias_m"] == [20.0, 50.0]
    co2 = m["contaminantes"]["co2"]
    assert len(co2["costo_tope"]) == 2 and len(co2["aditividad"]) == 2 and "recomendada_m" in m
    json.dumps(s.results, allow_nan=False)


def test_compare_variants(root):
    r = api.compare_variants(lengths=[100.0], lights=[(20, 20), "30/10"], replicas=1)
    m = r.results["metricas"]
    assert r.mode == "variantes" and len(m["escenarios"]) == 2 and m["cola_tipos"] == ["car"]
    assert len(m["escenarios"][0]["velocidad_kmh_por_carril"]) == 2


def test_errors_are_config_errors(root):
    with pytest.raises(api.ConfigError, match="--topes"):
        api.compare_emissions(bumps=["xx"], replicas=1)
    with pytest.raises(api.ConfigError, match="name solo se admite sin config"):
        api.simulate(root, name="x")
    with pytest.raises(api.ConfigError, match="al menos 2 topes"):
        api.bump_spacing(distances=[20], chain=1, replicas=1)
    with pytest.raises(api.ConfigError, match="no existe la carpeta"):
        api.simulate("no/existe")
    with pytest.raises(FileNotFoundError):
        read_results(root / "nada")


def test_api_does_not_print(root, capsys):
    api.compare_emissions(bumps=["sin", 40], replicas=1)
    out = capsys.readouterr()
    assert out.out == "" and out.err == ""


def test_json_flag_prints_only_the_results(root, capsys):
    folder = emissions(["--separacion", "20", "--replicas", "1", "--json"])
    out = capsys.readouterr()
    doc = json.loads(out.out)  # stdout es solo el JSON
    assert doc["modo"] == "separacion" and doc["carpeta"] == str(folder.resolve())
    assert "Separación mínima" in out.err  # lo demás, a stderr
    run(["--json"])
    assert json.loads(capsys.readouterr().out)["modo"] == "corrida"
    variants(["--largos", "100", "--replicas", "1", "--json"])
    assert json.loads(capsys.readouterr().out)["modo"] == "variantes"


def test_list_runs_and_rebuild(root):
    emis = api.compare_emissions(bumps=["sin", 40], replicas=1)
    spacing = api.bump_spacing(distances=[20, 40], replicas=1)
    api.simulate()
    assert {r["modo"] for r in list_runs()} == {"emisiones", "separacion", "corrida"}
    assert [r["modo"] for r in list_runs(mode="separacion")] == ["separacion"]
    assert len(list_runs(limit=2)) == 2 and list_runs()[0]["resultados_json"]
    assert list_runs(root / "no_hay") == []
    # una carpeta anterior sin resultados.json se reconstruye de sus datos, sin escribir nada
    for run_ in (emis, spacing):
        (run_.folder / RESULTS_NAME).unlink()
        doc = read_results(run_.folder)
        assert doc["modo"] == run_.mode and doc["metricas"] == run_.results["metricas"]
        assert not (run_.folder / RESULTS_NAME).exists()
    corrida = api.simulate()
    (corrida.folder / RESULTS_NAME).unlink()
    with pytest.raises(FileNotFoundError, match="vuelve a correrla"):
        read_results(corrida.folder)


def test_redraw_and_describe_config(root):
    r = api.bump_spacing(distances=[20, 40], replicas=1)
    path = api.redraw(r.folder)
    assert path.name == "separacion_topes-2.png" and path.stat().st_size > 10_000
    with pytest.raises(api.ConfigError, match="solo esas se redibujan"):
        api.redraw(api.simulate().folder)
    info = api.describe_config()
    assert info["tramo"]["largo_m"] == 150.0 and info["tramo"]["carriles"] == 2
    assert info["topes"][0]["posicion_m"] == 40.0 and info["ejecucion"]["semilla"] == 7
    assert [t["clave"] for t in info["tipos"]][0] == "car" and info["semaforo"]["activo"] is False
    json.dumps(info, allow_nan=False)
    assert api.describe_config(root)["config"] == str((root / CONFIG_NAME).resolve())
