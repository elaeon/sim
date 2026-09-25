import math
import re
import tomllib
from datetime import datetime
from pathlib import Path

import pytest

from trafico.cli import CSV_NAME, PAX_PLOT_NAME, PLOT_NAME, SUMMARY_NAME, run
from trafico.config import BUS, DEFAULT_SPECS, MAX_TYPES, SimConfig
from trafico.settings import (
    CONFIG_NAME,
    ConfigError,
    Target,
    config_copy_text,
    make_run_dir,
    parse_settings,
    project_root,
    resolve_target,
)

NOW = datetime(2026, 9, 24, 8, 30, 0)


def _named(run_dir: Path, name: str) -> bool:
    """La carpeta es <fecha-hora>_<name>, con un sufijo -N si otra corrida cayó en el mismo segundo."""
    return re.fullmatch(rf"\d{{8}}-\d{{6}}_{re.escape(name)}(-\d+)?", run_dir.name) is not None


def _parse(text: str):
    return parse_settings(text, Path("prueba.toml"))


SCOOTER = """
[demand]
scooter_rate = 6
[vehicles.scooter]
speed_kmh = 25
length = 1.5
gap_run = 1.5
gap_stop = 0.5
pax_min = 1
pax_max = 1
pax_mean = 1
pax_std = 0
"""


@pytest.mark.parametrize(
    "path", sorted(p for p in project_root().glob("*.toml") if p.name != "pyproject.toml"), ids=lambda p: p.name
)
def test_project_scenarios_are_valid(path):
    parse_settings(path.read_text(encoding="utf-8"), path)


def test_without_vehicles_section_uses_builtin_types():
    assert _parse("[road]\nlength = 200\n").sim == SimConfig()  # sin listas por carril: 2 carriles


def test_missing_keys_take_defaults_and_overrides_apply():
    s = _parse('[road]\nmax_line_speed = [50, 50, 50]\n[vehicles.bus]\ngap_run = 5\n[execution]\nworkers = 2\n')
    assert s.sim.lanes == 3 and s.sim.length == SimConfig().length
    assert s.sim.specs[BUS].gap_run == 5.0  # entero aceptado como número
    assert s.run.workers == 2 and s.run.seed is None


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("[road]\nlenght = 200\n", "claves desconocidas: [road] lenght"),
        ("[road]\nmax_line_speed = [40, true]\n", "[road] max_line_speed debe ser un número o una lista"),
        ("[road]\nmax_line_speed = [40, 0]\n", "[road] max_line_speed: cada valor debe ser mayor que 0"),
        (
            "[road]\nmax_line_speed = [40, 50, 40]\n[behavior]\ncongestion_factor = [0.2, 0.3]\n",
            "las listas por carril deben tener el mismo largo",
        ),
        ("[output]\ncsv = 1\n", "[output] csv debe ser true o false"),
        ("[road]\nlength = 10\n", "[road] length debe ser al menos"),
        ("[demand]\ncar_rate = 0\nbike_rate = 0\nbus_rate = 0\n", "al menos un tipo"),
        ("[traffic_light]\nred = 12.35\n", "múltiplo de 0.1"),
        ("[vehicles.car]\npax_mean = 9\n", "pax_mean debe estar entre"),
        ("[behavior]\ncongestion_factor = 1.0\n", "congestion_factor: cada valor debe estar en"),
        ("[behavior]\ncongestion_factor = [0.2, 1.0]\n", "congestion_factor: cada valor debe estar en"),
        ("[behavior]\ncongestion_factor = []\n", "congestion_factor debe ser un número o una lista"),
        ("road = 3\n[road.x]\n", "TOML inválido"),
    ],
)
def test_invalid_config_is_rejected(text, message):
    with pytest.raises(ConfigError, match=message.replace("[", r"\[").replace("]", r"\]")):
        _parse(text)


def test_new_vehicle_type_from_config_only():
    s = _parse(SCOOTER)
    assert [spec.key for spec in s.sim.specs] == ["car", "bike", "bus", "scooter"]
    scooter = s.sim.specs[-1]
    assert s.sim.rates[-1] == 6.0
    assert scooter.name == "scooter"  # por defecto, la clave
    assert scooter.can_change_lane  # por defecto, puede cambiar de carril
    assert scooter.speed_kmh == 25.0 and scooter.pax_max == 1

    named = _parse(SCOOTER + 'name = "patineta"\nlane_change = false\n').sim.specs[-1]
    assert named.name == "patineta" and not named.can_change_lane


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (SCOOTER.replace("pax_std = 0\n", ""), "falta la clave obligatoria [vehicles.scooter] pax_std"),
        (SCOOTER.replace("scooter_rate = 6\n", ""), "falta la clave obligatoria [demand] scooter_rate"),
        ("[demand]\ntruck_rate = 2\n", "define también su sección [vehicles.truck]"),
        (SCOOTER + 'name = "bici"\n', "se repite bici"),
        ("[vehicles]\nscooter = 3\n", "debe ser una sección [vehicles.scooter]"),
    ],
)
def test_invalid_new_vehicle_is_rejected(text, message):
    with pytest.raises(ConfigError, match=message.replace("[", r"\[").replace("]", r"\]")):
        _parse(text)


@pytest.mark.parametrize(
    ("text", "lanes", "limits", "factors"),
    [
        ("[behavior]\ncongestion_factor = [0.2, 0.3, 0.5]\n", 3, (math.inf,) * 3, (0.2, 0.3, 0.5)),
        ("[road]\nmax_line_speed = [40, 50, 40, 60]\n", 4, (40, 50, 40, 60), (0.0,) * 4),
        (
            "[road]\nmax_line_speed = [40, 50, 40]\n[behavior]\ncongestion_factor = [0.2, 0.3, 0.5]\n",
            3, (40, 50, 40), (0.2, 0.3, 0.5),
        ),
        ("[road]\nmax_line_speed = 30\n[behavior]\ncongestion_factor = [0.1, 0.2]\n", 2, (30, 30), (0.1, 0.2)),
        ("[behavior]\ncongestion_factor = 0.4\n", 2, (math.inf,) * 2, (0.4, 0.4)),  # sin listas: 2 carriles
    ],
)  # fmt: skip
def test_lanes_come_from_per_lane_lists(text, lanes, limits, factors):
    s = _parse(text)
    assert s.sim.lanes == lanes
    assert s.sim.lane_max_kmh == limits
    assert s.sim.lane_congestion == factors
    assert s.notices == ()


@pytest.mark.parametrize(
    ("lanes", "expected"),
    [(3, (0.2, 0.3, 0.5)), (5, (0.2, 0.3, 0.5, 0.5, 0.5)), (2, (0.2, 0.3)), (1, (0.2,))],
)
def test_legacy_lanes_key_still_replicates_old_copies(lanes, expected):
    """[road] lanes está obsoleto, pero las copias anteriores lo tienen: se acepta con un aviso
    y las listas se ajustan a él."""
    s = _parse(f"[road]\nlanes = {lanes}\n[behavior]\ncongestion_factor = [0.2, 0.3, 0.5]\n")
    assert s.sim.lanes == lanes
    assert s.sim.lane_congestion == expected
    assert "[road] lanes está obsoleto" in s.notices[0]


def test_congestion_notice_when_list_does_not_match_lanes():
    from trafico.cli import _congestion_lines

    lines = _congestion_lines(_parse("[road]\nlanes = 4\n[behavior]\ncongestion_factor = [0.2, 0.3]\n").sim)
    assert lines[0] == "Congestión por carril (0 = derecho): 0: 0.2 · 1: 0.3 · 2: 0.3 · 3: 0.3"
    assert "los carriles 2–3 usan el último (0.3)" in lines[1]
    lines = _congestion_lines(_parse("[road]\nlanes = 2\n[behavior]\ncongestion_factor = [0.2, 0.3, 0.5]\n").sim)
    assert "se ignoran los sobrantes (0.5)" in lines[1]
    assert _congestion_lines(_parse("[road]\nlanes = 2\n").sim) == []


def test_fixed_lane_per_type():
    s = _parse("[road]\nmax_line_speed = [50, 50, 50]\n[vehicles.bike]\nlane = 0\n[vehicles.bus]\nlane = 1\n")
    assert [spec.lane for spec in s.sim.specs] == [None, 0, 1]
    for text, message in [
        ("[vehicles.bus]\nlane = 2\n", "[vehicles.bus] lane debe estar entre 0 (carril derecho) y 1"),
        ("[vehicles.car]\nlane = 0\n", "[vehicles.car] lane solo aplica a tipos con lane_change = false"),
    ]:
        with pytest.raises(ConfigError, match=re.escape(message)):
            _parse(text)


def test_overtake_and_lane_change_thresholds_per_type():
    s = _parse(SCOOTER + "overtake = true\nlookahead = 50\nmin_advantage = 1\nlane_change_cooldown = 1.5\n")
    scooter = s.sim.specs[-1]
    assert scooter.overtake and scooter.lookahead == 50.0
    assert scooter.min_advantage == 1.0 and scooter.lane_change_cooldown == 1.5
    car = s.sim.specs[0]  # sin las claves, los de [behavior]
    assert not car.overtake and car.lookahead is None and car.min_advantage is None
    for text, message in [
        (SCOOTER + "lane_change = false\novertake = true\n",
         "[vehicles.scooter] overtake solo aplica a tipos con lane_change = true"),
        ("[vehicles.bus]\nlookahead = 40\n", "[vehicles.bus] lookahead solo aplica a tipos con lane_change = true"),
        ("[vehicles.car]\nmin_advantage = -1\n", "[vehicles.car] min_advantage no puede ser negativo"),
    ]:  # fmt: skip
        with pytest.raises(ConfigError, match=re.escape(message)):
            _parse(text)


def test_yellow_light_settings():
    s = _parse("[traffic_light]\nred = 25\ngreen = 35\nyellow = 3\n"
               "[behavior]\nyellow_approach = 40\nyellow_speed_factor = 0.6\n")  # fmt: skip
    assert s.sim.yellow == 3.0 and s.sim.cycle == 63.0
    assert s.sim.behavior.yellow_approach == 40.0 and s.sim.behavior.yellow_speed_factor == 0.6
    assert s.sim.light_label == "rojo 25 s / verde 35 s / amarillo 3 s"
    assert _parse("").sim.yellow == 0.0 and _parse("").sim.light_label == "rojo 30 s / verde 30 s"
    for text, message in [
        ("[traffic_light]\nyellow = -1\n", "[traffic_light] yellow no puede ser negativo"),
        ("[traffic_light]\nyellow = 3.05\n", "[traffic_light] yellow debe ser múltiplo de 0.1 s"),
        ("[behavior]\nyellow_speed_factor = 0\n", "[behavior] yellow_speed_factor debe estar en (0, 1]"),
        ("[behavior]\nyellow_approach = -5\n", "[behavior] yellow_approach no puede ser negativo"),
    ]:
        with pytest.raises(ConfigError, match=re.escape(message)):
            _parse(text)


def test_leader_slowdown_setting():
    assert _parse("").sim.behavior.leader_slowdown == 0.25
    assert _parse("[behavior]\nleader_slowdown = 0.4\n").sim.behavior.leader_slowdown == 0.4
    with pytest.raises(ConfigError, match=re.escape("[behavior] leader_slowdown debe estar en [0, 1)")):
        _parse("[behavior]\nleader_slowdown = 1\n")


CARGA_TOML = """
[demand]
carga_rate = 1
[vehicles.carga]
speed_kmh = 40
length = 10
length_std = 2
length_min = 8
gap_run = 4
gap_stop = 1.5
cargo_prob = 1
"""


def test_cargo_type_and_variable_length():
    carga = _parse(CARGA_TOML).sim.specs[-1]  # sin pax_*: solo lleva mercancía
    assert carga.cargo_prob == 1.0 and not carga.carries_passengers
    assert (carga.shortest, carga.length, carga.longest) == (8.0, 10.0, 16.0)
    assert _parse("[vehicles.car]\ncargo_prob = 0.2\n").sim.specs[0].cargo_prob == 0.2
    assert DEFAULT_SPECS[0].cargo_prob == 0 and DEFAULT_SPECS[0].longest == DEFAULT_SPECS[0].length
    for text, message in [
        (CARGA_TOML.replace("cargo_prob = 1", "cargo_prob = 0.5"),
         "falta la clave obligatoria [vehicles.carga] pax_min"),
        ("[vehicles.car]\ncargo_prob = 1.5\n", "[vehicles.car] cargo_prob debe estar entre 0 y 1"),
        ("[vehicles.car]\nlength_std = -1\n", "[vehicles.car] length_std no puede ser negativo"),
        ("[vehicles.car]\nlength_std = 1\nlength_min = 5\n", "[vehicles.car] requiere 0 < length_min ≤ length ≤ length_max"),
        ("[vehicles.car]\nlength_std = 2\n", "[vehicles.car] requiere 0 < length_min"),  # 4.5 − 3·2 < 0
    ]:  # fmt: skip
        with pytest.raises(ConfigError, match=re.escape(message)):
            _parse(text)


def test_summary_reports_cargo_share(root):
    text = FAST + CARGA_TOML.replace("carga_rate = 1", "carga_rate = 30") + "[vehicles.car]\ncargo_prob = 0.3\n"
    (root / CONFIG_NAME).write_text(text, encoding="utf-8")
    summary = (run([]) / SUMMARY_NAME).read_text()
    row = next(line for line in summary.splitlines() if line.startswith("Con mercancía (%)"))
    assert row.split()[-1] == "100.0"  # carga: toda con mercancía
    pax_row = next(line for line in summary.splitlines() if line.startswith("Pasajeros por vehículo"))
    assert pax_row.split()[-1] == "—"  # sin pasajeros


def test_initial_occupancy_setting():
    assert _parse("").sim.lane_initial_occupancy == (0.0, 0.0)
    s = _parse("[initial]\noccupancy = [0.2, 0.5, 0.5]\n").sim
    assert s.lanes == 3 and s.lane_initial_occupancy == (0.2, 0.5, 0.5)
    assert _parse("[road]\nmax_line_speed = [30, 50]\n[initial]\noccupancy = 0.4\n").sim.lane_initial_occupancy == (0.4, 0.4)
    for text, message in [
        ("[initial]\noccupancy = 1.2\n", "[initial] occupancy: cada valor debe estar en [0, 1]"),
        ("[road]\nmax_line_speed = [30, 50]\n[initial]\noccupancy = [0.1, 0.2, 0.3]\n",
         "[initial] occupancy tiene 3"),
    ]:  # fmt: skip
        with pytest.raises(ConfigError, match=re.escape(message)):
            _parse(text)


def test_reaction_time_settings():
    b = _parse("[behavior]\nreaction_min = 0.7\nreaction_max = 3\nreaction_mean = 1.5\nreaction_std = 0.5\n").sim.behavior
    assert (b.reaction_min, b.reaction_max, b.reaction_mean, b.reaction_std) == (0.7, 3.0, 1.5, 0.5)
    assert _parse("").sim.behavior.reaction_std is None  # por defecto, uniforme
    for text, message in [
        ("[behavior]\nreaction_mean = 2\n", "[behavior] reaction_mean requiere reaction_std"),
        ("[behavior]\nreaction_std = -1\n", "[behavior] reaction_std no puede ser negativo"),
        ("[behavior]\nreaction_mean = 9\nreaction_std = 1\n",
         "[behavior] reaction_mean debe estar entre reaction_min y reaction_max"),
    ]:  # fmt: skip
        with pytest.raises(ConfigError, match=re.escape(message)):
            _parse(text)


def test_bottleneck_settings():
    sim = _parse('[road]\nmax_line_speed = [30, 50, 50]\n[bottleneck]\nstop_lanes = [1, 2]\nstop_zone = [50, 150]\n').sim
    assert sim.bottleneck_lanes() == (1, 2) and sim.bottleneck_zone() == (50.0, 150.0)
    default = _parse("").sim
    assert default.bottleneck_lanes() == (0, 1) and default.bottleneck_zone() == (0.0, 200.0)
    assert not default.bottleneck_active
    for text, message in [
        ("[bottleneck]\nstop_lanes = [5]\n", "[bottleneck] stop_lanes: cada carril debe estar entre 0 y 1"),
        ("[bottleneck]\nstop_zone = [150, 250]\n", "[bottleneck] stop_zone requiere 0 ≤ inicio < fin ≤ 200"),
        ("[bottleneck]\nstop_zone = 50\n", "[bottleneck] stop_zone debe ser una lista [inicio, fin]"),
        ("[bottleneck]\nstop_lanes = 1\n", "[bottleneck] stop_lanes debe ser una lista de enteros"),
        # La probabilidad y la duración ya no van en [bottleneck]: el error dice a dónde se movieron.
        ("[bottleneck]\nstop_prob = 0.1\nstop_time_mean = 20\n",
         "en [bottleneck] solo quedan stop_lanes y stop_zone: la probabilidad y la duración van en cada "
         "[vehicles.<clave>] como bottleneck_prob, bottleneck_time_mean"),
    ]:  # fmt: skip
        with pytest.raises(ConfigError, match=re.escape(message)):
            _parse(text)


def test_bottleneck_per_type_settings():
    sim = _parse(CARGA_TOML + "bottleneck_prob = 0.3\nbottleneck_time_mean = 90\nbottleneck_time_std = 30\n"
                 "[vehicles.car]\nbottleneck_prob = 0.05\n").sim
    carga = len(sim.specs) - 1
    assert sim.bottleneck_prob(carga) == 0.3 and sim.bottleneck_time(carga) == (90.0, 30.0)
    assert sim.bottleneck_prob(0) == 0.05 and sim.bottleneck_time(0) == (30.0, 0.0)  # duración por defecto
    assert sim.bottleneck_prob(1) == 0.0 and sim.bottleneck_active
    for text, message in [
        ("[vehicles.car]\nbottleneck_prob = 1.5\n", "[vehicles.car] bottleneck_prob debe estar entre 0 y 1"),
        ("[vehicles.car]\nbottleneck_time_std = -2\n",
         "[vehicles.car] bottleneck_time_mean y bottleneck_time_std no pueden ser negativos"),
        ("[vehicles.bus]\nstop_position = 100\nbottleneck_prob = 0.5\n",
         "[vehicles.bus] bottleneck_prob no aplica a un tipo con parada propia"),
    ]:  # fmt: skip
        with pytest.raises(ConfigError, match=re.escape(message)):
            _parse(text)


def test_vehicle_type_limit():
    keys = [f"s{i}" for i in range(MAX_TYPES - len(DEFAULT_SPECS) + 1)]
    vehicle = SCOOTER[SCOOTER.index("[vehicles.scooter]") :]
    text = "[demand]\n" + "".join(f"{k}_rate = 1\n" for k in keys)
    text += "".join(vehicle.replace("scooter", k) for k in keys)
    with pytest.raises(ConfigError, match=f"hasta {MAX_TYPES} tipos"):
        _parse(text)


@pytest.mark.parametrize(
    "text",
    [
        "[road]\nlength = 300\n",  # sin secciones [execution] ni [output]
        "[execution] # ejecución\nrun = 2\n\n[output]\ncsv = false\n",  # sección con comentario
        '[output]\nname = ""   # vacío\ncsv = false\n[road]\nmax_line_speed = [40, 50]\n',  # reemplaza la línea existente
    ],
)
def test_copy_records_seed_and_name(tmp_path, text):
    settings = _parse(text)
    run_dir = tmp_path / "20260924-083000_hora_pico"
    copy = config_copy_text(settings, 987654, "hora_pico", run_dir, NOW)
    assert copy.startswith("# Copia de la configuración usada en la corrida 20260924-083000_hora_pico")
    reparsed = _parse(copy)
    assert reparsed.run.seed == 987654
    assert reparsed.run.name == "hora_pico"
    assert reparsed.sim == settings.sim
    # El resto del archivo queda igual salvo por las dos claves registradas.
    original, again_data = tomllib.loads(text), tomllib.loads(copy)
    original.setdefault("execution", {})["seed"] = 987654
    original.setdefault("output", {})["name"] = "hora_pico"
    assert again_data == original
    # Copiar la copia reemplaza el encabezado en lugar de apilarlo, y no duplica claves.
    again = config_copy_text(reparsed, 987654, "hora_pico", tmp_path / "20260924-083100_hora_pico", NOW)
    assert again.count("# Copia de") == 1
    assert tomllib.loads(again) == again_data


def test_run_dir_is_unique_and_sanitized(tmp_path):
    first = make_run_dir(tmp_path, "escenario A/1", NOW)
    second = make_run_dir(tmp_path, "escenario A/1", NOW)
    assert first.name == "20260924-083000_escenario_A_1"
    assert second.name == "20260924-083000_escenario_A_1-2"


@pytest.fixture
def root(tmp_path, monkeypatch):
    """Raíz de proyecto temporal con un config.toml rápido."""
    monkeypatch.setattr("trafico.settings.project_root", lambda: tmp_path)
    (tmp_path / CONFIG_NAME).write_text(FAST, encoding="utf-8")
    return tmp_path


FAST = "[execution]\nrun = 2\nreplicas = 2\nworkers = 2\n[output]\nprogress = false\n"


def test_resolve_target(root, monkeypatch):
    monkeypatch.chdir(root)
    (root / "escenario").mkdir()
    (root / "otro.toml").write_text("", encoding="utf-8")
    assert resolve_target(None) == Target(root / CONFIG_NAME, None, "corrida")
    assert resolve_target("hora_pico") == Target(root / CONFIG_NAME, "hora_pico", "corrida")
    assert resolve_target("escenario") == Target(Path("escenario") / CONFIG_NAME, None, "escenario")
    assert resolve_target(str(root / CONFIG_NAME)).config == root / CONFIG_NAME
    with pytest.raises(ConfigError, match="debe llamarse config.toml"):
        resolve_target("otro.toml")
    with pytest.raises(ConfigError, match="no existe la carpeta"):
        resolve_target("resultados/no_existe")


def test_name_argument_and_replication(root, capsys):
    first = run(["hora pico"])  # nombre con espacio: se sanea
    assert first.parent == root / "resultados" and _named(first, "hora_pico")
    assert {p.name for p in first.iterdir()} == {CONFIG_NAME, PLOT_NAME, PAX_PLOT_NAME, CSV_NAME, SUMMARY_NAME}
    assert (first / PLOT_NAME).stat().st_size > 10_000
    assert (first / PAX_PLOT_NAME).stat().st_size > 10_000
    lines = (first / CSV_NAME).read_text().splitlines()
    assert lines[0].startswith("t_sim_s,t_proceso_s,cum_pax_auto_media")
    assert "saturacion_carril1_media,saturacion_carril1_sd" in lines[0]  # 2 carriles
    assert lines[0].endswith("velocidad_kmh_carril1_media,velocidad_kmh_carril1_sd")
    assert len(lines) == 1 + 20  # 20 s simulados muestreados cada 1 s
    copy = tomllib.loads((first / CONFIG_NAME).read_text())
    assert copy["output"]["name"] == "hora_pico"
    assert f"semilla {copy['execution']['seed']}" in (first / SUMMARY_NAME).read_text()

    # Réplica desde la carpeta de resultados: conserva el nombre y los resultados.
    second = run([str(first)])
    assert second != first and _named(second, "hora_pico")
    assert (second / CSV_NAME).read_bytes() == (first / CSV_NAME).read_bytes()
    # Leer el config.toml de la copia directamente también conserva el nombre registrado.
    assert _named(run([str(first / CONFIG_NAME)]), "hora_pico")
    assert "Pasajeros que cruzan" in capsys.readouterr().out


def test_default_and_folder_names(root):
    assert _named(run([]), "corrida")
    scenario = root / "escenarios" / "lluvia"
    scenario.mkdir(parents=True)
    (scenario / CONFIG_NAME).write_text(FAST, encoding="utf-8")
    assert _named(run([str(scenario)]), "lluvia")
    (scenario / CONFIG_NAME).write_text(FAST + 'name = "tormenta"\n', encoding="utf-8")
    assert _named(run([str(scenario)]), "tormenta")


def test_cli_reports_config_errors(root, capsys):
    with pytest.raises(SystemExit) as exc:
        run(["resultados/no_existe"])
    assert exc.value.code == 2
    assert "no existe la carpeta" in capsys.readouterr().err
    (root / "vacia").mkdir()
    with pytest.raises(SystemExit):
        run([str(root / "vacia")])
    assert "no existe el archivo de configuración" in capsys.readouterr().err


def test_run_with_new_vehicle_type(tmp_path):
    folder = tmp_path / "con_scooter"
    folder.mkdir()
    (folder / CONFIG_NAME).write_text(
        SCOOTER
        + "[execution]\nrun = 2\nreplicas = 2\nworkers = 1\n"
        + f'[output]\ndir = "{tmp_path / "resultados"}"\nprogress = false\n',
        encoding="utf-8",
    )
    run_dir = run([str(folder)])
    assert _named(run_dir, "con_scooter")
    header = (run_dir / CSV_NAME).read_text().splitlines()[0]
    assert "cum_pax_scooter_media" in header
    assert "scooter" in (run_dir / SUMMARY_NAME).read_text()


def test_exclusive_lanes():
    s = _parse(
        "[road]\nmax_line_speed = [20, 40, 50]\n"
        "[vehicles.bike]\nlane = 0\nexclusive = true\n[vehicles.bus]\nlane = 1\nexclusive = true\n"
    )
    assert s.sim.reserved_lanes == {0, 1}
    assert s.sim.allowed_lanes(0) == (2,)
    assert s.sim.free_flow_kmh(0) == 50
    for text, message in [
        ("[vehicles.bike]\nexclusive = true\n", "[vehicles.bike] exclusive = true requiere un carril fijo (lane)"),
        (
            "[road]\nmax_line_speed = [40, 40]\n[vehicles.bike]\nlane = 0\nexclusive = true\n"
            "[vehicles.bus]\nlane = 1\nexclusive = true\n",
            "[vehicles.car] no le queda carril",
        ),
    ]:
        with pytest.raises(ConfigError, match=re.escape(message)):
            _parse(text)
    # Sin autos ni tipos sin carril fijo que participen, todos los carriles pueden ser exclusivos.
    _parse(
        "[road]\nmax_line_speed = [40, 40]\n[demand]\ncar_rate = 0\n[vehicles.bike]\nlane = 0\nexclusive = true\n"
        "[vehicles.bus]\nlane = 1\nexclusive = true\n"
    )


def test_bus_stop_settings():
    s = _parse("[vehicles.bus]\nstop_position = 150\nstop_time_mean = 25\nstop_time_std = 6\n")
    bus = s.sim.specs[BUS]
    assert (bus.stop_position, bus.stop_time_mean, bus.stop_time_std) == (150.0, 25.0, 6.0)
    assert s.sim.specs[0].stop_position is None  # los demás tipos no paran
    assert _parse("[vehicles.bus]\nstop_position = 150\n").sim.specs[BUS].stop_time_mean == 20.0
    for text, message in [
        ("[vehicles.bus]\nstop_position = 5\n", "[vehicles.bus] stop_position debe estar entre 12 "),
        ("[vehicles.bus]\nstop_position = 250\n", "y 200 m (el semáforo)"),
        ("[vehicles.bus]\nstop_position = 150\nstop_time_std = -1\n", "no pueden ser negativos"),
    ]:
        with pytest.raises(ConfigError, match=re.escape(message)):
            _parse(text)
