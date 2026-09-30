import math
import re
import tomllib
from datetime import datetime
from pathlib import Path

import pytest

from trafico.cli import CSV_NAME, PAX_PLOT_NAME, PLOT_NAME, SUMMARY_NAME, run
from trafico.config import BUS, DEFAULT_RATES, DEFAULT_SPECS, MAX_TYPES, Rate, SimConfig
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
scooter_rate = {min = 6, max = 6, mean = 6, std = 0}
[vehicles.scooter]
speed_kmh = {min = 25, max = 25, mean = 25, std = 0}
length = {min = 1.5, max = 1.5, mean = 1.5, std = 0}
gap_run = 1.5
gap_stop = 0.5
pax = {min = 1, max = 1, mean = 1, std = 0}
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
            "[road]\nmax_line_speed = [40, 50, 40]\n[initial]\noccupancy = [0.2, 0.3]\n",
            "las listas por carril deben tener el mismo largo",
        ),
        ("[output]\nseries_csv = 1\n", "[output] series_csv debe ser true o false"),
        ("[output]\nemisiones_csv = 1\n", "[output] emisiones_csv debe ser true o false"),
        ("[road]\nlength = 10\n", "[road] length debe ser al menos"),
        ("[demand]\ncar_rate = {min = 0, max = 0, mean = 0, std = 0}\nbike_rate = {min = 0, max = 0, mean = 0, std = 0}\nbus_rate = {min = 0, max = 0, mean = 0, std = 0}\n", "al menos un tipo"),
        ("[traffic_light]\nred = 12.35\n", "múltiplo de 0.1"),
        ("[vehicles.car]\npax = {min = 1, max = 6, mean = 9, std = 1}\n", "pax: mean debe estar entre min y max"),
        ("[vehicles.car]\npax = {min = 1, max = 6, mean = 2}\n", "falta la clave obligatoria [vehicles.car.pax] std"),
        ("[vehicles.car]\npax = {min = 1, max = 6.5, mean = 2, std = 1}\n", "[vehicles.car.pax] max debe ser"),
        ("[vehicles.car]\npax = 2\n", "[vehicles.car] pax debe ser un diccionario"),
        ("[vehicles.car]\npax_mean = 2\n", "los pasajeros van ahora en un diccionario"),
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
        (SCOOTER.replace("pax = {min = 1, max = 1, mean = 1, std = 0}\n", ""), "falta la clave obligatoria [vehicles.scooter] pax"),
        (SCOOTER.replace("scooter_rate = {min = 6, max = 6, mean = 6, std = 0}\n", ""), "falta la clave obligatoria [demand] scooter_rate"),
        ("[demand]\ntruck_rate = {min = 2, max = 2, mean = 2, std = 0}\n", "define también su sección [vehicles.truck]"),
        (SCOOTER + 'name = "bici"\n', "se repite bici"),
        ("[vehicles]\nscooter = 3\n", "debe ser una sección [vehicles.scooter]"),
    ],
)
def test_invalid_new_vehicle_is_rejected(text, message):
    with pytest.raises(ConfigError, match=message.replace("[", r"\[").replace("]", r"\]")):
        _parse(text)


@pytest.mark.parametrize(
    ("text", "lanes", "limits", "occupancy"),
    [
        ("[initial]\noccupancy = [0.2, 0.3, 0.5]\n", 3, (math.inf,) * 3, (0.2, 0.3, 0.5)),
        ("[road]\nmax_line_speed = [40, 50, 40, 60]\n", 4, (40, 50, 40, 60), (0.0,) * 4),
        (
            "[road]\nmax_line_speed = [40, 50, 40]\n[initial]\noccupancy = [0.2, 0.3, 0.5]\n",
            3, (40, 50, 40), (0.2, 0.3, 0.5),
        ),
        ("[road]\nmax_line_speed = 30\n[initial]\noccupancy = [0.1, 0.2]\n", 2, (30, 30), (0.1, 0.2)),
        ("[initial]\noccupancy = 0.4\n", 2, (math.inf,) * 2, (0.4, 0.4)),  # sin listas: 2 carriles
    ],
)  # fmt: skip
def test_lanes_come_from_per_lane_lists(text, lanes, limits, occupancy):
    s = _parse(text)
    assert s.sim.lanes == lanes
    assert s.sim.lane_max_kmh == limits
    assert s.sim.lane_initial_occupancy == occupancy
    assert s.notices == ()


@pytest.mark.parametrize(
    ("lanes", "expected"),
    [(3, (0.2, 0.3, 0.5)), (5, (0.2, 0.3, 0.5, 0.5, 0.5)), (2, (0.2, 0.3)), (1, (0.2,))],
)
def test_legacy_lanes_key_still_replicates_old_copies(lanes, expected):
    """[road] lanes está obsoleto, pero las copias anteriores lo tienen: se acepta con un aviso
    y las listas se ajustan a él."""
    s = _parse(f"[road]\nlanes = {lanes}\n[initial]\noccupancy = [0.2, 0.3, 0.5]\n")
    assert s.sim.lanes == lanes
    assert s.sim.lane_initial_occupancy == expected
    assert "[road] lanes está obsoleto" in s.notices[0]


def test_removed_saturation_threshold_is_ignored_in_old_copies():
    s = _parse("[output]\nsaturation_threshold = 0.9\n")
    assert any("saturation_threshold se eliminó" in n for n in s.notices)


def test_removed_show_is_ignored_in_old_copies():
    s = _parse("[output]\nshow = false\n")
    assert any("[output] show se eliminó" in n for n in s.notices)
    assert not hasattr(s.run, "show") and not _parse("").notices


def test_removed_slow_lane_key_is_ignored_in_old_copies():
    """[demand] slow_lane se eliminó: con "right" (lo que se hace ahora) se ignora con un aviso; otro valor es error."""
    s = _parse('[demand]\nslow_lane = "right"\n')
    assert any("slow_lane se eliminó" in n for n in s.notices)
    with pytest.raises(ConfigError, match="slow_lane se eliminó"):
        _parse('[demand]\nslow_lane = "random"\n')


def test_free_lanes_settings():
    assert _parse("").sim.free_lanes == ()
    exclusive = "[vehicles.bike]\nlane = 0\nexclusive = true\n[vehicles.bus]\nlane = 2\nexclusive = true\n"
    sim = _parse("[road]\nmax_line_speed = [20, 40, 50]\n[traffic_light]\nfree_lanes = [0, 2]\n" + exclusive).sim
    assert sim.free_lanes == (0, 2) and [sim.ignores_light(k) for k in range(3)] == [False, True, True]
    with pytest.raises(ConfigError, match="free_lanes: el carril 1 no es exclusivo de ningún tipo"):
        _parse("[road]\nmax_line_speed = [20, 40, 50]\n[traffic_light]\nfree_lanes = [1]\n" + exclusive)
    with pytest.raises(ConfigError, match=r"\[traffic_light\] free_lanes: cada carril debe estar entre 0 y 1"):
        _parse("[traffic_light]\nfree_lanes = [2]\n")
    with pytest.raises(ConfigError, match="free_lanes: hay carriles repetidos"):
        _parse("[traffic_light]\nfree_lanes = [0, 0]\n")


def test_traffic_light_can_be_disabled():
    """enabled = false, o red, green y yellow en 0, quitan el semáforo; con fases, green debe ser > 0."""
    assert _parse("").sim.has_light
    off = _parse("[traffic_light]\nenabled = false\n").sim
    assert not off.has_light and off.light_label == "sin semáforo" and off.line_name == "el final del tramo"
    assert not _parse("[traffic_light]\nred = 0\ngreen = 0\nyellow = 0\n").sim.has_light
    assert not _parse("[traffic_light]\nenabled = false\nred = 30\ngreen = 0\n").sim.has_light  # se ignoran
    with pytest.raises(ConfigError, match="green debe ser mayor que 0"):
        _parse("[traffic_light]\nred = 30\ngreen = 0\n")
    with pytest.raises(ConfigError, match="enabled debe ser true o false"):
        _parse("[traffic_light]\nenabled = 0\n")


def test_emission_settings():
    """El auto y el autobús traen coeficientes; una sección [vehicles.<clave>.emissions] los reemplaza (vacía, el
    tipo no emite). Solo emiten con accel y decel; si se escriben emisiones sin ellos, hay aviso."""
    s = _parse("")
    assert [e[0] for e in s.sim.specs[0].emissions] == ["co2", "nox", "voc"] and not s.sim.specs[0].emits
    assert [e[0] for e in s.sim.specs[2].emissions] == ["pm"]
    text = ("[vehicles.car]\naccel = 2.5\ndecel = 4.5\n[vehicles.car.emissions]\n"
            "co2 = [1, 2, 3, 4, 5, 6]\nnox = [1, 0, 0, 0, 0, 0]\nnox_decel = [0.5, 0, 0, 0, 0, 0]\n")  # fmt: skip
    car = _parse(text).sim.specs[0]
    assert car.emits and car.emission_coefs("co2") == ((1, 2, 3, 4, 5, 6), (1, 2, 3, 4, 5, 6))
    assert car.emission_coefs("nox")[1] == (0.5, 0, 0, 0, 0, 0) and car.emission_coefs("voc") is None
    assert _parse("[vehicles.car.emissions]\n").sim.specs[0].emissions == ()
    s = _parse("[vehicles.car.emissions]\nco2 = [1, 2, 3, 4, 5, 6]\n")
    assert any("sin accel y decel no se calculan las emisiones de auto" in n for n in s.notices)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("[vehicles.car.emissions]\nco2 = [1, 2, 3, 4, 5]\n", "co2 debe ser una lista de 6 números"),
        ("[vehicles.car.emissions]\nco = [1, 2, 3, 4, 5, 6]\n", "claves desconocidas: [vehicles.car.emissions] co"),
        ("[vehicles.car.emissions]\npm_decel = [1, 2, 3, 4, 5, 6]\n", "pm_decel requiere pm"),
    ],
)
def test_invalid_emissions_are_rejected(text, message):
    with pytest.raises(ConfigError, match=message.replace("[", r"\[").replace("]", r"\]")):
        _parse(text)


def test_speed_bump_settings():
    sim = _parse("").sim
    assert sim.speed_bump.position is None and sim.speed_bump_lanes() == ()
    sim = _parse("[road]\nmax_line_speed = [20, 40, 50]\n[speed_bump]\nposition = 60\nlanes = [1, 2]\n"
                 "[vehicles.car]\nspeed_bump_kmh = 10\n").sim  # fmt: skip
    assert sim.speed_bump.position == 60.0 and sim.speed_bump_lanes() == (1, 2)
    assert sim.specs[0].speed_bump_kmh == 10.0 and sim.specs[1].speed_bump_kmh is None
    assert _parse("[speed_bump]\nposition = 60\n").sim.speed_bump_lanes() == (0, 1)  # sin lanes: todos


def test_several_speed_bumps():
    """position acepta una lista: varios topes en los mismos carriles, ordenados de la entrada a la salida."""
    sim = _parse("[road]\nmax_line_speed = [20, 40, 50]\n[speed_bump]\nposition = [120, 40.5]\nlanes = [1]\n").sim
    assert sim.speed_bump.positions == (40.5, 120.0) and sim.speed_bump_lanes() == (1,)
    assert _parse("[speed_bump]\nposition = [60]\n").sim.speed_bump.positions == (60.0,)
    assert _parse("").sim.speed_bump.positions == () and _parse("").sim.speed_bump_lanes() == ()
    for text, message in (
        ("[speed_bump]\nposition = [60, 200]\n", "position debe estar entre 0 y 200 m"),
        ("[speed_bump]\nposition = [60, 60]\n", "position: hay topes en la misma posición"),
        ("[speed_bump]\nposition = [60, 'x']\n", "position debe ser un número o una lista de números"),
    ):
        with pytest.raises(ConfigError, match=re.escape(message)):
            _parse(text)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("[speed_bump]\nposition = 0\n", "[speed_bump] position debe estar entre 0 y 200 m"),
        ("[speed_bump]\nposition = 200\n", "[speed_bump] position debe estar entre 0 y 200 m"),
        ("[speed_bump]\nposition = 60\nlanes = [2]\n", "[speed_bump] lanes: cada carril debe estar entre 0 y 1"),
        ("[speed_bump]\nposition = 60\nlanes = [1, 1]\n", "[speed_bump] lanes: hay carriles repetidos"),
        ("[speed_bump]\nposition = 60\nlanes = []\n", "[speed_bump] lanes no puede estar vacía"),
        ("[speed_bump]\nlanes = [1]\n", "[speed_bump] lanes requiere position"),
        ("[vehicles.car]\nspeed_bump_kmh = 0\n", "[vehicles.car] speed_bump_kmh debe ser mayor que 0"),
    ],
)
def test_invalid_speed_bump_is_rejected(text, message):
    with pytest.raises(ConfigError, match=message.replace("[", r"\[").replace("]", r"\]")):
        _parse(text)


def test_removed_congestion_factor_is_ignored_in_old_copies():
    """[behavior] congestion_factor se eliminó: se ignora con un aviso, pero una lista sigue dando el número de
    carriles para que las copias anteriores carguen igual."""
    s = _parse("[behavior]\ncongestion_factor = [0.05, 0.5, 0.4]\n")
    assert s.sim.lanes == 3
    assert any("congestion_factor se eliminó" in n for n in s.notices)
    s = _parse("[road]\nmax_line_speed = [40, 50]\n[behavior]\ncongestion_factor = 0.3\n")
    assert s.sim.lanes == 2 and any("congestion_factor se eliminó" in n for n in s.notices)
    with pytest.raises(ConfigError, match="las listas por carril deben tener el mismo largo"):
        _parse("[road]\nmax_line_speed = [40, 50]\n[behavior]\ncongestion_factor = [0.1, 0.2, 0.3]\n")


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
carga_rate = {min = 1, max = 1, mean = 1, std = 0}
[vehicles.carga]
speed_kmh = {min = 40, max = 40, mean = 40, std = 0}
length = {min = 8, max = 16, mean = 10, std = 2}
gap_run = 4
gap_stop = 1.5
cargo_prob = 1
"""


def test_cargo_type_and_variable_length():
    carga = _parse(CARGA_TOML).sim.specs[-1]  # sin pax: solo lleva mercancía
    assert carga.cargo_prob == 1.0 and not carga.carries_passengers
    assert (carga.shortest, carga.length, carga.longest) == (8.0, 10.0, 16.0)
    assert _parse("[vehicles.car]\ncargo_prob = 0.2\n").sim.specs[0].cargo_prob == 0.2
    assert DEFAULT_SPECS[0].cargo_prob == 0 and DEFAULT_SPECS[0].longest == DEFAULT_SPECS[0].length
    for text, message in [
        (CARGA_TOML.replace("cargo_prob = 1", "cargo_prob = 0.5"),
         "falta la clave obligatoria [vehicles.carga] pax"),
        ("[vehicles.car]\ncargo_prob = 1.5\n", "[vehicles.car] cargo_prob debe estar entre 0 y 1"),
        ("[vehicles.car]\nlength = {min = 4, max = 5, mean = 4.5, std = -1}\n", "[vehicles.car] length: std no puede ser negativa"),
        ("[vehicles.car]\nlength = {min = 5, max = 6, mean = 4.5, std = 1}\n", "[vehicles.car] length: debe cumplirse 0 < min ≤ mean ≤ max"),
        ("[vehicles.car]\nlength = {min = 0, max = 6, mean = 4.5, std = 1}\n", "[vehicles.car] length: debe cumplirse 0 < min"),
        ("[vehicles.car]\nlength = {min = 4, max = 5, mean = 4.5}\n", "falta la clave obligatoria [vehicles.car.length] std"),
        ("[vehicles.car]\nlength = 4.5\n", "[vehicles.car] length debe ser un diccionario"),
        ("[vehicles.car]\nlength_std = 2\n", "el largo va ahora en un diccionario"),
        (SCOOTER.replace("length = {min = 1.5, max = 1.5, mean = 1.5, std = 0}\n", ""),
         "falta la clave obligatoria [vehicles.scooter] length"),
    ]:  # fmt: skip
        with pytest.raises(ConfigError, match=re.escape(message)):
            _parse(text)


def test_variable_speed_setting():
    car = _parse("[vehicles.car]\nspeed_kmh = {min = 60, max = 90, mean = 75, std = 8}\n").sim.specs[0]
    assert (car.slowest_kmh, car.speed_kmh, car.fastest_kmh, car.speed_std) == (60.0, 75.0, 90.0, 8.0)
    assert DEFAULT_SPECS[0].fastest_kmh == DEFAULT_SPECS[0].speed_kmh  # sin std, fija
    for text, message in [
        ("[vehicles.car]\nspeed_kmh = 80\n", "[vehicles.car] speed_kmh debe ser un diccionario"),
        ("[vehicles.car]\nspeed_kmh = {min = 60, max = 90, mean = 75}\n",
         "falta la clave obligatoria [vehicles.car.speed_kmh] std"),
        ("[vehicles.car]\nspeed_kmh = {min = 60, max = 90, mean = 75, std = -1}\n",
         "[vehicles.car] speed_kmh: std no puede ser negativa"),
        ("[vehicles.car]\nspeed_kmh = {min = 80, max = 90, mean = 75, std = 5}\n",
         "[vehicles.car] speed_kmh: debe cumplirse 0 < min ≤ mean ≤ max"),
        ("[vehicles.car]\nspeed_kmh = {min = 0, max = 90, mean = 75, std = 5}\n",
         "[vehicles.car] speed_kmh: debe cumplirse 0 < min"),
        (SCOOTER.replace("speed_kmh = {min = 25, max = 25, mean = 25, std = 0}\n", ""),
         "falta la clave obligatoria [vehicles.scooter] speed_kmh"),
    ]:  # fmt: skip
        with pytest.raises(ConfigError, match=re.escape(message)):
            _parse(text)


def test_pass_in_lane_setting():
    assert not _parse("").sim.specs[1].pass_in_lane  # por defecto, no
    bike = _parse("[vehicles.bike]\nabreast = 2\npass_in_lane = true\n").sim.specs[1]
    assert bike.pass_in_lane and bike.abreast == 2
    with pytest.raises(ConfigError, match=re.escape("[vehicles.bike] pass_in_lane = true requiere abreast ≥ 2")):
        _parse("[vehicles.bike]\npass_in_lane = true\n")


def test_queue_reaction_setting():
    assert not _parse("").sim.behavior.queue_reaction  # por defecto, como antes
    assert _parse("[behavior]\nqueue_reaction = true\n").sim.behavior.queue_reaction
    with pytest.raises(ConfigError, match=re.escape("[behavior] queue_reaction")):
        _parse("[behavior]\nqueue_reaction = 1\n")


def test_gradual_dynamics_settings():
    car = _parse("").sim.specs[0]
    assert car.accel is None and car.decel is None  # por defecto, instantáneos como antes
    car = _parse("[vehicles.car]\naccel = 2.5\ndecel = 4.5\n").sim.specs[0]
    assert (car.accel, car.decel) == (2.5, 4.5)
    for text, message in [
        ("[vehicles.car]\naccel = 0\n", "[vehicles.car] accel debe estar en (0, 20] m/s²"),
        ("[vehicles.car]\ndecel = 25\n", "[vehicles.car] decel debe estar en (0, 20] m/s²"),
    ]:
        with pytest.raises(ConfigError, match=re.escape(message)):
            _parse(text)


def test_exit_queue_settings():
    sim = _parse("").sim
    assert sim.lane_exit_capacity == (0.0, 0.0)  # por defecto, sin cola de salida
    sim = _parse("[exit]\ncapacity = [0, 15, 20]\nstorage = 8\n").sim
    assert sim.lanes == 3 and sim.lane_exit_capacity == (0.0, 15.0, 20.0) and sim.lane_exit_storage == (8.0,) * 3
    for text, message in [
        ("[exit]\ncapacity = -1\n", "[exit] capacity: cada valor debe ser ≥ 0"),
        ("[exit]\ncapacity = 10\nstorage = 0\n", "[exit] storage: cada valor debe ser > 0 m"),
        ("[road]\nmax_line_speed = [30, 50]\n[exit]\ncapacity = [1, 2, 3]\n", "[exit] capacity tiene 3"),
    ]:  # fmt: skip
        with pytest.raises(ConfigError, match=re.escape(message)):
            _parse(text)


def test_summary_reports_cargo_share(root):
    text = FAST + CARGA_TOML.replace("carga_rate = {min = 1, max = 1, mean = 1, std = 0}", "carga_rate = {min = 30, max = 30, mean = 30, std = 0}") + "[vehicles.car]\ncargo_prob = 0.3\n"
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
    b = _parse("[behavior]\nreaction = {min = 0.7, max = 3, mean = 1.5, std = 0.5}\n").sim.behavior
    assert (b.reaction_min, b.reaction_max, b.reaction_mean, b.reaction_std) == (0.7, 3.0, 1.5, 0.5)
    b = _parse("[behavior]\nreaction = {min = 0.7, max = 3}\n").sim.behavior  # sin mean ni std: uniforme
    assert (b.reaction_min, b.reaction_max, b.reaction_mean, b.reaction_std) == (0.7, 3.0, None, None)
    assert _parse("").sim.behavior.reaction_std is None  # por defecto, uniforme
    for text, message in [
        ("reaction = {min = 1, max = 3, mean = 2}", "[behavior] reaction: mean y std van juntas"),
        ("reaction = {min = 1, mean = 2, std = 1}", "falta la clave obligatoria [behavior.reaction] max"),
        ("reaction = {min = 1, max = 3, mean = 2, std = -1}", "[behavior] reaction: std no puede ser negativa"),
        ("reaction = {min = 1, max = 3, mean = 9, std = 1}", "[behavior] reaction: mean debe estar entre min y max"),
        ("reaction = {min = 3, max = 1}", "[behavior] reaction: debe cumplirse 0 < min ≤ max ≤ 600"),
        ("reaction = 2", "[behavior] reaction debe ser un diccionario"),
        ("reaction_min = 2", "la reacción va ahora en un diccionario"),
    ]:  # fmt: skip
        with pytest.raises(ConfigError, match=re.escape(message)):
            _parse(f"[behavior]\n{text}\n")


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
    text = "[demand]\n" + "".join(f"{k}_rate = {{min = 1, max = 1, mean = 1, std = 0}}\n" for k in keys)
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
    assert "velocidad_kmh_carril1_media,velocidad_kmh_carril1_sd" in lines[0]
    assert "cruzan_veh_min_carril1_media,cruzan_veh_min_carril1_sd" in lines[0]
    assert lines[0].endswith("cola_entrada_veh_carril1_media,cola_entrada_veh_carril1_sd")  # sin cola de salida
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
        "[road]\nmax_line_speed = [40, 40]\n[demand]\ncar_rate = {min = 0, max = 0, mean = 0, std = 0}\n[vehicles.bike]\nlane = 0\nexclusive = true\n"
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


def test_rate_is_a_dictionary():
    s = _parse("[demand]\nrate_interval = 30\ncar_rate = {min = 1, max = 20, mean = 15, std = 5}\n")
    assert s.sim.rate(0) == Rate(1.0, 20.0, 15.0, 5.0) and s.sim.rate_interval == 30.0
    assert s.sim.rates[0] == pytest.approx(Rate(1, 20, 15, 5).expected)
    assert s.sim.rate(1) == Rate.fixed(DEFAULT_RATES[1])  # sin la clave: la fija por defecto


@pytest.mark.parametrize(
    ("demand", "message"),
    [
        ("car_rate = 15", "car_rate debe ser un diccionario"),
        ("car_rate = {min = 1, max = 20, mean = 15}", "falta la clave obligatoria [demand.car_rate] std"),
        ('car_rate = {min = 1, max = "20", mean = 15, std = 5}', "[demand.car_rate] max debe ser"),
        ("car_rate = {min = 1, max = 20, mean = 15, std = 5, avg = 3}", "claves desconocidas: [demand.car_rate] avg"),
        ("car_rate = {min = 10, max = 20, mean = 25, std = 5}", "0 ≤ min ≤ mean ≤ max"),
        ("car_rate = {min = 1, max = 20, mean = 15, std = -1}", "std no puede ser negativa"),
        ("rate_interval = 0.05", "rate_interval debe ser"),
        ("rate_interval = 30.05", "rate_interval debe ser múltiplo"),
    ],
)
def test_invalid_rates(demand, message):
    with pytest.raises(ConfigError, match=re.escape(message)):
        _parse(f"[demand]\n{demand}\n")


def test_emission_sets_and_fuel_settings():
    """`source` toma los coeficientes de un conjunto con nombre (con `source_type`, los de otro tipo); los escritos
    a mano los reemplazan por contaminante. `fuel`, `mass_kg` y [fuel] price/currency."""
    from trafico.emission_sets import EMISSION_SETS

    sedema = EMISSION_SETS["sedema_cdmx_2018"]
    car = _parse('[vehicles.car.emissions]\nsource = "sedema_cdmx_2018"\n').sim.specs[0]
    assert car.emissions == sedema["car"] and car.emission_source == "sedema_cdmx_2018"
    assert car.fuel == "gasolina" and car.mass_kg == 1250.0  # los del auto incorporado
    assert _parse("").sim.specs[0].emission_source == "int_panis_2006"
    text = ('[vehicles.car]\nfuel = "glp"\nmass_kg = 2300\n[vehicles.car.emissions]\nsource = "sedema_cdmx_2018"\n'
            'source_type = "colectivo"\nnox = [1, 0, 0, 0, 0, 0]\nco2_decel = [2, 0, 0, 0, 0, 0]\npm = [3, 0, 0, 0, 0, 0]\n'
            '[fuel]\nprice = {gasolina = 24, glp = 11.5}\ncurrency = "USD"\n')  # fmt: skip
    sim = _parse(text).sim
    car = sim.specs[0]
    assert car.fuel == "glp" and car.mass_kg == 2300.0
    assert car.emission_coefs("co2") == (sedema["colectivo"][0][1], (2, 0, 0, 0, 0, 0))
    assert car.emission_coefs("nox") == ((1, 0, 0, 0, 0, 0),) * 2
    assert car.emission_coefs("voc") == sedema["colectivo"][2][1:] and car.emission_coefs("pm")[0][0] == 3
    assert sim.fuel_price == (("gasolina", 24.0), ("glp", 11.5)) and sim.price("glp") == 11.5
    assert sim.price("diesel") is None and sim.currency == "USD"


def test_fuel_notices():
    base = "[vehicles.car]\naccel = 2.5\ndecel = 4.5\n"
    assert not _parse(base + "[fuel]\nprice = {gasolina = 24}\n").notices
    notices = _parse(base).notices
    assert any("sin precio en [fuel] price para gasolina" in n for n in notices)
    # Sin participar, no hay aviso.
    assert not _parse(base + "[demand]\ncar_rate = {min = 0, max = 0, mean = 0, std = 0}\n").notices


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ('[vehicles.car.emissions]\nsource = "copert"\n', "no existe el conjunto 'copert'; hay int_panis_2006, sedema_cdmx_2018"),
        ('[vehicles.bike.emissions]\nsource = "sedema_cdmx_2018"\n', "el conjunto 'sedema_cdmx_2018' no tiene el tipo 'bike'"),
        ('[vehicles.car.emissions]\nsource = "int_panis_2006"\nsource_type = "taxi"\n', "no tiene el tipo 'taxi' (source_type)"),
        ('[vehicles.car.emissions]\nsource_type = "taxi"\n', "source_type requiere source"),
        ('[vehicles.car]\nfuel = "hidrogeno"\n', "fuel debe ser uno de gasolina, diesel, glp"),
        ("[vehicles.car]\nmass_kg = 0\n", "mass_kg debe ser mayor que 0"),
        ("[fuel]\nprice = {gasolina = -1}\n", "el precio de gasolina no puede ser negativo"),
        ("[fuel]\nprice = 24\n", "[fuel] price debe ser un diccionario"),
        ("[fuel]\nprice = {electricidad = 3}\n", "claves desconocidas: [fuel.price] electricidad"),
    ],
)
def test_invalid_fuel_and_sets_are_rejected(text, message):
    with pytest.raises(ConfigError, match=re.escape(message)):
        _parse(text)


@pytest.mark.parametrize("name", [CONFIG_NAME, "config.toml.example"])
def test_repository_configs_load_with_the_sedema_types(name):
    sim = parse_settings((project_root() / name).read_text(encoding="utf-8"), Path(name)).sim
    keys = [sp.key for sp in sim.specs]
    for key in ("taxi", "carga_ligera", "colectivo"):
        k = keys.index(key)
        assert sim.specs[k].emission_source == "sedema_cdmx_2018" and sim.specs[k].burns
        assert name == CONFIG_NAME or sim.rates[k] == 0  # el ejemplo los trae sin demanda; config.toml es del usuario
    assert sim.specs[0].emission_source == "sedema_cdmx_2018" and sim.price("gasolina") == 23.99


def test_several_traffic_lights():
    """[[traffic_light]]: un semáforo por sección, cada uno con su position, fases, fase inicial y free_lanes. El que
    no tiene position (o está en length) es el del final del tramo; los demás, intermedios."""
    text = """
[road]
length = 200
max_line_speed = [40, 50]
[[traffic_light]]
position = 120
red = 20
green = 25
yellow = 3
start_phase = "green"
[[traffic_light]]
position = 50
red = 15
green = 15
[[traffic_light]]
red = 30
green = 30
"""
    sim = _parse(text).sim
    assert [(lt.position, lt.red, lt.green, lt.yellow, lt.start_phase) for lt in sim.extra_lights] == [
        (50.0, 15.0, 15.0, 0.0, "red"), (120.0, 20.0, 25.0, 3.0, "green")]  # fmt: skip
    assert sim.has_light and (sim.red, sim.green, sim.start_phase) == (30.0, 30.0, "red")
    assert len(sim.lights) == 3 and sim.lights_label.count("·") == 2
    # Una sola sección con position < length: ese es el único semáforo, y el del final queda sin semáforo.
    sim = _parse("[road]\nlength = 200\n[traffic_light]\nposition = 80\nred = 10\ngreen = 10\n").sim
    assert not sim.has_light and [lt.position for lt in sim.inner_lights] == [80.0] and sim.flow_window == 20.0
    # position = length es el del final; sin [traffic_light] nada cambia.
    sim = _parse("[road]\nlength = 200\n[[traffic_light]]\nposition = 200\nred = 10\ngreen = 10\n").sim
    assert sim.has_light and sim.extra_lights == () and sim.red == 10.0
    assert _parse("").sim.extra_lights == () and _parse("").sim.has_light


def test_traffic_light_free_lanes_are_per_light():
    text = ("[road]\nlength = 200\nmax_line_speed = [40, 40, 50]\n[vehicles.bike]\nlane = 0\nexclusive = true\n"
            "[[traffic_light]]\nposition = 100\nfree_lanes = [0]\n[[traffic_light]]\n")  # fmt: skip
    sim = _parse(text).sim
    assert sim.extra_lights[0].free_lanes == (0,) and sim.free_lanes == ()
    k = [sp.key for sp in sim.specs].index("bike")
    assert sim.ignores_light(k, sim.extra_lights[0]) and not sim.ignores_light(k)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("[[traffic_light]]\nposition = 50\n[[traffic_light]]\nposition = 50\n", "hay semáforos en la misma posición"),
        ("[[traffic_light]]\nposition = 250\n", "cada semáforo intermedio debe estar entre 0 y 200 m"),
        ("[[traffic_light]]\nposition = 0\n", "cada semáforo intermedio debe estar entre 0 y 200 m"),
        ("[[traffic_light]]\nred = 10\n[[traffic_light]]\nposition = 200\n", "hay 2 semáforos en el final del tramo"),
        ("[[traffic_light]]\nposition = 50\nfoo = 1\n", "claves desconocidas en el semáforo 1 de [[traffic_light]]: foo"),
        ("[[traffic_light]]\nposition = 50\ngreen = 0\n", "[traffic_light] a 50 m green debe ser mayor que 0"),
        ("[[traffic_light]]\nposition = 50\nstart_phase = \"azul\"\n", 'a 50 m start_phase debe ser "red" o "green"'),
        ("[[traffic_light]]\nposition = 50\nred = 10.05\n", "a 50 m red debe ser múltiplo de 0.1 s"),
        ("[[traffic_light]]\nposition = 50\nfree_lanes = [0]\n", "a 50 m free_lanes: el carril 0 no es exclusivo"),
        ("[[traffic_light]]\nposition = 50\nfree_lanes = [5]\n", "a 50 m free_lanes: cada carril debe estar entre 0 y"),
        ("traffic_light = 3\n", "[traffic_light] debe ser una sección o una lista de secciones"),
    ],
)
def test_invalid_several_traffic_lights_are_rejected(text, message):
    with pytest.raises(ConfigError, match=re.escape(message)):
        _parse("[road]\nlength = 200\nmax_line_speed = [40, 40]\n" + text if not text.startswith("traffic_light") else text)


def test_plot_options_in_output(root):
    """mobility_plot, passengers_plot y emissions_plot: por defecto se dibujan las tres; en false no se escribe su PNG
    (el CSV de emisiones y el resto de las salidas sí)."""
    assert _parse("").run.mobility_plot and _parse("").run.emissions_plot and _parse("").run.passengers_plot
    with pytest.raises(ConfigError, match="mobility_plot debe ser true o false"):
        _parse("[output]\nmobility_plot = 1\n")
    emits = "[vehicles.car]\naccel = 2.5\ndecel = 4.5\n"
    (root / CONFIG_NAME).write_text(FAST + emits, encoding="utf-8")
    names = {p.name for p in run([]).iterdir()}
    assert {"movilidad_pasajeros.png", "emisiones_posicion.png", "emisiones_posicion.csv",
            "distribucion_pasajeros.png"} <= names  # fmt: skip
    off = FAST + "mobility_plot = false\nemissions_plot = false\npassengers_plot = false\n" + emits  # en [output]
    (root / CONFIG_NAME).write_text(off, encoding="utf-8")
    names = {p.name for p in run([]).iterdir()}
    assert not {"movilidad_pasajeros.png", "emisiones_posicion.png", "distribucion_pasajeros.png"} & names
    assert {"resumen.txt", "emisiones_posicion.csv", "series.csv"} <= names


def test_csv_options_in_output(root):
    """series_csv y emisiones_csv (por defecto, las dos): cada una escribe su archivo, independiente de las gráficas.
    `csv`, el nombre anterior de series_csv, sigue valiendo en las copias viejas, con un aviso."""
    assert _parse("").run.series_csv and _parse("").run.emisiones_csv and not hasattr(_parse("").run, "csv")
    s = _parse("[output]\ncsv = false\n")
    assert not s.run.series_csv and any("[output] csv se renombró series_csv" in n for n in s.notices)
    assert _parse("[output]\nseries_csv = false\n").run.series_csv is False and not _parse("").notices
    emits = "[vehicles.car]\naccel = 2.5\ndecel = 4.5\n"
    (root / CONFIG_NAME).write_text(FAST + emits, encoding="utf-8")
    assert {"series.csv", "emisiones_posicion.csv", "emisiones_posicion.png"} <= {p.name for p in run([]).iterdir()}
    off = FAST + "series_csv = false\nemisiones_csv = false\n" + emits
    (root / CONFIG_NAME).write_text(off, encoding="utf-8")
    names = {p.name for p in run([]).iterdir()}
    assert not {"series.csv", "emisiones_posicion.csv"} & names and "emisiones_posicion.png" in names
    (root / CONFIG_NAME).write_text(FAST + "csv = false\n" + emits, encoding="utf-8")  # copia anterior
    assert "series.csv" not in {p.name for p in run([]).iterdir()}
