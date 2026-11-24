"""Separación entre topes o semáforos: cuánto deben alejarse dos topes (`uv run traffic-emissions --spacing`) o dos
semáforos (`--light-spacing`) para que la huella de emisiones del primero se diluya antes del segundo.

Corre, con la misma semilla que `traffic`, la calle sin tope, con un solo tope en `P1` y con dos topes en `P1` y
`P1 + d` para cada separación `d` (o, con `--chain N`, con una cadena de N topes en `P1`, `P1 + d`, …, `P1 + (N − 1)·d`).
Del perfil de emisiones a lo largo de la calle (g/(m·h), todos los carriles) de cada
escenario se calculan, por contaminante:

  * aditividad R(d) = N(d) / (n·N1), con n los topes de la cadena (2 por omisión), N el exceso neto sobre la calle sin
    tope (∫ (E − base) dx) y N1 el de un tope solo: 1 = la cadena emite como n topes aislados; menos de 1, las estelas se
    solapan;
  * costo de cada tope añadido (N − N1) / ((n − 1)·N1), en topes aislados, y por metro de separación: cuánto emite de
    más un tope pegado a los anteriores y cuánto cuesta cada metro de zona lenta que añade;
  * huella L_h: distancia tras un tope solo hasta que el exceso baja de `umbral` × su pico;
  * media en el tramo [P1, Pn + L_h] contra la de la calle sin tope;
  * exceso medio entre el primer y el segundo tope (÷ pico de un tope solo);
  * gradiente medio |dE/dx| del perfil en todo ese tramo (sin ventanas interiores: no depende de dónde caiga su borde).

La separación mínima es la menor de la lista donde la cadena es aditiva (|R − 1| ≤ tolerancia) y el segundo tope queda
fuera de la huella del primero (d ≥ L_h).

Con semáforos (`kind = "semaforo"`) la calle no tiene los semáforos ni los topes de la configuración (salvo los topes que
se pidan fijos): cada escenario pone semáforos de ciclo fijo en P1, P1 + d, …, y con el desfase «rojo» cada uno pasa a rojo
cuando el anterior pasa a verde, así que el pelotón que sale de uno encuentra el siguiente en rojo. La huella de un
semáforo tiene dos lados: la cola y el frenado antes de él (L_antes) y la aceleración después (L_despues); el segundo
queda fuera de la huella del primero si d ≥ L_antes + L_despues, y el tramo de la media y del gradiente empieza en
P1 − L_antes. Se calcula además la demora: el tiempo de recorrido añadido por la cadena ÷ (n × el de un semáforo solo).
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from trafico.config import DT, POLLUTANT_LABELS, POLLUTANTS, Light, SimConfig
from trafico.emission_scenarios import DATA_NAME, META_NAME, scenario_bumps
from trafico.emissions import EMIS_BIN, unit
from trafico.settings import ConfigError, RunOptions, validate_config

PLOT_NAME = "bump_spacing.png"
TABLE_CSV = "bump_spacing.csv"
PROFILE_CSV = "spacing_profiles.csv"
SUMMARY_NAME = "summary.txt"
DEFAULT_DISTANCES = (10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0)
MODE = "separacion"
EDGE = 5.0  # m junto a cada tope que se dejan fuera del exceso medio entre topes
BUMP, LIGHT = "tope", "semaforo"
LIGHT_DISTANCES = (25.0, 50.0, 75.0, 100.0, 150.0, 200.0, 250.0, 300.0)
LIGHT_FIRST = 100.0  # m, primer semáforo por omisión: deja sitio a su cola
LIGHT_CYCLE = Light(0.0, red=30.0, green=30.0, yellow=3.0)  # semáforo por omisión si la configuración no tiene uno de ciclo fijo
LIGHT_ROOM = 150.0  # m tras el último semáforo al ampliar el tramo: su cola de salida y la aceleración
# Desfase entre semáforos consecutivos: el siguiente pasa a rojo al llegar el pelotón (el peor caso), a verde al llegar
# (onda verde) o tiene la misma fase.
OFFSETS = ("rojo", "verde", "igual")


@dataclass(frozen=True, slots=True)
class Nouns:
    one: str  # «bump»
    many: str  # «bumps»

    @property
    def a(self) -> str:
        return f"one {self.one}"

    @property
    def none(self) -> str:
        return f"no {self.one}"


NOUNS = {BUMP: Nouns("bump", "bumps"), LIGHT: Nouns("traffic light", "traffic lights")}


def kind_of(meta: dict) -> str:
    """Qué se separa en un barrido ya corrido: «tope» (también en las carpetas anteriores a los semáforos) o «semaforo»."""
    return meta.get("elemento", BUMP)


def first_position(meta: dict) -> float:
    """Posición (m) del primer tope o semáforo del barrido."""
    return meta["primer_semaforo_m"] if kind_of(meta) == LIGHT else meta["primer_tope_m"]


def plot_name(meta: dict) -> str:
    return PLOT_NAME if kind_of(meta) == BUMP else "light_spacing.png"


def table_name(meta: dict) -> str:
    return TABLE_CSV if kind_of(meta) == BUMP else "light_spacing.csv"


@dataclass(frozen=True, slots=True)
class Spacing:
    """Qué se compara."""

    first: float  # m, posición del primer tope
    distances: tuple[float, ...]  # m entre el primer y el segundo tope
    bump_lanes: tuple[int, ...] | None  # carriles de los topes (ya renumerados); None = los de la configuración
    light: bool | None  # semáforos activados en todos los escenarios; None = como en la configuración
    length: float | None  # m del tramo; None = [road] length
    tolerance: float  # |R − 1| máximo para considerar aditiva la cadena
    threshold: float  # fracción del pico de un tope bajo la cual el exceso cuenta como diluido
    replicas: int
    run: float  # s de proceso
    count: int = 2  # topes de la cadena (2 = una pareja)
    kind: str = BUMP  # «tope» | «semaforo»
    cycle: Light | None = None  # con semáforos: las fases de todos (rojo, verde, amarillo); la posición no se usa
    offset_mode: str = "rojo"  # con semáforos: desfase entre consecutivos (ver OFFSETS); uno por tramo separado por comas
    bumps: tuple[float, ...] = ()  # con semáforos: topes fijos en todos los escenarios (m)
    fixed_gaps: tuple[float, ...] = ()  # separaciones fijas (m) tras la que se barre: count − 2 valores; vacía = todas d

    @property
    def nouns(self) -> Nouns:
        return NOUNS[self.kind]

    @property
    def offsets(self) -> tuple[str, ...]:
        """Desfase de cada tramo de la cadena (count − 1): uno para todos o uno por tramo."""
        modes = tuple(self.offset_mode.split(","))
        return modes * (self.count - 1) if len(modes) == 1 else modes

    def gaps(self, d: float) -> tuple[float, ...]:
        """Separación de cada tramo de la cadena: d en el primero y, con `fixed_gaps`, esas en los siguientes."""
        return (d, *self.fixed_gaps) if self.fixed_gaps else (d,) * (self.count - 1)

    def positions(self, d: float) -> tuple[float, ...]:
        return tuple(self.first + g for g in np.cumsum((0.0, *self.gaps(d))).tolist())


def chain_span(meta: dict, d: float) -> float:
    """m entre el primer y el último elemento de la cadena con separación d (con las separaciones fijas, si hay)."""
    fixed = meta.get("separaciones_fijas_m") or []
    return d + sum(fixed) if fixed else (int(meta.get("topes_por_cadena", 2)) - 1) * d


def platoon_type(cfg: SimConfig) -> int:
    """El tipo que marca la llegada del pelotón: el de mayor tasa entre los que participan."""
    return max((k for k, rate in enumerate(cfg.rates) if rate > 0), key=lambda k: cfg.rates[k])


def travel_time(cfg: SimConfig, d: float) -> float:
    """s que tarda el tipo del pelotón (`platoon_type`) en recorrer d m desde detenido: acelera con su `accel` hasta su
    velocidad a flujo libre y sigue a esa velocidad (sin `accel`, sale ya a esa velocidad)."""
    k = platoon_type(cfg)
    v, a = cfg.free_flow_kmh(k) / 3.6, cfg.specs[k].accel
    if not a:
        return d / v
    reach = v * v / (2 * a)  # m hasta alcanzar la velocidad
    return float(np.sqrt(2 * d / a)) if d < reach else d / v + v / (2 * a)


def chain_lights(s: Spacing, positions: tuple[float, ...], cfg: SimConfig) -> tuple[Light, ...]:
    """Semáforos de ciclo fijo en esas posiciones, como `s.cycle`, empezando en rojo. Cada uno se retrasa respecto del
    anterior según el desfase de su tramo (`s.offsets`), con τ lo que tarda la cabeza del pelotón en recorrerlo desde
    detenida (`travel_time`): «rojo», rojo + τ (pasa a rojo justo cuando llega el pelotón que salió del anterior al
    ponerse en verde); «verde», τ (pasa a verde justo cuando llega: onda verde); «igual», 0. Módulo el ciclo, redondeado
    al paso."""
    c = s.cycle or LIGHT_CYCLE
    cycle = c.red + c.green + c.yellow
    shift, out = 0.0, []
    for i, p in enumerate(positions):
        if i:
            mode, trip = s.offsets[i - 1], travel_time(cfg, p - positions[i - 1])
            shift += {"rojo": c.red + trip, "verde": trip, "igual": 0.0}[mode]
        at = round(round((shift % cycle) / DT) * DT, 6)
        out.append(Light(p, red=c.red, green=c.green, yellow=c.yellow, start_phase="red",
                         offset=0.0 if at >= cycle else at))  # fmt: skip
    return tuple(out)


def build_configs(reduced: SimConfig, s: Spacing) -> tuple[list[tuple[str, SimConfig]], list[float]]:
    """([(nombre, configuración validada)], separaciones omitidas por no caber). Orden: sin tope, un tope en `first` y un
    escenario por separación con `count` topes. El último debe quedar al menos un intervalo de EMIS_BIN antes del final
    del tramo."""
    base = reduced if s.length is None else replace(reduced, length=s.length)
    base = base if s.light is None else base.with_lights(s.light)
    nn, flag = s.nouns, "--first-bump" if s.kind == BUMP else "--first-light"
    if s.kind == LIGHT:  # solo los semáforos del barrido y los topes fijos que se pidan
        base = replace(base, traffic_light=False, extra_lights=(),
                       speed_bumps=scenario_bumps(base, tuple(sorted(s.bumps)), s.bump_lanes))  # fmt: skip
    if not 0 < s.first < base.length:
        raise ConfigError(f"{flag} must be between 0 and {base.length:g} m (the segment length)")
    if s.count < 2:
        raise ConfigError(f"--chain must be at least 2 {'bumps' if s.kind == BUMP else 'lights'}")
    if s.fixed_gaps and len(s.fixed_gaps) != s.count - 2:
        raise ConfigError(f"--fixed-spacings: with --chain {s.count}, {s.count - 2} values are needed (the segments "
                          "after the swept one)")  # fmt: skip
    if len(s.offsets) != s.count - 1 or any(m not in OFFSETS for m in s.offsets):
        raise ConfigError(f"--offset: one of red, green or same for all segments, or one per segment separated by "
                          f"commas ({s.count - 1} with --chain {s.count})")  # fmt: skip
    kept = [d for d in s.distances if s.positions(d)[-1] <= base.length - EMIS_BIN]
    skipped = [d for d in s.distances if d not in kept]
    if not kept:
        noun = "bump" if s.kind == BUMP else "light"
        raise ConfigError(f"no spacing fits on the segment: the first {noun} is at {s.first:g} m and the segment is "
                          f"{base.length:g} m long (the last of {s.count} {noun}s must be {EMIS_BIN:g} m or more from the "
                          "end)")  # fmt: skip
    specs = [(nn.none, ()), (f"{nn.a} at {s.first:g} m", (s.first,))]
    specs += [(chain_name(s, d), s.positions(d)) for d in kept]
    out = []
    for name, positions in specs:
        if s.kind == LIGHT:
            cfg = replace(base, run=s.run, extra_lights=chain_lights(s, positions, base))
        else:
            cfg = replace(base, run=s.run, speed_bumps=scenario_bumps(base, positions, s.bump_lanes))
        try:
            validate_config(cfg, RunOptions(replicas=s.replicas))
        except ConfigError as exc:
            raise ConfigError(f"scenario «{name}»: {exc}") from None
        out.append((name, cfg))
    return out, skipped


def chain_name(s: Spacing, d: float) -> str:
    many = s.nouns.many
    if s.count == 2:
        return f"{many} at {s.first:g} and {s.first + d:g} m (d = {d:g} m)"
    if s.fixed_gaps:
        at = [f"{p:g}" for p in s.positions(d)]
        return f"{many} at {', '.join(at[:-1])} and {at[-1]} m (d = {d:g} m)"
    return f"{s.count} {many} from {s.first:g} m every {d:g} m"


def profiles(meta: dict, data: dict[str, np.ndarray]) -> np.ndarray:
    """Perfil de emisiones a lo largo de la calle, (escenario, contaminante de meta, intervalo), en g/(m·h) sumando los
    carriles."""
    pidx = [POLLUTANTS.index(p) for p in meta["contaminantes"]]
    pos = data["emissions_pos"].sum(axis=2) / EMIS_BIN / (meta["s_simulados"] / 3600.0)
    return pos[:, pidx, :]


def _first(distances: list[float], ok: np.ndarray) -> float | None:
    return next((d for d, good in zip(distances, ok) if good), None)


def travel_times(data: dict[str, np.ndarray]) -> np.ndarray | None:
    """Tiempo de recorrido medio de cada escenario (s), ponderado por los vehículos de cada tipo que cruzan; None si los
    datos no lo traen."""
    if "travel_time" not in data or "crossed_veh" not in data:
        return None
    tt, w = np.asarray(data["travel_time"], float), np.asarray(data["crossed_veh"], float)
    w = np.where(np.isfinite(tt), w, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nansum(np.nan_to_num(tt) * w, axis=1) / w.sum(axis=1)


def analyze(meta: dict, data: dict[str, np.ndarray]) -> dict:
    """Métricas de la separación entre topes o semáforos (ver el módulo). Por contaminante: arreglos con un valor por
    separación (NaN donde no aplica), la huella, el pico y las separaciones mínimas (None si ninguna de la lista
    cumple)."""
    prof = profiles(meta, data)
    x = (np.arange(prof.shape[2]) + 0.5) * EMIS_BIN
    first, dist = first_position(meta), list(meta["distancias_m"])
    tol, thr = meta["tolerancia"], meta["umbral"]
    n = int(meta.get("topes_por_cadena", 2))
    light = kind_of(meta) == LIGHT
    i0 = int(np.searchsorted(x, first))  # primer intervalo con el centro en P1 o después
    out: dict = {"distancias_m": dist, "x_m": x, "pollutants": {}}
    times = travel_times(data)
    delay = np.full(len(dist), np.nan)
    if times is not None and times[1] > times[0]:
        delay = (times[2:] - times[0]) / (n * (times[1] - times[0]))
    out["tiempo_recorrido_s"] = None if times is None else times.tolist()
    out["demora"] = delay
    crossed = np.asarray(data["crossed_veh"], float).sum(axis=1) if "crossed_veh" in data else None
    out["salen"] = (crossed[2:] / crossed[0] if crossed is not None and crossed[0] > 0
                    else np.full(len(dist), np.nan))  # fmt: skip
    blocked = np.zeros(len(dist), bool)  # la cola de la cadena llega a la entrada del tramo (con semáforos)
    for pi, pol in enumerate(meta["contaminantes"]):
        base, single = prof[0, pi], prof[1, pi]
        ex1 = single - base
        peak = float(ex1.max())
        n1 = float(ex1.sum() * EMIS_BIN)
        i_pk = int(np.argmax(ex1))
        # Huella después: desde el pico (con semáforos, desde P1 si el pico es la cola) hasta que el exceso baja de
        # umbral × pico. Antes: hasta el intervalo anterior al más lejano antes de P1 que lo supera (con las mismas
        # semillas, aguas arriba de la cola el exceso es 0); si ese es el primero, la cola llega a la entrada.
        start = max(i_pk, i0) if light else i_pk
        below = np.flatnonzero(ex1[start:] <= thr * peak)
        length_h = float(x[start + below[0]] - first) if below.size else float("nan")
        above = np.flatnonzero(ex1[:i0] > thr * peak)
        at_entry = bool(above.size and above[0] == 0)
        length_b = float(first) if at_entry else float(first - x[above[0] - 1]) if above.size else 0.0
        reaches_entry = light and at_entry
        lead = length_b if light else 0.0  # el tramo de la media y del gradiente empieza en P1 − lead
        after = (x >= first) & (x <= first + (length_h if np.isfinite(length_h) else x[-1]))
        g1 = float(np.abs(np.diff(single[after])).max() / EMIS_BIN) if after.sum() > 1 else float("nan")
        cols = {k: np.full(len(dist), np.nan) for k in
                ("aditividad", "media_tramo", "exceso_entre", "gradiente_tramo", "pendiente_max", "costo_tope", "costo_por_m")}
        for j, d in enumerate(dist):
            e, p2 = prof[2 + j, pi], first + d
            ex = e - base
            total = ex.sum() * EMIS_BIN
            cols["aditividad"][j] = total / (n * n1) if n1 > 0 else np.nan
            if n1 > 0:
                cols["costo_tope"][j] = (total - n1) / ((n - 1) * n1)
                cols["costo_por_m"][j] = 100 * cols["costo_tope"][j] / d
            end = first + chain_span(meta, d) + (length_h if np.isfinite(length_h) else 0.0)
            stretch = (x >= first - lead) & (x <= end)
            if stretch.any() and base[stretch].mean() > 0:
                cols["media_tramo"][j] = e[stretch].mean() / base[stretch].mean()
            if stretch.sum() > 1:
                slopes = np.abs(np.diff(e[stretch])) / EMIS_BIN
                cols["pendiente_max"][j] = slopes.max()
                cols["gradiente_tramo"][j] = slopes.mean()
            between = (x > first + EDGE) & (x < p2 - EDGE)
            if between.any():
                cols["exceso_entre"][j] = ex[between].mean() / peak if peak > 0 else np.nan
            blocked[j] |= light and peak > 0 and ex[0] > thr * peak
        reach = length_h + lead  # el segundo debe quedar fuera de la estela del primero y su cola, fuera también
        d_wake = _first(dist, np.array(dist) >= reach) if np.isfinite(reach) else None
        stable = np.full(len(dist), np.nan)
        if light:
            # Con el desfase «rojo» el pelotón entero se detiene en el siguiente semáforo (en uno aislado, solo los que
            # llegan en rojo): la pareja no tiende a dos semáforos aislados sino a la pareja lejana. Se pide que la
            # emisión ya no cambie al alejar el segundo: dentro de ±tol de la más separada en esa d y en todas las
            # mayores, sin contar la más separada sola ni separaciones donde la cola llega a la entrada.
            stable = cols["aditividad"] / cols["aditividad"][-1]
            ok = (np.abs(stable - 1) <= tol) & ~blocked
            ok = np.logical_and.accumulate(ok[::-1])[::-1]
            far = d_wake is not None
            d_add = next((d for d, good in zip(dist[:-1], ok[:-1]) if good), None) if far else None
        else:
            d_add = _first(dist, np.abs(cols["aditividad"] - 1) <= tol)
        both = None if d_add is None or d_wake is None else max(d_add, d_wake)
        out["pollutants"][pol] = {
            **cols, "estabilidad": stable, "pico": peak, "huella_m": length_h, "huella_antes_m": length_b,
            "cola_en_entrada": reaches_entry, "pendiente_tope_solo": g1, "neto_tope_solo": n1, "d_aditiva": d_add,
            "d_huella": d_wake, "d_minima": both,
        }  # fmt: skip
    found = [r["d_minima"] for r in out["pollutants"].values() if r["d_minima"] is not None]
    out["recomendada_m"] = max(found) if found else None
    lengths = [r["huella_m"] for r in out["pollutants"].values() if np.isfinite(r["huella_m"])]
    out["huella_m"] = max(lengths) if lengths else None
    out["huella_antes_m"] = max(r["huella_antes_m"] for r in out["pollutants"].values()) if out["pollutants"] else None
    out["cola_en_entrada"] = light and any(r["cola_en_entrada"] for r in out["pollutants"].values())
    out["cola_entrada_d"] = blocked
    return out


def metadata(base_meta: dict, s: Spacing, kept: list[float], skipped: list[float],
             cfg: SimConfig | None = None) -> dict:
    """`emission_scenarios.metadata` más lo propio del barrido, para describir, graficar y repetir."""
    meta = {
        **base_meta, "modo": MODE, "elemento": s.kind, "distancias_m": kept, "omitidas_m": skipped,
        "tolerancia": s.tolerance, "umbral": s.threshold, "topes_por_cadena": s.count,
    }  # fmt: skip
    if s.kind == BUMP:
        return {**meta, "primer_tope_m": s.first}
    c = s.cycle or LIGHT_CYCLE
    meta = {
        **meta, "primer_semaforo_m": s.first, "ciclo": {"rojo_s": c.red, "verde_s": c.green, "amarillo_s": c.yellow},
        "desfase": s.offset_mode, "topes_fijos_m": sorted(s.bumps), "separaciones_fijas_m": list(s.fixed_gaps),
    }  # fmt: skip
    if cfg is not None:  # el viaje que fija el desfase «rojo»: quién encabeza el pelotón y cuánto tarda en cada d
        meta["tipo_peloton"] = cfg.specs[platoon_type(cfg)].name
        meta["viaje_s"] = [round(travel_time(cfg, d), 2) for d in kept]
    return meta


def write_outputs(folder: Path, meta: dict, data: dict[str, np.ndarray]) -> str:
    """Datos crudos, parámetros, tablas y perfiles; devuelve el resumen en texto."""
    np.savez(folder / DATA_NAME, **data)
    (folder / META_NAME).write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result = analyze(meta, data)
    dist = result["distancias_m"]
    many = "bumps" if kind_of(meta) == BUMP else "lights"
    one = many[:-1]
    with open(folder / table_name(meta), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["pollutant", "spacing_m", "additivity", "segment_mean_x_base", f"excess_between_{many}_pct_peak",
                    "segment_gradient", "max_slope", f"added_{one}_cost", "cost_pct_per_m",
                    "extra_pct_vs_isolated", "delay_x_isolated"])  # fmt: skip
        for pol, r in result["pollutants"].items():
            for j, d in enumerate(dist):
                add = 100 * (r["aditividad"][j] - 1)
                w.writerow([pol, f"{d:g}", _num(r["aditividad"][j]), _num(r["media_tramo"][j]),
                            _num(100 * r["exceso_entre"][j]), _num(r["gradiente_tramo"][j]),
                            _num(r["pendiente_max"][j]), _num(r["costo_tope"][j]), _num(r["costo_por_m"][j]),
                            _num(add), _num(result["demora"][j])])  # fmt: skip
    prof = profiles(meta, data)
    with open(folder / PROFILE_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["x_start_m", "x_end_m"] + [f"{pol}_g_m_h_{i}" for pol in meta["contaminantes"]
                                                for i in range(len(meta["escenarios"]))])  # fmt: skip
        for b in range(prof.shape[2]):
            w.writerow([f"{b * EMIS_BIN:g}", f"{min((b + 1) * EMIS_BIN, meta['largo_m']):g}"]
                       + [f"{prof[i, p, b]:.6g}" for p in range(prof.shape[1]) for i in range(prof.shape[0])])  # fmt: skip
    return summary_text(meta, result)


def results_metrics(meta: dict, data: dict[str, np.ndarray]) -> dict:
    """Métricas de `resultados.json`: las de `analyze` por contaminante (un valor por separación) y la separación mínima."""
    r = analyze(meta, data)
    keep = ("aditividad", "media_tramo", "exceso_entre", "gradiente_tramo", "pendiente_max", "costo_tope", "costo_por_m",
            "estabilidad", "pico", "huella_m", "huella_antes_m", "pendiente_tope_solo", "neto_tope_solo", "d_aditiva",
            "d_huella", "d_minima")  # fmt: skip
    first = "primer_tope_m" if kind_of(meta) == BUMP else "primer_semaforo_m"
    return {
        "elemento": kind_of(meta), "topes_por_cadena": int(meta.get("topes_por_cadena", 2)), first: first_position(meta),
        "distancias_m": r["distancias_m"], "omitidas_m": meta.get("omitidas_m", []), "tolerancia": meta["tolerancia"],
        "umbral": meta["umbral"], "recomendada_m": r["recomendada_m"], "huella_m": r["huella_m"],
        "huella_antes_m": r["huella_antes_m"], "cola_en_entrada": r["cola_en_entrada"],
        "tiempo_recorrido_s": r["tiempo_recorrido_s"], "demora": r["demora"], "salen_x_base": r["salen"],
        "cola_hasta_la_entrada": r["cola_entrada_d"],
        "contaminantes": {pol: {k: v[k] for k in keep} for pol, v in r["pollutants"].items()},
        "parametros": meta,
    }  # fmt: skip


def _num(v: float) -> str:
    return "" if not np.isfinite(v) else f"{v:.4g}"


def _dist(d: float | None) -> str:
    return "none in the list" if d is None else f"{d:g} m"


def worst_case(meta: dict) -> bool:
    """Todos los semáforos de la cadena pasan a rojo al llegar el pelotón: tiene sentido buscar la separación mínima.
    Con onda verde o fases iguales en algún tramo, lo que se mide es a cuántos semáforos aislados equivale la cadena."""
    return set(meta.get("desfase", "rojo").split(",")) == {"rojo"}


def equivalence_text(meta: dict, result: dict) -> str:
    """«The chain emits like 2.1–2.2 isolated traffic lights (CO2)»: the range of n × R over the comparable spacings."""
    pol = "co2" if "co2" in result["pollutants"] else next(iter(result["pollutants"]))
    n = int(meta.get("topes_por_cadena", 2))
    ok = ~np.asarray(result.get("cola_entrada_d", np.zeros(len(result["distancias_m"]), bool)), bool)
    eq = n * np.asarray(result["pollutants"][pol]["aditividad"], float)[ok]
    eq = eq[np.isfinite(eq)]
    if not eq.size:
        return "No comparable spacings"
    what = "The pair" if n == 2 else f"The chain of {n}"
    span = f"{eq.min():.2f}" if np.isclose(eq.min(), eq.max(), atol=0.005) else f"{eq.min():.2f}–{eq.max():.2f}"
    return f"{what} emits like {span} isolated traffic lights ({POLLUTANT_LABELS[pol]})"


def cycle_text(meta: dict) -> str:
    """Phases and offsets of the lights of a sweep, e.g. «red 30 s / green 30 s / yellow 3 s, each one turns red when
    the platoon from the previous one arrives»."""
    c = meta["ciclo"]
    text = f"red {c['rojo_s']:g} s / green {c['verde_s']:g} s" + (
        f" / yellow {c['amarillo_s']:g} s" if c["amarillo_s"] > 0 else "")
    modes = meta["desfase"].split(",")
    words = {"rojo": "red when the platoon arrives", "verde": "green when the platoon arrives (green wave)",
             "igual": "in the same phase"}  # fmt: skip
    if len(set(modes)) == 1:
        return text + {"rojo": ", each one turns red when the platoon from the previous one arrives",
                       "verde": ", green wave: each one turns green when the platoon from the previous one arrives",
                       "igual": ", all in the same phase"}[modes[0]]  # fmt: skip
    ordinal = {2: "2nd", 3: "3rd"}
    return text + " · " + ", ".join(f"{ordinal.get(i + 2, f'{i + 2}th')} {words[m]}" for i, m in enumerate(modes))


def summary_text(meta: dict, result: dict) -> str:
    """Tabla por separación y contaminante, la huella de un tope (o semáforo) y las separaciones mínimas."""
    dist = result["distancias_m"]
    n = int(meta.get("topes_por_cadena", 2))
    light = kind_of(meta) == LIGHT
    nn = NOUNS[kind_of(meta)]
    chain = f"chain of {n} {nn.many}, " if n > 2 else ""
    lines = [
        f"Spacing between {nn.many} ({chain}first {nn.one} at {first_position(meta):g} m; mean of {meta['replicas']} "
        f"replicas, {meta['s_simulados']:,.0f} simulated s; tolerance {meta['tolerancia']:g}, threshold {meta['umbral']:g} "
        "of the peak):",
    ]  # fmt: skip
    if light:
        fixed = meta.get("topes_fijos_m") or []
        lines.append(f"Fixed-cycle traffic lights: {cycle_text(meta)} · segment {meta['largo_m']:g} m · "
                     + (f"fixed bumps at {', '.join(f'{p:g}' for p in fixed)} m" if fixed else "no bumps"))  # fmt: skip
    if meta["omitidas_m"]:
        lines.append("Skipped because they do not fit on the segment: " + ", ".join(f"{d:g}" for d in meta["omitidas_m"]) + " m")
    if result.get("cola_en_entrada"):
        lines.append("Warning: the queue of the first light reaches the entry of the segment and cuts off arrivals; use a "
                     "larger --first-light")  # fmt: skip
    for pol, r in result["pollutants"].items():
        name, scale = unit(pol)
        wake = (f"{r['huella_antes_m']:.0f} m before + {r['huella_m']:.0f} m after" if light
                else f"{r['huella_m']:.0f} m")  # fmt: skip
        if light:
            lines += _light_rows(pol, r, result, dist, wake, scale, name, n)
            continue
        lines += [
            "",
            f"{POLLUTANT_LABELS[pol]}: footprint of one {nn.one} {wake}, peak +{r['pico'] * scale:,.1f} {name}/(m·h) · "
            f"additive from {_dist(r['d_aditiva'])} · outside the footprint from {_dist(r['d_huella'])}",
            f"{'d (m)':>7}{'additivity':>12}{'mean/base':>12}{'excess between':>16}{'gradient':>12}"
            f"{f'{nn.one} cost':>{max(12, len(nn.one) + 8)}}{'% per m':>9}",
        ]  # fmt: skip
        for j, d in enumerate(dist):
            row = f"{d:>7g}{r['aditividad'][j]:>12.2f}{r['media_tramo'][j]:>12.2f}"
            if np.isfinite(r["exceso_entre"][j]):
                row += f"{100 * r['exceso_entre'][j]:>15.0f}%"
            else:
                row += f"{'—':>16}"
            grad = r["gradiente_tramo"][j] * scale
            row += f"{grad:>12.2f}" if np.isfinite(grad) else f"{'—':>12}"
            cost, per_m = r["costo_tope"][j], r["costo_por_m"][j]
            width = max(12, len(nn.one) + 8)
            row += f"{cost:>{width}.2f}{per_m:>9.2f}" if np.isfinite(cost) else f"{'—':>{width}}{'—':>9}"
            lines.append(row)
    rec = result["recomendada_m"]
    huella = result["huella_m"]
    if light and not worst_case(meta):
        return "\n".join(lines + ["", legend_light(dist), equivalence_text(meta, result)
                                   + " (with a green wave there is no minimum spacing to look for: see ≡ isolated)"])
    if light and huella is not None:
        wake = f" (footprint of one traffic light ≈ {result['huella_antes_m']:.0f} m before + {huella:.0f} m after)"
    else:
        wake = f" (footprint of one {nn.one} ≈ {huella:.0f} m)" if huella is not None else ""
    if light:
        legend = legend_light(dist)
    else:
        legend = (f"{nn.one} cost: what each added {nn.one} emits on top, in isolated {nn.many} (1 = like a single one); "
                  f"% per m: that cost per metre of spacing, in % of one isolated {nn.one}.")
    lines += [
        "",
        legend,
        f"Minimum spacing between {nn.many} to dilute the footprint: {_dist(rec)}{wake}"
        + ("" if rec is not None else
           f"; widen {'--light-spacing' if light else '--spacing'} with larger values"),
    ]  # fmt: skip
    return "\n".join(lines)


def legend_light(dist: list[float]) -> str:
    return ("additivity: extra emission of the chain ÷ (n × a single light); ≡ isolated: the same emission in single "
            f"lights (n × additivity); ÷ distant: the same emission ÷ that of the most separated chain (d = {dist[-1]:g} m); "
            "delay: added travel time ÷ (n × that of a single light); out: vehicles leaving the segment ÷ those of the "
            "street without a light; * the queue reaches the entry of the segment (cuts off arrivals: not comparable). The "
            "minimum spacing is the smallest from which the emission stays within ±tolerance of the most separated chain "
            "and the second light is outside the footprint.")


def _light_rows(pol: str, r: dict, result: dict, dist: list[float], wake: str, scale: float, name: str,
                n: int) -> list[str]:
    """Tabla de un contaminante en el barrido de semáforos."""
    delay, out_ratio, blocked = result["demora"], result["salen"], result["cola_entrada_d"]
    lines = [
        "",
        f"{POLLUTANT_LABELS[pol]}: footprint of one traffic light {wake}, peak +{r['pico'] * scale:,.1f} {name}/(m·h) · "
        f"stable from {_dist(r['d_aditiva'])} · outside the footprint from {_dist(r['d_huella'])} · with the most "
        f"separated chain, {r['aditividad'][-1]:.2f} × isolated lights",
        f"{'d (m)':>7}{'additivity':>12}{'≡ isolated':>12}{'÷ distant':>11}{'mean/base':>12}{'excess between':>16}"
        f"{'gradient':>11}{'delay':>9}{'out':>8}",
    ]  # fmt: skip

    def cell(v: float, width: int, fmt: str = ".2f") -> str:
        return f"{v:>{width}{fmt}}" if np.isfinite(v) else f"{'—':>{width}}"

    for j, d in enumerate(dist):
        ex = r["exceso_entre"][j]
        lines.append(f"{d:>7g}" + cell(r["aditividad"][j], 12) + cell(n * r["aditividad"][j], 12)
                     + cell(r["estabilidad"][j], 11)
                     + cell(r["media_tramo"][j], 12) + (f"{100 * ex:>15.0f}%" if np.isfinite(ex) else f"{'—':>16}")
                     + cell(r["gradiente_tramo"][j] * scale, 11) + cell(delay[j], 9) + cell(out_ratio[j], 8)
                     + (" *" if blocked[j] else ""))  # fmt: skip
    return lines
