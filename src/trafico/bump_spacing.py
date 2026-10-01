"""Separación entre topes: cuánto deben alejarse dos topes para que la huella de emisiones del primero se diluya antes
del segundo (`uv run trafico-emisiones --separacion`).

Corre, con la misma semilla que `trafico`, la calle sin tope, con un solo tope en `P1` y con dos topes en `P1` y
`P1 + d` para cada separación `d` (o, con `--cadena N`, con una cadena de N topes en `P1`, `P1 + d`, …, `P1 + (N − 1)·d`).
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
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from trafico.config import POLLUTANT_LABELS, POLLUTANTS, SimConfig
from trafico.emission_scenarios import DATA_NAME, META_NAME, scenario_bumps
from trafico.emissions import EMIS_BIN, unit
from trafico.settings import ConfigError, RunOptions, validate_config

PLOT_NAME = "separacion_topes.png"
TABLE_CSV = "separacion_topes.csv"
PROFILE_CSV = "perfiles_separacion.csv"
SUMMARY_NAME = "resumen.txt"
DEFAULT_DISTANCES = (10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0)
MODE = "separacion"
EDGE = 5.0  # m junto a cada tope que se dejan fuera del exceso medio entre topes


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


def build_configs(reduced: SimConfig, s: Spacing) -> tuple[list[tuple[str, SimConfig]], list[float]]:
    """([(nombre, configuración validada)], separaciones omitidas por no caber). Orden: sin tope, un tope en `first` y un
    escenario por separación con `count` topes. El último debe quedar al menos un intervalo de EMIS_BIN antes del final
    del tramo."""
    base = reduced if s.length is None else replace(reduced, length=s.length)
    base = base if s.light is None else base.with_lights(s.light)
    if not 0 < s.first < base.length:
        raise ConfigError(f"--primer-tope debe estar entre 0 y {base.length:g} m (largo del tramo)")
    if s.count < 2:
        raise ConfigError("--cadena debe ser de al menos 2 topes")
    last = s.count - 1  # separaciones entre el primer tope y el último
    kept = [d for d in s.distances if s.first + last * d <= base.length - EMIS_BIN]
    skipped = [d for d in s.distances if d not in kept]
    if not kept:
        raise ConfigError(f"ninguna separación cabe en el tramo: el primer tope está a {s.first:g} m y el tramo mide "
                          f"{base.length:g} m (el último de {s.count} topes debe quedar a {EMIS_BIN:g} m o más del final)")  # fmt: skip
    specs = [("sin tope", ()), (f"un tope a {s.first:g} m", (s.first,))]
    specs += [(chain_name(s, d), tuple(s.first + i * d for i in range(s.count))) for d in kept]
    out = []
    for name, positions in specs:
        cfg = replace(base, run=s.run, speed_bumps=scenario_bumps(base, positions, s.bump_lanes))
        try:
            validate_config(cfg, RunOptions(replicas=s.replicas))
        except ConfigError as exc:
            raise ConfigError(f"escenario «{name}»: {exc}") from None
        out.append((name, cfg))
    return out, skipped


def chain_name(s: Spacing, d: float) -> str:
    if s.count == 2:
        return f"topes a {s.first:g} y {s.first + d:g} m (d = {d:g} m)"
    return f"{s.count} topes desde {s.first:g} m cada {d:g} m"


def profiles(meta: dict, data: dict[str, np.ndarray]) -> np.ndarray:
    """Perfil de emisiones a lo largo de la calle, (escenario, contaminante de meta, intervalo), en g/(m·h) sumando los
    carriles."""
    pidx = [POLLUTANTS.index(p) for p in meta["contaminantes"]]
    pos = data["emissions_pos"].sum(axis=2) / EMIS_BIN / (meta["s_simulados"] / 3600.0)
    return pos[:, pidx, :]


def _first(distances: list[float], ok: np.ndarray) -> float | None:
    return next((d for d, good in zip(distances, ok) if good), None)


def analyze(meta: dict, data: dict[str, np.ndarray]) -> dict:
    """Métricas de la separación entre topes (ver el módulo). Por contaminante: arreglos con un valor por separación
    (NaN donde no aplica), la huella, el pico y las separaciones mínimas (None si ninguna de la lista cumple)."""
    prof = profiles(meta, data)
    x = (np.arange(prof.shape[2]) + 0.5) * EMIS_BIN
    first, dist = meta["primer_tope_m"], list(meta["distancias_m"])
    tol, thr = meta["tolerancia"], meta["umbral"]
    n = int(meta.get("topes_por_cadena", 2))
    out: dict = {"distancias_m": dist, "x_m": x, "pollutants": {}}
    for pi, pol in enumerate(meta["contaminantes"]):
        base, single = prof[0, pi], prof[1, pi]
        ex1 = single - base
        peak = float(ex1.max())
        n1 = float(ex1.sum() * EMIS_BIN)
        i_pk = int(np.argmax(ex1))
        below = np.flatnonzero(ex1[i_pk:] <= thr * peak)
        length_h = float(x[i_pk + below[0]] - first) if below.size else float("nan")
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
            end = first + (n - 1) * d + (length_h if np.isfinite(length_h) else 0.0)
            stretch = (x >= first) & (x <= end)
            if stretch.any() and base[stretch].mean() > 0:
                cols["media_tramo"][j] = e[stretch].mean() / base[stretch].mean()
            if stretch.sum() > 1:
                slopes = np.abs(np.diff(e[stretch])) / EMIS_BIN
                cols["pendiente_max"][j] = slopes.max()
                cols["gradiente_tramo"][j] = slopes.mean()
            between = (x > first + EDGE) & (x < p2 - EDGE)
            if between.any():
                cols["exceso_entre"][j] = ex[between].mean() / peak if peak > 0 else np.nan
        d_add = _first(dist, np.abs(cols["aditividad"] - 1) <= tol)
        d_wake = _first(dist, np.array(dist) >= length_h) if np.isfinite(length_h) else None
        both = None if d_add is None or d_wake is None else max(d_add, d_wake)
        out["pollutants"][pol] = {
            **cols, "pico": peak, "huella_m": length_h, "pendiente_tope_solo": g1, "neto_tope_solo": n1,
            "d_aditiva": d_add, "d_huella": d_wake, "d_minima": both,
        }  # fmt: skip
    found = [r["d_minima"] for r in out["pollutants"].values() if r["d_minima"] is not None]
    out["recomendada_m"] = max(found) if found else None
    lengths = [r["huella_m"] for r in out["pollutants"].values() if np.isfinite(r["huella_m"])]
    out["huella_m"] = max(lengths) if lengths else None
    return out


def metadata(base_meta: dict, s: Spacing, kept: list[float], skipped: list[float]) -> dict:
    """`emission_scenarios.metadata` más lo propio del barrido, para describir, graficar y repetir."""
    return {
        **base_meta, "modo": MODE, "primer_tope_m": s.first, "distancias_m": kept, "omitidas_m": skipped,
        "tolerancia": s.tolerance, "umbral": s.threshold, "topes_por_cadena": s.count,
    }  # fmt: skip


def write_outputs(folder: Path, meta: dict, data: dict[str, np.ndarray]) -> str:
    """Datos crudos, parámetros, tablas y perfiles; devuelve el resumen en texto."""
    np.savez(folder / DATA_NAME, **data)
    (folder / META_NAME).write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result = analyze(meta, data)
    dist = result["distancias_m"]
    with open(folder / TABLE_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["contaminante", "separacion_m", "aditividad", "media_tramo_x_base", "exceso_entre_topes_pct_pico",
                    "gradiente_tramo", "pendiente_maxima", "costo_tope_anadido", "costo_pct_por_m",
                    "adicional_pct_vs_aislados"])  # fmt: skip
        for pol, r in result["pollutants"].items():
            for j, d in enumerate(dist):
                add = 100 * (r["aditividad"][j] - 1)
                w.writerow([pol, f"{d:g}", _num(r["aditividad"][j]), _num(r["media_tramo"][j]),
                            _num(100 * r["exceso_entre"][j]), _num(r["gradiente_tramo"][j]),
                            _num(r["pendiente_max"][j]), _num(r["costo_tope"][j]), _num(r["costo_por_m"][j]),
                            _num(add)])  # fmt: skip
    prof = profiles(meta, data)
    with open(folder / PROFILE_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["x_inicio_m", "x_fin_m"] + [f"{pol}_g_m_h_{i}" for pol in meta["contaminantes"]
                                                for i in range(len(meta["escenarios"]))])  # fmt: skip
        for b in range(prof.shape[2]):
            w.writerow([f"{b * EMIS_BIN:g}", f"{min((b + 1) * EMIS_BIN, meta['largo_m']):g}"]
                       + [f"{prof[i, p, b]:.6g}" for p in range(prof.shape[1]) for i in range(prof.shape[0])])  # fmt: skip
    return summary_text(meta, result)


def results_metrics(meta: dict, data: dict[str, np.ndarray]) -> dict:
    """Métricas de `resultados.json`: las de `analyze` por contaminante (un valor por separación) y la separación mínima."""
    r = analyze(meta, data)
    keep = ("aditividad", "media_tramo", "exceso_entre", "gradiente_tramo", "pendiente_max", "costo_tope", "costo_por_m",
            "pico", "huella_m", "pendiente_tope_solo", "neto_tope_solo", "d_aditiva", "d_huella", "d_minima")  # fmt: skip
    return {
        "topes_por_cadena": int(meta.get("topes_por_cadena", 2)), "primer_tope_m": meta["primer_tope_m"],
        "distancias_m": r["distancias_m"], "omitidas_m": meta.get("omitidas_m", []), "tolerancia": meta["tolerancia"],
        "umbral": meta["umbral"], "recomendada_m": r["recomendada_m"], "huella_m": r["huella_m"],
        "contaminantes": {pol: {k: v[k] for k in keep} for pol, v in r["pollutants"].items()},
        "parametros": meta,
    }  # fmt: skip


def _num(v: float) -> str:
    return "" if not np.isfinite(v) else f"{v:.4g}"


def _dist(d: float | None) -> str:
    return "ninguna de la lista" if d is None else f"{d:g} m"


def summary_text(meta: dict, result: dict) -> str:
    """Tabla por separación y contaminante, la huella de un tope y las separaciones mínimas."""
    dist = result["distancias_m"]
    n = int(meta.get("topes_por_cadena", 2))
    chain = f"cadena de {n} topes, " if n > 2 else ""
    lines = [
        f"Separación entre topes ({chain}primer tope a {meta['primer_tope_m']:g} m; media de {meta['replicas']} réplicas, "
        f"{meta['s_simulados']:,.0f} s simulados; tolerancia {meta['tolerancia']:g}, umbral {meta['umbral']:g} del pico):",
    ]  # fmt: skip
    if meta["omitidas_m"]:
        lines.append("Omitidas por no caber en el tramo: " + ", ".join(f"{d:g}" for d in meta["omitidas_m"]) + " m")
    for pol, r in result["pollutants"].items():
        name, scale = unit(pol)
        lines += [
            "",
            f"{POLLUTANT_LABELS[pol]}: huella de un tope {r['huella_m']:.0f} m, pico +{r['pico'] * scale:,.1f} {name}/(m·h) · "
            f"aditiva desde {_dist(r['d_aditiva'])} · fuera de la huella desde {_dist(r['d_huella'])}",
            f"{'d (m)':>7}{'aditividad':>12}{'media/base':>12}{'exceso entre':>14}{'gradiente':>12}{'costo tope':>12}{'% por m':>9}",
        ]
        for j, d in enumerate(dist):
            row = f"{d:>7g}{r['aditividad'][j]:>12.2f}{r['media_tramo'][j]:>12.2f}"
            if np.isfinite(r["exceso_entre"][j]):
                row += f"{100 * r['exceso_entre'][j]:>13.0f}%"
            else:
                row += f"{'—':>14}"
            grad = r["gradiente_tramo"][j] * scale
            row += f"{grad:>12.2f}" if np.isfinite(grad) else f"{'—':>12}"
            cost, per_m = r["costo_tope"][j], r["costo_por_m"][j]
            lines.append(row + (f"{cost:>12.2f}{per_m:>9.2f}" if np.isfinite(cost) else f"{'—':>12}{'—':>9}"))
    rec = result["recomendada_m"]
    huella = result["huella_m"]
    lines += [
        "",
        "costo tope: lo que emite de más cada tope añadido, en topes aislados (1 = como uno solo); % por m: ese costo por"
        " metro de separación, en % de un tope aislado.",
        f"Separación mínima entre topes para diluir la huella: {_dist(rec)}"
        + (f" (huella de un tope ≈ {huella:.0f} m)" if huella is not None else "")
        + ("" if rec is not None else "; amplía --separacion con valores mayores"),
    ]  # fmt: skip
    return "\n".join(lines)
