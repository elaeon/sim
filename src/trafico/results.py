"""`resultados.json`: el resumen de una corrida en un formato estable para programas (scripts, agentes, un servidor).

Cada comando (`trafico`, `trafico-variantes`, `trafico-emisiones`, con o sin `--separacion`, y `trafico-calibrar`) lo
escribe al final en la
carpeta de resultados. Tiene siempre la misma cabecera y las métricas de cada modo bajo `"metricas"`:

    {"schema": 1, "modo": "corrida" | "variantes" | "emisiones" | "separacion" | "calibracion", "id": ..., "carpeta": ...,
     "creado": "AAAA-MM-DDTHH:MM:SS", "comando": ..., "semilla": ..., "replicas": ..., "s_simulados": ...,
     "archivos": [...], "metricas": {...}}

Reglas del formato (`SCHEMA`): las claves son ASCII en minúsculas con la unidad al final (`_s`, `_m`, `_kmh`, `_pct`);
una clave no cambia de significado ni de unidad; se pueden añadir claves nuevas sin subir `SCHEMA`, y solo se sube si
se quita o cambia una. NaN e infinitos se escriben como `null`.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

SCHEMA = 1
RESULTS_NAME = "resultados.json"
MODES = ("corrida", "variantes", "emisiones", "separacion", "calibracion")


def plain(obj):
    """`obj` con tipos de Python (listas, números, None en lugar de NaN) para `json`."""
    if isinstance(obj, dict):
        return {str(k): plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return plain(obj.tolist())
    if isinstance(obj, (np.integer, int)) and not isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def created(folder: Path) -> str | None:
    """Fecha y hora de la corrida, de su nombre `AAAAMMDD-HHMMSS_<nombre>`."""
    try:
        return datetime.strptime(folder.name[:15], "%Y%m%d-%H%M%S").isoformat()
    except ValueError:
        return None


def document(
    mode: str, folder: Path, *, command: str, seed: int | None, replicas: int, sim_seconds: float, metrics: dict,
) -> dict:  # fmt: skip
    """Cabecera común y métricas de un modo. `archivos` lista lo que hay en la carpeta (y el propio resultados.json)."""
    files = {p.name for p in folder.iterdir() if p.is_file()} | {RESULTS_NAME}
    return plain({
        "schema": SCHEMA, "modo": mode, "id": folder.name, "carpeta": folder.resolve(), "creado": created(folder),
        "comando": command, "semilla": seed, "replicas": replicas, "s_simulados": sim_seconds,
        "archivos": sorted(files), "metricas": metrics,
    })  # fmt: skip


def write_results(folder: Path, doc: dict) -> Path:
    path = folder / RESULTS_NAME
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def mode_of(folder: Path) -> str | None:
    """Modo de una carpeta de resultados sin `resultados.json` (de una versión anterior), por sus archivos."""
    if (folder / "calibracion.csv").is_file():
        return "calibracion"
    meta_path = folder / "escenarios.json"
    if not meta_path.is_file():
        return "corrida" if (folder / "resumen.txt").is_file() else None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("modo") == "separacion":
        return "separacion"
    return "variantes" if "largos_m" in meta else "emisiones"


def rebuild(folder: Path) -> dict | None:
    """El documento de una carpeta de comparación o de barrido sin `resultados.json`, a partir de sus datos crudos
    (`datos.npz` y `escenarios.json`); None si no se puede (las corridas de `trafico` no guardan datos crudos)."""
    mode = mode_of(folder)
    if mode == "separacion":
        from trafico.bump_spacing import results_metrics
        from trafico.emission_scenarios import load
    elif mode == "emisiones":
        from trafico.emission_scenarios import load, results_metrics
    elif mode == "variantes":
        from trafico.variants import load, results_metrics
    else:
        return None
    meta, data = load(folder)
    return document(mode, folder, command=meta.get("comando", ""), seed=meta.get("semilla"),
                    replicas=meta.get("replicas", 0), sim_seconds=meta.get("s_simulados", 0.0),
                    metrics=results_metrics(meta, data))  # fmt: skip


def read_results(folder: str | Path) -> dict:
    """`resultados.json` de una carpeta. Si no lo tiene (corrida de una versión anterior) y es una comparación o un
    barrido, lo reconstruye de sus datos sin escribir nada; para una corrida de `trafico` sin él, `FileNotFoundError`."""
    folder = Path(folder).expanduser()
    path = folder / RESULTS_NAME
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    if not folder.is_dir():
        raise FileNotFoundError(f"{folder} no es una carpeta de resultados")
    doc = rebuild(folder)
    if doc is None:
        raise FileNotFoundError(f"{folder} no tiene {RESULTS_NAME}: es de una versión anterior; vuelve a correrla")
    return doc


def list_runs(output_dir: str | Path | None = None, mode: str | None = None, limit: int | None = None) -> list[dict]:
    """Carpetas de resultados, de la más reciente a la más antigua: `id`, `carpeta`, `modo`, `creado` y si tiene
    `resultados.json`. `output_dir` por omisión es `[output] dir` de la configuración."""
    from trafico.settings import default_config_path, load_settings, resolve_output_dir

    base = Path(output_dir).expanduser() if output_dir else resolve_output_dir(load_settings(default_config_path()).run.output_dir)
    out = []
    for folder in sorted((p for p in base.iterdir() if p.is_dir()), reverse=True) if base.is_dir() else []:
        if created(folder) is None:
            continue
        has = (folder / RESULTS_NAME).is_file()
        kind = json.loads((folder / RESULTS_NAME).read_text(encoding="utf-8")).get("modo") if has else mode_of(folder)
        if kind is None or (mode is not None and kind != mode):
            continue
        out.append({"id": folder.name, "carpeta": str(folder.resolve()), "modo": kind, "creado": created(folder),
                    "resultados_json": has})  # fmt: skip
        if limit is not None and len(out) >= limit:
            break
    return out
