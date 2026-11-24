"""`results.json`: the summary of a run in a stable format for programs (scripts, agents, a server).

Each command (`traffic`, `traffic-variants`, `traffic-emissions`, with or without `--spacing`, and `traffic-calibrate`)
writes it at the end into the results folder. It always has the same header, with each mode's metrics under `"metricas"`:

    {"schema": 1, "modo": "corrida" | "variantes" | "emisiones" | "separacion" | "calibracion", "id": ..., "carpeta": ...,
     "creado": "YYYY-MM-DDTHH:MM:SS", "comando": ..., "semilla": ..., "replicas": ..., "s_simulados": ...,
     "archivos": [...], "metricas": {...}}

Format rules (`SCHEMA`): keys are lowercase ASCII with the unit at the end (`_s`, `_m`, `_kmh`, `_pct`); a key never
changes meaning or unit; new keys may be added without bumping `SCHEMA`, which is bumped only when one is removed or
changed. NaN and infinities are written as `null`.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

SCHEMA = 1
RESULTS_NAME = "results.json"
# Output files renamed to English on 2026-11-24: folders written before still use these names, so the readers
# fall back to them (see `existing`).
LEGACY_NAMES = {
    "results.json": "resultados.json",
    "scenarios.json": "escenarios.json",
    "variants.json": "variantes.json",
    "data.npz": "datos.npz",
    "summary.txt": "resumen.txt",
    "calibration.csv": "calibracion.csv",
}


def existing(folder: Path, name: str) -> Path:
    """Path of `name` in `folder`, or of its old Spanish name if only that one exists (folders from earlier versions)."""
    path = folder / name
    legacy = LEGACY_NAMES.get(name)
    if not path.is_file() and legacy is not None and (folder / legacy).is_file():
        return folder / legacy
    return path
MODES = ("corrida", "variantes", "emisiones", "separacion", "calibracion")


def plain(obj):
    """`obj` with Python types (lists, numbers, None instead of NaN) for `json`."""
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
    """Date and time of the run, from its name `YYYYMMDD-HHMMSS_<name>`."""
    try:
        return datetime.strptime(folder.name[:15], "%Y%m%d-%H%M%S").isoformat()
    except ValueError:
        return None


def document(
    mode: str, folder: Path, *, command: str, seed: int | None, replicas: int, sim_seconds: float, metrics: dict,
) -> dict:  # fmt: skip
    """Common header and the metrics of a mode. `archivos` lists what is in the folder (including results.json itself)."""
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
    """Mode of a results folder without `results.json` (from an older version), from its files."""
    if existing(folder, "calibration.csv").is_file():
        return "calibracion"
    meta_path = existing(folder, "scenarios.json")
    if not meta_path.is_file():
        return "corrida" if existing(folder, "summary.txt").is_file() else None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("modo") == "separacion":
        return "separacion"
    return "variantes" if "largos_m" in meta else "emisiones"


def rebuild(folder: Path) -> dict | None:
    """The document of a comparison or sweep folder without `results.json`, from its raw data
    (`data.npz` and `scenarios.json`); None if that is not possible (`traffic` runs keep no raw data)."""
    mode = mode_of(folder)
    if mode == "separacion":
        from traffic.bump_spacing import results_metrics
        from traffic.emission_scenarios import load
    elif mode == "emisiones":
        from traffic.emission_scenarios import load, results_metrics
    elif mode == "variantes":
        from traffic.variants import load, results_metrics
    else:
        return None
    meta, data = load(folder)
    return document(mode, folder, command=meta.get("comando", ""), seed=meta.get("semilla"),
                    replicas=meta.get("replicas", 0), sim_seconds=meta.get("s_simulados", 0.0),
                    metrics=results_metrics(meta, data))  # fmt: skip


def read_results(folder: str | Path) -> dict:
    """`results.json` of a folder. If it has none (a run from an older version) and it is a comparison or a
    sweep, it is rebuilt from its data without writing anything; for a `traffic` run without it, `FileNotFoundError`."""
    folder = Path(folder).expanduser()
    path = existing(folder, RESULTS_NAME)
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    if not folder.is_dir():
        raise FileNotFoundError(f"{folder} is not a results folder")
    doc = rebuild(folder)
    if doc is None:
        raise FileNotFoundError(f"{folder} has no {RESULTS_NAME}: it is from an earlier version; run it again")
    return doc


def list_runs(output_dir: str | Path | None = None, mode: str | None = None, limit: int | None = None) -> list[dict]:
    """Results folders, newest first: `id`, `carpeta`, `modo`, `creado` and whether it has
    `results.json`. `output_dir` defaults to `[output] dir` from the configuration."""
    from traffic.settings import default_config_path, load_settings, resolve_output_dir

    base = Path(output_dir).expanduser() if output_dir else resolve_output_dir(load_settings(default_config_path()).run.output_dir)
    out = []
    for folder in sorted((p for p in base.iterdir() if p.is_dir()), reverse=True) if base.is_dir() else []:
        if created(folder) is None:
            continue
        path = existing(folder, RESULTS_NAME)
        has = path.is_file()
        kind = json.loads(path.read_text(encoding="utf-8")).get("modo") if has else mode_of(folder)
        if kind is None or (mode is not None and kind != mode):
            continue
        out.append({"id": folder.name, "carpeta": str(folder.resolve()), "modo": kind, "creado": created(folder),
                    "resultados_json": has})  # fmt: skip
        if limit is not None and len(out) >= limit:
            break
    return out
