"""Small shared helpers: palette, results folders and video encoder choice.

Copied from the traffic package (trafico/plotting.py, trafico/settings.py, trafico/movement.py)
so that rail stays independent of it.
"""

from __future__ import annotations

import re
import subprocess
from datetime import datetime
from pathlib import Path

# Same palette as the traffic plots.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"


def project_root() -> Path:
    """Project root (where pyproject.toml is); the current directory if the package is not installed editable."""
    root = Path(__file__).resolve().parents[2]
    if (root / "pyproject.toml").is_file() and (root / "src" / "rail").is_dir():
        return root
    return Path.cwd()


def safe_name(name: str) -> str:
    """Folder-safe name: letters, digits, '.', '_' and '-'."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "run"


def make_run_dir(base: Path, name: str, now: datetime) -> Path:
    """Create `base/<date-time>_<name>`; if it already exists, add a numeric suffix."""
    stem = f"{now:%Y%m%d-%H%M%S}_{safe_name(name)}"
    run_dir = base / stem
    k = 2
    while run_dir.exists():
        run_dir = base / f"{stem}-{k}"
        k += 1
    run_dir.mkdir(parents=True)
    return run_dir


def free_path(path: Path) -> Path:
    """Never overwrite: if the file exists, add a numeric suffix."""
    k = 2
    candidate = path
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}-{k}{path.suffix}")
        k += 1
    return candidate


def h264_encoder(ffmpeg: str) -> str:
    """Available H.264 encoder (libx264 or, e.g. on Fedora, libopenh264); MPEG-4 if there is none."""
    listing = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    return next((enc for enc in ("libx264", "libopenh264") if f" {enc} " in listing), "mpeg4")
