"""Colour themes of the rail figures (route.png and the video).

`default` is the palette shared with the traffic plots. `cream_crimson` is copied from the
`cream_crimson` theme of multi-dashboard/exploracion/estilo.py: brand crimson #993138 on a cream
background #FAF6EE, warm inks, and its first categorical hues (crimson, blue, ochre, teal), which
that module reports as validated with the dataviz skill's palette validator on the cream surface.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Theme:
    name: str
    surface: str  # figure and axes background
    ink: str  # primary text, hollow station markers
    ink_2: str  # secondary text
    muted: str  # tick labels, credits
    grid: str
    baseline: str  # axes lines, leader lines
    track: str  # the line not travelled yet
    travelled: str  # the line already travelled (video)
    train: str  # train marker (video)
    highlight: str  # highlighted stations (e.g. Chepe Express stops)
    highlight_ink: str  # their labels, readable on the surface
    landmark: str
    terrain: str  # fill under the elevation profile
    fonts: tuple[str, ...] = ()  # preferred sans-serif families; empty keeps matplotlib's default


THEMES = {
    "default": Theme(
        name="default", surface="#fcfcfb", ink="#0b0b0b", ink_2="#52514e", muted="#898781", grid="#e1e0d9",
        baseline="#c3c2b7", track="#c3c2b7", travelled="#2a78d6", train="#e34948", highlight="#1baf7a",
        highlight_ink="#0e7a52", landmark="#eb6834", terrain="#e8e2d0",
    ),
    "cream_crimson": Theme(
        name="cream_crimson", surface="#FAF6EE", ink="#2B1D1F", ink_2="#5C4A4B", muted="#857372", grid="#E8DFD3",
        baseline="#CDBFB0", track="#A69B91", travelled="#2464B4", train="#0A8C80", highlight="#993138",
        highlight_ink="#993138", landmark="#A87406", terrain="#EDE3D3",
        fonts=("Roboto", "Open Sans", "DejaVu Sans"),
    ),
}
DEFAULT_THEME = THEMES["default"]


def get_theme(name: str) -> Theme:
    try:
        return THEMES[name]
    except KeyError:
        raise ValueError(f"unknown style {name!r}; choose one of {', '.join(THEMES)}") from None


@contextmanager
def fonts(theme: Theme):
    """Use the theme's fonts while building and saving a figure (only the installed ones, to avoid warnings)."""
    import matplotlib as mpl
    from matplotlib import font_manager

    installed = {f.name for f in font_manager.fontManager.ttflist}
    families = [f for f in theme.fonts if f in installed]
    if not families:
        yield
        return
    with mpl.rc_context({"font.family": "sans-serif", "font.sans-serif": families}):
        yield
