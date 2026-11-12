"""rail: simulate a train along a curated route and write a timetable, an overview and a video."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from datetime import datetime
from pathlib import Path

from rail.route import RouteError, load_route, orient
from rail.sim import clock, duration, simulate, timetable
from rail.elevation import elevation_path
from rail.track import realistic_route, track_path
from rail.common import make_run_dir, project_root
from rail.theme import THEMES, get_theme

DEFAULT_ROUTE = Path("examples") / "route_chepe.toml"
DURATION_S = 60.0
REALISTIC_DURATION_S = 90.0  # the slow-motion section needs more video time
RESULTS_DIR = "resultados"  # same folder as the traffic runs


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rail",
        description=(
            "Simulate a train along a route file (stations, landmarks, speed limits) and save "
            f"summary.txt, timetable.csv, route.png and a landscape video in {RESULTS_DIR}/<date-time>_<name>/."
        ),
    )
    p.add_argument("route_file", nargs="?", type=Path, default=None,
                   help=f"route TOML file (default: {DEFAULT_ROUTE})")  # fmt: skip
    p.add_argument("--route", default=None, metavar="FROM-TO",
                   help="direction of travel, from the route file's 'ends': e.g. mochis-chihuahua (default, the "
                        "file's order) or chihuahua-mochis")  # fmt: skip
    p.add_argument("--name", default=None, help="results folder suffix (default: the route file name without 'route_')")
    p.add_argument("--duration", type=float, default=None,
                   help=f"video length in seconds (default: {DURATION_S:g}; {REALISTIC_DURATION_S:g} with --realistic)")
    p.add_argument("--fps", type=int, default=25, help="video frames per second (default: 25)")
    p.add_argument("--format", choices=("webm", "mp4", "gif"), default="webm", help="video format (default: webm)")
    p.add_argument("--depart", default="06:00", help="departure time of day, HH:MM (default: 06:00)")
    p.add_argument("--no-video", action="store_true", help="skip the video (timetable and route.png only)")
    p.add_argument("--style", choices=tuple(THEMES), default="default",
                   help="colour style of route.png and the video: default or cream_crimson (crimson on cream; "
                        "default: default)")  # fmt: skip
    p.add_argument("--realistic", action="store_true",
                   help="use the real track from OpenStreetMap (downloaded once, see [realistic] in the route file), "
                        "with a zoomed inset and slow motion through the scenic section")  # fmt: skip
    return p


def parse_clock(text: str) -> float:
    """'HH:MM' as seconds after midnight."""
    try:
        hours, minutes = (int(part) for part in text.split(":"))
    except ValueError as exc:
        raise ValueError(f"--depart must be HH:MM, got {text!r}") from exc
    if not (0 <= hours < 24 and 0 <= minutes < 60):
        raise ValueError(f"--depart must be HH:MM, got {text!r}")
    return hours * 3600.0 + minutes * 60.0


def summary_text(route, journey, depart_s: float) -> str:
    total = journey.duration_s
    lines = [
        route.name,
        f"distance            {route.length_km:.0f} km",
        f"departure           {clock(depart_s)}",
        f"arrival             {clock(depart_s + total)}",
        f"total time          {duration(total)}",
        f"stops               {len(journey.stops) - 2} intermediate, "
        f"{sum(s.departure_s - s.arrival_s for s in journey.stops) / 60:.0f} min dwelling",
        f"average speed       {route.length_km / (total / 3600):.1f} km/h (including stops)",
    ]
    if route.track is not None:
        lines.append(f"track               {route.track.source}")
    if route.track is not None and route.track.elevation_m is not None:
        lines.append(f"elevation           {route.track.elevation_source}")
    slow = route.slow_section()
    if slow is not None:
        lines.append(f"slow motion         x{slow[2]:g}, {route.realistic.slow_from} - {route.realistic.slow_to} "
                     f"(km {slow[0]:.0f}-{slow[1]:.0f})")  # fmt: skip
    stops = {s.name: s for s in journey.stops}
    if "Los Mochis" in stops and "Creel" in stops:  # Chepe Express section, used for calibration
        a, b = sorted((stops["Los Mochis"], stops["Creel"]), key=lambda s: s.arrival_s)
        label = f"{a.name} - {b.name}"
        lines.append(f"{label:<19} {duration(b.arrival_s - a.departure_s)} (published Chepe Express: ~9h 40min)")
    return "\n".join(lines) + "\n"


def run(argv: list[str] | None = None) -> Path:
    args = build_parser().parse_args(argv)
    root = project_root()
    route_path = args.route_file or root / DEFAULT_ROUTE
    try:
        route = load_route(route_path)
        depart_s = parse_clock(args.depart)
        if args.realistic:
            route = realistic_route(route)  # snapping needs the file's order; reverse afterwards
        route = orient(route, args.route)
    except (OSError, RouteError, ValueError) as exc:
        sys.exit(f"Error: {exc}")
    journey = simulate(route)
    name = args.name or (route_path.stem.removeprefix("route_") + (f"_{route.route_id}" if route.reversed else "")
                         + ("_realistic" if args.realistic else ""))  # fmt: skip
    run_dir = make_run_dir(root / RESULTS_DIR, name, datetime.now())
    (run_dir / route_path.name).write_text(route.text, encoding="utf-8")
    if args.realistic:
        shutil.copyfile(track_path(route), run_dir / track_path(route).name)
        if route.realistic.elevation_file is not None:
            shutil.copyfile(elevation_path(route), run_dir / elevation_path(route).name)
    (run_dir / "summary.txt").write_text(summary_text(route, journey, depart_s), encoding="utf-8")
    rows = timetable(journey, depart_s)
    with open(run_dir / "timetable.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    from rail.video import plot_route, render_journey

    theme = get_theme(args.style)
    plot_route(route, run_dir / "route.png", theme)
    if not args.no_video:
        seconds = args.duration or (REALISTIC_DURATION_S if args.realistic else DURATION_S)
        render_journey(route, journey, run_dir / "journey", duration_s=seconds, fps=args.fps,
                       fmt=args.format, depart_s=depart_s, theme=theme)  # fmt: skip
    print((run_dir / "summary.txt").read_text(encoding="utf-8"), end="")
    print(f"Results in {run_dir}")
    return run_dir


def main(argv: list[str] | None = None) -> None:
    run(argv)
