"""Command-line interface for stlinspector: `python cli.py part.stl --report out.json`."""

from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.table import Table

from core import InspectionReport, inspect_mesh, load_mesh

console = Console()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description="Inspect and validate an STL part file.",
    )
    parser.add_argument("path", help="Path to the STL file to inspect")
    parser.add_argument(
        "--report",
        metavar="OUT_JSON",
        help="Write the full report as JSON to this path",
    )
    return parser


def _print_report(report: InspectionReport, path: str) -> None:
    table = Table(title=f"Inspection report: {path}")
    table.add_column("Metric")
    table.add_column("Value")

    bb_min, bb_max = report.bounding_box
    table.add_row("Bounding box (min)", str([round(v, 4) for v in bb_min]))
    table.add_row("Bounding box (max)", str([round(v, 4) for v in bb_max]))
    table.add_row("Volume", f"{report.volume:.4f}")
    table.add_row("Surface area", f"{report.surface_area:.4f}")
    table.add_row("Triangle count", str(report.triangle_count))
    console.print(table)

    if report.issues:
        console.print(f"[red]Issues found: {', '.join(report.issues)}[/red]")
    else:
        console.print("[green]All checks passed[/green]")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        mesh = load_mesh(args.path)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]Error: {exc}[/red]")
        return 2

    report = inspect_mesh(mesh)
    _print_report(report, args.path)

    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)

    return 1 if report.issues else 0


if __name__ == "__main__":
    sys.exit(main())
