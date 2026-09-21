"""Command line for collider checks and USD to MJCF conversion."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaac_usd_to_mjcf.check import check_asset
from isaac_usd_to_mjcf.export import export_mjcf
from isaac_usd_to_mjcf.read import load_asset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="isaac-usd-to-mjcf")
    commands = parser.add_subparsers(dest="command", required=True)

    check = commands.add_parser("check", help="Report collider scale, bounds, and approximation issues.")
    check.add_argument("usd", nargs="+", type=Path)

    convert = commands.add_parser("convert", help="Write model.xml, meshes, and collider_report.json.")
    convert.add_argument("usd", type=Path)
    convert.add_argument("-o", "--output", required=True, type=Path)
    convert.add_argument(
        "--compile",
        action="store_true",
        help="Load the MJCF with MuJoCo after writing it.",
    )

    args = parser.parse_args(argv)
    if args.command == "check":
        return _check(args.usd)
    return _convert(args.usd, args.output, args.compile)


def _check(paths: list[Path]) -> int:
    for path in paths:
        asset = load_asset(path)
        findings = check_asset(asset)
        print(f"{path}")
        if not findings:
            print("  no collider findings")
            continue
        for finding in findings:
            print(f"  {finding.severity:7} {finding.code:28} {finding.path}")
            print(f"          {finding.message}")
    return 0


def _convert(path: Path, output: Path, compile_model: bool) -> int:
    asset = load_asset(path)
    check_asset(asset)
    xml_path = export_mjcf(asset, output)
    warnings = [finding for finding in asset.findings if finding.severity == "warning"]
    print(f"wrote {xml_path} ({len(warnings)} collider warning(s))")
    if compile_model:
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(xml_path))
        print(f"mujoco compiled nbody={model.nbody} ngeom={model.ngeom} njnt={model.njnt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
