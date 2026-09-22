"""Convert one simulation file to another format."""

from __future__ import annotations

from pathlib import Path

from palatial_sim_file_converters.check import check_asset
from palatial_sim_file_converters.export import export_mjcf
from palatial_sim_file_converters.glb import write_glb
from palatial_sim_file_converters.mjcf_read import load_mjcf
from palatial_sim_file_converters.read import load_asset
from palatial_sim_file_converters.urdf import load_urdf, write_urdf
from palatial_sim_file_converters.usd_write import write_usd

_BY_SUFFIX = {
    ".usd": "usd",
    ".usda": "usd",
    ".usdc": "usd",
    ".mjcf": "mjcf",
    ".xml": "mjcf",
    ".urdf": "urdf",
    ".glb": "glb",
}


def convert_file(source: str | Path, output: str | Path) -> Path:
    source = Path(source)
    output = Path(output)
    source_format = _format(source)
    output_format = _format(output)
    if source_format is None or output_format is None:
        raise ValueError(
            "Use .usd/.usda/.usdc, .xml/.mjcf, .urdf, or .glb. "
            f"Got {source.name} -> {output.name}."
        )
    if source_format == "glb":
        raise ValueError("GLB is written from USD, MJCF, or URDF. It is not a source format.")
    asset = {"usd": load_asset, "mjcf": load_mjcf, "urdf": load_urdf}[source_format](source)
    if source_format == "usd":
        check_asset(asset)
    writer = {"usd": write_usd, "mjcf": export_mjcf, "urdf": write_urdf, "glb": write_glb}[output_format]
    return Path(writer(asset, output))


def _format(path: Path) -> str | None:
    return _BY_SUFFIX.get(path.suffix.lower())
