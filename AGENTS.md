# Agent notes

This repository is `palatial-sim-file-converters`. Import `palatial_sim_file_converters`. Convert with `sim-convert SOURCE OUTPUT`. `isaac-usd-to-mjcf check` only prints collider findings.

## Where a change goes

Code lives in `src/palatial_sim_file_converters/`. Conversion passes through `model.py`. `convert.py` chooses the reader and writer from the file suffix. Add a format there. Keep the package flat.

| Format | Read | Write |
| --- | --- | --- |
| USD | `read.py` | `usd_write.py` |
| MJCF | `mjcf_read.py` | `export.py` |
| URDF | `urdf.py` | `urdf.py` |
| GLB | output only | `glb.py` |

In-memory entry points are `load_usda`, `parse_mjcf`, `parse_urdf`, `mjcf_tree`, `urdf_tree`, `build_stage`, and `glb_document`. File writers are thin wrappers around those.

## Checks

```sh
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Tests live in `tests/` and assert the scene in memory. Do not write temporary files, glob meshes, or store an absolute path. Reports and manifests store the filename only.

Hold an OpenUSD stage in a variable before calling `Export` or `ExportToString`. Releasing the stage first invalidates the layer.

If a target format cannot represent a shape, write the closest geom and a warning.

## Builds these APIs were checked against

| API | Version |
| --- | --- |
| MuJoCo | 3.13.0 |
| OpenUSD (`usd-core`) | 26.08 |
| URDF | 1.0 |
| glTF | 2.0 |

The package requires `usd-core>=25.5`. URDF output uses `xyz` and `rpy`, with box, cylinder, sphere, and mesh. Capsules are written as meshes. The 1.1 quaternion and capsule elements, and the 1.2 joint-limit attributes, are not written.

## Pull request

Run `pytest`. Add a line under the next version in `CHANGELOG.md`. Open one pull request for one change.

If you are an agent, say so in the pull request. Describe the issues you found and the steps to reproduce them.

Human setup and review steps are in [CONTRIBUTING.md](CONTRIBUTING.md).
