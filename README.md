# palatial-sim-file-converters

Convert one simulation file to another.

```sh
pip install "palatial-sim-file-converters @ git+https://github.com/PalatialSim/palatial-sim-file-converters.git@v0.2.2"
sim-convert asset.usd asset.xml
```

The output extension chooses the format.

```sh
sim-convert asset.usd asset.xml      # USD to MJCF
sim-convert asset.xml asset.usda     # MJCF to USD
sim-convert asset.urdf asset.xml     # URDF to MJCF
sim-convert asset.usd asset.urdf     # USD to URDF
sim-convert asset.xml asset.urdf     # MJCF to URDF
sim-convert asset.urdf asset.usda    # URDF to USD
sim-convert asset.usd asset.glb      # USD to GLB
```

You can start from `.usd`, `.usda`, `.usdc`, `.xml`, `.mjcf`, or `.urdf`. You can write any of those, plus `.glb`.

## What you get

- An MJCF or URDF file, with meshes in a `meshes/` folder next to it.
- A `.glb`, plus `manifest.json` in the same folder. The manifest holds bodies, mass, inertia, joints, and colliders. The GLB holds the visible meshes.
- A `.usd`, `.usda`, or `.usdc` scene in meters, kilograms, and Z-up.

Add `--compile` to load an MJCF result in MuJoCo. That needs the `compile` extra:

```sh
pip install "palatial-sim-file-converters[compile] @ git+https://github.com/PalatialSim/palatial-sim-file-converters.git@v0.2.2"
sim-convert asset.usd asset.xml --compile
```

## Check an Isaac asset

```sh
isaac-usd-to-mjcf check asset.usd
```

This prints collider warnings. It does not write a file. `isaac-usd-to-mjcf convert asset.usd -o out/` writes `model.xml`, `meshes/`, and `collider_report.json`.

## Good to know

- `.xml` means MJCF.
- A mesh collider becomes a MuJoCo convex hull. An `sdf` collider, or a triangle mesh with no approximation, becomes a MuJoCo SDF of those triangles.
- A sphere, box, capsule, or cylinder stays that shape when the scale is uniform.
- URDF has no ball joint, plane, distance joint, or SDF. Those become a close substitute, and the tool prints a warning.
- GLB is an output only. It cannot be the input file.

## Versions

v0.2.2 was checked against these builds:

| API | Version |
| --- | --- |
| MuJoCo | 3.13.0 |
| OpenUSD (`usd-core`) | 26.08 |
| glTF | 2.0 |

`--compile` loads the MJCF in MuJoCo 3.13.0. USD reading and writing uses UsdPhysics from OpenUSD 26.08. The package requires `usd-core>=25.5`. Those USD schemas are the ones Isaac Sim writes. This release was not run inside an Isaac Sim install. Isaac Sim 5.1.0 ships OpenUSD 24.05, and Isaac Sim 6.0.0 ships OpenUSD 25.11.

This package is [MIT licensed](LICENSE).

## Contributing

Clone the repo, install the dev extra, and run the tests:

```sh
git clone https://github.com/PalatialSim/palatial-sim-file-converters.git
cd palatial-sim-file-converters
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

A conversion goes through `model.py`. Add the reader or writer for your format, cover it with a test in `tests/`, and add a line to `CHANGELOG.md`. Then open a pull request.

The full guide is [CONTRIBUTING.md](CONTRIBUTING.md). Issues go to [GitHub](https://github.com/PalatialSim/palatial-sim-file-converters/issues).
