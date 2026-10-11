# Changelog

## Unreleased

Preserve static scene supports and visual geometry in USD conversion, traverse referenced instance contents, and keep fixed world geometry distinct from kinematic bodies through USD, MJCF, URDF and GLB output.

Verify converted referenced floor/table supports with a local MuJoCo drop-and-settle regression and a missing-support control.

Keep geometry below disabled nested rigid-body APIs attached to its nearest enabled ancestor.

Keep independent static joint endpoints outside enabled rigid-body hierarchies in USD. Refuse MJCF and URDF joint layouts that would animate an authored static child instead of silently changing its physical role.

## 0.2.2 — 2026-09-22

Record the builds this release was checked against: MuJoCo 3.13.0 and OpenUSD 26.08 (`usd-core` 26.8).

## 0.2.1 — 2026-09-22

Release under the MIT license. Add a contributing guide. Reports and manifests name the file, not the directory it was read from.

## 0.2.0 — 2026-09-22

Convert between USD, MJCF, and URDF in either direction, and write a GLB with `manifest.json`, from one command: `sim-convert INPUT OUTPUT`.

The package is `palatial-sim-file-converters`. `isaac-usd-to-mjcf check` and `isaac-usd-to-mjcf convert` still check Isaac colliders and write an MJCF bundle.

## 0.1.0 — 2026-09-22

Convert an Isaac Sim USD asset to MuJoCo MJCF. Collider scale and orientation are baked in, and UsdPhysics shapes and joints are mapped onto MuJoCo geoms.
