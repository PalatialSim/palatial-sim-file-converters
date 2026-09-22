# Changelog

## 0.2.1 — 2026-09-22

Release under the MIT license. Add a contributing guide. Reports and manifests name the file, not the directory it was read from.

## 0.2.0 — 2026-09-22

Convert between USD, MJCF, and URDF in either direction, and write a GLB with `manifest.json`, from one command: `sim-convert INPUT OUTPUT`.

The package is `palatial-sim-file-converters`. `isaac-usd-to-mjcf check` and `isaac-usd-to-mjcf convert` still check Isaac colliders and write an MJCF bundle.

## 0.1.0 — 2026-09-22

Convert an Isaac Sim USD asset to MuJoCo MJCF. Collider scale and orientation are baked in, and UsdPhysics shapes and joints are mapped onto MuJoCo geoms.
