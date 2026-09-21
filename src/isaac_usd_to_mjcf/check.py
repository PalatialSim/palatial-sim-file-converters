"""Collider checks that the USD stage does not answer by itself.

Isaac often stores a unit cube, or a convex fragment, and puts the real size on
a scale op. PhysX ``sdf`` and ``convexDecomposition`` are cooked at runtime, so
the stage mesh is not the shape the simulator collides with.
"""

from __future__ import annotations

import numpy as np

from isaac_usd_to_mjcf.model import Asset, Finding

_UNIT = 1e-3
_MIN_EXTENT = 1e-5
_RATIO_LOW = 0.8
_RATIO_HIGH = 1.25


def check_asset(asset: Asset) -> list[Finding]:
    findings: list[Finding] = []
    for body in asset.bodies:
        if not body.colliders:
            findings.append(
                Finding(
                    "warning",
                    "missing_collider",
                    body.path,
                    "Rigid body has no enabled collider.",
                )
            )
            continue
        dedicated = [collider for collider in body.colliders if collider.dedicated]
        visual_collision = [collider for collider in body.colliders if not collider.dedicated]
        if dedicated and visual_collision:
            findings.append(
                Finding(
                    "warning",
                    "double_collision",
                    body.path,
                    "The visual mesh collides and dedicated collider prims collide too.",
                )
            )
        for collider in body.colliders:
            findings.extend(_collider_findings(collider))
        if dedicated and body.visuals:
            findings.extend(_bounds_findings(body.path, dedicated, body.visuals))
    for joint in asset.joints:
        if joint.note:
            findings.append(Finding("warning", "joint_approximation", joint.path, joint.note))
        if joint.kind == "distance" and joint.range is None:
            findings.append(
                Finding(
                    "warning",
                    "distance_unlimited",
                    joint.path,
                    "Distance joint limits are unset. USD uses -1 for an inactive limit, so no MuJoCo tendon was written.",
                )
            )
    asset.findings = findings
    return findings


def _collider_findings(collider) -> list[Finding]:
    findings: list[Finding] = []
    extent = collider.world_max - collider.world_min
    if np.any(extent < _MIN_EXTENT):
        findings.append(
            Finding(
                "warning",
                "degenerate_collider",
                collider.path,
                f"World extent {np.array2string(extent, precision=6)} is below {_MIN_EXTENT} m.",
            )
        )
    non_unit = [
        op
        for op in collider.scales
        if any(abs(component - 1.0) > _UNIT for component in op.scale)
    ]
    own = [op for op in non_unit if op.prim == collider.path]
    inherited = [op for op in non_unit if op.prim != collider.path]
    if collider.unit_cube and own:
        scale = own[-1].scale
        findings.append(
            Finding(
                "info",
                "unit_cube_scale",
                collider.path,
                "Mesh is a unit cube; its size is the scale "
                f"({scale[0]:.6g}, {scale[1]:.6g}, {scale[2]:.6g}).",
            )
        )
    if inherited:
        described = ", ".join(
            f"{op.prim} ({op.scale[0]:.6g}, {op.scale[1]:.6g}, {op.scale[2]:.6g})"
            for op in inherited
        )
        findings.append(
            Finding(
                "warning",
                "ancestor_scale",
                collider.path,
                f"Collider inherits a non-unit scale from {described}.",
            )
        )
    if len(non_unit) >= 2:
        findings.append(
            Finding(
                "warning",
                "stacked_scale",
                collider.path,
                "More than one non-unit scale sits between this collider and its rigid body.",
            )
        )
    for op in collider.scales:
        if any(component < -_UNIT for component in op.scale):
            findings.append(
                Finding(
                    "warning",
                    "negative_scale",
                    op.prim,
                    f"Negative scale {op.scale} mirrors the collider.",
                )
            )
            break
    if collider.approximation == "sdf":
        findings.append(
            Finding(
                "warning",
                "physx_sdf",
                collider.path,
                "PhysX approximation is sdf. MJCF gets a MuJoCo SDF of this triangle mesh, not the cooked PhysX SDF.",
            )
        )
    elif collider.approximation == "none":
        findings.append(
            Finding(
                "warning",
                "triangle_mesh_as_sdf",
                collider.path,
                "Approximation none is a triangle mesh. MuJoCo has no triangle-mesh collider, so this exports an SDF of the same mesh.",
            )
        )
    elif collider.approximation == "convexDecomposition":
        findings.append(
            Finding(
                "warning",
                "physx_convex_decomposition",
                collider.path,
                "PhysX convexDecomposition is cooked at runtime. The stage has no hulls, so MJCF uses one convex hull of this mesh.",
            )
        )
    elif collider.approximation == "meshSimplification":
        findings.append(
            Finding(
                "warning",
                "mesh_simplification_unavailable",
                collider.path,
                "meshSimplification is cooked at runtime and is not stored. MJCF uses the convex hull of the source mesh.",
            )
        )
    elif collider.approximation == "boundingCube":
        findings.append(
            Finding(
                "info",
                "generated_bounding_cube",
                collider.path,
                "boundingCube replaces the mesh with a box of its local axis-aligned bounds.",
            )
        )
    elif collider.approximation == "boundingSphere":
        findings.append(
            Finding(
                "info",
                "generated_bounding_sphere",
                collider.path,
                "boundingSphere replaces the mesh with the sphere centered on those bounds that contains every vertex.",
            )
        )
    elif collider.kind == "cone":
        findings.append(
            Finding(
                "info",
                "generated_cone",
                collider.path,
                "MuJoCo has no cone geom. The UsdGeom cone is tessellated; collision uses its convex hull.",
            )
        )
    return findings


def _bounds_findings(body_path: str, colliders, visuals) -> list[Finding]:
    collider_min = np.min([collider.world_min for collider in colliders], axis=0)
    collider_max = np.max([collider.world_max for collider in colliders], axis=0)
    visual_min = np.min([visual.world_min for visual in visuals], axis=0)
    visual_max = np.max([visual.world_max for visual in visuals], axis=0)
    collider_extent = collider_max - collider_min
    visual_extent = visual_max - visual_min
    ratios = []
    for index, (collider_size, visual_size) in enumerate(zip(collider_extent, visual_extent)):
        if visual_size < _MIN_EXTENT:
            continue
        ratio = float(collider_size / visual_size)
        ratios.append(( "xyz"[index], ratio))
    mismatched = [item for item in ratios if item[1] < _RATIO_LOW or item[1] > _RATIO_HIGH]
    if not mismatched:
        return []
    rendered = ", ".join(f"{axis} {ratio:.3f}" for axis, ratio in mismatched)
    return [
        Finding(
            "warning",
            "bounds_mismatch",
            body_path,
            "Dedicated collider bounds differ from the visual mesh "
            f"(collider/visual axis ratios: {rendered}).",
        )
    ]
