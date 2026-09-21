"""Read rigid bodies, colliders, and joints from an Isaac Sim USD stage."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from isaac_usd_to_mjcf.model import Asset, Body, Collider, Joint, ScaleOp, Visual
from isaac_usd_to_mjcf.primitives import (
    aabb,
    bounding_sphere,
    capsule_mesh,
    cone_mesh,
    cylinder_mesh,
)
from isaac_usd_to_mjcf.xform import (
    Y_UP_TO_Z_UP,
    gf_matrix,
    gf_quat,
    invert_rigid,
    matrix_to_quat,
    rotate_vector,
    transform_points,
)

_AXIS = {
    "X": np.array([1.0, 0.0, 0.0]),
    "Y": np.array([0.0, 1.0, 0.0]),
    "Z": np.array([0.0, 0.0, 1.0]),
}


def load_asset(path: str | Path) -> Asset:
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise FileNotFoundError(path)
    meters = float(UsdGeom.GetStageMetersPerUnit(stage) or 1.0)
    kilograms = float(UsdPhysics.GetStageKilogramsPerUnit(stage) or 1.0)
    up_axis = UsdGeom.GetStageUpAxis(stage) or "Z"
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    bodies = _bodies(stage, cache, meters, kilograms, up_axis)
    by_path = {body.path: body for body in bodies}
    _attach_geometry(stage, cache, by_path, meters, up_axis)
    joints = _joints(stage, by_path, meters)
    return Asset(
        name=Path(path).stem,
        source=str(path),
        up_axis=up_axis,
        meters_per_unit=meters,
        bodies=bodies,
        joints=joints,
        gravity=_gravity(stage, meters),
        filtered_pairs=_filtered_pairs(stage, set(by_path)),
    )


def _bodies(stage, cache, meters: float, kilograms: float, up_axis: str) -> list[Body]:
    bodies: list[Body] = []
    used: set[str] = set()
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        rigid_api = UsdPhysics.RigidBodyAPI(prim)
        enabled = rigid_api.GetRigidBodyEnabledAttr().Get()
        if enabled is False:
            continue
        rigid, pos, quat = _rigid_pose(cache, prim, meters, up_axis)
        name = _unique(prim.GetName(), used)
        mass, com, inertia, axes = _mass(prim, cache, rigid, meters, kilograms, up_axis)
        kinematic = bool(rigid_api.GetKinematicEnabledAttr().Get())
        bodies.append(
            Body(
                path=str(prim.GetPath()),
                name=name,
                pos=pos,
                quat=quat,
                rigid=rigid,
                mass=mass,
                com=com,
                diaginertia=inertia,
                principal_quat=axes,
                kinematic=kinematic,
            )
        )
    return bodies


def _attach_geometry(stage, cache, bodies: dict[str, Body], meters: float, up_axis: str) -> None:
    body_paths = set(bodies)
    for prim in stage.Traverse():
        owner = _owner(prim, body_paths)
        if owner is None:
            continue
        body = bodies[owner]
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            collider = _collider(prim, body, cache, meters, up_axis)
            if collider is not None:
                body.colliders.append(collider)
            continue
        if prim.GetTypeName() == "Mesh" and not _guide(prim):
            visual = _visual(prim, body, cache, meters, up_axis)
            if visual is not None:
                body.visuals.append(visual)


def _collider(prim, body: Body, cache, meters: float, up_axis: str) -> Collider | None:
    collision = UsdPhysics.CollisionAPI(prim)
    enabled = collision.GetCollisionEnabledAttr().Get()
    if enabled is False:
        return None
    approximation = None
    if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
        approximation = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
        approximation = str(approximation) if approximation else None
    world = _world_matrix(cache, prim, meters, up_axis)
    # Row-vector change of frame: p_body = p_world * inv(body).
    to_body = world @ invert_rigid(body.rigid)
    friction = _friction(prim)
    scales = _scale_ops(prim, body.path)
    kind = prim.GetTypeName()
    dedicated = "collider" in str(prim.GetPath()).lower()
    if kind == "Mesh":
        points, faces = _mesh(prim)
        if points is None:
            return None
        return _mesh_collider(
            prim,
            points,
            faces,
            approximation,
            dedicated,
            scales,
            friction,
            world,
            to_body,
        )
    primitive = _primitive(prim, to_body)
    if primitive is None:
        return None
    pos, quat, size, geom_type, mesh_local, local_faces = primitive
    if mesh_local is not None:
        world_points = transform_points(world, mesh_local)
        local_points = transform_points(to_body, mesh_local)
        world_min, world_max = world_points.min(axis=0), world_points.max(axis=0)
    else:
        local_points = None
        world_min, world_max = _primitive_world_bounds(prim, world)
    return Collider(
        path=str(prim.GetPath()),
        kind=kind[0].lower() + kind[1:],
        geom_type=geom_type,
        approximation=approximation,
        dedicated=True,
        unit_cube=False,
        scales=scales,
        world_min=world_min,
        world_max=world_max,
        friction=friction,
        local_points=local_points,
        faces=local_faces,
        primitive_pos=pos,
        primitive_quat=quat,
        primitive_size=size,
    )


def _visual(prim, body: Body, cache, meters: float, up_axis: str) -> Visual | None:
    points, faces = _mesh(prim)
    if points is None:
        return None
    world = _world_matrix(cache, prim, meters, up_axis)
    world_points = transform_points(world, points)
    local_points = transform_points(world @ invert_rigid(body.rigid), points)
    return Visual(
        path=str(prim.GetPath()),
        local_points=local_points,
        faces=faces,
        world_min=world_points.min(axis=0),
        world_max=world_points.max(axis=0),
    )


def _mesh_collider(prim, points, faces, approximation, dedicated, scales, friction, world, to_body):
    """Map a mesh collider using the UsdPhysics approximation token.

    ``convexHull`` becomes a MuJoCo mesh geom, which collides as a convex hull.
    ``none`` is a triangle mesh; MuJoCo has no triangle-mesh collider, so the
    same triangles are exported as an SDF. ``boundingCube`` and
    ``boundingSphere`` replace the mesh with a generated primitive.
    """
    token = approximation or "none"
    world_points = transform_points(world, points)
    common = dict(
        path=str(prim.GetPath()),
        approximation=token,
        dedicated=dedicated,
        unit_cube=_unit_cube(points),
        scales=scales,
        world_min=world_points.min(axis=0),
        world_max=world_points.max(axis=0),
        friction=friction,
    )
    if token == "boundingCube":
        center, half = aabb(points)
        pos, quat, size = _scaled_primitive(to_body, center, half)
        return Collider(kind="cube", geom_type="box", primitive_pos=pos, primitive_quat=quat, primitive_size=size, **common)
    if token == "boundingSphere":
        center, radius = bounding_sphere(points)
        pos, quat, size = _scaled_primitive(to_body, center, np.array([radius, radius, radius]))
        geom = "sphere" if _uniform(size) else "ellipsoid"
        return Collider(kind="sphere", geom_type=geom, primitive_pos=pos, primitive_quat=quat, primitive_size=size if geom == "ellipsoid" else (float(np.mean(size)),), **common)
    geom_type = "sdf" if token in {"sdf", "none"} else "mesh"
    return Collider(
        kind="mesh",
        geom_type=geom_type,
        local_points=transform_points(to_body, points),
        faces=faces,
        **common,
    )


def _scaled_primitive(to_body: np.ndarray, center: np.ndarray, half: np.ndarray):
    linear = to_body[:3, :3]
    scale = np.abs(np.linalg.norm(linear, axis=1))
    pos = transform_points(to_body, center.reshape(1, 3))[0]
    quat = matrix_to_quat(_orthonormalize(linear))
    size = np.abs(half) * scale
    return (
        (float(pos[0]), float(pos[1]), float(pos[2])),
        quat,
        tuple(float(value) for value in size),
    )


def _uniform(values, tol: float = 1e-4) -> bool:
    array = np.abs(np.array(values, dtype=np.float64))
    return float(array.max() - array.min()) <= tol * max(float(array.max()), 1e-8)


def _primitive(prim, to_body: np.ndarray):
    kind = prim.GetTypeName()
    linear = to_body[:3, :3]
    # Row-vector matrices store each local axis scale on a row.
    axis_scale = np.abs(np.linalg.norm(linear, axis=1))
    quat = matrix_to_quat(_orthonormalize(linear))
    pos = tuple(float(value) for value in to_body[3, :3])
    if kind == "Sphere":
        radius = float(UsdGeom.Sphere(prim).GetRadiusAttr().Get() or 1.0)
        radii = axis_scale * radius
        if _uniform(radii):
            return pos, quat, (float(np.mean(radii)),), "sphere", None, None
        return pos, quat, tuple(float(value) for value in radii), "ellipsoid", None, None
    if kind == "Cube":
        edge = float(UsdGeom.Cube(prim).GetSizeAttr().Get() or 2.0)
        half = axis_scale * edge * 0.5
        return pos, quat, tuple(float(value) for value in half), "box", None, None
    if kind == "Plane":
        plane = UsdGeom.Plane(prim)
        width = float(plane.GetWidthAttr().Get() or 0.0)
        length = float(plane.GetLengthAttr().Get() or 0.0)
        axis = str(plane.GetAxisAttr().Get() or "Z")
        quat = _axis_quat(quat, axis)
        perpendicular = _perpendicular_scales(axis_scale, axis)
        # MuJoCo planes are infinite. size is the rendered half-extent only.
        size = (
            width * 0.5 * perpendicular[0],
            length * 0.5 * perpendicular[1],
            0.01,
        )
        return pos, quat, size, "plane", None, None
    if kind == "Cone":
        cone = UsdGeom.Cone(prim)
        radius = float(cone.GetRadiusAttr().Get() or 1.0)
        height = float(cone.GetHeightAttr().Get() or 2.0)
        axis = str(cone.GetAxisAttr().Get() or "Z")
        # Authored size only. to_body applies the prim scale, including a
        # non-uniform scale that turns the circular base into an ellipse.
        mesh_local, faces = cone_mesh(radius, height, axis)
        return None, None, None, "mesh", mesh_local, faces
    if kind in {"Capsule", "Cylinder"}:
        geom = UsdGeom.Capsule(prim) if kind == "Capsule" else UsdGeom.Cylinder(prim)
        radius = float(geom.GetRadiusAttr().Get() or (0.5 if kind == "Capsule" else 1.0))
        height = float(geom.GetHeightAttr().Get() or (1.0 if kind == "Capsule" else 2.0))
        axis = str(geom.GetAxisAttr().Get() or "Z")
        radial = _perpendicular_scales(axis_scale, axis)
        axial = _axial_scale(axis_scale, axis)
        if _uniform(radial):
            quat = _axis_quat(quat, axis)
            radial_scale = float(np.mean(radial))
            size = (radius * radial_scale, height * axial * 0.5)
            geom_type = "capsule" if kind == "Capsule" else "cylinder"
            return pos, quat, size, geom_type, None, None
        if kind == "Capsule":
            mesh_local, faces = capsule_mesh(radius, radius, height, axis)
        else:
            mesh_local, faces = cylinder_mesh(radius, radius, height, axis)
        return None, None, None, "mesh", mesh_local, faces
    return None


def _primitive_world_bounds(prim, world: np.ndarray):
    kind = prim.GetTypeName()
    if kind == "Cube":
        edge = float(UsdGeom.Cube(prim).GetSizeAttr().Get() or 2.0) * 0.5
        corners = np.array(
            [[x, y, z] for x in (-edge, edge) for y in (-edge, edge) for z in (-edge, edge)]
        )
        transformed = transform_points(world, corners)
        return transformed.min(axis=0), transformed.max(axis=0)
    if kind == "Sphere":
        radius = float(UsdGeom.Sphere(prim).GetRadiusAttr().Get() or 1.0)
        extent = radius * np.linalg.norm(world[:3, :3], axis=1)
        center = world[3, :3]
        return center - extent, center + extent
    if kind == "Plane":
        plane = UsdGeom.Plane(prim)
        reach = max(float(plane.GetWidthAttr().Get() or 2.0), float(plane.GetLengthAttr().Get() or 2.0))
        center = world[3, :3]
        return center - reach, center + reach
    radius = float((UsdGeom.Capsule(prim) if kind == "Capsule" else UsdGeom.Cylinder(prim)).GetRadiusAttr().Get() or 1.0)
    reach = radius * float(np.max(np.linalg.norm(world[:3, :3], axis=1)))
    center = world[3, :3]
    return center - reach, center + reach


def _joints(stage, bodies: dict[str, Body], meters: float) -> list[Joint]:
    joints: list[Joint] = []
    used: set[str] = set()
    for prim in stage.Traverse():
        kind = _joint_kind(prim)
        if kind is None:
            continue
        api = UsdPhysics.Joint(prim)
        if api.GetJointEnabledAttr().Get() is False:
            continue
        parent = _target(api.GetBody0Rel())
        child = _target(api.GetBody1Rel())
        if child not in bodies:
            continue
        if parent is not None and parent not in bodies:
            parent = None
        anchor = _local_point(api.GetLocalPos1Attr().Get(), meters)
        parent_anchor = _local_point(api.GetLocalPos0Attr().Get(), meters)
        axis = _joint_axis(prim, api)
        limits = _limits(prim, kind, meters)
        stiffness, damping, springref = _drive(prim, kind, meters)
        collision = api.GetCollisionEnabledAttr().Get()
        dofs = None
        note = None
        export_kind = kind
        if kind == "spherical":
            export_kind = "ball"
            limits, note = _spherical_limits(prim)
        elif kind == "distance":
            limits = _distance_limits(prim, meters)
        elif kind == "generic":
            export_kind, axis, limits, dofs = _generic_joint(prim, api, meters)
        joints.append(
            Joint(
                name=_unique(prim.GetName(), used),
                path=str(prim.GetPath()),
                kind=export_kind,
                parent=parent,
                child=child,
                axis=axis,
                anchor=anchor,
                range=limits,
                stiffness=stiffness,
                damping=damping,
                springref=springref,
                disable_collision=collision is False,
                parent_anchor=parent_anchor,
                dofs=dofs,
                note=note,
            )
        )
    return joints


def _joint_kind(prim) -> str | None:
    schema = prim.GetTypeName()
    return {
        "PhysicsFixedJoint": "fixed",
        "PhysicsRevoluteJoint": "revolute",
        "PhysicsPrismaticJoint": "prismatic",
        "PhysicsSphericalJoint": "spherical",
        "PhysicsDistanceJoint": "distance",
        "PhysicsJoint": "generic",
    }.get(schema)


def _joint_axis(prim, api: UsdPhysics.Joint) -> tuple[float, float, float]:
    token = "Z"
    if prim.HasAPI(UsdPhysics.RevoluteJoint) or prim.GetTypeName() == "PhysicsRevoluteJoint":
        token = str(UsdPhysics.RevoluteJoint(prim).GetAxisAttr().Get() or "Z")
    elif prim.GetTypeName() == "PhysicsPrismaticJoint":
        token = str(UsdPhysics.PrismaticJoint(prim).GetAxisAttr().Get() or "Z")
    direction = _AXIS.get(token, _AXIS["Z"])
    rotation = gf_quat(api.GetLocalRot1Attr().Get())
    rotated = rotate_vector(rotation, direction)
    length = np.linalg.norm(rotated)
    if length == 0.0:
        rotated = direction
    else:
        rotated = rotated / length
    return (float(rotated[0]), float(rotated[1]), float(rotated[2]))


def _limits(prim, kind: str, meters: float) -> tuple[float, float] | None:
    if kind == "revolute":
        joint = UsdPhysics.RevoluteJoint(prim)
        lower = joint.GetLowerLimitAttr().Get()
        upper = joint.GetUpperLimitAttr().Get()
        if lower is None or upper is None:
            return None
        return (np.deg2rad(float(lower)), np.deg2rad(float(upper)))
    if kind == "prismatic":
        joint = UsdPhysics.PrismaticJoint(prim)
        lower = joint.GetLowerLimitAttr().Get()
        upper = joint.GetUpperLimitAttr().Get()
        if lower is None or upper is None:
            return None
        return (float(lower) * meters, float(upper) * meters)
    return None


def _spherical_limits(prim):
    joint = UsdPhysics.SphericalJoint(prim)
    angles = []
    for attribute in (joint.GetConeAngle0LimitAttr(), joint.GetConeAngle1LimitAttr()):
        value = attribute.Get()
        if value is not None and float(value) >= 0.0:
            angles.append(float(value))
    if not angles:
        return None, None
    note = None
    if len(angles) == 2 and abs(angles[0] - angles[1]) > 1e-3:
        note = (
            "Spherical cone angles differ "
            f"({angles[0]:.6g} and {angles[1]:.6g} degrees). "
            "MuJoCo ball joints have one circular cone, so the smaller angle is used."
        )
    return (0.0, float(np.deg2rad(min(angles)))), note


def _distance_limits(prim, meters: float):
    joint = UsdPhysics.DistanceJoint(prim)
    minimum = joint.GetMinDistanceAttr().Get()
    maximum = joint.GetMaxDistanceAttr().Get()

    def active(value):
        return value is not None and float(value) >= 0.0

    low = float(minimum) * meters if active(minimum) else None
    high = float(maximum) * meters if active(maximum) else None
    if low is None and high is None:
        return None
    if low is None:
        low = 0.0
    if high is None:
        high = max(low, 1.0e6)
    if low > high:
        low, high = high, low
    return (low, high)


_GENERIC_DOFS = (
    ("transX", "slide", np.array([1.0, 0.0, 0.0]), False),
    ("transY", "slide", np.array([0.0, 1.0, 0.0]), False),
    ("transZ", "slide", np.array([0.0, 0.0, 1.0]), False),
    ("rotX", "hinge", np.array([1.0, 0.0, 0.0]), True),
    ("rotY", "hinge", np.array([0.0, 1.0, 0.0]), True),
    ("rotZ", "hinge", np.array([0.0, 0.0, 1.0]), True),
)


def _generic_joint(prim, api: UsdPhysics.Joint, meters: float):
    """UsdPhysics joints with LimitAPI are D6 joints.

    A degree of freedom with no LimitAPI is locked. A limit spanning all
    finite numbers, or with low > high, is free. Anything else is a ranged
    hinge or slide. Three free rotations and no translations become one ball
    joint, which MuJoCo can represent without a gimbal chain.
    """
    rotation = gf_quat(api.GetLocalRot1Attr().Get())
    unlocked = []
    for instance, mj_type, direction, angular in _GENERIC_DOFS:
        if not prim.HasAPI(UsdPhysics.LimitAPI, instance):
            continue
        limit = UsdPhysics.LimitAPI(prim, instance)
        span = _limit_span(limit.GetLowAttr().Get(), limit.GetHighAttr().Get(), meters, angular)
        rotated = rotate_vector(rotation, direction)
        length = float(np.linalg.norm(rotated))
        if length == 0.0:
            rotated = direction
        else:
            rotated = rotated / length
        unlocked.append(
            {
                "type": mj_type,
                "axis": (float(rotated[0]), float(rotated[1]), float(rotated[2])),
                "range": span,
            }
        )
    hinges = [dof for dof in unlocked if dof["type"] == "hinge"]
    slides = [dof for dof in unlocked if dof["type"] == "slide"]
    if not unlocked:
        return "fixed", (0.0, 0.0, 1.0), None, None
    if len(hinges) == 3 and not slides and all(dof["range"] is None for dof in hinges):
        return "ball", (0.0, 0.0, 1.0), None, None
    if len(unlocked) == 1:
        dof = unlocked[0]
        kind = "revolute" if dof["type"] == "hinge" else "prismatic"
        return kind, dof["axis"], dof["range"], None
    return "compound", unlocked[0]["axis"], unlocked[0]["range"], unlocked


def _limit_span(low, high, meters: float, angular: bool):
    if low is None or high is None:
        return None
    low = float(low)
    high = float(high)
    if not np.isfinite(low) or not np.isfinite(high) or low > high:
        return None
    if angular:
        return (float(np.deg2rad(low)), float(np.deg2rad(high)))
    return (low * meters, high * meters)


def _perpendicular_scales(axis_scale, axis: str) -> tuple[float, float]:
    if axis == "X":
        return (float(axis_scale[1]), float(axis_scale[2]))
    if axis == "Y":
        return (float(axis_scale[0]), float(axis_scale[2]))
    return (float(axis_scale[0]), float(axis_scale[1]))


def _axial_scale(axis_scale, axis: str) -> float:
    return float(axis_scale[{"X": 0, "Y": 1, "Z": 2}[axis]])


def _gravity(stage, meters: float) -> tuple[float, float, float]:
    for prim in stage.Traverse():
        if prim.GetTypeName() != "PhysicsScene":
            continue
        scene = UsdPhysics.Scene(prim)
        if not scene.GetGravityMagnitudeAttr().IsAuthored():
            continue
        magnitude = scene.GetGravityMagnitudeAttr().Get()
        if magnitude is None or not np.isfinite(float(magnitude)):
            continue
        direction = scene.GetGravityDirectionAttr().Get()
        vector = np.array(
            [0.0, 0.0, -1.0] if direction is None else [float(value) for value in direction],
            dtype=np.float64,
        )
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            vector = np.array([0.0, 0.0, -1.0])
            norm = 1.0
        vector = vector / norm * float(magnitude) * meters
        return (float(vector[0]), float(vector[1]), float(vector[2]))
    return (0.0, 0.0, -9.81)


def _filtered_pairs(stage, body_paths: set[str]) -> list[tuple[str, str]]:
    pairs = []
    seen: set[tuple[str, str]] = set()
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.FilteredPairsAPI):
            continue
        relation = prim.GetRelationship("physics:filteredPairs")
        if not relation:
            continue
        owner = str(prim.GetPath()) if str(prim.GetPath()) in body_paths else _owner(prim, body_paths)
        if owner is None:
            continue
        for target in relation.GetTargets():
            other_path = str(target)
            if other_path not in body_paths:
                other_prim = stage.GetPrimAtPath(target)
                other_path = _owner(other_prim, body_paths) if other_prim else None
            if not other_path or other_path == owner:
                continue
            key = tuple(sorted((owner, other_path)))
            if key in seen:
                continue
            seen.add(key)
            pairs.append(key)
    return pairs


def _drive(prim, kind: str, meters: float):
    instance = "linear" if kind == "prismatic" else "angular"
    if not prim.HasAPI(UsdPhysics.DriveAPI, instance):
        return None, None, None
    drive = UsdPhysics.DriveAPI(prim, instance)
    stiffness = _finite(drive.GetStiffnessAttr().Get())
    damping = _finite(drive.GetDampingAttr().Get())
    target = drive.GetTargetPositionAttr().Get()
    springref = None
    if target is not None and np.isfinite(float(target)):
        springref = float(target)
        if kind in {"revolute", "spherical"}:
            springref = float(np.deg2rad(springref))
        elif kind == "prismatic":
            springref *= meters
    return stiffness, damping, springref


def _mass(prim, cache, rigid: np.ndarray, meters: float, kilograms: float, up_axis: str):
    if not prim.HasAPI(UsdPhysics.MassAPI):
        return None, None, None, None
    api = UsdPhysics.MassAPI(prim)
    mass = _finite(api.GetMassAttr().Get())
    if mass is not None:
        mass *= kilograms
    com = api.GetCenterOfMassAttr().Get()
    com_m = None
    if com is not None and all(np.isfinite(float(component)) for component in com):
        local = np.array([[float(com[0]), float(com[1]), float(com[2])]])
        world = _world_matrix(cache, prim, meters, up_axis)
        # COM is authored in prim-local space. Map it through the same matrix
        # the mesh uses, then express it in the rigid body frame.
        mapped = transform_points(world @ invert_rigid(rigid), local)[0]
        com_m = (float(mapped[0]), float(mapped[1]), float(mapped[2]))
    inertia = api.GetDiagonalInertiaAttr().Get()
    inertia_m = None
    if inertia is not None and all(np.isfinite(float(component)) for component in inertia):
        scale = meters * meters * kilograms
        inertia_m = tuple(max(float(value) * scale, 0.0) for value in inertia)
    axes = api.GetPrincipalAxesAttr().Get()
    principal = gf_quat(axes) if axes is not None else None
    if principal is not None and abs(float(np.linalg.norm(principal)) - 1.0) > 1e-3:
        principal = None
    return mass, com_m, inertia_m, principal


def _friction(prim) -> tuple[float, float, float] | None:
    current = prim
    while current and current.IsValid():
        for name in ("material:binding:physics", "material:binding"):
            relation = current.GetRelationship(name)
            if not relation:
                continue
            targets = relation.GetTargets()
            if not targets:
                continue
            material = current.GetStage().GetPrimAtPath(targets[0])
            if material and material.HasAPI(UsdPhysics.MaterialAPI):
                api = UsdPhysics.MaterialAPI(material)
                static = float(api.GetStaticFrictionAttr().Get() or 0.0)
                dynamic = float(api.GetDynamicFrictionAttr().Get() or 0.0)
                restitution = float(api.GetRestitutionAttr().Get() or 0.0)
                return static, dynamic, restitution
        current = current.GetParent()
    return None


def _scale_ops(prim, stop: str) -> list[ScaleOp]:
    found: list[ScaleOp] = []
    current = prim
    while current and current.IsValid() and str(current.GetPath()) != stop:
        xformable = UsdGeom.Xformable(current)
        if xformable:
            for op in xformable.GetOrderedXformOps():
                if op.GetOpType() != UsdGeom.XformOp.TypeScale:
                    continue
                value = op.Get()
                if value is None:
                    continue
                found.append(
                    ScaleOp(
                        str(current.GetPath()),
                        (float(value[0]), float(value[1]), float(value[2])),
                    )
                )
        current = current.GetParent()
    return found


def _mesh(prim):
    mesh = UsdGeom.Mesh(prim)
    points = mesh.GetPointsAttr().Get()
    counts = mesh.GetFaceVertexCountsAttr().Get()
    indices = mesh.GetFaceVertexIndicesAttr().Get()
    if not points or not counts or not indices:
        return None, None
    coordinates = np.array(points, dtype=np.float64)
    faces: list[tuple[int, int, int]] = []
    cursor = 0
    index_list = list(indices)
    for count in counts:
        polygon = index_list[cursor : cursor + int(count)]
        cursor += int(count)
        for offset in range(1, len(polygon) - 1):
            faces.append((int(polygon[0]), int(polygon[offset]), int(polygon[offset + 1])))
    if not faces:
        return None, None
    return coordinates, np.array(faces, dtype=np.int32)


def _world_matrix(cache, prim, meters: float, up_axis: str) -> np.ndarray:
    matrix = gf_matrix(cache.GetLocalToWorldTransform(prim))
    if up_axis == "Y":
        matrix = matrix @ Y_UP_TO_Z_UP
    matrix = matrix.copy()
    matrix[:3, :3] *= meters
    matrix[3, :3] *= meters
    return matrix


def _rigid_pose(cache, prim, meters: float, up_axis: str):
    raw = gf_matrix(cache.GetLocalToWorldTransform(prim))
    if up_axis == "Y":
        raw = raw @ Y_UP_TO_Z_UP
    rotation = _rotation_of(raw)
    rigid = np.eye(4)
    rigid[:3, :3] = rotation
    rigid[3, :3] = raw[3, :3] * meters
    quat = matrix_to_quat(rotation.T)
    pos = (float(rigid[3, 0]), float(rigid[3, 1]), float(rigid[3, 2]))
    return rigid, pos, quat


def _rotation_of(matrix: np.ndarray) -> np.ndarray:
    gf = Gf.Matrix4d(
        float(matrix[0, 0]), float(matrix[0, 1]), float(matrix[0, 2]), float(matrix[0, 3]),
        float(matrix[1, 0]), float(matrix[1, 1]), float(matrix[1, 2]), float(matrix[1, 3]),
        float(matrix[2, 0]), float(matrix[2, 1]), float(matrix[2, 2]), float(matrix[2, 3]),
        float(matrix[3, 0]), float(matrix[3, 1]), float(matrix[3, 2]), float(matrix[3, 3]),
    )
    extracted = gf.ExtractRotationMatrix()
    rotation = np.array(
        [[extracted[row][col] for col in range(3)] for row in range(3)],
        dtype=np.float64,
    )
    if np.linalg.det(rotation) < 0.0:
        rotation[:, 0] *= -1.0
    return rotation


def _orthonormalize(linear: np.ndarray) -> np.ndarray:
    gf = Gf.Matrix4d(
        float(linear[0, 0]), float(linear[0, 1]), float(linear[0, 2]), 0.0,
        float(linear[1, 0]), float(linear[1, 1]), float(linear[1, 2]), 0.0,
        float(linear[2, 0]), float(linear[2, 1]), float(linear[2, 2]), 0.0,
        0.0, 0.0, 0.0, 1.0,
    )
    extracted = gf.ExtractRotationMatrix()
    rotation = np.array(
        [[extracted[row][col] for col in range(3)] for row in range(3)],
        dtype=np.float64,
    )
    if np.linalg.det(rotation) < 0.0:
        rotation[:, 0] *= -1.0
    return rotation


def _axis_quat(quat, axis: str):
    if axis == "X":
        extra = (np.sqrt(0.5), 0.0, np.sqrt(0.5), 0.0)
    elif axis == "Y":
        extra = (np.sqrt(0.5), -np.sqrt(0.5), 0.0, 0.0)
    else:
        return quat
    return _mul_quat(quat, extra)


def _mul_quat(lhs, rhs):
    w1, x1, y1, z1 = lhs
    w2, x2, y2, z2 = rhs
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def _unit_cube(points: np.ndarray) -> bool:
    if len(points) < 8:
        return False
    low = points.min(axis=0)
    high = points.max(axis=0)
    return np.allclose(low, -0.5, atol=1e-3) and np.allclose(high, 0.5, atol=1e-3)


def _owner(prim, body_paths: set[str]) -> str | None:
    current = prim.GetParent()
    while current and current.IsValid():
        path = str(current.GetPath())
        if path in body_paths:
            return path
        current = current.GetParent()
    return None


def _guide(prim) -> bool:
    purpose = UsdGeom.Imageable(prim).GetPurposeAttr().Get()
    return purpose == UsdGeom.Tokens.guide


def _target(relation) -> str | None:
    targets = relation.GetTargets()
    if not targets:
        return None
    return str(targets[0])


def _local_point(value, meters: float) -> tuple[float, float, float]:
    if value is None:
        return (0.0, 0.0, 0.0)
    return (float(value[0]) * meters, float(value[1]) * meters, float(value[2]) * meters)


def _finite(value):
    if value is None:
        return None
    number = float(value)
    if not np.isfinite(number) or number == 0.0:
        return None
    return number


def _unique(name: str, used: set[str]) -> str:
    cleaned = "".join(char if char.isalnum() or char == "_" else "_" for char in name)
    cleaned = cleaned.strip("_") or "part"
    if cleaned[0].isdigit():
        cleaned = f"p_{cleaned}"
    candidate = cleaned
    index = 2
    while candidate in used:
        candidate = f"{cleaned}_{index}"
        index += 1
    used.add(candidate)
    return candidate
