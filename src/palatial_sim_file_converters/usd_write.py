"""Write the shared scene as an OpenUSD stage with UsdPhysics.

Analytic MuJoCo geoms become the matching UsdGeom primitive. Mesh and SDF
geoms become triangle meshes with ``convexHull`` or ``sdf`` approximation.
Revolute and prismatic limits are degrees and stage units. A ball joint is a
spherical joint whose cone is the MuJoCo range. A distance joint is not a
parent. Filtered pairs become ``physics:filteredPairs``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from palatial_sim_file_converters.frame import (
    align_axis,
    child_bodies,
    kinematic_joints,
    near_identity,
    relative_pose,
    root_bodies,
)
from palatial_sim_file_converters.model import Asset, Body, Joint
from palatial_sim_file_converters.primitives import box_mesh, sphere_mesh


def write_usd(asset: Asset, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    _gravity(stage, asset.gravity)
    incoming = kinematic_joints(asset)
    written: dict[str, str] = {}
    for body in root_bodies(asset, incoming):
        _emit_body(stage, body, None, "/World", asset, incoming, written)
    for joint in asset.joints:
        _emit_joint(stage, joint, written)
    _filtered(stage, asset, written)
    stage.GetRootLayer().Save()
    return path


def _emit_body(stage, body: Body, parent: Body | None, parent_path: str, asset, incoming, written):
    path = f"{parent_path}/{_token(body.name)}"
    written[body.path] = path
    xform = UsdGeom.Xform.Define(stage, path)
    position, quat = relative_pose(body, parent)
    _set_pose(xform, position, quat)
    rigid = UsdPhysics.RigidBodyAPI.Apply(xform.GetPrim())
    if body.kinematic:
        rigid.CreateKinematicEnabledAttr().Set(True)
    if body.mass is not None:
        mass = UsdPhysics.MassAPI.Apply(xform.GetPrim())
        mass.CreateMassAttr().Set(float(body.mass))
        if body.com is not None:
            mass.CreateCenterOfMassAttr().Set(_vec3(body.com))
        if body.diaginertia is not None:
            mass.CreateDiagonalInertiaAttr().Set(_vec3(body.diaginertia))
        if body.principal_quat is not None:
            mass.CreatePrincipalAxesAttr().Set(_quat(body.principal_quat))
    for index, visual in enumerate(body.visuals):
        _emit_shape(stage, xform.GetPrim(), visual, f"visual_{index}", collide=False)
    for index, collider in enumerate(body.colliders):
        _emit_shape(stage, xform.GetPrim(), collider, f"collision_{index}", collide=True)
    for child in child_bodies(asset, body, incoming):
        _emit_body(stage, child, body, path, asset, incoming, written)


def _emit_shape(stage, body_prim, shape, name: str, collide: bool) -> None:
    points, faces, primitive = _shape_surface(shape)
    path = f"{body_prim.GetPath()}/{_token(name)}"
    if points is not None:
        mesh = UsdGeom.Mesh.Define(stage, path)
        mesh.CreatePointsAttr().Set([_vec3(point) for point in points])
        mesh.CreateFaceVertexCountsAttr().Set([3] * len(faces))
        mesh.CreateFaceVertexIndicesAttr().Set([int(index) for face in faces for index in face])
        mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
        prim = mesh.GetPrim()
    else:
        prim = _primitive(stage, path, shape)
    if collide:
        UsdPhysics.CollisionAPI.Apply(prim)
        if points is not None:
            approximation = UsdPhysics.MeshCollisionAPI.Apply(prim)
            token = "sdf" if shape.geom_type == "sdf" else "convexHull"
            approximation.CreateApproximationAttr().Set(token)
        if getattr(shape, "friction", None):
            _bind_friction(stage, prim, shape.friction, name)


def _shape_surface(shape):
    if shape.local_points is not None:
        return shape.local_points, shape.faces, False
    geom_type = shape.geom_type
    size = shape.primitive_size or (1.0,)
    quat = shape.primitive_quat or (1.0, 0.0, 0.0, 0.0)
    rotated = not near_identity(quat)
    if geom_type == "ellipsoid" or (geom_type == "box" and rotated and not _uniform(size)):
        if geom_type == "box":
            points, faces = box_mesh(size[:3])
        else:
            points, faces = sphere_mesh(1.0)
            points = points * np.asarray(size[:3], dtype=np.float64)
        points = _posed(points, shape.primitive_pos, quat)
        return points, faces, False
    if geom_type == "mesh":
        return None, None, False
    return None, None, True


def _primitive(stage, path: str, shape):
    geom_type = shape.geom_type
    size = shape.primitive_size or (0.01,)
    if geom_type == "sphere":
        prim = UsdGeom.Sphere.Define(stage, path)
        prim.CreateRadiusAttr().Set(float(size[0]))
    elif geom_type == "ellipsoid":
        prim = UsdGeom.Sphere.Define(stage, path)
        prim.CreateRadiusAttr().Set(1.0)
        _set_pose(prim, shape.primitive_pos, shape.primitive_quat, scale=size[:3])
        return prim.GetPrim()
    elif geom_type == "box":
        prim = UsdGeom.Cube.Define(stage, path)
        if _uniform(size):
            prim.CreateSizeAttr().Set(float(size[0]) * 2.0)
        else:
            prim.CreateSizeAttr().Set(2.0)
            _set_pose(prim, shape.primitive_pos, shape.primitive_quat, scale=size[:3])
            return prim.GetPrim()
    elif geom_type == "capsule":
        prim = UsdGeom.Capsule.Define(stage, path)
        prim.CreateRadiusAttr().Set(float(size[0]))
        prim.CreateHeightAttr().Set(float(size[1]) * 2.0)
        prim.CreateAxisAttr().Set(UsdGeom.Tokens.z)
    elif geom_type == "cylinder":
        prim = UsdGeom.Cylinder.Define(stage, path)
        prim.CreateRadiusAttr().Set(float(size[0]))
        prim.CreateHeightAttr().Set(float(size[1]) * 2.0)
        prim.CreateAxisAttr().Set(UsdGeom.Tokens.z)
    elif geom_type == "plane":
        prim = UsdGeom.Plane.Define(stage, path)
        prim.CreateWidthAttr().Set(float(size[0]) * 2.0)
        prim.CreateLengthAttr().Set(float(size[1]) * 2.0)
        prim.CreateAxisAttr().Set(UsdGeom.Tokens.z)
    else:
        points, faces = sphere_mesh(float(size[0]) if size else 0.01)
        mesh = UsdGeom.Mesh.Define(stage, path)
        mesh.CreatePointsAttr().Set([_vec3(point) for point in points])
        mesh.CreateFaceVertexCountsAttr().Set([3] * len(faces))
        mesh.CreateFaceVertexIndicesAttr().Set([int(index) for face in faces for index in face])
        prim = mesh
    _set_pose(prim, shape.primitive_pos, shape.primitive_quat)
    return prim.GetPrim()


def _emit_joint(stage, joint: Joint, written: dict[str, str]) -> None:
    if joint.child not in written:
        return
    if joint.kind == "fixed" and joint.parent is None:
        return
    path = f"/World/joints/{_token(joint.name)}"
    kind = joint.kind
    if kind == "revolute":
        prim = UsdPhysics.RevoluteJoint.Define(stage, path)
        _axis_and_limits(prim, joint, angular=True)
    elif kind == "prismatic":
        prim = UsdPhysics.PrismaticJoint.Define(stage, path)
        _axis_and_limits(prim, joint, angular=False)
    elif kind == "ball":
        prim = UsdPhysics.SphericalJoint.Define(stage, path)
        prim.CreateAxisAttr().Set("Z")
        if joint.range is not None:
            angle = float(np.rad2deg(joint.range[1]))
            prim.CreateConeAngle0LimitAttr().Set(angle)
            prim.CreateConeAngle1LimitAttr().Set(angle)
    elif kind == "distance":
        prim = UsdPhysics.DistanceJoint.Define(stage, path)
        if joint.range is not None:
            prim.CreateMinDistanceAttr().Set(float(joint.range[0]))
            prim.CreateMaxDistanceAttr().Set(float(joint.range[1]))
    elif kind == "compound" and joint.dofs:
        prim = UsdPhysics.Joint.Define(stage, path)
        _compound_limits(prim, joint)
    elif kind == "fixed":
        prim = UsdPhysics.FixedJoint.Define(stage, path)
    else:
        return
    if joint.parent is not None and joint.parent in written:
        prim.CreateBody0Rel().SetTargets([Sdf.Path(written[joint.parent])])
    prim.CreateBody1Rel().SetTargets([Sdf.Path(written[joint.child])])
    prim.CreateLocalPos0Attr().Set(_vec3(joint.parent_anchor))
    prim.CreateLocalPos1Attr().Set(_vec3(joint.anchor))
    if joint.disable_collision:
        prim.CreateCollisionEnabledAttr().Set(False)
    _drive(prim.GetPrim(), joint)


def _axis_and_limits(prim, joint: Joint, angular: bool) -> None:
    prim.CreateAxisAttr().Set("X")
    prim.CreateLocalRot1Attr().Set(_quat(align_axis(np.array([1.0, 0.0, 0.0]), joint.axis)))
    if joint.range is None:
        return
    lower, upper = joint.range
    if angular:
        lower, upper = np.rad2deg(lower), np.rad2deg(upper)
    prim.CreateLowerLimitAttr().Set(float(lower))
    prim.CreateUpperLimitAttr().Set(float(upper))


def _compound_limits(prim, joint: Joint) -> None:
    for dof in joint.dofs or []:
        instance = _dof_instance(dof)
        if instance is None:
            continue
        limit = UsdPhysics.LimitAPI.Apply(prim.GetPrim(), instance)
        span = dof.get("range")
        if span is None:
            limit.CreateLowAttr().Set(float("-inf"))
            limit.CreateHighAttr().Set(float("inf"))
            continue
        lower, upper = span
        if dof["type"] == "hinge":
            lower, upper = np.rad2deg(lower), np.rad2deg(upper)
        limit.CreateLowAttr().Set(float(lower))
        limit.CreateHighAttr().Set(float(upper))


def _dof_instance(dof) -> str | None:
    axis = np.asarray(dof["axis"], dtype=np.float64)
    length = float(np.linalg.norm(axis))
    if length == 0.0:
        return None
    axis = axis / length
    names = "XYZ"
    for index, name in enumerate(names):
        probe = np.zeros(3)
        probe[index] = 1.0
        if np.allclose(np.abs(axis), probe, atol=1e-4):
            prefix = "rot" if dof["type"] == "hinge" else "trans"
            return f"{prefix}{name}"
    return None


def _drive(prim, joint: Joint) -> None:
    if joint.stiffness is None and joint.damping is None and joint.springref is None:
        return
    instance = "linear" if joint.kind == "prismatic" else "angular"
    drive = UsdPhysics.DriveAPI.Apply(prim, instance)
    drive.CreateTypeAttr().Set("force")
    if joint.stiffness is not None:
        drive.CreateStiffnessAttr().Set(float(joint.stiffness))
    if joint.damping is not None:
        drive.CreateDampingAttr().Set(float(joint.damping))
    if joint.springref is not None:
        target = float(np.rad2deg(joint.springref)) if instance == "angular" else float(joint.springref)
        drive.CreateTargetPositionAttr().Set(target)


def _filtered(stage, asset: Asset, written: dict[str, str]) -> None:
    for index, (first, second) in enumerate(asset.filtered_pairs):
        if first not in written or second not in written:
            continue
        prim = stage.GetPrimAtPath(written[first])
        UsdPhysics.FilteredPairsAPI.Apply(prim)
        relation = prim.GetRelationship("physics:filteredPairs")
        if not relation:
            relation = prim.CreateRelationship("physics:filteredPairs", custom=True)
        targets = list(relation.GetTargets())
        targets.append(Sdf.Path(written[second]))
        relation.SetTargets(targets)
        del index


def _bind_friction(stage, prim, friction, name: str) -> None:
    static, dynamic, restitution = friction
    material_path = f"/World/PhysicsMaterials/{_token(name)}"
    material = stage.DefinePrim(material_path, "Material")
    api = UsdPhysics.MaterialAPI.Apply(material)
    api.CreateStaticFrictionAttr().Set(float(static))
    api.CreateDynamicFrictionAttr().Set(float(dynamic))
    api.CreateRestitutionAttr().Set(float(restitution))
    relation = prim.CreateRelationship("material:binding:physics", custom=True)
    relation.SetTargets([Sdf.Path(material_path)])


def _gravity(stage, gravity) -> None:
    scene = UsdPhysics.Scene.Define(stage, "/physicsScene")
    vector = np.asarray(gravity, dtype=np.float64)
    magnitude = float(np.linalg.norm(vector))
    direction = np.array([0.0, 0.0, -1.0]) if magnitude == 0.0 else vector / magnitude
    scene.CreateGravityDirectionAttr().Set(_vec3(direction))
    scene.CreateGravityMagnitudeAttr().Set(magnitude)


def _set_pose(xform, pos, quat, scale=None) -> None:
    if pos is None and quat is None and scale is None:
        return
    position = pos or (0.0, 0.0, 0.0)
    rotation = quat or (1.0, 0.0, 0.0, 0.0)
    order = []
    translate = xform.AddTranslateOp()
    translate.Set(Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])))
    order.append(translate)
    if not near_identity(rotation):
        orient = xform.AddOrientOp()
        orient.Set(_quat(rotation))
        order.append(orient)
    if scale is not None:
        scale_op = xform.AddScaleOp()
        scale_op.Set(_vec3(scale))
        order.append(scale_op)
    xform.SetXformOpOrder(order)


def _posed(points, pos, quat):
    from palatial_sim_file_converters.frame import apply_pose

    return apply_pose(points, pos or (0.0, 0.0, 0.0), quat)


def _uniform(size, tol: float = 1e-6) -> bool:
    values = np.abs(np.asarray(size, dtype=np.float64))
    return float(values.max() - values.min()) <= tol


def _vec3(values):
    return Gf.Vec3f(float(values[0]), float(values[1]), float(values[2]))


def _quat(values):
    return Gf.Quatf(float(values[0]), Gf.Vec3f(float(values[1]), float(values[2]), float(values[3])))


def _token(name: str) -> str:
    cleaned = "".join(char if char.isalnum() or char == "_" else "_" for char in name)
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"n_{cleaned}"
    return cleaned

