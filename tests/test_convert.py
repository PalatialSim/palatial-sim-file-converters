"""Round trips between USD, MJCF, URDF, and GLB."""

import struct
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from palatial_sim_file_converters.export import mjcf_tree
from palatial_sim_file_converters.glb import glb_document
from palatial_sim_file_converters.mjcf_read import parse_mjcf
from palatial_sim_file_converters.read import load_usda
from palatial_sim_file_converters.urdf import parse_urdf, urdf_tree
from palatial_sim_file_converters.usd_write import build_stage


def test_static_supports_and_disabled_collisions_survive_usd_conversion():
    from pxr import Usd, UsdGeom, UsdPhysics
    from palatial_sim_file_converters.read import asset_from_stage

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, "Z")
    floor = UsdGeom.Cube.Define(stage, "/World/Floor")
    floor.AddTranslateOp().Set((1, 2, -0.1))
    floor.AddScaleOp().Set((4, 3, 0.1))
    UsdPhysics.CollisionAPI.Apply(floor.GetPrim())
    support = UsdGeom.Xform.Define(stage, "/World/FixedSupport")
    UsdPhysics.RigidBodyAPI.Apply(support.GetPrim()).CreateRigidBodyEnabledAttr(False)
    UsdPhysics.CollisionAPI.Apply(UsdGeom.Cube.Define(stage, "/World/FixedSupport/Shape").GetPrim())
    decoration = UsdGeom.Cube.Define(stage, "/World/Decoration")
    UsdPhysics.CollisionAPI.Apply(decoration.GetPrim()).CreateCollisionEnabledAttr(False)
    prop = UsdGeom.Xform.Define(stage, "/World/Prop")
    prop.AddTranslateOp().Set((0, 0, 4))
    UsdPhysics.RigidBodyAPI.Apply(prop.GetPrim())
    UsdPhysics.CollisionAPI.Apply(UsdGeom.Sphere.Define(stage, "/World/Prop/Shape").GetPrim())

    asset = asset_from_stage(stage, "scene.usda")
    bodies = {body.name: body for body in asset.bodies}
    assert bodies["Floor"].static and not bodies["Floor"].kinematic
    assert bodies["FixedSupport"].static
    assert not bodies["Prop"].static
    assert not bodies["Decoration"].colliders and len(bodies["Decoration"].visuals) == 1
    assert np.allclose(bodies["Floor"].colliders[0].world_min, [-3, -1, -0.2])
    root = mjcf_tree(asset)
    assert root.find(".//body[@name='Floor']/freejoint") is None
    assert root.find(".//body[@name='FixedSupport']/freejoint") is None
    assert root.find(".//body[@name='Prop']/freejoint") is not None
    assert root.find(".//body[@name='Decoration']/geom").get("contype") == "0"
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    assert model.nv == 6
    assert model.ngeom == 4

    restored_stage = build_stage(asset)
    restored = asset_from_stage(restored_stage, "restored.usda")
    restored_bodies = {body.name: body for body in restored.bodies}
    assert restored_bodies["Floor"].static
    assert len(restored_bodies["FixedSupport"].colliders) == 1
    assert not restored_bodies["Prop"].static
    assert np.allclose(restored_bodies["Floor"].colliders[0].world_min, [-3, -1, -0.2])
    robot = urdf_tree(asset)
    fixed = robot.find("joint[@name='Floor_world']")
    assert fixed is not None and fixed.get("type") == "fixed"
    _, manifest = glb_document(asset)
    assert next(body for body in manifest["bodies"] if body["name"] == "Floor")["static"] is True


def test_repeated_usd_instances_keep_bodies_colliders_and_world_poses():
    from pxr import Usd, UsdGeom, UsdPhysics
    from palatial_sim_file_converters.read import asset_from_stage

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, "Z")
    stage.CreateClassPrim("/Template")
    body = UsdGeom.Xform.Define(stage, "/Template/Body")
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    UsdPhysics.CollisionAPI.Apply(UsdGeom.Cube.Define(stage, "/Template/Body/Shape").GetPrim())
    for name, position in [("Left", (2, 0, 1)), ("Right", (-2, 0, 1))]:
        instance = UsdGeom.Xform.Define(stage, f"/World/{name}")
        instance.GetPrim().GetReferences().AddInternalReference("/Template")
        instance.GetPrim().SetInstanceable(True)
        instance.AddTranslateOp().Set(position)

    asset = asset_from_stage(stage, "instances.usda")
    assert {body.path for body in asset.bodies} == {"/World/Left/Body", "/World/Right/Body"}
    assert len({body.name for body in asset.bodies}) == 2
    assert all(len(body.colliders) == 1 for body in asset.bodies)
    assert sorted(body.pos for body in asset.bodies) == [(-2.0, 0.0, 1.0), (2.0, 0.0, 1.0)]
    model = pytest.importorskip("mujoco").MjModel.from_xml_string(ET.tostring(mjcf_tree(asset), encoding="unicode"))
    assert model.nv == 12 and model.ngeom == 2


def test_converted_instance_supports_hold_dynamic_props_in_mujoco():
    from pxr import Usd, UsdGeom, UsdPhysics
    from palatial_sim_file_converters.read import asset_from_stage

    mujoco = pytest.importorskip("mujoco")
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, "Z")
    physics = UsdPhysics.Scene.Define(stage, "/Physics")
    physics.CreateGravityDirectionAttr((0, 0, -1))
    physics.CreateGravityMagnitudeAttr(9.81)
    stage.CreateClassPrim("/SupportTemplate")
    shape = UsdGeom.Cube.Define(stage, "/SupportTemplate/Shape")
    UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
    for name, position, scale in [
        ("Floor", (0, 0, -0.05), (2, 2, 0.05)),
        ("Table", (0, 0, 0.75), (0.4, 0.25, 0.02)),
    ]:
        support = UsdGeom.Xform.Define(stage, f"/World/{name}")
        support.GetPrim().GetReferences().AddInternalReference("/SupportTemplate")
        support.GetPrim().SetInstanceable(True)
        support.AddTranslateOp().Set(position)
        support.AddScaleOp().Set(scale)
    for name, x in [("TableProp", 0), ("FloorProp", 1)]:
        prop = UsdGeom.Xform.Define(stage, f"/World/{name}")
        prop.AddTranslateOp().Set((x, 0, 1.1))
        UsdPhysics.RigidBodyAPI.Apply(prop.GetPrim())
        sphere = UsdGeom.Sphere.Define(stage, f"/World/{name}/Shape")
        sphere.CreateRadiusAttr(0.03)
        UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())

    asset = asset_from_stage(stage, "support-fixture.usda")
    xml = mjcf_tree(asset)
    model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert model.nv == 12 and data.ncon == 0
    by_path = {body.path: body for body in asset.bodies}
    body_ids = {
        path: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body.name)
        for path, body in by_path.items()
    }
    settled_positions = []
    for step in range(2000):
        mujoco.mj_step(model, data)
        if step >= 1750:
            settled_positions.append(data.xpos.copy())
    for prop, support, height in [
        ("TableProp", "Table", 0.8), ("FloorProp", "Floor", 0.03),
    ]:
        prop_id = body_ids[f"/World/{prop}"]
        support_id = body_ids[f"/World/{support}/Shape"]
        assert data.xpos[prop_id, 2] == pytest.approx(height, abs=0.001)
        assert np.ptp(np.array(settled_positions)[:, prop_id, 2]) < 1e-6
        assert any(
            {int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])}
            == {prop_id, support_id}
            for c in data.contact
        )
    assert np.isfinite(data.qpos).all() and np.max(np.abs(data.qvel)) < 1e-5
    assert min(c.dist for c in data.contact) > -0.001

    # Removing precisely the converted static supports must make both props fall.
    world = xml.find("worldbody")
    static_names = {body.name for body in asset.bodies if body.static}
    for body in list(world):
        if body.get("name") in static_names:
            world.remove(body)
    unsupported = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
    falling = mujoco.MjData(unsupported)
    for _ in range(1000):
        mujoco.mj_step(unsupported, falling)
    assert all(falling.qpos[int(unsupported.jnt_qposadr[joint]) + 2] < -1
               for joint in range(unsupported.njnt))
    assert falling.ncon == 0


def test_fixed_mjcf_geometry_round_trips_without_new_free_motion():
    asset = parse_mjcf('<mujoco><worldbody><geom type="box" size="2 2 .1"/><body name="fixed"><geom type="sphere" size=".2"/></body></worldbody></mujoco>')
    assert all(body.static and not body.kinematic for body in asset.bodies)
    stage = build_stage(asset)
    restored = load_usda(stage.GetRootLayer().ExportToString())
    root = mjcf_tree(restored)
    assert not root.findall(".//freejoint")
    model = pytest.importorskip("mujoco").MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    assert model.nv == 0 and model.ngeom == 2


def test_visual_visibility_is_inherited_without_disabling_physical_colliders():
    from pxr import Usd, UsdGeom, UsdPhysics
    from palatial_sim_file_converters.read import asset_from_stage

    stage = Usd.Stage.CreateInMemory()
    for path, setting in [("/World/Hidden", "visibility"), ("/World/Guides", "purpose")]:
        parent = UsdGeom.Xform.Define(stage, path)
        if setting == "visibility":
            parent.CreateVisibilityAttr("invisible")
        else:
            parent.CreatePurposeAttr("guide")
        UsdGeom.Cube.Define(stage, path + "/Helper")
        collider = UsdGeom.Cube.Define(stage, path + "/PhysicalShape")
        UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
    direct_body = UsdGeom.Cube.Define(stage, "/World/DirectBody")
    UsdPhysics.RigidBodyAPI.Apply(direct_body.GetPrim())
    UsdPhysics.CollisionAPI.Apply(direct_body.GetPrim())
    asset = asset_from_stage(stage, "scene.usda")
    assert {body.path for body in asset.bodies} == {"/World/DirectBody", "/World/Hidden/PhysicalShape", "/World/Guides/PhysicalShape"}
    assert all(len(body.colliders) == 1 and not body.visuals for body in asset.bodies)
    assert next(body for body in asset.bodies if body.path == "/World/DirectBody").static is False
    hidden = [body for body in asset.bodies if body.static]
    assert all(body.colliders[0].visible is False for body in hidden)
    xml = mjcf_tree(asset)
    for body in hidden:
        geom = xml.find(f".//body[@name='{body.name}']/geom")
        assert geom.get("group") == "3" and geom.get("rgba") == "0 0 0 0"
        assert geom.get("contype") != "0"
    restored_stage = build_stage(asset)
    restored = asset_from_stage(restored_stage, "restored.usda")
    assert all(not body.colliders[0].visible for body in restored.bodies if body.static)
    payload, manifest = glb_document(asset)
    length = struct.unpack_from("<I", payload, 12)[0]
    import json
    document = json.loads(payload[20:20 + length])
    assert all("mesh" not in node for node in document["nodes"] if node["name"] in {body.name for body in hidden})
    assert all(not body["colliders"][0]["visible"] for body in manifest["bodies"] if body["static"])
    mjcf_restored = parse_mjcf(ET.tostring(xml, encoding="unicode"))
    assert all(not body.colliders[0].visible for body in mjcf_restored.bodies if body.static)

MJCF = """<mujoco model="sample">
  <compiler angle="radian"/>
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <body name="base" pos="1 0 0">
      <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
      <site name="base_site" pos="0 0 0"/>
      <geom name="ball" type="sphere" size="0.033" pos="0 0 0.01" friction="0.4"/>
      <geom name="block" type="box" size="0.1 0.2 0.3"/>
      <body name="lid" pos="0 0 0.2">
        <inertial pos="0 0 0" mass="0.2" diaginertia="0.001 0.001 0.001"/>
        <joint name="hinge" type="hinge" axis="0 0 1" range="-0.5 1.0"/>
        <geom type="capsule" size="0.02 0.05"/>
      </body>
    </body>
    <body name="bob" pos="0.4 0 0">
      <freejoint/>
      <inertial pos="0 0 0" mass="0.1" diaginertia="0.0001 0.0001 0.0001"/>
      <site name="bob_site" pos="0 0 0"/>
      <geom type="sphere" size="0.02"/>
    </body>
    <body name="cup" pos="0 1 0">
      <inertial pos="0 0 0" mass="0.1" diaginertia="0.0001 0.0001 0.0001"/>
      <joint name="swing" type="ball" range="0 0.3"/>
      <geom type="sphere" size="0.02"/>
    </body>
  </worldbody>
  <tendon>
    <spatial name="gap" range="0.1 0.4">
      <site site="base_site"/>
      <site site="bob_site"/>
    </spatial>
  </tendon>
  <contact>
    <exclude body1="base" body2="lid"/>
  </contact>
</mujoco>
"""

URDF = """<robot name="arm">
  <link name="base">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="1"/>
      <inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><cylinder radius="0.05" length="0.2"/></geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><box size="0.2 0.4 0.6"/></geometry>
    </collision>
  </link>
  <link name="tip">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="0.1"/>
      <inertia ixx="0.0001" iyy="0.0001" izz="0.0001" ixy="0" ixz="0" iyz="0"/>
    </inertial>
    <collision>
      <geometry><sphere radius="0.02"/></geometry>
    </collision>
  </link>
  <joint name="slide" type="prismatic">
    <parent link="base"/>
    <child link="tip"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-0.02" upper="0.05" effort="1" velocity="1"/>
  </joint>
</robot>
"""


def test_mjcf_usd_round_trip():
    stage = build_stage(parse_mjcf(MJCF))
    asset = load_usda(stage.GetRootLayer().ExportToString())
    assert "/" not in asset.source
    by_name = {body.name: body for body in asset.bodies}
    sphere = next(collider for collider in by_name["base"].colliders if collider.geom_type == "sphere")
    box = next(collider for collider in by_name["base"].colliders if collider.geom_type == "box")
    assert sphere.primitive_size[0] == pytest.approx(0.033)
    assert sphere.primitive_pos[2] == pytest.approx(0.01)
    assert sphere.friction[0] == pytest.approx(0.4)
    assert np.allclose(box.primitive_size, [0.1, 0.2, 0.3])
    assert by_name["base"].mass == pytest.approx(1.0)
    hinge = next(joint for joint in asset.joints if joint.kind == "revolute")
    assert np.allclose(hinge.axis, [0.0, 0.0, 1.0], atol=1e-5)
    assert np.allclose(hinge.range, [-0.5, 1.0], atol=1e-5)
    ball = next(joint for joint in asset.joints if joint.kind == "ball")
    assert ball.range[1] == pytest.approx(0.3)
    distance = next(joint for joint in asset.joints if joint.kind == "distance")
    assert np.allclose(distance.range, [0.1, 0.4])
    names = {body.path: body.name for body in asset.bodies}
    assert {tuple(sorted((names[a], names[b]))) for a, b in asset.filtered_pairs} == {("base", "lid")}
    assert np.allclose(asset.gravity, [0.0, 0.0, -9.81])

    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(ET.tostring(mjcf_tree(asset), encoding="unicode"))
    assert model.nbody > 2


def test_urdf_to_mjcf_and_usd():
    root = mjcf_tree(parse_urdf(URDF))
    box = root.find(".//geom[@type='box']")
    assert np.allclose([float(value) for value in box.get("size").split()], [0.1, 0.2, 0.3])
    cylinder = root.find(".//geom[@type='cylinder']")
    assert cylinder.get("contype") == "0"
    assert np.allclose([float(value) for value in cylinder.get("size").split()], [0.05, 0.1])
    slide = root.find(".//joint[@name='slide']")
    assert slide.get("type") == "slide"
    assert np.allclose([float(value) for value in slide.get("range").split()], [-0.02, 0.05])
    mujoco = pytest.importorskip("mujoco")
    mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))

    stage = build_stage(parse_urdf(URDF))
    asset = load_usda(stage.GetRootLayer().ExportToString())
    prismatic = next(joint for joint in asset.joints if joint.kind == "prismatic")
    assert np.allclose(prismatic.range, [-0.02, 0.05])
    assert np.allclose(prismatic.axis, [0.0, 0.0, 1.0], atol=1e-5)


def test_usd_to_urdf_and_glb():
    asset = load_usda(
        """#usda 1.0
(
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "World"
{
    def Xform "Base" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    )
    {
        float physics:mass = 0.2
        float3 physics:diagonalInertia = (0.0002, 0.0002, 0.0002)
        def Sphere "shape" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {
            double radius = 0.05
        }
    }
    def Xform "Lid" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
    )
    {
        float physics:mass = 0.1
        float3 physics:diagonalInertia = (0.0001, 0.0001, 0.0001)
        double3 xformOp:translate = (0, 0, 0.2)
        uniform token[] xformOpOrder = ["xformOp:translate"]
        def Sphere "shape" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {
            double radius = 0.02
        }
    }
    def PhysicsRevoluteJoint "hinge"
    {
        rel physics:body0 = </World/Base>
        rel physics:body1 = </World/Lid>
        uniform token physics:axis = "Z"
        float physics:lowerLimit = -50
        float physics:upperLimit = 110
    }
}
"""
    )
    robot = urdf_tree(asset)
    assert {link.get("name") for link in robot.findall("link")} >= {"Base", "Lid"}
    joint = robot.find("joint")
    assert joint.get("type") == "revolute"
    limit = joint.find("limit")
    assert float(limit.get("lower")) == pytest.approx(-50)
    assert float(limit.get("upper")) == pytest.approx(110)

    payload, manifest = glb_document(asset)
    magic, version, _length = struct.unpack("<III", payload[:12])
    assert magic == 0x46546C67
    assert version == 2
    assert "/" not in manifest["source"]
    masses = {body["name"]: body["mass"] for body in manifest["bodies"]}
    assert masses["Base"] == pytest.approx(0.2)
    assert masses["Lid"] == pytest.approx(0.1)
    assert manifest["joints"][0]["type"] == "revolute"
