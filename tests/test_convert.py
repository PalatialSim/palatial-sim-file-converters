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
