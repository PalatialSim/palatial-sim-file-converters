# palatial-sim-file-converters

Convert one simulation file to another. USD, MJCF, and URDF read and write in every direction. USD, MJCF, and URDF also write a GLB plus `manifest.json` containing the rigid bodies, inertias, joints, and collider metadata.

```sh
pip install -e ".[dev]"
sim-convert asset.usd asset.xml
sim-convert asset.xml asset.usda
sim-convert asset.urdf asset.xml
sim-convert asset.usd asset.urdf
sim-convert asset.usd asset.glb
```

`asset.glb` is accompanied by `manifest.json` in the same directory. MJCF mesh files are written to `meshes/` next to the XML. URDF meshes are written the same way.

`isaac-usd-to-mjcf check asset.usd` still reports collider problems on an Isaac USD stage. `isaac-usd-to-mjcf convert asset.usd -o out/` writes the MJCF bundle and `collider_report.json`.

Install the `dev` extra to run the tests (`pytest`). The `compile` extra, included by `dev`, lets `--compile` load an MJCF result in MuJoCo.

Isaac stores many colliders as a unit cube or a convex fragment. The size and orientation are on `xformOp:scale` and `xformOp:orient`, including scales inherited from a parent `Colliders` prim. The USD reader bakes that transform into the collider and reports when it does not match the visual mesh. `isaac-usd-to-mjcf convert` writes `model.xml`, `meshes/*.obj`, and `collider_report.json`.

## What is checked

- **ancestor scale** — a parent prim scales the collider (the golf-ball `1.8` and football `1.53` cases). The MJCF still uses that scale, because that is the shape PhysX simulates.
- **bounds mismatch** — the dedicated collider's world bounds and the visual mesh disagree by more than 25% on any axis.
- **unit cube scale** — the mesh is the cube from -0.5 to 0.5, so dropping the scale op drops the shape.
- **stacked or negative scale** — more than one non-unit scale, or a mirror.
- **double collision** — the visual mesh and a dedicated collider are both enabled.
- **PhysX sdf** — exported as a MuJoCo `sdf` geom of the same triangle mesh. It is not the cooked PhysX SDF.
- **PhysX convexDecomposition** — the stage does not contain the cooked hulls. MJCF uses one convex hull of the source mesh.
- **missing or degenerate collider** — a rigid body with nothing to collide, or an extent below 0.01 mm.

## Shape and joint mapping

| USD | MJCF |
| --- | --- |
| Sphere, uniform scale | `sphere` |
| Sphere, non-uniform scale | `ellipsoid` |
| Cube | `box` (half the edge, times scale) |
| Capsule or cylinder, uniform radius | analytic `capsule` or `cylinder` |
| Capsule or cylinder, stretched radius | generated mesh, then the convex hull |
| Cone | generated convex mesh (MuJoCo has no cone) |
| Plane | infinite `plane`; the USD axis becomes +Z |
| `convexHull` mesh | `mesh` (MuJoCo convexifies it) |
| `none` or PhysX `sdf` | MuJoCo `sdf` of the same triangles |
| `convexDecomposition`, `meshSimplification` | one convex hull, with a warning |
| `boundingCube`, `boundingSphere` | generated box or containing sphere |
| Fixed joint | nested body, no joint |
| Revolute, prismatic | `hinge`, `slide` |
| Spherical | `ball`; the smaller cone angle, from 0 |
| Distance | spatial tendon between sites, not a parent link |
| Generic joint | locked axes omitted; one free axis is a hinge or slide; three free rotations are a ball; mixed axes are stacked joints |
| Filtered pairs, or `collisionEnabled = false` | `contact/exclude` |

Mass, center of mass, and principal inertia are carried over when the authored values are finite. A missing center of mass or a zero principal-axes quaternion is left at the body origin rather than written as NaN. Revolute and spherical limits and drive targets are converted from degrees to radians. Scene gravity is used only when its magnitude is authored.

MJCF and URDF use the same scene. MJCF angles are read in the compiler's unit and stored as radians. URDF revolute limits are degrees. A URDF ball joint is three revolute joints, a plane is a thin box, and a distance tendon is omitted, each with a warning. GLB stores the render meshes; the physics fields live in `manifest.json`.
