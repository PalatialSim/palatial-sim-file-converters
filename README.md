# isaac-usd-to-mjcf

Convert an Isaac Sim USD asset to a MuJoCo MJCF bundle, and report collider problems before they become a wrong contact shape.

Isaac stores many colliders as a unit cube or a convex fragment. The size and orientation are on `xformOp:scale` and `xformOp:orient`, including scales inherited from a parent `Colliders` prim. This package bakes that full transform into the MJCF geom. It also reports when those colliders do not match the visual mesh.

```sh
pip install -e .
isaac-usd-to-mjcf check asset.usd
isaac-usd-to-mjcf convert asset.usd -o out/ --compile
```

`convert` writes `model.xml`, `meshes/*.obj`, and `collider_report.json`.

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
