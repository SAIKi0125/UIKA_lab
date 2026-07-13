# UIKA Primitive Collision Model Design

## Goal

Generate a primitive-collision UIKA URDF from the current master robot geometry and use it in every Isaac Lab UIKA environment. Preserve the current visual meshes, inertial properties, joints, limits, and link frames while replacing mesh collision geometry with boxes and spheres.

The MuJoCo model under `/home/saiki/project/deploy_UIKA` is outside this change.

## Source and Generated Assets

Keep `assets/uika/urdf/uika.urdf` as the authoritative mesh-collision source model. Add:

- `assets/uika/urdf/uika_simple_collision.urdf`, the generated primitive-collision model.
- `scripts/generate_uika_simple_collision_urdf.py`, adapted from `origin/teacherstudent` to regenerate the output from the current source URDF and current mesh files.

The generated file must retain every non-collision XML element from the current source URDF. It must not copy the older robot geometry, joint origins, axes, or inertial values from the teacher-student branch.

## Collision Generation

Use the teacher-student branch's robust mesh-projection algorithm:

- Load each link mesh without geometry processing.
- Use the 2nd and 98th vertex percentiles for default robust bounds.
- Represent the base and hip links as axis-aligned boxes with a 6 mm size margin.
- Represent thigh and calf links as PCA-oriented boxes.
  - Use the 1st and 99th percentiles on the long axis.
  - Use the 10th and 90th percentiles on both short axes.
  - Add a 6 mm size margin.
- Represent each foot as a sphere centered at the robust-bound center.
  - Radius is half the maximum robust-bound extent plus a 2 mm margin.

For the current master foot meshes, the expected generated values are approximately:

- `FL_foot` and `RL_foot`: center `(-0.012595, -0.004778, 0.005902)` m.
- `FR_foot` and `RR_foot`: center `(-0.012595, 0.004715, 0.006002)` m.
- All feet: radius `0.025275` m.

All visual elements remain STL meshes. Mass and inertia remain those of the current source URDF rather than being recomputed from the primitive collision geometry.

## Runtime Integration

Change `UIKA_CFG.spawn.asset_path` to `uika_simple_collision.urdf`. Because Velocity, Flat, Lower, and Parkour environments all consume the shared `UIKA_CFG`, they will use identical primitive collision geometry.

Isaac Lab hashes the selected URDF contents when converting it to USD. The new asset path and generated file contents therefore trigger conversion without adding `force_usd_conversion` or requiring manual cache deletion.

## Generator Behavior and Errors

The generator reads the source URDF and current meshes from repository-relative paths, updates only each link's collision origin and geometry, and writes deterministic XML.

Generation must fail clearly when the source URDF or an expected link mesh is missing. Unsupported links must not silently retain mesh collisions. The committed output must be reproducible by rerunning the generator in an environment containing NumPy and trimesh.

## Verification

Add focused tests that parse both URDFs and verify:

- `UIKA_CFG` selects `uika_simple_collision.urdf`.
- Every generated link has exactly one primitive collision geometry.
- Base, hip, thigh, and calf links use boxes.
- Foot links use spheres with the expected centers and radius.
- No generated collision geometry references a mesh.
- Every visual geometry remains a mesh.
- Link names, inertial elements, joint elements, visual elements, and other non-collision content match the current source URDF.
- Rerunning the generator produces identical output.

Run focused tests, XML parsing, Python compilation, and diff checks. A simulator rollout is not required for this asset-structure change.
