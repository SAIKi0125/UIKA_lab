# UIKAC Compatible Asset Preview Design

## Goal

Prepare a corrected UIKAC asset candidate for inspection without modifying the current training asset. The candidate adopts the new UIKAC geometry, joint origins, mass, and inertia while preserving the joint naming and control semantics expected by the existing UIKA environments and policies.

## Candidate Location

Create the preview under `/tmp/uikac_compatible_preview`. The current files under `source/himloco_lab/himloco_lab/assets/uika` remain unchanged until the user approves the candidate.

## Compatibility Transform

1. Copy `UIKAC.urdf` and all referenced meshes into the preview directory.
2. Rename back-leg links, joints, parents, children, and mesh references from `BL/BR` to `RL/RR`.
3. Reverse only the seven joint axes whose direction is opposite to the current UIKA convention: `FL_thigh`, `FR_hip`, `FR_thigh`, `FR_calf`, `RL_hip`, `RL_thigh`, and `RR_calf`.
4. Preserve the new axis inclination after sign correction rather than replacing it with the old geometry's axis vector.
5. Restore the current UIKA position, effort, and velocity limits for all twelve revolute joints.
6. Convert mesh references to paths local to the preview package so Isaac Sim can import it independently.

## Dynamics

The mass, center of mass, and inertia tensors in `UIKAC.urdf` are authoritative and remain unchanged. The SolidWorks CSV is used only as a reference because its inertia values correspond to a different set of CAD link masses.

Validation checks:

- positive link masses;
- positive-definite inertia tensors;
- principal-moment triangle inequalities;
- center of mass inside each link mesh bounding box;
- radius of gyration compatible with mesh dimensions;
- finite total mass and left/right symmetry where expected.

## Verification

- XML and ROS `check_urdf` parsing;
- automated comparison against current joint names, joint direction conventions, and limits;
- Isaac Sim 5.1 URDF-to-USD conversion with all meshes resolved;
- forward-kinematics comparison for normal and Lower target postures;
- no tracked training asset changes before explicit user approval.

## Approval Boundary

The preview URDF, meshes, validation report, and generated USD are delivered for inspection. Only after explicit approval will the candidate replace the current repository asset and proceed through environment smoke tests.
