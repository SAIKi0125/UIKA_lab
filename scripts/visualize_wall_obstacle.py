# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Load a 1000mm wide, 50mm long, 300mm high wall obstacle in Isaac Sim."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Visualize the parkour wall obstacle.")
parser.add_argument(
    "--steps",
    type=int,
    default=None,
    help="Run this many sim steps, then exit. Use 0 to only load the scene and exit.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext

from himloco_lab.terrains.wall_obstacle import WALL_CENTER_POS_M, WALL_SIZE_M


def design_scene():
    """Spawn a ground plane and one static wall obstacle."""
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/GroundPlane", ground_cfg)

    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.9, 0.9, 0.9))
    light_cfg.func("/World/Light", light_cfg)

    wall_cfg = sim_utils.CuboidCfg(size=WALL_SIZE_M,
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.9, dynamic_friction=0.8),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.78, 0.18, 0.12), roughness=0.65),
    )
    wall_cfg.func("/World/WallObstacle", wall_cfg, translation=WALL_CENTER_POS_M)


def main():
    """Run the visualization app."""
    sim_cfg = SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=(1.3, -1.8, 0.85), target=(0.0, 0.0, 0.16))

    design_scene()
    sim.reset()

    print(
        "[INFO] Wall obstacle loaded: "
        "1000mm width, 50mm length, 300mm height; "
        f"size_m={WALL_SIZE_M}, center_m={WALL_CENTER_POS_M}",
        flush=True,
    )

    if args_cli.steps == 0:
        os._exit(0)

    step_count = 0
    while simulation_app.is_running():
        sim.step()
        step_count += 1
        if args_cli.steps is not None and step_count >= args_cli.steps:
            break

    sim.stop()


if __name__ == "__main__":
    main()
    simulation_app.close(wait_for_replicator=False)
