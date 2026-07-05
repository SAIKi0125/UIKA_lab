# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visualize Extreme Parkour waypoint markers in Isaac Sim."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Visualize Extreme Parkour waypoint markers.")
parser.add_argument("--task", type=str, default="UIKA-Parkour-Velocity-Play", help="Name of the parkour task.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to simulate.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument(
    "--surface_offset",
    type=float,
    default=None,
    help="Override waypoint marker surface offset in meters.",
)
parser.add_argument("--steps", type=int, default=None, help="Run this many sim steps, then exit.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import himloco_lab.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def _refresh_waypoint_markers(env):
    command = env.unwrapped.command_manager._terms["base_velocity"]
    translations, marker_indices = command._debug_waypoint_markers()
    command.waypoint_visualizer.visualize(translations=translations, marker_indices=marker_indices)
    return translations, marker_indices


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.commands.base_velocity.debug_vis = True
    if args_cli.surface_offset is not None:
        env_cfg.commands.base_velocity.waypoint_marker_surface_offset = args_cli.surface_offset

    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    command = env.unwrapped.command_manager._terms["base_velocity"]
    command._set_debug_vis_impl(True)
    translations, marker_indices = _refresh_waypoint_markers(env)
    print(
        "[INFO] Waypoint markers: "
        f"num_envs={env.unwrapped.num_envs}, "
        f"markers={translations.shape[0]}, "
        f"highlighted={int((marker_indices == 1).sum().item())}, "
        f"z_min={float(translations[:, 2].min()):.3f}, "
        f"z_max={float(translations[:, 2].max()):.3f}",
        flush=True,
    )

    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    step_count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            env.step(actions)
            _refresh_waypoint_markers(env)
        step_count += 1
        if args_cli.steps is not None and step_count >= args_cli.steps:
            break

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
