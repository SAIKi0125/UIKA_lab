"""Show UIKA in the current lower crouch pose with a fixed base."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent / "himloco_rsl_rl"))
from isaacsim_compat import configure_isaacsim_pip_extensions  # noqa: E402

configure_isaacsim_pip_extensions()

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Show UIKA lower crouch pose with fixed base suspended in air.")
parser.add_argument("--base_height", type=float, default=0.5, help="Fixed base height in meters.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of UIKA instances to spawn.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass

from himloco_lab.assets.uika import UIKA_CFG as ROBOT_CFG
from himloco_lab.tasks.locomotion.robots.uika.velocity_env_cfg import UIKA_LOWER_JOINT_POS_TARGET


@configclass
class UIKACrouchSceneCfg(InteractiveSceneCfg):
    """Scene with a fixed-base UIKA held in the lower crouch pose."""

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )
    robot: ArticulationCfg = ROBOT_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=ROBOT_CFG.spawn.replace(fix_base=True),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, args_cli.base_height),
            joint_pos=UIKA_LOWER_JOINT_POS_TARGET,
            joint_vel={".*": 0.0},
        ),
    )


def _lower_pose_tensors(robot, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = torch.zeros_like(robot.data.default_joint_vel)
    for joint_name, target in UIKA_LOWER_JOINT_POS_TARGET.items():
        joint_pos[:, robot.joint_names.index(joint_name)] = target
    return joint_pos.to(device), joint_vel.to(device)


def run_simulator(sim: SimulationContext, scene: InteractiveScene):
    robot = scene["robot"]
    sim_dt = sim.get_physics_dt()
    joint_pos, joint_vel = _lower_pose_tensors(robot, sim.device)

    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.set_joint_position_target(joint_pos)
    scene.write_data_to_sim()
    scene.reset()

    print("[INFO] UIKA lower crouch pose loaded.")
    print(f"[INFO] Fixed base height: {args_cli.base_height:.3f} m")
    print("[INFO] Close the Isaac Sim window or press Ctrl+C to exit.")

    while simulation_app.is_running():
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        robot.set_joint_position_target(joint_pos)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([1.6, -2.2, 1.2], [0.0, 0.0, args_cli.base_height])

    scene_cfg = UIKACrouchSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    run_simulator(sim, scene)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
