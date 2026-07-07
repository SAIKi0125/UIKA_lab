# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint of an RL agent trained with HimLoco RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaacsim_compat import configure_isaacsim_pip_extensions

configure_isaacsim_pip_extensions()

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint with the takeoff RSL-RL agent.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during playback.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument(
    "--env_cfg_entry_point", type=str, default="play_env_cfg_entry_point", help="Name of the play environment configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append HimLoco RSL-RL cli arguments
cli_args.add_himloco_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import himloco_lab.tasks  # noqa: F401
from himloco_lab.rsl_rl import HIMOnPolicyRunner, HimlocoVecEnvWrapper
from himloco_lab.rsl_rl.config import HIMOnPolicyRunnerCfg
from himloco_lab.utils import export_himloco_policy_as_jit, export_himloco_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from rsl_rl.runners import OnPolicyRunner

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _strip_deprecated_standard_rsl_rl_keys(runner_cfg: dict) -> dict:
    for model_key in ("actor", "critic"):
        model_cfg = runner_cfg.get(model_key)
        if model_cfg is None:
            continue
        for deprecated_key in ("stochastic", "init_noise_std", "noise_std_type", "state_dependent_std"):
            model_cfg.pop(deprecated_key, None)
    return runner_cfg


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: HIMOnPolicyRunnerCfg | RslRlOnPolicyRunnerCfg):
    """Play with HimLoco RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_himloco_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg.device = env_cfg.sim.device

    use_himloco_runner = isinstance(agent_cfg, HIMOnPolicyRunnerCfg)

    # specify directory for logging experiments
    if use_himloco_runner:
        log_root_path = os.path.join("logs", "himloco_rsl_rl", agent_cfg.experiment_name)
    else:
        log_root_path = os.path.join("logs/rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    
    # get checkpoint path
    if args_cli.checkpoint and (os.path.isabs(args_cli.checkpoint) or os.path.exists(args_cli.checkpoint)):
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during playback.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    if use_himloco_runner:
        # wrap around environment for HimLoco RSL-RL
        env = HimlocoVecEnvWrapper(
            env,
            history_length=agent_cfg.history_length,
            privileged_history_length=agent_cfg.privileged_history_length,
        )

        print(f"[INFO] Environment wrapped successfully")
        print(f"[INFO] num_envs: {env.num_envs}")
        print(f"[INFO] num_one_step_obs: {env.num_one_step_obs}")
        print(f"[INFO] history_length: {env.history_length}")
        print(f"[INFO] num_obs (total): {env.num_obs}")
        if env.num_one_step_privileged_obs is not None:
            print(f"[INFO] num_one_step_privileged_obs: {env.num_one_step_privileged_obs}")
            print(f"[INFO] privileged_history_length: {env.privileged_history_length}")
            print(f"[INFO] num_privileged_obs (total): {env.num_privileged_obs}")
        print(f"[INFO] num_actions: {env.num_actions}")
    else:
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        print(f"[INFO] Standard RSL-RL environment wrapped successfully")
        print(f"[INFO] num_envs: {env.num_envs}")

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    
    # create runner
    if use_himloco_runner:
        runner = HIMOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        runner_cfg = _strip_deprecated_standard_rsl_rl_keys(agent_cfg.to_dict())
        runner = OnPolicyRunner(env, runner_cfg, log_dir=None, device=agent_cfg.device)
    
    # load the checkpoint
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.device if use_himloco_runner else env.unwrapped.device)

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    if use_himloco_runner:
        # export HimLoco dual network (encoder + policy) as TorchScript JIT and ONNX
        print(f"[INFO] Exporting HimLoco dual network to: {export_model_dir}")
        export_himloco_policy_as_jit(
            runner.alg.actor_critic,
            path=export_model_dir,
            policy_filename="policy.pt",
        )
        export_himloco_policy_as_onnx(
            runner.alg.actor_critic,
            path=export_model_dir,
            encoder_filename="encoder.onnx",
            policy_filename="policy.onnx",
            verbose=False,
            skip_if_unavailable=True,
        )
    else:
        print(f"[INFO] Exporting standard RSL-RL policy to: {export_model_dir}")
        runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
        runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0
    
    # simulate environment
    print("[INFO] Starting playback...")
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            if use_himloco_runner:
                obs, privileged_obs, rewards, dones, infos, termination_ids, termination_privileged_obs = env.step(actions)
            else:
                obs, rewards, dones, infos = env.step(actions)
        
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
