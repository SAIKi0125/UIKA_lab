"""Print per-term observations for an Isaac Lab task."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Print observation terms for a HimLoco/Isaac Lab task.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default="UIKA-Velocity")
parser.add_argument("--env_idx", type=int, default=0)
parser.add_argument("--steps", type=int, default=1)
parser.add_argument("--precision", type=int, default=4)
parser.add_argument("--max_items", type=int, default=256)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg

import himloco_lab.tasks  # noqa: F401


def _format_tensor(values: torch.Tensor, max_items: int, precision: int) -> str:
    flat = values.flatten().detach().cpu()
    if flat.numel() > max_items:
        shown = flat[:max_items]
        suffix = f" ... ({flat.numel()} total)"
    else:
        shown = flat
        suffix = ""
    items = ", ".join(f"{x:.{precision}f}" for x in shown.tolist())
    return f"[{items}]{suffix}"


def _print_group(env, group_name: str, env_idx: int, max_items: int, precision: int):
    obs_manager = env.unwrapped.observation_manager
    terms = obs_manager.compute_group(group_name, update_history=False)
    term_names = obs_manager.active_terms[group_name]

    print(f"\n[{group_name}]")
    if isinstance(terms, dict):
        for term_name in term_names:
            term = terms[term_name]
            print(
                f"{term_name:28s} shape={tuple(term.shape[1:])!s:12s} "
                f"env[{env_idx}]={_format_tensor(term[env_idx], max_items, precision)}"
            )
        return

    start = 0
    term_dims = obs_manager.group_obs_term_dim[group_name]
    for term_name, term_dim in zip(term_names, term_dims):
        width = int(torch.tensor(term_dim).prod().item())
        term = terms[:, start : start + width]
        start += width
        print(
            f"{term_name:28s} shape={tuple(term_dim)!s:12s} "
            f"env[{env_idx}]={_format_tensor(term[env_idx], max_items, precision)}"
        )

    print(f"{'TOTAL':28s} shape={tuple(terms.shape[1:])!s:12s} env[{env_idx}]={_format_tensor(terms[env_idx], max_items, precision)}")


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    if not 0 <= args_cli.env_idx < env.unwrapped.num_envs:
        raise ValueError(f"--env_idx must be in [0, {env.unwrapped.num_envs - 1}]")

    with torch.inference_mode():
        for _ in range(max(args_cli.steps - 1, 0)):
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            env.step(actions)

        obs_manager = env.unwrapped.observation_manager
        print(f"task={args_cli.task}")
        print(f"num_envs={env.unwrapped.num_envs}, env_idx={args_cli.env_idx}, device={env.unwrapped.device}")
        print(f"active_terms={obs_manager.active_terms}")
        print(f"group_obs_dim={obs_manager.group_obs_dim}")
        print(f"group_obs_term_dim={obs_manager.group_obs_term_dim}")

        for group_name in obs_manager.active_terms:
            _print_group(env, group_name, args_cli.env_idx, args_cli.max_items, args_cli.precision)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
