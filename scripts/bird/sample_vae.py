"""Load a trained VAE and visualize prior-sampled trajectories.

By default loads the most recent training run under /home/esd/LOGS/vae/, samples
N trajectories from N(0, I), de-normalizes, and writes one PNG per sample to
/home/esd/LOGS/vae_samples/ (same 4-panel layout as data/bird preview/).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pythae.models import AutoModel


N_CHANNELS = 6  # must match train_vae.py


def integrate_pos_from_vel(vel: np.ndarray, dur_s: float) -> np.ndarray:
    """vel: (T, 3) NED velocities. Returns (T, 3) with pos[-1] = 0."""
    T = vel.shape[0]
    dt = max(dur_s, 1e-3) / max(T - 1, 1)
    cum = np.cumsum(0.5 * (vel[:-1] + vel[1:]) * dt, axis=0)
    pos = np.concatenate([np.zeros((1, 3), dtype=vel.dtype), cum.astype(vel.dtype)], axis=0)
    pos = pos - pos[-1]
    return pos


def find_latest_run(out_dir: str) -> str:
    runs = sorted(glob.glob(os.path.join(out_dir, "VAE_training_*", "final_model")))
    if not runs:
        sys.exit(f"no trained model under {out_dir}/VAE_training_*/final_model")
    return runs[-1]


def plot_segment(out_path: str, pos: np.ndarray, vel: np.ndarray, wind: np.ndarray, dur_s: float, title: str):
    """Same 4-panel layout as augment_descents.py.

    NED convention: PD positive down, so we plot alt = -PD.
    Time axis is reconstructed from the per-sample duration (uniform spacing).
    """
    n = pos.shape[0]
    t_s = np.linspace(0, max(dur_s, 1e-3), n)
    pn, pe, pd = pos[:, 0], pos[:, 1], pos[:, 2]
    alt = -pd
    horiz_speed = np.linalg.norm(vel[:, :2], axis=1)
    vert_speed = -vel[:, 2]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    ax = axes[0, 0]
    ax.plot(pe, pn, lw=1.2)
    ax.scatter([pe[0]], [pn[0]], c="g", s=30, label="start")
    ax.scatter([pe[-1]], [pn[-1]], c="r", s=30, label="end")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title("top view")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(np.sqrt((pn - pn[0]) ** 2 + (pe - pe[0]) ** 2), alt, lw=1.2)
    ax.set_xlabel("horizontal dist (m)")
    ax.set_ylabel("altitude (m)")
    ax.set_title("side view")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(t_s, alt)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("altitude (m)")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(t_s, horiz_speed, label="|v_horiz|")
    ax.plot(t_s, vert_speed, label="v_up")
    if wind.size > 0:
        ax.plot(t_s, np.linalg.norm(wind[:, :2], axis=1), "--", alpha=0.6, label="|wind_horiz|")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("speed (m/s)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=80)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=str, default=None,
                    help="path to .../final_model/. defaults to the latest run under --vae-root")
    ap.add_argument("--vae-root", type=str, default="/home/esd/LOGS/vae")
    ap.add_argument("--norm-stats", type=str, default="/home/esd/LOGS/vae/norm_stats.json")
    ap.add_argument("--out-dir", type=str, default="/home/esd/LOGS/vae_samples")
    ap.add_argument("--n", type=int, default=12, help="number of trajectories to sample")
    ap.add_argument("--T", type=int, default=128)
    ap.add_argument("--seed", type=int, default=None,
                    help="if set, fix the latent sampling seed for reproducibility")
    args = ap.parse_args()

    run_dir = args.run_dir or find_latest_run(args.vae_root)
    print(f"loading model: {run_dir}")
    model = AutoModel.load_from_folder(run_dir).eval()
    device = next(model.parameters()).device
    print(f"  latent_dim={model.latent_dim}  input_dim={model.input_dim}  device={device}")

    with open(args.norm_stats) as f:
        stats = json.load(f)
    cm = np.asarray(stats["channel_mean"], dtype=np.float32)
    cs = np.asarray(stats["channel_std"], dtype=np.float32)
    dm = float(stats["dur_mean"])
    ds = float(stats["dur_std"])

    if args.seed is not None:
        torch.manual_seed(args.seed)
    with torch.no_grad():
        z = torch.randn(args.n, model.latent_dim, device=device)
        x = model.decoder(z).reconstruction.cpu().numpy()

    expected = args.T * N_CHANNELS + 1
    if x.shape[1] != expected:
        sys.exit(f"output dim {x.shape[1]} != T*{N_CHANNELS} + 1 = {expected}; check --T")

    traj_norm = x[:, : args.T * N_CHANNELS].reshape(args.n, args.T, N_CHANNELS)
    dur_norm = x[:, -1]
    traj = traj_norm * cs + cm  # vel(3) + wind(3)
    dur = dur_norm * ds + dm

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"writing {args.n} samples to {args.out_dir}/")
    for i in range(args.n):
        vel = traj[i, :, 0:3]
        wind = traj[i, :, 3:6]
        pos = integrate_pos_from_vel(vel, float(dur[i]))
        out_path = os.path.join(args.out_dir, f"sample_{i:03d}.png")
        plot_segment(
            out_path,
            pos,
            vel,
            wind,
            float(dur[i]),
            f"VAE sample #{i}  dur={float(dur[i]):.1f}s  "
            f"drop={float(pos[:,2].max()-pos[:,2].min()):.1f}m  "
            f"end=({pos[-1,0]:+.1f},{pos[-1,1]:+.1f},{pos[-1,2]:+.1f})",
        )

    # also write one combined overlay
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i in range(args.n):
        vel = traj[i, :, 0:3]
        pos = integrate_pos_from_vel(vel, float(dur[i]))
        pn, pe, pd = pos[:, 0], pos[:, 1], pos[:, 2]
        alt = -pd
        horiz_speed = np.linalg.norm(vel[:, :2], axis=1)
        t_s = np.linspace(0, max(float(dur[i]), 1e-3), pos.shape[0])
        axes[0].plot(pe, pn, lw=1.0, alpha=0.7)
        axes[1].plot(np.sqrt((pn - pn[0]) ** 2 + (pe - pe[0]) ** 2), alt, lw=1.0, alpha=0.7)
        axes[2].plot(t_s, horiz_speed, lw=1.0, alpha=0.7)
    axes[0].set_title("top view (E-N)")
    axes[0].set_xlabel("East (m)")
    axes[0].set_ylabel("North (m)")
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[1].set_title("side view")
    axes[1].set_xlabel("horizontal dist (m)")
    axes[1].set_ylabel("altitude (m)")
    axes[2].set_title("horizontal speed")
    axes[2].set_xlabel("t (s)")
    axes[2].set_ylabel("|v_horiz| (m/s)")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    overlay_path = os.path.join(args.out_dir, "_overlay.png")
    fig.savefig(overlay_path, dpi=80)
    plt.close(fig)
    print(f"overlay -> {overlay_path}")

    # endpoint is exactly 0 by construction (we shift integrated pos so pos[-1] = 0).
    # Useful diagnostic: how big is the "natural" drift before we zero it?
    natural_ends = []
    for i in range(args.n):
        v = traj[i, :, 0:3]
        T_ = v.shape[0]
        dt = max(float(dur[i]), 1e-3) / max(T_ - 1, 1)
        end = np.cumsum(0.5 * (v[:-1] + v[1:]) * dt, axis=0)[-1]
        natural_ends.append(np.linalg.norm(end))
    natural_ends = np.asarray(natural_ends)
    print(
        f"\nintegrated end-of-trajectory drift before centering: "
        f"mean={natural_ends.mean():.2f}m  max={natural_ends.max():.2f}m  "
        f"(then shifted so pos[-1] = 0 exactly)"
    )


if __name__ == "__main__":
    main()
