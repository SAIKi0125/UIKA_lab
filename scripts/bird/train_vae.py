"""Train a VAE on bird-descent trajectories using the pythae library.

We supply only the (a) MLP encoder / decoder adapted to (T, 9) trajectories and
(b) a small Dataset wrapper. pythae handles the VAE loss (ELBO), KL annealing,
checkpointing, and samples from the prior.

Outputs go to /home/esd/LOGS/vae/:
  - pythae's own training_<timestamp>/ directory with checkpoints + logs
  - norm_stats.json (so we can de-normalize at inference time)
  - samples_vs_train.png (sanity check: prior samples overlaid on training data)
  - loss_curve.png
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, random_split

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pythae.data.datasets import DatasetOutput
from pythae.models import VAE, VAEConfig
from pythae.models.base.base_utils import ModelOutput
from pythae.models.nn import BaseDecoder, BaseEncoder
from pythae.pipelines import TrainingPipeline
from pythae.trainers import BaseTrainerConfig


# -----------------------------------------------------------------------------
# data pipeline (resample variable-length segments to fixed T, normalize)
# -----------------------------------------------------------------------------


def _resample_to_T(arr: np.ndarray, T: int) -> np.ndarray:
    n = arr.shape[0]
    if n == T:
        return arr.astype(np.float32)
    src = np.linspace(0.0, 1.0, n, dtype=np.float32)
    dst = np.linspace(0.0, 1.0, T, dtype=np.float32)
    out = np.empty((T, arr.shape[1]), dtype=np.float32)
    for k in range(arr.shape[1]):
        out[:, k] = np.interp(dst, src, arr[:, k])
    return out


# channel layout: [vel(3) + wind(3)] = 6
# position is reconstructed at inference time by integrating velocity and
# subtracting the endpoint, which guarantees pos[-1] = 0 (touchdown at origin)
N_CHANNELS = 6


def load_npz(npz_path: str, T: int):
    d = np.load(npz_path, allow_pickle=True)
    vel_obj, wind_obj, t_obj = d["vel_ned"], d["wind_ned"], d["t_us"]
    n = len(vel_obj)
    trajs = np.empty((n, T, N_CHANNELS), dtype=np.float32)
    durations = np.empty((n,), dtype=np.float32)
    for i in range(n):
        joined = np.concatenate(
            [vel_obj[i].astype(np.float32), wind_obj[i].astype(np.float32)],
            axis=1,
        )
        trajs[i] = _resample_to_T(joined, T)
        durations[i] = float((t_obj[i][-1] - t_obj[i][0]) * 1e-6)
    return trajs, durations


def compute_stats(trajs: np.ndarray, durations: np.ndarray) -> dict:
    flat = trajs.reshape(-1, N_CHANNELS)
    means = flat.mean(axis=0)
    stds = flat.std(axis=0)
    stds = np.where(stds < 1e-3, 1.0, stds)
    return {
        "channel_mean": means.tolist(),
        "channel_std": stds.tolist(),
        "dur_mean": float(durations.mean()),
        "dur_std": float(max(durations.std(), 1e-3)),
        "n_channels": N_CHANNELS,
        "channel_layout": "vel(3)+wind(3); pos integrated at inference",
    }


def normalize(trajs, durations, stats):
    cm = np.asarray(stats["channel_mean"], dtype=np.float32)
    cs = np.asarray(stats["channel_std"], dtype=np.float32)
    ntraj = (trajs - cm) / cs
    ndur = (durations - stats["dur_mean"]) / stats["dur_std"]
    return ntraj, ndur


def denormalize(ntraj, ndur, stats):
    cm = np.asarray(stats["channel_mean"], dtype=np.float32)
    cs = np.asarray(stats["channel_std"], dtype=np.float32)
    traj = ntraj * cs + cm
    dur = ndur * stats["dur_std"] + stats["dur_mean"]
    return traj, dur


def integrate_pos_from_vel(vel: np.ndarray, dur_s: float) -> np.ndarray:
    """vel: (T, 3) NED velocities. Returns (T, 3) position with pos[-1] = 0.

    Trapezoidal integration along uniform time steps, then shift so the last
    sample lands at the origin (i.e. touchdown at ground).
    """
    T = vel.shape[0]
    dt = max(dur_s, 1e-3) / max(T - 1, 1)
    cum = np.cumsum(0.5 * (vel[:-1] + vel[1:]) * dt, axis=0)
    pos = np.concatenate([np.zeros((1, 3), dtype=vel.dtype), cum.astype(vel.dtype)], axis=0)
    pos = pos - pos[-1]
    return pos


class BirdDataset(Dataset):
    """Returns DatasetOutput so it works with pythae's trainer.

    pythae expects a flat tensor under key 'data'. We pack (T*9 + 1,)
    where the last entry is the duration scalar.
    """

    def __init__(self, ntraj, ndur):
        x_traj = torch.from_numpy(ntraj.astype(np.float32)).reshape(ntraj.shape[0], -1)
        x_dur = torch.from_numpy(ndur.astype(np.float32)).unsqueeze(-1)
        self.x = torch.cat([x_traj, x_dur], dim=-1)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return DatasetOutput(data=self.x[idx])


# -----------------------------------------------------------------------------
# MLP encoder / decoder (subclass pythae's BaseEncoder / BaseDecoder)
# -----------------------------------------------------------------------------


def _mlp(in_dim: int, hidden: list[int], out_dim: int) -> nn.Sequential:
    dims = [in_dim] + hidden + [out_dim]
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)


class MLPEncoder(BaseEncoder):
    def __init__(self, x_dim: int, hidden: list[int], latent_dim: int):
        BaseEncoder.__init__(self)
        self.trunk = _mlp(x_dim, hidden, hidden[-1])
        self.act = nn.SiLU()
        self.fc_mu = nn.Linear(hidden[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden[-1], latent_dim)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        h = self.act(self.trunk(x))
        return ModelOutput(embedding=self.fc_mu(h), log_covariance=self.fc_logvar(h))


class MLPDecoder(BaseDecoder):
    def __init__(self, latent_dim: int, hidden: list[int], x_dim: int):
        BaseDecoder.__init__(self)
        self.net = _mlp(latent_dim, hidden[::-1], x_dim)

    def forward(self, z: torch.Tensor) -> ModelOutput:
        return ModelOutput(reconstruction=self.net(z))


# -----------------------------------------------------------------------------
# main training entry
# -----------------------------------------------------------------------------


def find_latest_run(out_dir: str) -> str | None:
    runs = sorted(glob.glob(os.path.join(out_dir, "VAE_training_*")))
    return runs[-1] if runs else None


def find_best_checkpoint(run_dir: str) -> str | None:
    candidates = sorted(glob.glob(os.path.join(run_dir, "**", "model.pt"), recursive=True))
    # prefer "final_model" over "checkpoint_*"
    final = [c for c in candidates if "final_model" in c]
    return final[-1] if final else (candidates[-1] if candidates else None)


def plot_curves(traj_np, dur_np, train_traj_np, train_dur_np, out_path: str, n_plot: int):
    """traj_np / train_traj_np: (N, T, 6) = vel(3) + wind(3) in real units.
    Position is integrated from velocity for plotting.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i in range(min(n_plot, traj_np.shape[0])):
        vel = traj_np[i, :, :3]
        pos = integrate_pos_from_vel(vel, float(dur_np[i]))
        pn, pe, pd = pos[:, 0], pos[:, 1], pos[:, 2]
        alt = -pd
        horiz_speed = np.linalg.norm(vel[:, :2], axis=1)
        t_s = np.linspace(0, max(0.1, float(dur_np[i])), pos.shape[0])
        axes[0].plot(pe, pn, color="C0", lw=1.0, alpha=0.7, label="vae" if i == 0 else None)
        horiz = np.sqrt((pn - pn[0]) ** 2 + (pe - pe[0]) ** 2)
        axes[1].plot(horiz, alt, color="C0", lw=1.0, alpha=0.7)
        axes[2].plot(t_s, horiz_speed, color="C0", lw=1.0, alpha=0.7)

    n_train = min(8, train_traj_np.shape[0])
    for i in range(n_train):
        vel = train_traj_np[i, :, :3]
        pos = integrate_pos_from_vel(vel, float(train_dur_np[i]))
        pn, pe, pd = pos[:, 0], pos[:, 1], pos[:, 2]
        alt = -pd
        horiz_speed = np.linalg.norm(vel[:, :2], axis=1)
        t_s = np.linspace(0, max(0.1, float(train_dur_np[i])), pos.shape[0])
        axes[0].plot(pe, pn, color="C3", lw=1.0, alpha=0.7, label="train" if i == 0 else None)
        horiz = np.sqrt((pn - pn[0]) ** 2 + (pe - pe[0]) ** 2)
        axes[1].plot(horiz, alt, color="C3", lw=1.0, alpha=0.7)
        axes[2].plot(t_s, horiz_speed, color="C3", lw=1.0, alpha=0.7)

    axes[0].set_title("top view (E-N)")
    axes[0].set_xlabel("East (m)")
    axes[0].set_ylabel("North (m)")
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[0].legend(fontsize=8)
    axes[1].set_title("side view")
    axes[1].set_xlabel("horizontal dist (m)")
    axes[1].set_ylabel("altitude (m)")
    axes[2].set_title("horizontal speed")
    axes[2].set_xlabel("t (s)")
    axes[2].set_ylabel("|v_horiz| (m/s)")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=80)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="input", type=str, default="/home/esd/LOGS/augmented_segments.npz")
    ap.add_argument("--out-dir", type=str, default="/home/esd/LOGS/vae")
    ap.add_argument("--T", type=int, default=128)
    ap.add_argument("--latent-dim", type=int, default=16)
    ap.add_argument("--hidden", type=int, nargs="+", default=[512, 256])
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-samples-preview", type=int, default=12)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"loading {args.input}  T={args.T}  channels={N_CHANNELS} (vel+wind)")
    trajs, durations = load_npz(args.input, args.T)
    print(f"  {trajs.shape[0]} samples  shape={trajs.shape}  dur=[{durations.min():.1f},{durations.max():.1f}]s")

    stats = compute_stats(trajs, durations)
    with open(os.path.join(args.out_dir, "norm_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    ntraj, ndur = normalize(trajs, durations, stats)

    full = BirdDataset(ntraj, ndur)
    n_val = max(1, int(len(full) * args.val_frac))
    n_train = len(full) - n_val
    train_ds, val_ds = random_split(full, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))

    x_dim = args.T * N_CHANNELS + 1
    encoder = MLPEncoder(x_dim, list(args.hidden), args.latent_dim)
    decoder = MLPDecoder(args.latent_dim, list(args.hidden), x_dim)

    model_cfg = VAEConfig(
        input_dim=(x_dim,),
        latent_dim=args.latent_dim,
        reconstruction_loss="mse",
    )
    model = VAE(model_cfg, encoder=encoder, decoder=decoder)
    print(f"model params: {sum(p.numel() for p in model.parameters())}")

    trainer_cfg = BaseTrainerConfig(
        output_dir=args.out_dir,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        seed=args.seed,
        steps_saving=None,  # only save final
        keep_best_on_train=False,
    )

    pipeline = TrainingPipeline(model=model, training_config=trainer_cfg)
    pipeline(train_data=train_ds, eval_data=val_ds)

    # locate the run pythae just created
    run_dir = find_latest_run(args.out_dir)
    print(f"\npythae run dir: {run_dir}")

    # generate samples from the prior
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        z = torch.randn(args.n_samples_preview, args.latent_dim, device=device)
        x_recon = model.decoder(z).reconstruction.cpu().numpy()

    sampled_traj_flat = x_recon[:, : args.T * N_CHANNELS]
    sampled_dur = x_recon[:, -1]
    sampled_traj = sampled_traj_flat.reshape(args.n_samples_preview, args.T, N_CHANNELS)
    sampled_traj, sampled_dur = denormalize(sampled_traj, sampled_dur, stats)

    train_traj_dn, train_dur_dn = denormalize(ntraj, ndur, stats)
    plot_curves(
        sampled_traj,
        sampled_dur,
        train_traj_dn,
        train_dur_dn,
        os.path.join(args.out_dir, "samples_vs_train.png"),
        n_plot=args.n_samples_preview,
    )
    print(f"preview -> {os.path.join(args.out_dir, 'samples_vs_train.png')}")
    print(f"norm_stats -> {os.path.join(args.out_dir, 'norm_stats.json')}")


if __name__ == "__main__":
    main()
