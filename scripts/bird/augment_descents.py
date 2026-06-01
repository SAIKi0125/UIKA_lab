"""Augment extracted descent segments and write previews.

Reads raw_segments.npz, applies:
  - random yaw rotation (around vertical NED-Down axis), rotates pos+vel+wind together
  - time scaling +/- max_time_warp (resamples via linear interp on the original timeline)
  - small Gaussian noise on position and velocity

Outputs:
  - augmented .npz (object arrays of variable length, same schema as input + an
    'orig_idx' column pointing to which raw segment each augmented sample came from)
  - PNG previews (top view, side view, height + speed curves) for sanity checking
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def yaw_rotate(vec_xyz: np.ndarray, yaw: float) -> np.ndarray:
    """Rotate (..., 3) vectors about the NED-Down (z) axis by `yaw` rad.

    NED convention: z points down. We rotate the horizontal (N, E) plane.
    """
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=vec_xyz.dtype)
    return vec_xyz @ R.T


def time_warp(
    t_us: np.ndarray,
    pos: np.ndarray,
    vel: np.ndarray,
    wind: np.ndarray,
    scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stretch/compress the timeline by `scale` (>1 = slower bird).

    We resample on a new uniformly spaced grid spanning the original duration
    times `scale`, with the same number of samples. Velocity magnitude is
    divided by `scale` to keep position-velocity consistency.
    """
    n = len(t_us)
    t0 = t_us[0]
    dur_us = t_us[-1] - t0
    new_t = (t_us - t0) * scale + t0  # warped timestamps
    # resample everything onto an evenly-spaced grid of n samples on [t0, t0+dur*scale]
    sample_t = np.linspace(t0, t0 + dur_us * scale, n)
    pos_w = np.empty_like(pos)
    vel_w = np.empty_like(vel)
    wind_w = np.empty_like(wind)
    for k in range(3):
        pos_w[:, k] = np.interp(sample_t, new_t, pos[:, k])
        vel_w[:, k] = np.interp(sample_t, new_t, vel[:, k]) / scale
        wind_w[:, k] = np.interp(sample_t, new_t, wind[:, k])
    return sample_t.astype(t_us.dtype), pos_w, vel_w, wind_w


def augment_one(
    t_us: np.ndarray,
    pos: np.ndarray,
    vel: np.ndarray,
    wind: np.ndarray,
    rng: np.random.Generator,
    *,
    max_time_warp: float,
    pos_noise: float,
    vel_noise: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    yaw = rng.uniform(-np.pi, np.pi)
    pos_r = yaw_rotate(pos, yaw)
    vel_r = yaw_rotate(vel, yaw)
    wind_r = yaw_rotate(wind, yaw)

    scale = float(rng.uniform(1.0 - max_time_warp, 1.0 + max_time_warp))
    t_w, pos_w, vel_w, wind_w = time_warp(t_us, pos_r, vel_r, wind_r, scale)

    if pos_noise > 0:
        noise = rng.normal(0, pos_noise, size=pos_w.shape).astype(pos_w.dtype)
        # taper noise to zero at the touchdown so endpoint stays at the ground
        n = pos_w.shape[0]
        taper = np.linspace(1.0, 0.0, n, dtype=pos_w.dtype)[:, None]
        pos_w = pos_w + noise * taper
    if vel_noise > 0:
        vel_w = vel_w + rng.normal(0, vel_noise, size=vel_w.shape).astype(vel_w.dtype)

    # enforce exact ground contact at endpoint (yaw rotation preserves the
    # all-zero point, but time warp + interpolation can drift by ~1e-6)
    pos_w = pos_w - pos_w[-1]

    return t_w, pos_w, vel_w, wind_w


def plot_segment(
    out_path: str,
    pos: np.ndarray,
    vel: np.ndarray,
    wind: np.ndarray,
    t_us: np.ndarray,
    title: str,
):
    """Top view, side view, altitude curve, speed curve. NED -> plot z up."""
    t_s = (t_us - t_us[0]) * 1e-6
    pn, pe, pd = pos[:, 0], pos[:, 1], pos[:, 2]
    alt = -pd  # NED: down -> up
    horiz_speed = np.linalg.norm(vel[:, :2], axis=1)
    vert_speed = -vel[:, 2]  # downward to upward sign

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
    ax.plot(t_s, alt, label="alt")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("altitude (m)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(t_s, horiz_speed, label="|v_horiz|")
    ax.plot(t_s, vert_speed, label="v_up")
    if wind.size > 0:
        ax.plot(
            t_s, np.linalg.norm(wind[:, :2], axis=1), "--", alpha=0.6, label="|wind_horiz|"
        )
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
    ap.add_argument(
        "--in",
        dest="input",
        type=str,
        default="/home/esd/LOGS/raw_segments.npz",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="/home/esd/LOGS/augmented_segments.npz",
    )
    ap.add_argument(
        "--preview-dir",
        type=str,
        default="/home/esd/LOGS/preview",
    )
    ap.add_argument("--aug", type=int, default=20, help="augmented copies per raw segment")
    ap.add_argument("--max-time-warp", type=float, default=0.10)
    ap.add_argument("--pos-noise", type=float, default=0.05)
    ap.add_argument("--vel-noise", type=float, default=0.05)
    ap.add_argument(
        "--n-preview-raw",
        type=int,
        default=10,
        help="how many raw segments to plot",
    )
    ap.add_argument(
        "--n-preview-aug-per-raw",
        type=int,
        default=2,
        help="for each previewed raw, how many augmented variants to plot",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not os.path.isfile(args.input):
        print(f"input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    data = np.load(args.input, allow_pickle=True)
    raw_t = data["t_us"]
    raw_pos = data["pos_ned"]
    raw_vel = data["vel_ned"]
    raw_wind = data["wind_ned"]
    sources = data["sources"]
    n_raw = len(raw_t)
    print(f"loaded {n_raw} raw segments")

    rng = np.random.default_rng(args.seed)
    aug_t, aug_pos, aug_vel, aug_wind, aug_origin = [], [], [], [], []

    os.makedirs(args.preview_dir, exist_ok=True)
    n_preview_raw = min(args.n_preview_raw, n_raw)
    preview_raw_idx = rng.choice(n_raw, size=n_preview_raw, replace=False)
    preview_raw_idx = set(int(x) for x in preview_raw_idx)

    print(f"augmenting x{args.aug} per segment...")
    for i in range(n_raw):
        t_us = raw_t[i]
        pos = raw_pos[i]
        vel = raw_vel[i]
        wind = raw_wind[i]

        if i in preview_raw_idx:
            plot_segment(
                os.path.join(args.preview_dir, f"raw_{i:03d}_{sources[i]}.png"),
                pos,
                vel,
                wind,
                t_us,
                f"raw segment #{i} ({sources[i]})  n={len(t_us)}",
            )

        for k in range(args.aug):
            tw, pw, vw, ww = augment_one(
                t_us,
                pos,
                vel,
                wind,
                rng,
                max_time_warp=args.max_time_warp,
                pos_noise=args.pos_noise,
                vel_noise=args.vel_noise,
            )
            aug_t.append(tw)
            aug_pos.append(pw)
            aug_vel.append(vw)
            aug_wind.append(ww)
            aug_origin.append(i)

            if i in preview_raw_idx and k < args.n_preview_aug_per_raw:
                plot_segment(
                    os.path.join(args.preview_dir, f"aug_{i:03d}_v{k}.png"),
                    pw,
                    vw,
                    ww,
                    tw,
                    f"aug variant #{k} of raw #{i}",
                )

    print(f"saving {len(aug_t)} augmented segments -> {args.out}")
    np.savez(
        args.out,
        sources=sources,
        orig_idx=np.asarray(aug_origin, dtype=np.int32),
        t_us=np.array(aug_t, dtype=object),
        pos_ned=np.array(aug_pos, dtype=object),
        vel_ned=np.array(aug_vel, dtype=object),
        wind_ned=np.array(aug_wind, dtype=object),
        params=np.array(
            {
                "aug": args.aug,
                "max_time_warp": args.max_time_warp,
                "pos_noise": args.pos_noise,
                "vel_noise": args.vel_noise,
                "seed": args.seed,
            }
        ),
    )
    print(f"previews -> {args.preview_dir}")


if __name__ == "__main__":
    main()
