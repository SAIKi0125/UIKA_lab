"""Extract descending segments from ArduPilot dataflash logs.

Reads .BIN files via pymavlink, joins XKF1 (pos/vel NED) with DCM (3D wind)
on TimeUS, then runs a state machine to slice "descent + decelerate" segments.

Output: a single .npz containing variable-length segments stored as object arrays.
"""

from __future__ import annotations

import argparse
import os
import sys
from glob import glob

import numpy as np
from pymavlink import DFReader


def read_bin(path: str) -> dict[str, np.ndarray]:
    """Parse a single .BIN file, return time-series dicts for XKF1 and DCM."""
    r = DFReader.DFReader_binary(path)
    xkf1_t, xkf1_pos, xkf1_vel = [], [], []
    dcm_t, dcm_wind = [], []
    while True:
        m = r.recv_match(type=["XKF1", "DCM"])
        if m is None:
            break
        t = m.get_type()
        d = m.to_dict()
        if t == "XKF1":
            xkf1_t.append(d["TimeUS"])
            xkf1_pos.append([d["PN"], d["PE"], d["PD"]])
            xkf1_vel.append([d["VN"], d["VE"], d["VD"]])
        else:
            dcm_t.append(d["TimeUS"])
            dcm_wind.append([d["VWN"], d["VWE"], d["VWD"]])
    return {
        "xkf1_t": np.asarray(xkf1_t, dtype=np.int64),
        "xkf1_pos": np.asarray(xkf1_pos, dtype=np.float32),
        "xkf1_vel": np.asarray(xkf1_vel, dtype=np.float32),
        "dcm_t": np.asarray(dcm_t, dtype=np.int64),
        "dcm_wind": np.asarray(dcm_wind, dtype=np.float32),
    }


def join_wind(xkf1_t: np.ndarray, dcm_t: np.ndarray, dcm_wind: np.ndarray) -> np.ndarray:
    """For every XKF1 timestamp, pick the most recent DCM wind sample (zero-order hold).

    Returns array of shape (N, 3). If DCM is empty, returns zeros.
    """
    if dcm_t.size == 0:
        return np.zeros((xkf1_t.size, 3), dtype=np.float32)
    idx = np.searchsorted(dcm_t, xkf1_t, side="right") - 1
    idx = np.clip(idx, 0, dcm_t.size - 1)
    return dcm_wind[idx]


def find_landings(
    t_us: np.ndarray,
    pos: np.ndarray,
    vel: np.ndarray,
    *,
    ground_tol_m: float = 3.0,
    min_drop_m: float = 3.0,
    min_dur_s: float = 3.0,
    max_dur_s: float = 60.0,
    min_gap_s: float = 5.0,
) -> list[tuple[int, int]]:
    """Locate landing trajectories from any altitude peak down to ground contact.

    Each touchdown event (entering "near ground") counts as one landing,
    including touch-and-go. For each touchdown the descent start is the
    altitude peak in the window between the previous touchdown (or log start)
    and the current touchdown.

    Conventions: PD is NED Down -> PD_max corresponds to the lowest altitude
    in the log, which we treat as ground level.

    Algorithm:
      1. ground = PD >= (PD_max - ground_tol_m)
      2. touchdown_idx = every rising edge of `ground` (False -> True), AND
         a final touchdown if the log ends near ground.
      3. for the i-th touchdown td[i], the search window is
         (end_of_previous_ground_run, td[i]]. The descent start is the
         argmin of PD inside that window (i.e. highest altitude peak between
         leaving the ground last time and touching down again).
      4. Keep segments with duration in [min_dur_s, max_dur_s] and
         drop >= min_drop_m. Ignore segments that never actually leave the
         ground (would yield drop < min_drop_m).
    """
    if t_us.size < 10:
        return []

    pd = pos[:, 2]
    pd_max = float(pd.max())
    n = len(pd)
    near_ground = pd >= (pd_max - ground_tol_m)

    # rising edges: index i where near_ground[i] is True and near_ground[i-1] is False
    rising = np.where((~near_ground[:-1]) & near_ground[1:])[0] + 1
    # falling edges: index i where near_ground[i] is False and near_ground[i-1] is True
    # (used to compute "end of previous ground run" -> start of next search window)
    falling = np.where(near_ground[:-1] & (~near_ground[1:]))[0] + 1

    if rising.size == 0:
        return []

    # debounce: drop rising edges that follow another rising edge within
    # min_gap_s (prevents bouncing/lift-and-touch within a single landing
    # from being counted as multiple touchdowns)
    if rising.size > 1:
        keep = [int(rising[0])]
        for r in rising[1:]:
            if (t_us[r] - t_us[keep[-1]]) * 1e-6 >= min_gap_s:
                keep.append(int(r))
        rising = np.asarray(keep)

    segments: list[tuple[int, int]] = []
    for k, td in enumerate(rising):
        # search window: from end of previous ground run (or log start) to td
        prev_falling = falling[falling < td]
        if prev_falling.size > 0:
            window_start = int(prev_falling[-1])
        else:
            window_start = 0
        if td - window_start < 10:
            continue
        peak_offset = int(np.argmin(pd[window_start : td + 1]))
        start = window_start + peak_offset
        end = int(td)
        if end - start < 10:
            continue
        dur_s = (t_us[end] - t_us[start]) * 1e-6
        drop = pd[end] - pd[start]
        if drop < min_drop_m or dur_s < min_dur_s or dur_s > max_dur_s:
            continue
        segments.append((start, end))
    return segments


def extract_segments(bin_path: str, **detect_kwargs) -> list[dict]:
    """Pipeline: parse → join → segment → return list of dicts."""
    raw = read_bin(bin_path)
    if raw["xkf1_t"].size == 0:
        return []
    wind = join_wind(raw["xkf1_t"], raw["dcm_t"], raw["dcm_wind"])
    segs = find_landings(raw["xkf1_t"], raw["xkf1_pos"], raw["xkf1_vel"], **detect_kwargs)
    out = []
    for s, e in segs:
        sl = slice(s, e + 1)
        pos = raw["xkf1_pos"][sl].copy()
        # shift so the touchdown point is exactly at origin (ground = z = 0)
        pos -= pos[-1]
        out.append(
            {
                "source": os.path.basename(bin_path),
                "t_us": raw["xkf1_t"][sl].copy(),
                "pos_ned": pos,
                "vel_ned": raw["xkf1_vel"][sl].copy(),
                "wind_ned": wind[sl].copy(),
            }
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--logs",
        type=str,
        default="/home/esd/LOGS/*.BIN",
        help="glob pattern for input .BIN files",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="/home/esd/LOGS/raw_segments.npz",
    )
    ap.add_argument("--ground-tol", type=float, default=3.0,
                    help="frames with PD within this much of PD_max count as 'near ground'")
    ap.add_argument("--min-drop", type=float, default=3.0)
    ap.add_argument("--min-dur", type=float, default=3.0)
    ap.add_argument("--max-dur", type=float, default=60.0)
    ap.add_argument("--min-gap", type=float, default=5.0,
                    help="ignore touchdown events spaced less than this from the previous one")
    args = ap.parse_args()

    files = sorted(glob(args.logs))
    if not files:
        print(f"no files matched: {args.logs}", file=sys.stderr)
        sys.exit(1)

    all_segments: list[dict] = []
    print(f"scanning {len(files)} file(s)...")
    for f in files:
        try:
            segs = extract_segments(
                f,
                ground_tol_m=args.ground_tol,
                min_drop_m=args.min_drop,
                min_dur_s=args.min_dur,
                max_dur_s=args.max_dur,
                min_gap_s=args.min_gap,
            )
        except Exception as ex:
            print(f"  {os.path.basename(f)}: ERROR {ex}")
            continue
        durs = [(s["t_us"][-1] - s["t_us"][0]) * 1e-6 for s in segs]
        drops = [s["pos_ned"][:, 2].max() - s["pos_ned"][0, 2] for s in segs]
        print(
            f"  {os.path.basename(f):16s} -> {len(segs):3d} segs"
            + (f"  dur=[{min(durs):.1f},{max(durs):.1f}]s" if durs else "")
            + (f"  drop=[{min(drops):.1f},{max(drops):.1f}]m" if drops else "")
        )
        all_segments.extend(segs)

    if not all_segments:
        print("no segments found; relax detection thresholds", file=sys.stderr)
        sys.exit(2)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    # save as object arrays since lengths vary
    np.savez(
        args.out,
        sources=np.array([s["source"] for s in all_segments]),
        t_us=np.array([s["t_us"] for s in all_segments], dtype=object),
        pos_ned=np.array([s["pos_ned"] for s in all_segments], dtype=object),
        vel_ned=np.array([s["vel_ned"] for s in all_segments], dtype=object),
        wind_ned=np.array([s["wind_ned"] for s in all_segments], dtype=object),
        detect_params=np.array(
            {
                "ground_tol_m": args.ground_tol,
                "min_drop_m": args.min_drop,
                "min_dur_s": args.min_dur,
                "max_dur_s": args.max_dur,
                "min_gap_s": args.min_gap,
            }
        ),
    )
    print(f"\nsaved {len(all_segments)} segments -> {args.out}")


if __name__ == "__main__":
    main()
