#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METRIC_TITLES = {
    "joint_err": "Joint Angle Error (rad)",
    "joint_vel_err": "Joint Velocity Error (rad/s)",
    "root_pos_err": "Root Position Error (m)",
    "root_rot_err": "Root Rotation Error (rad)",
    "root_vel_err": "Root Velocity Error (m/s)",
    "root_ang_vel_err": "Root Angular Velocity Error (rad/s)",
    "key_body_err": "Key Body Position Error (m)",
}


def load_npz(path: Path):
    with np.load(path, allow_pickle=True) as data:
        logs = data["logs"]
        body_names = data["key_body_names"].tolist()
    return logs, body_names


def series_from_logs(logs: np.ndarray, key: str) -> np.ndarray:
    values = [entry[key] for entry in logs]
    sample = values[0]
    if isinstance(sample, np.ndarray) and sample.ndim > 0:
        return np.stack(values, axis=0)
    if isinstance(sample, (list, tuple)):
        return np.stack([np.asarray(v, dtype=np.float32) for v in values], axis=0)
    return np.asarray(values, dtype=np.float32).squeeze()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot tracking error metrics from humanoid_mimic_errors.npz")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("humanoid_mimic_errors.npz"),
        help="Path to the saved npz log file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("humanoid_mimic_errors.png"),
        help="Destination PNG for the combined plot",
    )
    parser.add_argument(
        "--per-link-output",
        type=Path,
        default=Path("humanoid_keybody_errors.png"),
        help="Destination PNG for per-key-body error plot",
    )
    parser.add_argument("--show", action="store_true", help="Display the plots after saving")
    args = parser.parse_args()

    logs, body_names = load_npz(args.input)
    timesteps = np.arange(len(logs))

    metrics = [
        "joint_err",
        "joint_vel_err",
        "key_body_err",
        "root_pos_err",
        "root_rot_err",
        "root_vel_err",
        "root_ang_vel_err",
    ]

    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 14), sharex=True)
    for ax, key in zip(axes, metrics):
        values = series_from_logs(logs, key)
        ax.plot(timesteps, values, linewidth=1.2)
        ax.set_ylabel(METRIC_TITLES.get(key, key))
        ax.grid(True, linestyle="--", alpha=0.3)
    axes[-1].set_xlabel("Simulation Step")
    fig.suptitle("TWIST Tracking Errors (Isaac Gym Playback)", fontsize=16, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(args.output, dpi=200)
    print(f"Saved summary figure to {args.output}")

    # Per-key-body plot
    per_link = series_from_logs(logs, "key_body_err_per_link")
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    for idx, name in enumerate(body_names):
        ax2.plot(timesteps, per_link[:, idx], label=name, linewidth=1.0)
    ax2.set_xlabel("Simulation Step")
    ax2.set_ylabel("Key Body Position Error (m)")
    ax2.set_title("Per-Key-Body Tracking Error")
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.legend(loc="upper right", fontsize=8, ncol=2)
    fig2.tight_layout()
    fig2.savefig(args.per_link_output, dpi=200)
    print(f"Saved per-key-body figure to {args.per_link_output}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
