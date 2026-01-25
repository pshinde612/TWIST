#!/usr/bin/env python3
"""
Drive the TWIST Isaac Gym environment with the student policy and log tracking errors.
"""
import os
import sys
from typing import Any, Dict, List

import numpy as np

# --------------------------------------------------------------------------- #
# Path setup so we can import legged_gym and isaacgym without installing them #
# --------------------------------------------------------------------------- #
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
PKG_ROOT = os.path.join(REPO_ROOT, "legged_gym")
ISAAC_PY_ROOT = os.path.join(REPO_ROOT, "isaacgym", "python")

for path in (REPO_ROOT, PKG_ROOT, ISAAC_PY_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)


# Import isaacgym before torch to avoid the dependency warning.
import isaacgym  # noqa: E402,F401
import torch 
from legged_gym import envs  # noqa: E402,F401 - registers environments
from legged_gym.gym_utils import task_registry  # noqa: E402
 # noqa: E402


def _to_numpy(x: torch.Tensor) -> float:
    return x.detach().cpu().numpy().item()


def main() -> None:
    motion_file = os.path.join(REPO_ROOT, "aiming1_subject1_unitree_g1.pkl")
    if not os.path.isfile(motion_file):
        raise FileNotFoundError(
            f"Motion file not found: {motion_file}. Update motion_file to the correct path."
        )

    env_cfg, _ = task_registry.get_cfgs("g1_stu_rl")
    env_cfg.motion.motion_curriculum = False
    env_cfg.motion.motion_file = motion_file
    env_cfg.env.num_envs = 1
    env_cfg.env.headless = True
    env_cfg.viewer.pos = np.array([2.5, 0.0, 1.3])
    env_cfg.viewer.lookat = np.array([0.0, 0.0, 1.0])

    env, _ = task_registry.make_env("g1_stu_rl", args=None, env_cfg=env_cfg)
    reset_out = env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

    policy = torch.jit.load("assets/twist_general_motion_tracker.pt", map_location=env.device)
    key_body_names = env.cfg.motion.key_bodies

    logs: List[Dict[str, Any]] = []
    steps = min(2000, int(env.max_episode_length))

    for _ in range(steps):
        actions = policy(obs)
        obs, privileged_obs, rewards, dones, infos = env.step(actions)

        key_scalar, key_diffs = env._error_tracking_keybody_pos()
        key_diffs_per_link = key_diffs.squeeze(0).detach().cpu().numpy()

        logs.append(
            {
                "joint_err": _to_numpy(env._error_tracking_joint_dof().mean()),
                "joint_vel_err": _to_numpy(env._error_tracking_joint_vel().mean()),
                "root_pos_err": _to_numpy(env._error_tracking_root_translation().mean()),
                "root_rot_err": _to_numpy(env._error_tracking_root_rotation().mean()),
                "root_vel_err": _to_numpy(env._error_tracking_root_vel().mean()),
                "root_ang_vel_err": _to_numpy(env._error_tracking_root_ang_vel().mean()),
                "key_body_err": _to_numpy(key_scalar.mean()),
                "key_body_err_per_link": key_diffs_per_link,
            }
        )

        if dones.any():
            break

    np.savez(
        "humanoid_mimic_errors.npz",
        logs=np.array(logs, dtype=object),
        key_body_names=np.array(key_body_names, dtype=object),
    )
    print("Saved metrics to humanoid_mimic_errors.npz")


if __name__ == "__main__":
    main()
