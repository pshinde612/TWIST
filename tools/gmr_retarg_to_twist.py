#!/usr/bin/env python3
"""
Retarget a BVH motion with GMR and export a TWIST-compatible pickle.

This script wraps the standard GMR export pipeline but augments the saved data
with the additional fields expected by TWIST's MotionLib:
  - root_pos (T, 3)
  - root_rot (T, 4) in XYZW order
  - dof_pos (T, 23) matching TWIST's G1 joint ordering
  - local_body_pos (T, 38, 3) expressed in the pelvis frame
  - link_body_list (38 body names)
"""
import argparse
import os
import time
import pickle
from typing import List

import mujoco as mj
import mujoco.viewer as mjv
import numpy as np
from rich import print
from tqdm import tqdm

from general_motion_retargeting import (
    GeneralMotionRetargeting as GMR,
    RobotMotionViewer,
    ROBOT_XML_DICT,
)
from general_motion_retargeting.utils.lafan1 import load_lafan1_file
from benchmark_logger_gmr import BenchmarkLogger


def _load_model(robot_key: str) -> mj.MjModel:
    xml_path = str(ROBOT_XML_DICT[robot_key])
    return mj.MjModel.from_xml_path(xml_path)


def _lookup_body_ids(model: mj.MjModel, body_names: List[str]) -> List[int]:
    ids = []
    for name in body_names:
        bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise ValueError(f"Body '{name}' not found in MJCF; check naming.")
        ids.append(bid)
    return ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bvh_file", required=True, type=str, help="BVH motion file to load.")
    parser.add_argument(
        "--robot",
        choices=[
            "unitree_g1",
            "unitree_g1_with_hands",
            "booster_t1",
            "stanford_toddy",
            "fourier_n1",
            "engineai_pm01",
        ],
        default="unitree_g1",
    )
    parser.add_argument("--record_video", action="store_true", default=False)
    parser.add_argument("--video_path", type=str, default="videos/example.mp4")
    parser.add_argument("--rate_limit", action="store_true", default=False)
    parser.add_argument("--save_path", default=None, help="Path to save the robot motion.")
    parser.add_argument("--log_file", default=None, help="Path to save the benchmark log file csv.")
    args = parser.parse_args()

    if args.save_path is not None:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    lafan_frames, actual_human_height = load_lafan1_file(args.bvh_file)

    retargeter = GMR(src_human="bvh", tgt_robot=args.robot, actual_human_height=actual_human_height)

    model = _load_model(args.robot)
    data = mj.MjData(model)

    logger = None
    if args.log_file:
        try:
            logger = BenchmarkLogger(model, data, args.log_file)
            print(f"Logging benchmark data to {args.log_file}")
        except Exception as exc:
            print(f"[bold red]Error initializing logger:[/bold red] {exc}")

    motion_fps = 30
    robot_motion_viewer = RobotMotionViewer(
        robot_type=args.robot,
        motion_fps=motion_fps,
        transparent_robot=0,
        record_video=args.record_video,
        video_path=args.video_path,
    )

    fps_counter = 0
    fps_start_time = time.time()
    fps_display_interval = 2.0
    print(f"mocap_frame_rate: {motion_fps}")

    pbar = tqdm(total=len(lafan_frames), desc="Retargeting")

    body_names = [
        "pelvis",
        "left_hip_pitch_link",
        "left_hip_roll_link",
        "left_hip_yaw_link",
        "left_knee_link",
        "left_ankle_pitch_link",
        "left_ankle_roll_link",
        "left_toe_link",
        "pelvis_contour_link",
        "right_hip_pitch_link",
        "right_hip_roll_link",
        "right_hip_yaw_link",
        "right_knee_link",
        "right_ankle_pitch_link",
        "right_ankle_roll_link",
        "right_toe_link",
        "waist_yaw_link",
        "waist_roll_link",
        "torso_link",
        "head_link",
        "head_mocap",
        "imu_in_torso",
        "left_shoulder_pitch_link",
        "left_shoulder_roll_link",
        "left_shoulder_yaw_link",
        "left_elbow_link",
        "left_wrist_roll_link",
        "left_wrist_pitch_link",
        "left_wrist_yaw_link",
        "left_rubber_hand",
        "right_shoulder_pitch_link",
        "right_shoulder_roll_link",
        "right_shoulder_yaw_link",
        "right_elbow_link",
        "right_wrist_roll_link",
        "right_wrist_pitch_link",
        "right_wrist_yaw_link",
        "right_rubber_hand",
    ]
    body_ids = _lookup_body_ids(model, body_names)
    pelvis_id = body_ids[0]

    root_pos_acc = []
    root_rot_acc = []
    dof_pos_acc = []
    local_body_pos_acc = []

    viz_start_time = time.time()
    for i, smplx_data in enumerate(lafan_frames):
        fps_counter += 1
        now = time.time()
        if now - fps_start_time >= fps_display_interval:
            actual_fps = fps_counter / (now - fps_start_time)
            print(f"Actual rendering FPS: {actual_fps:.2f}")
            fps_counter = 0
            fps_start_time = now

        pbar.update(1)

        start_time = time.time()
        qpos = retargeter.retarget(smplx_data)
        ik_time = time.time() - start_time
        elapsed_time = now - viz_start_time

        if logger:
            logger.log(elapsed_time, ik_time, qpos)

        robot_motion_viewer.step(
            root_pos=qpos[:3],
            root_rot=qpos[3:7],
            dof_pos=qpos[7:],
            human_motion_data=retargeter.scaled_human_data,
            rate_limit=args.rate_limit,
        )

        if args.save_path is not None:
            q = qpos.copy()
            root_pos_acc.append(q[:3].astype(np.float32))
            root_rot_acc.append(np.array([q[4], q[5], q[6], q[3]], dtype=np.float32))

            g1_dof_indices = [
                7, 8, 9, 10, 11, 12,
                13, 14, 15, 16, 17, 18,
                19, 20, 21,
                22, 23, 24, 25,
                29, 30, 31, 32,
            ]
            dof_pos_acc.append(q[g1_dof_indices].astype(np.float32))

            data.qpos[:] = q
            mj.mj_forward(model, data)

            pelvis_pos = data.xpos[pelvis_id]
            pelvis_quat = data.xquat[pelvis_id]

            rot_mat = np.empty(9, dtype=np.float64)
            mj.mju_quat2Mat(rot_mat, pelvis_quat)
            R = rot_mat.reshape(3, 3)

            pelvis_T = pelvis_pos.astype(np.float64)
            frame_local = []
            for bid in body_ids:
                world = data.xpos[bid].astype(np.float64)
                local = R.T @ (world - pelvis_T)
                frame_local.append(local.astype(np.float32))
            local_body_pos_acc.append(frame_local)

    pbar.close()

    if args.save_path is not None:
        motion_data = {
            "fps": motion_fps,
            "root_pos": np.array(root_pos_acc, dtype=np.float32),
            "root_rot": np.array(root_rot_acc, dtype=np.float32),
            "dof_pos": np.array(dof_pos_acc, dtype=np.float32),
            "local_body_pos": np.array(local_body_pos_acc, dtype=np.float32),
            "link_body_list": body_names,
        }
        with open(args.save_path, "wb") as f:
            pickle.dump(motion_data, f)
        print(f"[green]Saved to {args.save_path}[/green]")

    if logger:
        logger.close()

    robot_motion_viewer.close()


if __name__ == "__main__":
    main()
