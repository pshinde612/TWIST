"""
Single-threaded TWIST pipeline runner (sim or real).

- No async/multiprocessing.
- Retargeting can be a stub or BVH-based (see --retargeter).
- Optionally logs sim/robot state to an .npz file for later analysis.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import typer

import mujoco
import mujoco.viewer as mjv

from data_utils.rot_utils import (
    euler_from_quaternion,
    quat_rotate_inverse_torch,
)
from data_utils.rot_utils import quatToEuler

try:
    from projects.g1_full_body_kinematic.sew_mimic_g1_full_body_bvh import SEWMimicG1FullBody
    from projects.shared_devices.offline_lafan_bvh_reader_full_body import LafanBVHReaderV2
except Exception:  # pragma: no cover - optional dependency for BVH retargeting
    SEWMimicG1FullBody = None
    LafanBVHReaderV2 = None


# Retargeting stubs + helpers

@dataclass
class RetargetFrame:
    root_pos: np.ndarray      # (3,)
    root_rot: np.ndarray      # (4,) quaternion
    root_vel: np.ndarray      # (3,)
    root_ang_vel: np.ndarray  # (3,)
    dof_pos: np.ndarray       # (23,) for G1 body dofs (no wrists)


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q)
    if norm < 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return (q / norm).astype(np.float32)


def _quat_conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float32)


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ], dtype=np.float32)


def _quat_to_axis_angle(q: np.ndarray) -> Tuple[np.ndarray, float]:
    q = _quat_normalize(q)
    if q[3] < 0.0:
        q = -q
    angle = 2.0 * np.arccos(np.clip(q[3], -1.0, 1.0))
    s = np.sqrt(max(1.0 - q[3] * q[3], 0.0))
    if s < 1e-6:
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        axis = q[:3] / s
    return axis.astype(np.float32), float(angle)


def _angular_velocity_from_quats(q_prev: np.ndarray, q_curr: np.ndarray, dt: float) -> np.ndarray:
    if dt <= 0.0:
        return np.zeros(3, dtype=np.float32)
    q_prev = _quat_normalize(q_prev)
    q_curr = _quat_normalize(q_curr)
    q_rel = _quat_mul(_quat_conjugate(q_prev), q_curr)
    axis, angle = _quat_to_axis_angle(q_rel)
    return (axis * (angle / dt)).astype(np.float32)


class BaseRetargeter:
    def get_frame(self, frame_index: int, elapsed_time: float, dt: float) -> Optional[RetargetFrame]:
        raise NotImplementedError


class StubRetargeter(BaseRetargeter):
    def __init__(self, human_data: np.ndarray) -> None:
        self.human_data = human_data

    def get_frame(self, frame_index: int, elapsed_time: float, dt: float) -> Optional[RetargetFrame]:
        return retarget_frame_from_human(self.human_data, frame_index, dt)


class BvhRetargeter(BaseRetargeter):
    def __init__(
        self,
        bvh_path: Path,
        mapping_path: Optional[Path],
        unit_scale: float,
        playback_speed: float,
        loop_mode: str,
        yaw_offset: float,
        remove_world_offset: bool,
        retarget_xml: Path,
        control_rate_hz: float,
    ) -> None:
        if SEWMimicG1FullBody is None or LafanBVHReaderV2 is None:
            raise RuntimeError(
                "BVH retargeting requires the installed 'projects' package. "
                "Make sure it is available in your Python environment."
            )

        if not retarget_xml.exists():
            raise FileNotFoundError(f"Retarget XML not found: {retarget_xml}")

        self.bvh_reader = LafanBVHReaderV2(
            bvh_path=bvh_path,
            playback_speed=playback_speed,
            loop_mode=loop_mode,
            unit_scale=unit_scale,
            yaw_offset=yaw_offset,
            remove_world_offset=remove_world_offset,
            mapping_path=mapping_path,
        )

        self.model = mujoco.MjModel.from_xml_path(str(retarget_xml))
        self.data = mujoco.MjData(self.model)
        if control_rate_hz > 0:
            self.model.opt.timestep = 1.0 / control_rate_hz

        self.controller = SEWMimicG1FullBody(
            self.model,
            self.data,
            debug=False,
            timing_enabled=False,
            control_rate_hz=float(control_rate_hz),
            elbow_filter_cutoff_hz=8.0,
        )
        self.controller.setup_mocap_and_scaling(
            mocap_body_name="pelvis_mocap_mover",
            mocap_cartesian_scale=np.array([1.0, 1.0, 1.0]),
            mocap_offset=np.array([0.0, 0.0, -1.28]),
            shoulder_width_scale=0.9,
            hip_width_scale=0.9,
        )

        self._prev_root_pos = None
        self._prev_root_rot = None

    def get_frame(self, frame_index: int, elapsed_time: float, dt: float) -> Optional[RetargetFrame]:
        action_dict = self.bvh_reader.get_action_dict_at_time(elapsed_time)
        if action_dict is None:
            return None

        self.controller.update_control_from_csv_data(
            action_dict,
            engaged=True,
            enable_lower_body_correction=True,
        )
        self.controller.apply_kinematic_control()
        mujoco.mj_forward(self.model, self.data)

        root_pos = self.data.mocap_pos[0].copy().astype(np.float32)
        root_quat = self.data.mocap_quat[0].copy().astype(np.float32)  # wxyz
        root_rot = np.array([root_quat[1], root_quat[2], root_quat[3], root_quat[0]], dtype=np.float32)  # xyzw

        action_29dof = self.controller.get_current_29dof_joint_actions()
        left_leg = action_29dof["left_leg"]
        right_leg = action_29dof["right_leg"]
        torso = action_29dof["torso"]
        left_arm = action_29dof["left_arm"][:4]
        right_arm = action_29dof["right_arm"][:4]
        dof_pos = np.concatenate([left_leg, right_leg, torso, left_arm, right_arm]).astype(np.float32)

        if self._prev_root_pos is None or self._prev_root_rot is None:
            root_vel = np.zeros(3, dtype=np.float32)
            root_ang_vel = np.zeros(3, dtype=np.float32)
        else:
            root_vel = (root_pos - self._prev_root_pos) / max(dt, 1e-6)
            root_ang_vel = _angular_velocity_from_quats(self._prev_root_rot, root_rot, dt)

        self._prev_root_pos = root_pos
        self._prev_root_rot = root_rot

        return RetargetFrame(root_pos, root_rot, root_vel, root_ang_vel, dof_pos)


def retarget_frame_from_human(human_data: np.ndarray, idx: int, dt: float) -> RetargetFrame:
    """
    Placeholder retargeter. Replace this with your own human->robot mapping.

    Expected to return a single robot-frame sample at index `idx`.
    """
    # TODO: Replace with real retargeting.
    root_pos = np.array([0.0, 0.0, 0.793], dtype=np.float32)
    root_rot = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    root_vel = np.zeros(3, dtype=np.float32)
    root_ang_vel = np.zeros(3, dtype=np.float32)
    dof_pos = np.zeros(23, dtype=np.float32)
    return RetargetFrame(root_pos, root_rot, root_vel, root_ang_vel, dof_pos)


def load_human_pose_file(path: str) -> np.ndarray:
    """
    Minimal loader for human pose data.
    Replace this to match your file format (npz, pkl, csv, etc.).
    """
    # TODO: Replace this loader. For now, allow .npy as a simple placeholder.
    if path.endswith(".npy"):
        return np.load(path)
    # Fall back to empty array if file format unsupported; retargeter stub ignores it.
    return np.zeros((1, 1), dtype=np.float32)



# Mimic obs construction

def build_mimic_obs_from_frame(frame: RetargetFrame, device: torch.device) -> np.ndarray:
    """Build mimic_obs vector identical to server_high_level_motion_lib.py."""
    root_pos = torch.tensor(frame.root_pos, device=device).view(1, 1, 3)
    root_rot = torch.tensor(frame.root_rot, device=device).view(1, 1, 4)
    root_vel = torch.tensor(frame.root_vel, device=device).view(1, 1, 3)
    root_ang_vel = torch.tensor(frame.root_ang_vel, device=device).view(1, 1, 3)

    # rpy from quat
    roll, pitch, yaw = euler_from_quaternion(root_rot)
    roll = roll.reshape(1, -1, 1)
    pitch = pitch.reshape(1, -1, 1)
    yaw = yaw.reshape(1, -1, 1)

    # velocities in root frame
    root_vel = quat_rotate_inverse_torch(root_rot, root_vel).reshape(1, -1, 3)
    root_ang_vel = quat_rotate_inverse_torch(root_rot, root_ang_vel).reshape(1, -1, 3)

    # Insert wrist slots into dof_pos (G1 expects 25 with wrists at [19,24])
    dof_pos = torch.tensor(frame.dof_pos, device=device).view(1, 1, -1)
    dof_pos_with_wrist = torch.zeros(1, 1, 25, device=device)
    wrist_ids = [19, 24]
    other_ids = [i for i in range(25) if i not in wrist_ids]
    dof_pos_with_wrist[..., other_ids] = dof_pos

    mimic_obs = torch.cat(
        (
            root_pos[..., 2:3],
            roll, pitch, yaw,
            root_vel,
            root_ang_vel[..., 2:3],
            dof_pos_with_wrist,
        ),
        dim=-1,
    )[:, 0:1]

    return mimic_obs.reshape(1, -1).detach().cpu().numpy().squeeze()


def extract_mimic_obs_to_body_and_wrist(mimic_obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    total_degrees = 33
    wrist_ids = [27, 32]
    other_ids = [f for f in range(total_degrees) if f not in wrist_ids]
    policy_target = mimic_obs[other_ids]
    wrist_dof_pos = mimic_obs[wrist_ids]
    return policy_target, wrist_dof_pos


def aggregate_wrist_dof_pos(body_dof_pos: np.ndarray, wrist_dof_pos: np.ndarray) -> np.ndarray:
    total_degrees = 25
    wrist_ids = [19, 24]
    other_ids = [f for f in range(total_degrees) if f not in wrist_ids]
    whole_body_pd_target = np.zeros(total_degrees, dtype=np.float32)
    whole_body_pd_target[other_ids] = body_dof_pos
    whole_body_pd_target[wrist_ids] = wrist_dof_pos
    return whole_body_pd_target



# Sim runner
class TwistSimRunner:
    def __init__(
        self,
        xml_file: str,
        policy_path: str,
        device: str,
        vis: bool,
        record_video: bool,
    ) -> None:
        self.device = device
        self.policy = torch.jit.load(policy_path, map_location=device)
        self.policy.eval()

        self.model = mujoco.MjModel.from_xml_path(xml_file)
        self.model.opt.timestep = 0.001
        self.data = mujoco.MjData(self.model)

        self.viewer = None
        if vis:
            self.viewer = mjv.launch_passive(self.model, self.data, show_left_ui=False, show_right_ui=False)
            self.viewer.cam.distance = 2.0

        self.record_video = record_video
        self.num_actions = 23
        self.sim_dt = 0.001
        self.sim_decimation = 20

        self.default_dof_pos = np.array([
            -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,
            -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.4, 0.0, 1.2,
            0.0, -0.4, 0.0, 1.2,
        ], dtype=np.float32)

        self.mujoco_default_dof_pos = np.concatenate([
            np.array([0, 0, 0.793]),
            np.array([0, 0, 0, 1]),
            np.array([
                -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,
                -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,
                0.0, 0.0, 0.0,
                0.0, 0.2, 0.0, 1.2, 0.0,
                0.0, -0.2, 0.0, 1.2, 0.0,
            ], dtype=np.float32)
        ])

        self.stiffness = np.array([
            100, 100, 100, 150, 40, 40,
            100, 100, 100, 150, 40, 40,
            150, 150, 150,
            40, 40, 40, 40, 20,
            40, 40, 40, 40, 20,
        ], dtype=np.float32)
        self.damping = np.array([
            2, 2, 2, 4, 2, 2,
            2, 2, 2, 4, 2, 2,
            4, 4, 4,
            5, 5, 5, 5, 1,
            5, 5, 5, 5, 1,
        ], dtype=np.float32)
        self.torque_limits = np.array([
            88, 139, 88, 139, 50, 50,
            88, 139, 88, 139, 50, 50,
            88, 50, 50,
            25, 25, 25, 25, 25,
            25, 25, 25, 25, 25,
        ], dtype=np.float32)
        self.action_scale = 0.5
        self.ankle_idx = [4, 5, 10, 11]
        self.last_action = np.zeros(self.num_actions, dtype=np.float32)

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.mujoco_default_dof_pos
        mujoco.mj_forward(self.model, self.data)

    def extract_data(self):
        qpos = self.data.qpos.astype(np.float32)
        qvel = self.data.qvel.astype(np.float32)

        body_ids = [0, 1, 2, 3, 4, 5,
                    6, 7, 8, 9, 10, 11,
                    12, 13, 14,
                    15, 16, 17, 18,
                    20, 21, 22, 23]
        wrist_ids = [19, 24]

        whole_body_dof = qpos[7:]
        whole_body_dof_vel = qvel[6:]
        body_dof_pos = qpos[[f + 7 for f in body_ids]]
        body_dof_vel = qvel[[f + 6 for f in body_ids]]
        wrist_dof_pos = qpos[[f + 7 for f in wrist_ids]]
        wrist_dof_vel = qvel[[f + 6 for f in wrist_ids]]

        quat = self.data.sensor('orientation').data.astype(np.float32)
        ang_vel = self.data.sensor('angular-velocity').data.astype(np.float32)
        return (whole_body_dof, whole_body_dof_vel, body_dof_pos, body_dof_vel,
                wrist_dof_pos, wrist_dof_vel, quat, ang_vel)

    def step(self, mimic_obs: np.ndarray) -> Dict[str, np.ndarray]:
        whole_body_dof, whole_body_dof_vel, body_dof_pos, body_dof_vel, wrist_dof_pos, _, quat, ang_vel = self.extract_data()

        rpy = quatToEuler(quat)
        obs_body_dof_vel = body_dof_vel.copy()
        obs_body_dof_vel[self.ankle_idx] = 0.0
        obs_proprio = np.concatenate([
            ang_vel * 0.25,
            rpy[:2],
            (body_dof_pos - self.default_dof_pos),
            obs_body_dof_vel * 0.05,
            self.last_action,
        ])

        action_mimic, wrist_dof_pos = extract_mimic_obs_to_body_and_wrist(mimic_obs)
        obs_full = np.concatenate([action_mimic, obs_proprio])

        obs_tensor = torch.from_numpy(obs_full).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            raw_action = self.policy(obs_tensor).cpu().numpy().squeeze()
        self.last_action = raw_action

        raw_action = np.clip(raw_action, -10.0, 10.0)
        scaled_actions = raw_action * self.action_scale
        pd_target = scaled_actions + self.default_dof_pos
        pd_target = aggregate_wrist_dof_pos(pd_target, wrist_dof_pos)

        torque = (pd_target - whole_body_dof) * self.stiffness - whole_body_dof_vel * self.damping
        torque = np.clip(torque, -self.torque_limits, self.torque_limits)
        self.data.ctrl[:] = torque
        mujoco.mj_step(self.model, self.data)

        if self.viewer is not None:
            pelvis_pos = self.data.xpos[self.model.body("pelvis").id]
            self.viewer.cam.lookat = pelvis_pos
            self.viewer.sync()

        return {
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "ctrl": self.data.ctrl.copy(),
            "pd_target": pd_target.copy(),
        }



# CLI
app = typer.Typer(add_completion=False)


@app.command()
def run(
    input_human_file: str = typer.Option("/home/pshinde31/Desktop/arm_teleop/lafan_dataset/aiming1_subject1.bvh", help="Human pose data file (format up to you)."),
    mode: str = typer.Option("sim", help="Run mode: sim (real is TODO)."),
    retargeter: str = typer.Option("bvh", help="Retargeter: stub or bvh."),
    policy_path: str = typer.Option("assets/twist_general_motion_tracker.pt", help="TorchScript policy."),
    xml_file: str = typer.Option("assets/g1/g1_sim2sim_with_wrist_roll.xml", help="MuJoCo XML (sim only)."),
    retarget_xml: str = typer.Option(
        "/home/pshinde31/GitHub/SEW-Geometric-Teleop/projects/g1_full_body_kinematic/g1/g1_29dof_mocap_ctrl_auto_gen.xml",
        help="MuJoCo XML used for BVH retargeter (must include mocap bodies).",
    ),
    device: str = typer.Option("cuda", help="cpu or cuda."),
    num_steps: int = typer.Option(1000, help="Number of control steps to run."),
    control_dt: float = typer.Option(0.02, help="Control timestep (seconds)."),
    log_path: str = typer.Option("", help="Optional .npz output path for logging."),
    vis: bool = typer.Option(True, help="Enable MuJoCo viewer (sim only)."),
    bvh_mapping_file: str = typer.Option("/home/pshinde31/GitHub/SEW-Geometric-Teleop/projects/g1_full_body_kinematic/g1/bvh_to_g1.json", help="Optional BVH->G1 mapping JSON."),
    bvh_unit_scale: float = typer.Option(0.01, help="Scale applied to BVH positions."),
    bvh_playback_speed: float = typer.Option(1.0, help="BVH playback speed multiplier."),
    bvh_loop_mode: str = typer.Option("once", help="BVH loop mode: once or loop."),
    bvh_yaw_offset: float = typer.Option(0.0, help="Yaw offset (degrees) applied to BVH."),
    bvh_remove_world_offset: bool = typer.Option(True, help="Remove initial BVH world offset."),
) -> None:
    """Run TWIST in a single-threaded loop (sim or real)."""
    if mode not in {"sim"}:
        raise typer.BadParameter("mode must be 'sim' for now")
    if retargeter not in {"stub", "bvh"}:
        raise typer.BadParameter("retargeter must be 'stub' or 'bvh'")

    device_t = torch.device(device)
    if retargeter == "bvh":
        retargeter_obj: BaseRetargeter = BvhRetargeter(
            bvh_path=Path(input_human_file),
            mapping_path=Path(bvh_mapping_file) if bvh_mapping_file else None,
            unit_scale=bvh_unit_scale,
            playback_speed=bvh_playback_speed,
            loop_mode=bvh_loop_mode,
            yaw_offset=bvh_yaw_offset,
            remove_world_offset=bvh_remove_world_offset,
            retarget_xml=Path(retarget_xml),
            control_rate_hz=1.0 / control_dt if control_dt > 0 else 50.0,
        )
    else:
        human_data = load_human_pose_file(input_human_file)
        retargeter_obj = StubRetargeter(human_data)

    log_records: List[Dict[str, np.ndarray]] = []

    if mode == "sim":
        runner = TwistSimRunner(xml_file=xml_file, policy_path=policy_path, device=device, vis=vis, record_video=False)
        runner.reset()
        for i in range(num_steps):
            t0 = time.time()
            elapsed_time = i * control_dt
            frame = retargeter_obj.get_frame(i, elapsed_time, control_dt)
            if frame is None:
                break
            mimic_obs = build_mimic_obs_from_frame(frame, device_t)
            step_log = runner.step(mimic_obs)
            if log_path:
                step_log["time"] = np.array([i * control_dt], dtype=np.float32)
                log_records.append(step_log)

            elapsed = time.time() - t0
            if elapsed < control_dt:
                time.sleep(control_dt - elapsed)

    else:
        raise NotImplementedError("Real-robot mode is not wired in yet.")

    if log_path:
        # Flatten record list into arrays of shape (T, ...)
        keys = sorted({k for rec in log_records for k in rec.keys()})
        out: Dict[str, np.ndarray] = {}
        for k in keys:
            out[k] = np.stack([rec[k] for rec in log_records], axis=0)
        np.savez(log_path, **out)


if __name__ == "__main__":
    app()
