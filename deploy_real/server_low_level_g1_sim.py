import argparse
import csv
import json
import time
import numpy as np
import redis
import mujoco
import torch
from rich import print
from collections import deque
import mujoco.viewer as mjv
from tqdm import tqdm
from data_utils.params import DEFAULT_MIMIC_OBS
import os
from data_utils.rot_utils import quatToEuler

def _format_error_bar(value, max_value=1.0, width=20):
    if value is None:
        return " " * width, 0.0
    v = float(max(0.0, min(value, max_value)))
    filled = int(round((v / max_value) * width)) if max_value > 0 else 0
    bar = "█" * filled + "░" * (width - filled)
    return bar, v


def _make_csv_header(prefix: str, length: int):
    return ["frame_idx", "time"] + [f"{prefix}_{i}" for i in range(length)]


def _coerce_length(arr, length: int):
    if arr is None:
        return np.full(length, np.nan, dtype=np.float32)
    arr = np.asarray(arr, dtype=np.float32).flatten()
    if arr.size == length:
        return arr
    if arr.size > length:
        return arr[:length]
    out = np.full(length, np.nan, dtype=np.float32)
    out[:arr.size] = arr
    return out


class CsvLogger:
    def __init__(self, path, header):
        self.path = path
        self.file = open(path, "w", newline="")
        self.writer = csv.writer(self.file)
        if header:
            self.writer.writerow(header)

    def log(self, row):
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        self.file.close()

def draw_root_velocity(mujoco_model, mujoco_data, mujoco_viewer, tgt_root_vel, init_geom_id, root_name, rgba_velocity=[1, 1, 0, 1]):
    """
    Draws an arrow representing velocity, for debug/visualization.
    """
    mujoco_viewer.user_scn.ngeom = init_geom_id
    root_body_id = mujoco_model.body(root_name).id
    root_pos = mujoco_data.xpos[root_body_id]
    root_vel = tgt_root_vel
    vel_scale = 1.0

    mujoco.mjv_initGeom(
        mujoco_viewer.user_scn.geoms[mujoco_viewer.user_scn.ngeom],
        type=mujoco.mjtGeom.mjGEOM_ARROW,
        size=np.zeros(3),
        pos=np.zeros(3),
        mat=np.zeros(9),
        rgba=rgba_velocity,
    )
    mujoco.mjv_connector(
        mujoco_viewer.user_scn.geoms[mujoco_viewer.user_scn.ngeom],
        type=mujoco.mjtGeom.mjGEOM_ARROW,
        width=0.01,
        from_=root_pos,
        to=root_pos + vel_scale * np.array(root_vel),
    )
    mujoco_viewer.user_scn.ngeom += 1
    return mujoco_viewer.user_scn.ngeom


# -------------------------------------------------------------------
# Main low-level policy controller that:
#   - reads mimic obs from Redis
#   - feeds into policy
#   - runs the sim
# -------------------------------------------------------------------
def extract_mimic_obs_to_body_and_wrist(mimic_obs):
    total_degrees = 33
    wrist_ids = [27, 32]
    other_ids = [f for f in range(total_degrees) if f not in wrist_ids]
    policy_target = mimic_obs[other_ids]
    wrist_dof_pos = mimic_obs[wrist_ids]
    
    return policy_target, wrist_dof_pos

def aggregate_wrist_dof_pos(body_dof_pos, wrist_dof_pos):
    total_degrees = 25
    wrist_ids = [19, 24]
    other_ids = [f for f in range(total_degrees) if f not in wrist_ids]
    whole_body_pd_target = np.zeros(total_degrees)
    whole_body_pd_target[other_ids] = body_dof_pos
    whole_body_pd_target[wrist_ids] = wrist_dof_pos
    
    return whole_body_pd_target
    
class RealTimePolicyController:
    def __init__(self, 
                 xml_file, 
                 policy_path, 
                 device='cuda', 
                 record_video=False,
                 log_dir=None,
                 log_prefix="g1_sim",
                 headless=False):
        
        self.redis_client = None
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        except Exception as e:
            print(f"Error connecting to Redis: {e}")

        self.device = device

        # Load policy
        self.policy = torch.jit.load(policy_path, map_location=device)
        print(f"Policy loaded from {policy_path}")

        # Create MuJoCo sim
        self.model = mujoco.MjModel.from_xml_path(xml_file)
        self.model.opt.timestep = 0.001
        self.data = mujoco.MjData(self.model)
        
        # Print DoF names in order
        print("Degrees of Freedom (DoF) names and their order:")
        for i in range(self.model.nv):  # 'nv' is the number of DoFs
            dof_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, self.model.dof_jntid[i])
            print(f"DoF {i}: {dof_name}")

        # print("Body names and their IDs:")
        # for i in range(self.model.nbody):  # 'nbody' is the number of bodies
        #     body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
        #     print(f"Body ID {i}: {body_name}")
        
        print("Motor (Actuator) names and their IDs:")
        for i in range(self.model.nu):  # 'nu' is the number of actuators (motors)
            motor_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            print(f"Motor ID {i}: {motor_name}")
            

        self.viewer = None
        if not headless:
            self.viewer = mjv.launch_passive(self.model, self.data, show_left_ui=False, show_right_ui=False)
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = 0
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 0
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = 0
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = 0
            self.viewer.cam.distance = 2.0

        # Example defaults & placeholders
        self.num_actions = 23
        self.sim_duration = 100000.0
        self.sim_dt = 0.001
        self.sim_decimation = 20

        self.last_action = np.zeros(self.num_actions, dtype=np.float32)

        # PD Gains, etc. (adapt as needed)
        self.default_dof_pos = np.array([
                -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,  # left leg (6)
                -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,  # right leg (6)
                0.0, 0.0, 0.0, # torso (1)
                0.0, 0.4, 0.0, 1.2,
                0.0, -0.4, 0.0, 1.2,
            ])
        """
        mimic_obs = np.concatenate([
        root_pos[2:3],      # just the z for height
        rpy,                # roll, pitch, yaw
        root_vel_relative,  # local root vel
        dof_pos])
        """
        self.default_mimic_obs = DEFAULT_MIMIC_OBS["g1"]
        self.mujoco_default_dof_pos = np.concatenate([
            np.array([0, 0, 0.793]),
            np.array([0, 0, 0, 1]),
             np.array([-0.2, 0.0, 0.0, 0.4, -0.2, 0.0,  # left leg (6)
                -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,  # right leg (6)
                0.0, 0.0, 0.0, # torso (1)
                0.0, 0.2, 0.0, 1.2, 0.0, # left arm (4)
                0.0, -0.2, 0.0, 1.2, 0.0, # right arm (4)
                ])
        ])
        self.stiffness = np.array([
                100, 100, 100, 150, 40, 40,
                100, 100, 100, 150, 40, 40,
                150, 150, 150,
                40, 40, 40, 40, 20,
                40, 40, 40, 40, 20,
            ])
        self.damping = np.array([
                2, 2, 2, 4, 2, 2,
                2, 2, 2, 4, 2, 2,
                4, 4, 4,
                5, 5, 5, 5, 1,
                5, 5, 5, 5, 1,
            ])
        self.torque_limits = np.array([
                88, 139, 88, 139, 50, 50,
                88, 139, 88, 139, 50, 50,
                88, 50, 50,
                25, 25, 25, 25, 25,
                25, 25, 25, 25, 25,
            ])
        
        self.action_scale = 0.5

        
        self.ankle_idx = [4, 5, 10, 11]
        
        # For multi-step history
        self.n_mimic_obs = 31
        self.n_proprio = self.n_mimic_obs + 3 + 2 + 3*self.num_actions
        self.proprio_history_buf = deque(maxlen=10)
        for _ in range(10):
            self.proprio_history_buf.append(np.zeros(self.n_proprio))

        self.record_video = record_video if not headless else False
        self.headless = headless
        self.log_dir = log_dir
        self.log_prefix = log_prefix
        self.qpos_ref_logger = None
        self.pd_target_logger = None
        self.qpos_sim_logger = None
        self.control_dt = self.sim_dt * self.sim_decimation
        self._pending_log = None
        self.last_joint_error = None
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            self._init_loggers()

    def _init_loggers(self):
        qpos_len = int(self.model.nq)
        qpos_ref_path = os.path.join(self.log_dir, f"{self.log_prefix}_qpos_ref.csv")
        pd_target_path = os.path.join(self.log_dir, f"{self.log_prefix}_pd_target.csv")
        qpos_sim_path = os.path.join(self.log_dir, f"{self.log_prefix}_qpos_sim.csv")
        self.qpos_ref_logger = CsvLogger(qpos_ref_path, _make_csv_header("qpos", qpos_len))
        self.pd_target_logger = CsvLogger(pd_target_path, _make_csv_header("qpos", qpos_len))
        self.qpos_sim_logger = CsvLogger(qpos_sim_path, _make_csv_header("qpos", qpos_len))

    def _fetch_ref_qpos(self, fallback_frame_idx: int, fallback_time: float):
        frame_idx = fallback_frame_idx
        time_ref = fallback_time
        qpos_ref = None
        if self.redis_client is None:
            return qpos_ref, frame_idx, time_ref
        msg = self.redis_client.get("qpos_ref_g1")
        if msg is None:
            return qpos_ref, frame_idx, time_ref
        payload = json.loads(msg)
        if isinstance(payload, dict):
            frame_idx = int(payload.get("frame", frame_idx))
            time_ref = float(payload.get("time", time_ref))
            qpos_ref = payload.get("qpos", None)
        else:
            qpos_ref = payload
        return qpos_ref, frame_idx, time_ref

    def _flush_pending_log(self):
        if self._pending_log is None:
            return
        frame_idx = self._pending_log["frame_idx"]
        time_ref = self._pending_log["time"]
        qpos_ref = _coerce_length(self._pending_log["qpos_ref"], int(self.model.nq))
        pd_target = _coerce_length(self._pending_log["pd_target"], int(self.stiffness.shape[0]))
        qpos_sim = _coerce_length(self._pending_log["qpos_sim"], int(self.model.nq))
        qpos_root = None
        if self._pending_log.get("qpos_ref") is not None and len(qpos_ref) >= 7:
            qpos_root = qpos_ref[:7]
        elif self._pending_log.get("qpos_sim_start") is not None:
            qpos_root = _coerce_length(self._pending_log["qpos_sim_start"], int(self.model.nq))[:7]
        if qpos_root is None or len(qpos_root) < 7:
            qpos_root = np.full(7, np.nan, dtype=np.float32)
        qpos_pd_target = np.concatenate([qpos_root, pd_target]).astype(np.float32)
        if self.qpos_ref_logger:
            self.qpos_ref_logger.log([frame_idx, time_ref, *qpos_ref.tolist()])
        if self.pd_target_logger:
            self.pd_target_logger.log([frame_idx, time_ref, *qpos_pd_target.tolist()])
        if self.qpos_sim_logger:
            self.qpos_sim_logger.log([frame_idx, time_ref, *qpos_sim.tolist()])
        self._pending_log = None
    
    def _update_error_overlay(self):
        if self.viewer is None:
            return
        bar, v = _format_error_bar(self.last_joint_error, max_value=1.0, width=18)
        title = "Mean Joint Error"
        body = f"{v:.4f} rad\\n{bar}"
        try:
            if hasattr(self.viewer, "overlay"):
                self.viewer.overlay[mujoco.mjtGridPos.mjGRID_TOPRIGHT] = (title, body)
            if hasattr(self.viewer, "_overlay"):
                self.viewer._overlay[mujoco.mjtGridPos.mjGRID_TOPRIGHT] = (title, body)
        except Exception:
            pass

    def extract_data(self):
        qpos = self.data.qpos.astype(np.float32)
        qvel = self.data.qvel.astype(np.float32)
        
        body_ids = [0,1,2,3,4,5,
                    6,7,8,9,10,11,
                    12,13,14,
                    15,16,17,18,# 19
                    20,21,22,23, # 24
                    ]
        wrist_ids = [19, 24]
        
        whole_body_dof = qpos[7:]
        whole_body_dof_vel = qvel[6:]
        body_dof_pos = qpos[[f+7 for f in body_ids]]
        body_dof_vel = qvel[[f+6 for f in body_ids]]
        wrist_dof_pos = qpos[[f+7 for f in wrist_ids]]
        wrist_dof_vel = qvel[[f+6 for f in wrist_ids]]

        quat = self.data.sensor('orientation').data.astype(np.float32)
        ang_vel = self.data.sensor('angular-velocity').data.astype(np.float32)
        return whole_body_dof, whole_body_dof_vel, body_dof_pos, body_dof_vel, wrist_dof_pos, wrist_dof_vel, quat, ang_vel

    def reset_sim(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def reset(self, mujoco_dof_pos=None):
        # body & hand
        self.data.qpos[:] = mujoco_dof_pos
        mujoco.mj_forward(self.model, self.data)
       
    def run(self):
        # Optionally record video
        if self.record_video:
            import imageio
            video_name = "debug_sim.mp4"
            print(f"Saving video to {video_name}")
            mp4_writer = imageio.get_writer(video_name, fps=50)
        else:
            mp4_writer = None

        self.reset_sim()
        self.reset(self.mujoco_default_dof_pos)

        steps = int(self.sim_duration / self.sim_dt)
        pbar = tqdm(range(steps), desc="Simulating...")

        # send initial proprio to redis
        proprio_json = json.dumps(self.proprio_history_buf[0].tolist())
        self.redis_client.set("state_body_g1", proprio_json)
        self.redis_client.set("state_hand_g1", json.dumps(np.zeros(14).tolist()))
        try:
            for i in pbar:
                
                t_start = time.time()
                whole_body_dof, whole_body_dof_vel, body_dof_pos, body_dof_vel, wrist_dof_pos, wrist_dof_vel, quat, ang_vel = self.extract_data()
                
                if i % self.sim_decimation == 0:
                    
                    # Build a "proprio" vector for your policy, e.g.:
                    rpy = quatToEuler(quat)
                    obs_body_dof_vel = body_dof_vel.copy()
                    obs_body_dof_vel[self.ankle_idx] = 0.
                    obs_proprio = np.concatenate([
                        ang_vel * 0.25,
                        rpy[:2],
                        (body_dof_pos - self.default_dof_pos),
                        obs_body_dof_vel * 0.05,
                        self.last_action
                    ])
                    # send proprio to redis
                    self.redis_client.set("state_body_g1", json.dumps(obs_proprio.tolist()))
                    self.redis_client.set("state_hand_g1", json.dumps(np.zeros(14).tolist()))

                    # Try to get the latest mimic obs from Redis
                    try:
                        action_mimic_json = self.redis_client.get("action_mimic_g1")
                        if action_mimic_json is not None:
                            action_mimic_list = json.loads(action_mimic_json)
                            action_mimic = np.array(action_mimic_list, dtype=np.float32)
                            action_mimic, wrist_dof_pos = extract_mimic_obs_to_body_and_wrist(action_mimic)
                        else:
                            raise Exception("cannot get action mimic from redis")
                    except:
                        raise Exception("cannot get action mimic from redis")

                    control_step_idx = i // self.sim_decimation
                    control_time = control_step_idx * self.control_dt
                    qpos_ref, ref_frame_idx, ref_time = self._fetch_ref_qpos(control_step_idx, control_time)

                    obs_full = np.concatenate([action_mimic, obs_proprio])
                    obs_hist = np.array(self.proprio_history_buf).flatten()
                    obs_buf = np.concatenate([obs_full, obs_hist])
                    self.proprio_history_buf.append(obs_full)

                    obs_tensor = torch.from_numpy(obs_buf).float().unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        raw_action = self.policy(obs_tensor).cpu().numpy().squeeze()
                    
                    self.last_action = raw_action
                    raw_action = np.clip(raw_action, -10., 10.)
                    scaled_actions = raw_action * self.action_scale
                    pd_target = scaled_actions + self.default_dof_pos
                    pd_target = aggregate_wrist_dof_pos(pd_target, wrist_dof_pos)
                    if qpos_ref is not None:
                        qpos_ref_arr = np.asarray(qpos_ref, dtype=np.float32).flatten()
                        if qpos_ref_arr.size >= (7 + pd_target.size):
                            self.last_joint_error = float(np.mean(np.abs(pd_target - qpos_ref_arr[7:7 + pd_target.size])))
                        else:
                            self.last_joint_error = None
                    if self.log_dir:
                        self._pending_log = {
                            "frame_idx": ref_frame_idx,
                            "time": ref_time,
                            "qpos_ref": qpos_ref,
                            "pd_target": pd_target.copy(),
                            "qpos_sim_start": self.data.qpos.astype(np.float32).copy(),
                        }
                    if self.viewer is not None:
                        # debug draw velocity arrow if you want
                        self.viewer.user_scn.ngeom = 0
                        draw_root_velocity(self.model, self.data, self.viewer, [0,0,0], 0, "pelvis", [1,0,0,1])
                        
                        # make camera follow the pelvis
                        pelvis_pos = self.data.xpos[self.model.body("pelvis").id]
                        self.viewer.cam.lookat = pelvis_pos
                        self._update_error_overlay()
                        self.viewer.sync()
                        if mp4_writer is not None:
                            img = self.viewer.read_pixels()
                            mp4_writer.append_data(img)

                # PD control
                torque = (pd_target - whole_body_dof) * self.stiffness - whole_body_dof_vel * self.damping
                torque = np.clip(torque, -self.torque_limits, self.torque_limits)
                
                self.data.ctrl[:] = torque
                
                mujoco.mj_step(self.model, self.data)
                if self.log_dir and (i + 1) % self.sim_decimation == 0 and self._pending_log is not None:
                    self._pending_log["qpos_sim"] = self.data.qpos.astype(np.float32).copy()
                    self._flush_pending_log()
                # sleep to maintain real-time pace
                elapsed = time.time() - t_start
                if elapsed < self.sim_dt:
                    time.sleep(self.sim_dt - elapsed)
        except Exception as e:
            print(f"Error in run: {e}")
            pass
        finally:
            if mp4_writer is not None:
                mp4_writer.close()
                print("Video saved")

            if self.viewer is not None:
                self.viewer.close()
            if self.qpos_ref_logger:
                self.qpos_ref_logger.close()
            if self.pd_target_logger:
                self.pd_target_logger.close()
            if self.qpos_sim_logger:
                self.qpos_sim_logger.close()


def main_low_level_sim(args):
    controller = RealTimePolicyController(
        xml_file=args.xml_file,
        policy_path=args.policy_path,
        device='cuda',
        record_video=args.record_video,
        log_dir=args.log_dir,
        log_prefix=args.log_prefix,
        headless=args.headless,
    )
    controller.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    HERE = os.path.dirname(os.path.abspath(__file__))
    
    parser.add_argument("--xml_file", default=os.path.join(HERE, "../assets/g1/g1_sim2sim_with_wrist_roll.xml"), help="Mujoco XML file")
    
    parser.add_argument("--policy_path",  help="Path to the policy",
                        default="/home/pshinde31/Desktop/arm_teleop/TWIST/assets/twist_general_motion_tracker.pt"
                        )
                        
    parser.add_argument("--record_video", action="store_true", help="Record a video")
    parser.add_argument("--log_dir", default=None, help="Directory to write CSV logs")
    parser.add_argument("--log_prefix", default="g1_sim", help="Prefix for CSV log files")
    parser.add_argument("--headless", action="store_true", help="Run without MuJoCo viewer")
    args = parser.parse_args()

    args.record_proprio = True
    
    main_low_level_sim(args)
