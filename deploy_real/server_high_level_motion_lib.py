#!/usr/bin/env python
import argparse
import time
import redis
import json
import numpy as np
import isaacgym
import torch
from rich import print
import os
import mujoco
from mujoco.viewer import launch_passive
# ---------------------------------------------------------------------
# Example imports: adapt to your actual file structure
# ---------------------------------------------------------------------
from pose.utils.motion_lib_pkl import MotionLib
from data_utils.rot_utils import euler_from_quaternion, quat_rotate_inverse, quat_rotate_inverse_torch

from data_utils.params import DEFAULT_MIMIC_OBS, DEFAULT_ACTION_HAND

# ---------------------------------------------------------------------
# A small helper to replicate "mimic obs" logic from your code
# ---------------------------------------------------------------------
def build_mimic_obs(
    motion_lib: MotionLib,
    t_step: int,
    control_dt: float,
    tar_obs_steps,
    robot_type: str = "g1"
):
    """0.132, -0.229, -0.035,  0.135, -0.196, │                                                                                           │
│ │                │      -0.001,  0.149, -0.283,  0.099, -0.009, -0.033,  0.144,  0.263,             │                                                                                           │
│ │                │   │   1.44 ,  0.453,  1.151,  0.142, -1.474, -0.273,  1.171],                    │                                                                                           │
│ │                │     dtype=float32)                                                               │                                                                                           │
│ │                )                                                                                  │                                                                                           │
│ │ root_ang_vel = tensor([[[0., 0., 0.]]], device='cuda:0')                                          │                                                                                           │
│ │     root_pos = tensor([[[-0.0157, -0.0064,  0.8412]]], device='cuda:0')                           │                                                                                           │
│ │     root_rot = tensor([[[ 0.0172, -0.0504, -0.0106,  0.9985]]], device='cuda:0')                  │                                                                                           │
│ │     root_vel = tensor([[[0., 0., 0.]]], device='cuda:0')                                          │                                                                                           │
│ ╰────────────────────────────────────────────────────────────────────────────────
    Build the mimic_obs at time-step t_step, referencing the code in MimicRunner.
    """
    device = torch.device("cuda")
    # Build times
    motion_times = torch.tensor([t_step * control_dt], device=device).unsqueeze(-1)
    obs_motion_times = tar_obs_steps * control_dt + motion_times
    obs_motion_times = obs_motion_times.flatten()
    
    # Suppose we only have a single motion in the .pkl
    motion_ids = torch.zeros(len(tar_obs_steps), dtype=torch.int, device=device)
    
    # Retrieve motion frames
    root_pos, root_rot, root_vel, root_ang_vel, dof_pos, _, body_pos = motion_lib.calc_motion_frame(motion_ids, obs_motion_times)

    # Convert to euler (roll, pitch, yaw)
    roll, pitch, yaw = euler_from_quaternion(root_rot)
    roll = roll.reshape(1, -1, 1)
    pitch = pitch.reshape(1, -1, 1)
    yaw = yaw.reshape(1, -1, 1)

    # Transform velocities to root frame
    root_vel = quat_rotate_inverse_torch(root_rot, root_vel).reshape(1, -1, 3)
    root_ang_vel = quat_rotate_inverse_torch(root_rot, root_ang_vel).reshape(1, -1, 3)

    root_pos = root_pos.reshape(1, -1, 3)
    dof_pos = dof_pos.reshape(1, -1, dof_pos.shape[-1])
    
    if robot_type == "g1":
        dof_pos_with_wrist = torch.zeros(25, device=device).reshape(1, 1, 25)
        wrist_ids = [19, 24]
        other_ids = [f for f in range(25) if f not in wrist_ids]
        dof_pos_with_wrist[..., other_ids] = dof_pos
        dof_pos = dof_pos_with_wrist
        
    mimic_obs_buf = torch.cat((
                root_pos[..., 2:3],
                roll, pitch, yaw,
                root_vel,
                root_ang_vel[..., 2:3],
                dof_pos
            ), dim=-1)[:, 0:1]  # shape (1, 1, ?)
    mimic_obs_buf = mimic_obs_buf.reshape(1, -1)
    
    return mimic_obs_buf.detach().cpu().numpy().squeeze(), root_pos.detach().cpu().numpy().squeeze(), \
        root_rot.detach().cpu().numpy().squeeze(), dof_pos.detach().cpu().numpy().squeeze(), \
            root_vel.detach().cpu().numpy().squeeze(), root_ang_vel.detach().cpu().numpy().squeeze()


def main(args, xml_file, robot_base):

    if args.vis:
        sim_model = mujoco.MjModel.from_xml_path(xml_file)
        sim_data = mujoco.MjData(sim_model)
        viewer = launch_passive(model=sim_model, data=sim_data, show_left_ui=False, show_right_ui=False)
        
        # Print DoF names in order
        print("Degrees of Freedom (DoF) names and their order:")
        for i in range(sim_model.nv):  # 'nv' is the number of DoFs
            dof_name = mujoco.mj_id2name(sim_model, mujoco.mjtObj.mjOBJ_JOINT, sim_model.dof_jntid[i])
            print(f"DoF {i}: {dof_name}")

        # print("Body names and their IDs:")
        # for i in range(self.model.nbody):  # 'nbody' is the number of bodies
        #     body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
        #     print(f"Body ID {i}: {body_name}")
        
        print("Motor (Actuator) names and their IDs:")
        for i in range(sim_model.nu):  # 'nu' is the number of actuators (motors)
            motor_name = mujoco.mj_id2name(sim_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            print(f"Motor ID {i}: {motor_name}")
            
    # 1. Connect to Redis
    redis_client = redis.Redis(host="localhost", port=6379, db=0)

    # 2. Load motion library
    device = "cuda" if torch.cuda.is_available() else "cpu"
    motion_lib = MotionLib(args.motion_file, device=device)
    
    # 3. Prepare the steps array
    tar_obs_steps = [int(x.strip()) for x in args.steps.split(",")]
    tar_obs_steps_tensor = torch.tensor(tar_obs_steps, device=device, dtype=torch.int)

    # 4. Loop over time steps and publish mimic obs
    control_dt = 0.02
    # compute num_steps based on motion length
    motion_id = torch.tensor([0], device=device, dtype=torch.long)
    motion_length = motion_lib.get_motion_length(motion_id)
    num_steps = int(motion_length / control_dt)
    
    print(f"[Motion Server] Streaming for {num_steps} steps at dt={control_dt:.3f} seconds...")

    last_mimic_obs = DEFAULT_MIMIC_OBS[args.robot]
    vis_root_vel = False
    vis_root_ang_vel = False
    if vis_root_vel:
        root_vel_list = []
    if vis_root_ang_vel:
        root_ang_vel_list = []
        
    try:
        for t_step in range(num_steps):
            t0 = time.time()

            # Build a mimic obs from the motion library
            mimic_obs, root_pos, root_rot, dof_pos, root_vel, root_ang_vel = build_mimic_obs(
                motion_lib=motion_lib,
                t_step=t_step,
                control_dt=control_dt,
                tar_obs_steps=tar_obs_steps_tensor,
                robot_type=args.robot
            )
            if vis_root_vel:
                root_vel_list.append(root_vel)
            if vis_root_ang_vel:
                root_ang_vel_list.append(root_ang_vel)

            # Convert to JSON (list) to put into Redis
            mimic_obs_list = mimic_obs.tolist() if mimic_obs.ndim == 1 else mimic_obs.flatten().tolist()
            redis_client.set(f"action_mimic_{args.robot}", json.dumps(mimic_obs_list))
            redis_client.set(f"action_hand_{args.robot}", json.dumps(DEFAULT_ACTION_HAND[args.robot].tolist()))
            # Publish reference qpos (MuJoCo order: pos(3), quat(wxyz), dof_pos(25))
            root_rot_wxyz = root_rot[[3, 0, 1, 2]]
            qpos_ref = np.concatenate([root_pos, root_rot_wxyz, dof_pos]).astype(np.float32)
            redis_client.set(
                f"qpos_ref_{args.robot}",
                json.dumps({
                    "frame": t_step,
                    "time": t_step * control_dt,
                    "qpos": qpos_ref.tolist(),
                }),
            )
            last_mimic_obs = mimic_obs
            # Print or log it
            print(f"Step {t_step:4d} => mimic_obs shape = {mimic_obs.shape} published...", end="\r")

            if args.vis:
                sim_data.qpos[:3] = root_pos
                # filp rot
                # root_rot = root_rot[[1,2,3,0]]
                sim_data.qpos[3:7] = root_rot_wxyz
                sim_data.qpos[7:] = dof_pos
                mujoco.mj_forward(sim_model, sim_data)
                robot_base_pos = sim_data.xpos[sim_model.body(robot_base).id]
                viewer.cam.lookat = robot_base_pos
                # set distance to pelvis
                viewer.cam.distance = 2.0
                viewer.sync()
                
            # Sleep to maintain real-time pace
            elapsed = time.time() - t0
            if elapsed < control_dt:
                time.sleep(control_dt - elapsed)
        
    except KeyboardInterrupt:
        print("[Motion Server] Keyboard interrupt. Interpolating to default mimic_obs...")
        # do linear interpolation to the last mimic_obs
        time_back_to_default = 2.0
        for i in range(int(time_back_to_default / control_dt)):
            interp_mimic_obs = last_mimic_obs + (DEFAULT_MIMIC_OBS[args.robot] - last_mimic_obs) * (i / (time_back_to_default / control_dt))
            redis_client.set(f"action_mimic_{args.robot}", json.dumps(interp_mimic_obs.tolist()))
            redis_client.set(f"action_hand_{args.robot}", json.dumps(DEFAULT_ACTION_HAND[args.robot].tolist()))
            time.sleep(control_dt)
        redis_client.set(f"action_mimic_{args.robot}", json.dumps(DEFAULT_MIMIC_OBS[args.robot].tolist()))
        redis_client.set(f"action_hand_{args.robot}", json.dumps(DEFAULT_ACTION_HAND[args.robot].tolist()))
        last_mimic_obs = DEFAULT_MIMIC_OBS[args.robot]
        exit()
    finally:
        print("[Motion Server] Exiting...Interpolating to default mimic_obs...")
        # do linear interpolation to the last mimic_obs
        time_back_to_default = 2.0
        for i in range(int(time_back_to_default / control_dt)):
            interp_mimic_obs = last_mimic_obs + (DEFAULT_MIMIC_OBS[args.robot] - last_mimic_obs) * (i / (time_back_to_default / control_dt))
            redis_client.set(f"action_mimic_{args.robot}", json.dumps(interp_mimic_obs.tolist()))
            redis_client.set(f"action_hand_{args.robot}", json.dumps(DEFAULT_ACTION_HAND[args.robot].tolist()))
            time.sleep(control_dt)
        redis_client.set(f"action_mimic_{args.robot}", json.dumps(DEFAULT_MIMIC_OBS[args.robot].tolist()))
        redis_client.set(f"action_hand_{args.robot}", json.dumps(DEFAULT_ACTION_HAND[args.robot].tolist()))
        last_mimic_obs = DEFAULT_MIMIC_OBS[args.robot]
        exit()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion_file", help="Path to your *.pkl motion file for MotionLib", 
                        default="/home/yanjieze/projects/g1_wbc/humanoid-motion-imitation/track_dataset/twist_motion_dataset/mocap/0.pkl")
    parser.add_argument("--robot", type=str, default="g1", choices=["g1"])
    parser.add_argument("--steps", type=str,
                        default="1",
                        help="Comma-separated steps for future frames (tar_obs_steps)")
    parser.add_argument("--vis", action="store_true", help="Visualize the motion")
    args = parser.parse_args()

    # Respect CLI flag for visualization
    args.vis = True
    
    print("Robot type: ", args.robot)
    print("Motion file: ", args.motion_file)
    print("Steps: ", args.steps)
    
    HERE = os.path.dirname(os.path.abspath(__file__))
    
    if args.robot == "g1":
        xml_file = f"{HERE}/../assets/g1/g1_mocap_with_wrist_roll.xml"
        robot_base = "pelvis"
    else:
        raise ValueError(f"robot type {args.robot} not supported")
    
    
    main(args, xml_file, robot_base)
