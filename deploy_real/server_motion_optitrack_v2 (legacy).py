import argparse
import json
import time
import numpy as np
import redis
from collections import deque
from rich import print
from scipy.spatial.transform import Rotation as R
import mujoco
import os
from robot_retarget.mink_retarget import MinkRetarget
from robot_retarget.inverse_dynamics import get_id_data
from robot_retarget.utils.draw import draw_frame
from robot_retarget.optitrack_datastream import setup_optitrack
from data_utils.params import DEFAULT_MIMIC_OBS, DEX31_QPOS_OPEN, DEX31_QPOS_CLOSE
from loop_rate_limiters import RateLimiter
from data_utils.rot_utils import quatToEuler, quat_rotate_inverse
from robot_control.joycon_wrapper import JoyConController
from robot_control.speaker import Speaker

# Constants kept consistent with the original script
BUFFER_START_IDX = 0  # The buffer can ignore a few frames at startup
LOOKAHEAD = 2         # How many frames to look ahead for acceleration estimation
RESET_THRESHOLD = 120
# SEED_BUFFER_SAMPLES = 10  # How many seed frames to collect at startup
SEED_BUFFER_SAMPLES = 3

class ResetError(Exception):
    """
    Used to raise a reset signal to the upper layer when needed.
    """
    pass


class ViconDataBuffer:
    """
    Stores a sequence of frames (frame_time, qpos) from OptiTrack (Vicon),
    and performs retargeting on push().
    """
    def __init__(self, client, model_file, ik_config, ik_match_table):
        self.client = client
        self.ik_config = ik_config
        self.mink_retarget = MinkRetarget(model_file, ik_match_table, 
                                          scale=ik_config["scale"],
                                          ground=ik_config["ground_height"],
                                          feet_offset=ik_config["feet_offset"],
                                          )

        self._buffer = []

    def push(self):
        # Read from OptiTrack
        vicon_data = self.client.get_frame()
        frame_number = self.client.frame_number()
        # print("frame_number: ", frame_number)
        # Perform retarget
        self.mink_retarget.update_targets(vicon_data)
        curr_error = self.mink_retarget.error()
        qpos = self.mink_retarget.retarget(vicon_data)
        next_error = self.mink_retarget.error()

        # Iterate retargeting until the error converges
        num_iter = 0
        max_iter = 20
        while curr_error - next_error > 0.001:
            curr_error = next_error
            qpos = self.mink_retarget.retarget(vicon_data)
            next_error = self.mink_retarget.error()
            num_iter += 1
            if num_iter > max_iter:
                break
        # Store (timestamp, retargeted qpos) in the buffer
        self._buffer.append((frame_number / 120.0, qpos.copy(), vicon_data.copy()))

    def pushN(self, n):
        """Read n frames continuously."""
        start_len = self.length
        while self.length - start_len < n:
            self.push()

    def pop(self):
        assert self.length > 0, "buffer is empty"
        return self._buffer[-1]

    def clear(self):
        self._buffer = []

    @property
    def length(self):
        return len(self._buffer)

    @property
    def times(self):
        return np.array([frame_time for frame_time, _, _ in self._buffer])

    @property
    def qpos(self):
        return np.array([qpos for _, qpos, _ in self._buffer])

    @property
    def mocap_data(self):
        return np.array([mocap_data for _, _, mocap_data in self._buffer])

    def __getitem__(self, idx):
        return self._buffer[idx]


def _get_mimic_obs(qpos, qdot):
    """
    Keep consistent with the original script: process converting qpos, qdot to mimic_obs.
    No line is simplified to keep parity with the original script.
    """
    root_pos = qpos[:3]
    dof_pos  = qpos[7:]
    root_rot = qpos[3:7]

    roll, pitch, yaw = quatToEuler(root_rot)
    root_height = root_pos[2:3]

    root_vel = qdot[:3]
    root_rot = root_rot.reshape(1, 4)
    root_vel = root_vel.reshape(1, 3)
    # Note the order [1,2,3,0] here
    root_rot = root_rot[:, [1,2,3,0]]
    root_vel_relative = quat_rotate_inverse(root_rot, root_vel).reshape(3)
    root_ang_vel = qdot[3:6]
    root_ang_vel_relative = quat_rotate_inverse(root_rot, root_ang_vel).reshape(3)
    root_ang_vel_relative_yaw = root_ang_vel_relative[2:3]
    
    mimic_obs = np.concatenate([
        root_height,           # z, 1
        [roll, pitch, yaw],    # roll, pitch, yaw, 3
        root_vel_relative,     # root vel in local frame, 3
        root_ang_vel_relative_yaw,          # root ang vel in local frame, 1
        dof_pos                # Joint angles, 21 for t1, 23 for g1
    ])
    # print("shape of mimic_obs: ", mimic_obs.shape)
    return mimic_obs


class OptiTrackMimicObsServer:
    """
    Used to:
    1. Connect to OptiTrack
    2. Continuously get mocap data and retarget
    3. Get (qpos, qdot) via get_id_data
    4. Call _get_mimic_obs
    5. Publish via Redis
    """
    def __init__(self, vicon_host, ik_config_path, xml_file, robot_type, vis=False, use_hand=False):
        # Initialize Vicon
        self.client = setup_optitrack(vicon_host)
        print(f"Client initialized")
        # Load IK config
        with open(ik_config_path) as f:
            self.ik_config = json.load(f)
        self.ik_match_table = self.ik_config.pop("ik_match_table")
        # Fix array types in ik_match_table
        for key, val in self.ik_match_table.items():
            self.ik_match_table[key][-2] = np.array(val[-2])
            self.ik_match_table[key][-1] = np.array(val[-1])

        # get robot type and base
        self.robot_type = robot_type
        if self.robot_type == "g1":
            self.robot_base = "pelvis"
        elif self.robot_type == "t1":
            self.robot_base = "Waist"
        elif self.robot_type == "toddy":
            self.robot_base = "waist_link"
        else:
            raise ValueError(f"robot type {self.robot_type} not supported")
        
        self.use_hand = use_hand
        # assert self.use_hand and self.robot_type == "g1", "hand control is only supported for g1"
        
        # ViconDataBuffer stores multiple frames
        self.vicon_data_buffer = ViconDataBuffer(
            self.client, 
            xml_file, 
            self.ik_config, 
            self.ik_match_table
        )

        # Redis connection (ensure local/remote Redis is running)
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)

        # Collect a few frames first to avoid unstable data at startup
        self.vicon_data_buffer.pushN(SEED_BUFFER_SAMPLES)
        self.curr_buffer_idx = BUFFER_START_IDX
        
        self.last_mimic_obs = DEFAULT_MIMIC_OBS[self.robot_type]
        
        # joycon controller
        self.joycon_controller = JoyConController()
        
        # speaker
        self.speaker = Speaker()
        
        # Qdot smoothing
        qdot_window_size = 5
        self.qdot_window_size = qdot_window_size
        self.qdot_history = deque(maxlen=qdot_window_size)
        
        
        self.vis = vis
        if self.vis:
            self.sim_model = mujoco.MjModel.from_xml_path(xml_file)
            self.sim_data = mujoco.MjData(self.sim_model)
            self.viewer = mujoco.viewer.launch_passive(model=self.sim_model, data=self.sim_data, show_left_ui=False, show_right_ui=False)
            
            # Print DoF names in order
            print("Degrees of Freedom (DoF) names and their order:")
            for i in range(self.sim_model.nv):  # 'nv' is the number of DoFs
                dof_name = mujoco.mj_id2name(self.sim_model, mujoco.mjtObj.mjOBJ_JOINT, self.sim_model.dof_jntid[i])
                print(f"DoF {i}: {dof_name}")

            # print("Body names and their IDs:")
            # for i in range(self.model.nbody):  # 'nbody' is the number of bodies
            #     body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            #     print(f"Body ID {i}: {body_name}")
            
            print("Motor (Actuator) names and their IDs:")
            for i in range(self.sim_model.nu):  # 'nu' is the number of actuators (motors)
                motor_name = mujoco.mj_id2name(self.sim_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                print(f"Motor ID {i}: {motor_name}")

    def teleop_mode(self, freq=50):
        """
        Pull data from Vicon at fixed frequency freq (default 30Hz),
        compute mimic obs, and publish via Redis.
        """
        print(f"[Teleop Mode] Start running at {freq} Hz ...")
        self.speaker.speak(f"Start Teleoperation Mode")
        
        rate = RateLimiter(frequency=freq, warn=False)
        step_count = 0
        while self.flag_teleop_mode and self.running:
            joycon_state = self.joycon_controller.get_state()
            self.flag_teleop_mode = joycon_state["right"]["x"] == 1
            if joycon_state["right"]["b"] == 1:
                self.running = False
            
            t_start = time.time()
            self.vicon_data_buffer.push()
            self.curr_buffer_idx += 1
            
            # use LOOKAHEAD + get_id_data to get qpos, qdot
            times = self.vicon_data_buffer.times
            qpos  = self.vicon_data_buffer.qpos
            
            if self.curr_buffer_idx + LOOKAHEAD >= len(times):
                # if buffer is not long enough, wait for next frame
                continue
            else:
                # get a window to calculate acceleration
                window_size = 100
                start_idx = max(
                    0, 
                    self.curr_buffer_idx + LOOKAHEAD + 1 - window_size
                )
                # use get_id_data to calculate smoother (q0, qdot0)
                q0, qdot0 = get_id_data(
                    times[start_idx : self.curr_buffer_idx + LOOKAHEAD + 1],
                    qpos[start_idx : self.curr_buffer_idx + LOOKAHEAD + 1],
                    t0=times[self.curr_buffer_idx + LOOKAHEAD - 2]
                )
                
                # --- SMOOTH qdot0 via 1D conv over history ---
                self.qdot_history.append(qdot0)
                if len(self.qdot_history) == self.qdot_window_size:
                    hist = np.stack(self.qdot_history, axis=0)  # (W, D)
                    kernel = np.ones(self.qdot_window_size) / self.qdot_window_size
                    # smoothed: weighted sum over window
                    qdot0_sm = hist.T.dot(kernel)              # (D,)
                else:
                    qdot0_sm = qdot0
                # ----------------------------------------------
                
                # get mimic_obs
                mimic_obs = _get_mimic_obs(q0, qdot0_sm)

                if step_count == 0: # if newly into teleop mode, we need to do a safe transition
                    self.safe_transition_mode(self.last_mimic_obs, mimic_obs, freq=freq, seconds=1.0)
                    
                self.redis_client.set(f"action_mimic_{self.robot_type}", json.dumps(mimic_obs.tolist()))

                self.last_mimic_obs = mimic_obs
                
                if self.use_hand:
                    if joycon_state["left"]["zl"]:
                        left_q_target = DEX31_QPOS_OPEN["left"]
                    else:
                        left_q_target = DEX31_QPOS_CLOSE["left"]
                    if joycon_state["right"]["zr"]:
                        right_q_target = DEX31_QPOS_OPEN["right"]
                    else:
                        right_q_target = DEX31_QPOS_CLOSE["right"]
                    dex31_qpos = np.concatenate([left_q_target, right_q_target])
                    self.redis_client.set(f"action_hand_{self.robot_type}", json.dumps(dex31_qpos.tolist()))

                step_count += 1
                
            if self.vis:
                self.vis_qpos(q0)

            # Keep fixed frequency
            rate.sleep()
            fps = 1 / (time.time() - t_start)
            # if self.curr_buffer_idx % 100 == 0:
            #     print(f"FPS: {fps}")
        self.speaker.speak("Finished Teleoperation Mode")
    
    def safe_stand_mode(self, freq=50, seconds=3.0):
        print("[Into safe stand mode...]")
        self.speaker.speak("Start Safe Standing Mode")
        # do linear interpolation to the last mimic_obs
        time_back_to_default = seconds
        for i in range(int(time_back_to_default * freq)):
            interp_mimic_obs = self.last_mimic_obs + (DEFAULT_MIMIC_OBS[self.robot_type] - self.last_mimic_obs) * (i / (time_back_to_default * freq))
            self.redis_client.set(f"action_mimic_{self.robot_type}", json.dumps(interp_mimic_obs.tolist()))
            time.sleep(1.0 / freq)
        self.redis_client.set(f"action_mimic_{self.robot_type}", json.dumps(DEFAULT_MIMIC_OBS[self.robot_type].tolist()))
        self.last_mimic_obs = DEFAULT_MIMIC_OBS[self.robot_type]
        
        print("[Finished safe stand mode...]")
        self.speaker.speak("Finished Safe Standing Mode")
        
    def safe_transition_mode(self, current_mimic_obs, target_mimic_obs, freq=50, seconds=1.0):
        print("[Into safe transition mode...]")
        self.speaker.speak("Start Transition Mode")
        # do linear interpolation to the last mimic_obs
        time_back_to_default = seconds
        for i in range(int(time_back_to_default * freq)):
            interp_mimic_obs = current_mimic_obs + (target_mimic_obs - current_mimic_obs) * (i / (time_back_to_default * freq))
            self.redis_client.set(f"action_mimic_{self.robot_type}", json.dumps(interp_mimic_obs.tolist()))
            time.sleep(1.0 / freq)
        self.redis_client.set(f"action_mimic_{self.robot_type}", json.dumps(target_mimic_obs.tolist()))
        self.speaker.speak("Finished Transition Mode")
        
    def vis_qpos(self, qpos):
        if self.vis:
            self.viewer.user_scn.ngeom = 0
            # Draw the task targets for reference
            mocap_data = self.vicon_data_buffer.mocap_data[-1]
            for robot_link, ik_data in self.ik_match_table.items():
                if ik_data[0] not in mocap_data:
                    continue
                draw_frame(
                    self.ik_config["scale"] * np.array(mocap_data[ik_data[0]][0])
                    - self.vicon_data_buffer.mink_retarget.ground,
                    R.from_quat(
                        mocap_data[ik_data[0]][1], scalar_first=True
                    ).as_matrix(),
                    self.viewer,
                    0.1,
                    orientation_correction=R.from_quat(ik_data[-1], scalar_first=True),
                )
                
            self.sim_data.qpos[:] = qpos
            mujoco.mj_forward(self.sim_model, self.sim_data)

            # camera follow the pelvis
            robot_base_pos = self.sim_data.xpos[self.sim_model.body(self.robot_base).id]
            self.viewer.cam.lookat = robot_base_pos
            # set distance to pelvis
            self.viewer.cam.distance = 2.0
            # Visualize at fixed FPS.
            self.viewer.sync()
            
            
            
    def main_loop(self, freq=50):
        
        self.safe_stand_mode(freq, seconds=2.0)
        
        self.running = True
        try:
            while self.running:
                
                joycon_state = self.joycon_controller.get_state()
                self.flag_teleop_mode = joycon_state["right"]["x"] == 1
                if joycon_state["right"]["b"] == 1:
                    self.running = False
                
                if self.flag_teleop_mode:
                    try:
                        self.teleop_mode(freq)
                    except Exception as e:
                        print(f"exit teleop mode: {e}")
                        print(f"Into safe stand mode...")
                        self.safe_stand_mode(freq, seconds=3.0)
                        break
                else:
                    try:
                        t_start = time.time()
                        self.vicon_data_buffer.push()
                        self.curr_buffer_idx += 1
                        
                        times = self.vicon_data_buffer.times
                        qpos  = self.vicon_data_buffer.qpos
                        
                        if self.curr_buffer_idx + LOOKAHEAD >= len(times):
                            # If the buffer is not long enough, wait for the next frame
                            pass
                        else:
                            # Take a window to compute acceleration
                            window_size = 100
                            start_idx = max(
                                0, 
                                self.curr_buffer_idx + LOOKAHEAD + 1 - window_size
                            )
                            # Use get_id_data to compute smoother (q0, qdot0)
                            q0, qdot0 = get_id_data(
                                times[start_idx : self.curr_buffer_idx + LOOKAHEAD + 1],
                                qpos[start_idx : self.curr_buffer_idx + LOOKAHEAD + 1],
                                t0=times[self.curr_buffer_idx + LOOKAHEAD - 2]
                            )
                            self.qdot_history.append(qdot0)
                        
                        
                        # keep sending last_mimic_obs to maintain the motion
                        self.redis_client.set(f"action_mimic_{self.robot_type}", json.dumps(self.last_mimic_obs.tolist()))
                    
                        if self.vis:
                            self.vis_qpos(q0)

                        # keep fixed frequency
                        elapsed_time = time.time() - t_start
                        if elapsed_time < 1.0 / freq:
                            time.sleep(1.0 / freq - elapsed_time)
                        fps = 1 / (time.time() - t_start)
                        if self.curr_buffer_idx % 100 == 0:
                            print(f"FPS: {fps}")
                    except Exception as e:
                        print(f"exit non-teleop mode: {e}")
                        print(f"Into safe stand mode...")
                        self.safe_stand_mode(freq, seconds=3.0)
                        break
        except Exception as e:
            print(f"main loop error: {e}")
            time_back_to_default = 3.0
            for i in range(int(time_back_to_default * freq)):
                interp_mimic_obs = self.last_mimic_obs + (DEFAULT_MIMIC_OBS[self.robot_type] - self.last_mimic_obs) * (i / (time_back_to_default * freq))
                self.redis_client.set(f"action_mimic_{self.robot_type}", json.dumps(interp_mimic_obs.tolist()))
                time.sleep(1.0 / freq)
            self.redis_client.set(f"action_mimic_{self.robot_type}", json.dumps(DEFAULT_MIMIC_OBS[self.robot_type].tolist()))
            self.last_mimic_obs = DEFAULT_MIMIC_OBS[self.robot_type]
            
        self.safe_stand_mode(freq, seconds=3.0)
        exit()

def main():
    parser = argparse.ArgumentParser()
    HERE = os.path.dirname(os.path.abspath(__file__))
    # parser.add_argument("--host", default="192.168.110.120", help="OptiTrack Server Address")
    parser.add_argument("--host", default="192.168.110.119", help="OptiTrack Server Address")
    
    parser.add_argument("--robot", default="g1", help="robot type", choices=["g1", "t1", "toddy"])
    
    parser.add_argument("--use_wrist_roll", action="store_true", 
        help="whether to use wrist roll"
    )
    
 
    parser.add_argument("--vis", action="store_true", 
        help="whether to visualize the simulation"
    )
    
    parser.add_argument("--use_hand", action="store_true", 
        help="whether to use hand"
    )
    
    args = parser.parse_args()
    
    args.vis = True
    args.use_wrist_roll = True
    # args.use_hand = True

    if args.robot == "g1":
        ik_config = f"{HERE}/robot_retarget/ik_configs/optitrack_to_g1.json"
        if args.use_wrist_roll:
            xml_file = f"{HERE}/../assets/g1/g1_mocap_with_wrist_roll.xml"
        else:
            xml_file = f"{HERE}/../assets/g1/g1_mocap.xml"
    elif args.robot == "t1":
        ik_config = f"{HERE}/robot_retarget/ik_configs/optitrack_to_t1.json"
        xml_file = f"{HERE}/../assets/t1/t1_mocap.xml"
    elif args.robot == "toddy":
        ik_config = f"{HERE}/robot_retarget/ik_configs/optitrack_to_toddy.json"
        xml_file = f"{HERE}/../assets/toddy/toddy_mocap.xml"
    else:
        raise ValueError(f"robot type {args.robot} not supported")
    
    server = OptiTrackMimicObsServer(
        vicon_host=args.host,
        ik_config_path=ik_config,
        xml_file=xml_file,
        robot_type=args.robot,
        vis=args.vis,
        use_hand=args.use_hand,
    )
    server.main_loop()

if __name__ == "__main__":
    main()
# Stored from
