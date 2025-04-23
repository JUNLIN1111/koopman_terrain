#--------------- Junlin Wu 2025/4/15 ---------------#
#--------------- Sim to sim (deploy go2 walking legged gym to Mujoco) ---------------#

import torch
import torch.nn as nn
import numpy as np
import mujoco
import mujoco.viewer
import time
import os
import glob

# Define the Actor Network (inherit nn.Sequential to match the state_dict keys)
class Actor(nn.Sequential):
    def __init__(self, input_dim=48, output_dim=12, hidden_dims=[512, 256, 128]):# According to LeggedRobotCfgPPO 
        
        layers = []
        prev_dim = input_dim # temporary 48

        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ELU()) # activate function ELU()
            prev_dim = dim          # create the input and hidden layers step by step 

        layers.append(nn.Linear(prev_dim, output_dim)) # create output layer 
        layers.append(nn.Tanh()) # Bounded the action in (-1,1)
        super().__init__(*layers)

#  Converts a quaternion into a gravity vector projected into the robot’s local frame
def get_gravity_orientation(quaternion):

    qw, qx, qy, qz = quaternion
    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation
# PD control
def pd_control(target_q, q, kp, target_dq, dq, kd):

    return (target_q - q) * kp + (target_dq - dq) * kd

# 参数
model_dir = "D:\\koopman_robot\\Koopman_optimization_rl_policy\\pretrain_model"
xml_path = "D:\\koopman_robot\\Koopman_optimization_rl_policy\\resourse\go2\scene_hard_ground.xml"
simulation_duration = 30.0  # 仿真时长 [s]
simulation_dt = 0.002      # 仿真步长 [s]
control_decimation = 10    # 控制更新频率
action_scale = 0.25  

# 调整 PD 参数
kps = np.array([100.0] * 12, dtype=np.float32)  # 降低刚性
kds = np.array([10.0] * 12, dtype=np.float32)   # 增加阻尼
default_angles = np.array([
    0.1,   # FL_hip_joint
    0.1,   # RL_hip_joint
    -0.1,  # FR_hip_joint
    -0.1,  # RR_hip_joint
    0.8,   # FL_thigh_joint
    1.0,   # RL_thigh_joint
    0.8,   # FR_thigh_joint
    1.0,   # RR_thigh_joint
    -1.5,  # FL_calf_joint
    -1.5,  # RL_calf_joint
    -1.5,  # FR_calf_joint
    -1.5   # RR_calf_joint
], dtype=np.float32)

# 观测缩放因子（从 normalization 配置）
obs_scales = {
    "lin_vel": 2.0,
    "ang_vel": 0.25,
    "dof_pos": 1.0,
    "dof_vel": 0.05
}
commands_scale = np.array([0.0,0.0,0.0], dtype=np.float32)  # 对应 [lin_vel_x, lin_vel_y, yaw_rate]
clip_observations = 100.0
num_obs = 48 #dim of obs (include last act)
num_actions = 12 # dim of act (actually is the target trajectory)
cmd = np.array([2.0, 0.0, 0.0], dtype=np.float32)
action = np.zeros(num_actions, dtype=np.float32)
target_dof_pos = default_angles.copy()
obs = np.zeros(num_obs, dtype=np.float32)
counter = 0

# 自动搜索模型文件
def find_model_file(model_dir, pattern="model_*.p*"):
    model_files = glob.glob(os.path.join(model_dir, pattern))
    if not model_files:
        raise FileNotFoundError(f"No model files (.pt or .pth) found in {model_dir} with pattern {pattern}")
    model_files.sort(key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x)))) if any(c.isdigit() for c in os.path.basename(x)) else 0)
    return model_files[-1]

# 确定模型路径
try:
    # pth_path = find_model_file(model_dir)
    pth_path = "D:\\koopman_robot\\Koopman_optimization_rl_policy\\pretrain_model\\model_10000.pt"
    print(f"Found model file: {pth_path}")
except FileNotFoundError as e:
    print(e)
    print(f"Please check if {model_dir} exists and contains .pt or .pth files.")
    print("Directory contents:", os.listdir(model_dir))
    exit(1)

# 检查 XML 文件
if not os.path.exists(xml_path):
    raise FileNotFoundError(f"XML file not found at {xml_path}")

# 加载模型
try:
    policy = Actor(input_dim=48, output_dim=12, hidden_dims=[512, 256, 128])
    checkpoint = torch.load(pth_path, map_location="cpu")
    
    # 调试：打印 state_dict 键
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    print("State dict keys:", list(state_dict.keys())[:10])
    
    # 提取 Actor 权重
    actor_state_dict = {k.replace("actor.", ""): v for k, v in state_dict.items() if k.startswith("actor.")}
    if not actor_state_dict:
        actor_state_dict = state_dict
    
    # 调试：打印加载的键
    print("Actor state dict keys:", list(actor_state_dict.keys())[:10])
    
    policy.load_state_dict(actor_state_dict)
    policy.eval()
    print("Policy loaded successfully")
except Exception as e:
    raise RuntimeError(f"Failed to load policy from {pth_path}: {str(e)}")

# 加载 MuJoCo 模型
try:
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt
except Exception as e:
    raise RuntimeError(f"Failed to load MuJoCo model from {xml_path}: {str(e)}")

# 仿真循环
with mujoco.viewer.launch_passive(m, d) as viewer:
    start = time.time()
    zero_obs = torch.zeros((1, 48))
    with torch.no_grad():
        print("Zero obs action:", policy(zero_obs).numpy())
    while viewer.is_running() and time.time() - start < simulation_duration:
        step_start = time.time()

        # PD 控制
        tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
        d.ctrl[:] = tau

        # 仿真一步
        mujoco.mj_step(m, d)

        counter += 1
        if counter % control_decimation == 0:
            # 更新 obs
            base_lin_vel = d.qvel[:3]  # 线速度
            base_ang_vel = d.qvel[3:6]  # 角速度
            quat = d.qpos[3:7]  # 四元数
            projected_gravity = get_gravity_orientation(quat)
            dof_pos = d.qpos[7:]  # 关节位置
            dof_vel = d.qvel[6:]  # 关节速度

            # 填充 obs
            obs[0:3] = base_lin_vel * obs_scales["lin_vel"]
            obs[3:6] = base_ang_vel * obs_scales["ang_vel"]
            obs[6:9] = projected_gravity
            obs[9:12] = cmd * commands_scale
            obs[12:24] = (dof_pos - default_angles) * obs_scales["dof_pos"]
            obs[24:36] = dof_vel * obs_scales["dof_vel"]
            obs[36:48] = action

            # 裁剪观测值
            obs = np.clip(obs, -clip_observations, clip_observations)

            obs_tensor = torch.from_numpy(obs).unsqueeze(0)

            # 推理 action
            try:
                with torch.no_grad():
                    alpha = 0.2
                    action = policy(obs_tensor).numpy().squeeze()
                    new_action = policy(obs_tensor).numpy().squeeze()
                    action = alpha * action + (1 - alpha) * new_action
                target_dof_pos = action * action_scale + default_angles
                

                if counter % 100 == 0:
                    print(f"Step {counter}:")
                    print(f"  Obs lin_vel = {obs[0:3]}, ang_vel = {obs[3:6]}")
                    joint_error = np.abs(d.qpos[7:] - target_dof_pos)
                    print(f"Joint error (mean): {np.mean(joint_error):.3f}")
                    print(f"  Action = {action[:3]}, Target DOF = {target_dof_pos[:3]}")
                    print(f"  Target DOF = {target_dof_pos[:3]}, Actual DOF = {d.qpos[7:10]}")
                    print(f"  Obs range: min = {np.min(obs)}, max = {np.max(obs)}")

            except Exception as e:
                print(f"Error during inference: {str(e)}")
                break

        viewer.sync()

        # 时间控制
        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

        # print("Sample policy output for zero obs:")
        # zero_obs = torch.zeros((1, 48))
        # print(policy(zero_obs))
        # time.sleep(0.1)
print("Simulation completed")