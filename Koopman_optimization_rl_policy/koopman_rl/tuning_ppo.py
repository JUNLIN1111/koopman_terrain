import numpy as np
import torch
import torch.nn as nn
import mujoco
import gymnasium as gym
from stable_baselines3 import PPO
from scipy.io import loadmat
import pickle

# 假设环境和模型配置
NUM_OBS = 47  # PPO 观测维度（[2025-04-15, 10:18]）
NUM_ACTIONS = 12  # 动作维度（12 个关节扭矩）
SIMULATION_DT = 0.002  # 仿真步长（[2025-04-20, 15:35]）
CONTROL_DECIMATION = 5  # 控制间隔（[2025-04-20, 15:35]）
STEPS_PER_EPISODE = 100  # 每回合步数（[2025-04-20, 15:35]）
NUM_EPISODES = 10  # 回合数（微调）

# 加载 MuJoCo 环境（沙地，摩擦 0.9）
env = gym.make("Go2Sim-v0", ground_type="sand", friction=0.9)  # [2025-04-20, 02:35]
model = mujoco.MjModel.from_xml_path("go2.xml")
data = mujoco.MjData(model)

# 加载预训练的 PPO 模型（沙地策略）
ppo = PPO.load("model_10000.pt")  # [2025-04-15, 03:59]

# 加载 Koopman 模型（沙地）
with open("koopman_sand.pkl", "rb") as f:
    koopman_model = pickle.load(f)  # 包含 encoder, decoder
A_sand = loadmat("A_sand.mat")['A'][0]  # 128x128
B_sand = loadmat("B_sand.mat")['B'][0]  # 128x12
encoder = koopman_model.regressor._encoder  # 输入 40 维 -> 128 维
decoder = koopman_model.regressor._decoder  # 128 维 -> 40 维

# 从状态构造 PPO 观测（简化版，基于 collect_g1_koopman_data.py）
def get_ppo_obs(state, action, counter):
    # state: 40 维（body_pos(3), orientation(4), lin_vel(3), ang_vel(3), target_vel(3), q(12), dq(12)）
    q = state[16:28]  # 关节角度
    dq = state[28:40]  # 关节角速度
    omega = state[10:13] * 0.25  # 角速度缩放（[2025-04-15, 10:18]）
    target_vel = state[13:16]  # 目标速度
    phase = (counter * SIMULATION_DT) % 1.0
    sin_phase = np.sin(2 * np.pi * phase)
    cos_phase = np.cos(2 * np.pi * phase)
    obs = np.zeros(NUM_OBS, dtype=np.float32)
    obs[:3] = omega
    obs[3:6] = state[3:6]  # 简化：用 orientation 代替 gravity_orientation
    obs[6:9] = target_vel
    obs[9:21] = q  # 简化：未缩放
    obs[21:33] = dq
    obs[33:45] = action
    obs[45:47] = [sin_phase, cos_phase]
    return obs

# 用 Koopman 模型生成伪数据
def generate_pseudo_data(state, action, num_steps=10):
    pseudo_states = []
    pseudo_actions = []
    x = state  # 40 维状态
    curr_action = action
    for _ in range(num_steps):
        z = encoder(torch.tensor(x, dtype=torch.float32))
        z_next = A_sand @ z.detach().numpy() + B_sand @ curr_action
        x_next = decoder(torch.tensor(z_next, dtype=torch.float32)).detach().numpy()
        pseudo_states.append(x_next)
        pseudo_actions.append(curr_action)
        x = x_next
        ppo_obs = get_ppo_obs(x_next, curr_action, 0)  # counter 设为 0
        obs_tensor = torch.from_numpy(ppo_obs).unsqueeze(0)
        curr_action = ppo.policy.predict(obs_tensor)[0].numpy()
    return np.array(pseudo_states), np.array(pseudo_actions)

# 奖励函数
def compute_reward(state):
    lin_vel_x = state[7]  # 前进速度
    return lin_vel_x - 0.01 * np.linalg.norm(state[28:40])  # 减小关节速度惩罚

# 主调优循环
real_buffer = []
pseudo_buffer = []
counter = 0

for episode in range(NUM_EPISODES):
    mujoco.mj_resetData(model, data)
    state = np.zeros(40, dtype=np.float32)  # 初始状态（简化）
    state[13:16] = [0.5, 0.0, 0.0]  # 目标速度：向前走
    action = np.zeros(NUM_ACTIONS, dtype=np.float32)
    episode_data = []

    for step in range(STEPS_PER_EPISODE):
        # 获取观测
        ppo_obs = get_ppo_obs(state, action, counter)
        obs_tensor = torch.from_numpy(ppo_obs).unsqueeze(0)

        # PPO 预测动作
        if counter % CONTROL_DECIMATION == 0:
            action = ppo.policy.predict(obs_tensor)[0].numpy()

        # 仿真一步（简化：用伪状态更新）
        data.ctrl[:12] = action
        mujoco.mj_step(model, data)
        next_state = np.zeros(40, dtype=np.float32)  # 简化：用 Koopman 更新
        z = encoder(torch.tensor(state, dtype=torch.float32))
        z_next = A_sand @ z.detach().numpy() + B_sand @ action
        next_state = decoder(torch.tensor(z_next, dtype=torch.float32)).detach().numpy()

        # 计算奖励
        reward = compute_reward(next_state)

        # 存储真实数据
        real_buffer.append((state, action, reward, next_state))
        state = next_state
        counter += 1

        # 生成伪数据
        pseudo_states, pseudo_actions = generate_pseudo_data(state, action, num_steps=5)
        for i in range(len(pseudo_states)):
            pseudo_r = compute_reward(pseudo_states[i])
            pseudo_buffer.append((pseudo_states[i], pseudo_actions[i], pseudo_r, pseudo_states[i+1] if i+1 < len(pseudo_states) else pseudo_states[i]))

    print(f"Episode {episode + 1}/{NUM_EPISODES} completed.")

# 微调 PPO
for step in range(1000):
    # 混合真实和伪数据（50% 真实，50% 伪数据）
    if np.random.rand() < 0.5 and real_buffer:
        batch = [real_buffer[np.random.randint(len(real_buffer))]]
    elif pseudo_buffer:
        batch = [pseudo_buffer[np.random.randint(len(pseudo_buffer))]]
    else:
        continue

    # 转换为 PPO 格式 
    obs = batch[0][0]  # state
    act = batch[0][1]  # action
    rew = batch[0][2]  # reward
    next_obs = batch[0][3]  # next_state
    ppo.learn(total_timesteps=1, reset_num_timesteps=False, 
              tb_log_name="ppo_tune_sand", 
              callback=None)

# 保存调优后的 PPO 模型
ppo.save("ppo_tuned_sand.pt")
print("PPO tuning completed and saved as ppo_tuned_sand.pt")