
# --------------- Dyna framework using Koopman --------------- #
# --------------- Junlin Wu --------------- #

import numpy as np
import torch
import torch.nn as nn
import mujoco
import gymnasium as gym
from stable_baselines3 import PPO
import pykoopman as pk
import os

# Dyna 框架相关
class KoopmanDynamicsModel:
    def __init__(self, koopman_model=None, encoder=None, decoder=None):
        self.koopman_model = koopman_model  # Koopman 模型
        self.encoder = encoder  # 硬地的 encoder
        self.decoder = decoder  # 硬地的 decoder
        self.A = None  # Koopman 矩阵 A
        self.B = None  # Koopman 矩阵 B

    def fit(self, observations):
        # 收集的数据：observations 包含状态 (40 维) 和控制 (12 维)
        if self.koopman_model is None:
            # 初始化新的 Koopman 模型，只拟合 A 和 B
            dlk_regressor = pk.regression.NNDMDc(
                mode='Dissipative_control',
                n=40,  # 状态维度（[2025-04-20, 21:35]）
                m=12,  # 控制维度
                dt=0.002,
                config_encoder=dict(input_size=40, hidden_sizes=[512, 256], output_size=128, activations='relu'),
                config_decoder=dict(input_size=128, hidden_sizes=[256, 512], output_size=40, activations='relu'),
                batch_size=64,
                trainer_kwargs=dict(max_epochs=10)  # 快速拟合
            )
            self.koopman_model = pk.Koopman(regressor=dlk_regressor)

        # 用沙地数据拟合 A 和 B（encoder 和 decoder 固定）
        self.koopman_model.fit(observations)
        self.A = self.koopman_model.regressor.net.A.detach().numpy()  # 128x128
        self.B = self.koopman_model.regressor.net.B.detach().numpy()  # 128x12

    def predict(self, state, action):
        # 用 encoder 映射状态
        z = self.encoder(torch.tensor(state, dtype=torch.float32))
        # 用新的 A 和 B 预测下一状态
        z_next = self.A @ z.detach().numpy() + self.B @ action
        # 用 decoder 解码回状态空间
        state_next = self.decoder(torch.tensor(z_next, dtype=torch.float32)).detach().numpy()
        return state_next

# 环境和模型配置
NUM_OBS = 47  # PPO 观测维度
NUM_ACTIONS = 12  # 动作维度
SIMULATION_DT = 0.002  # 仿真步长
CONTROL_DECIMATION = 5  # 控制间隔
STEPS_PER_EPISODE = 100  # 每回合步数
NUM_EPISODES = 10  # 回合数（微调）

# 加载 MuJoCo 环境（沙地，摩擦 0.9）
env = gym.make("Go2Sim-v0", ground_type="sand", friction=0.9)  # [2025-04-20, 02:35]
model = mujoco.MjModel.from_xml_path("go2.xml")
data = mujoco.MjData(model)

# 加载硬地的 PPO 模型
ppo = PPO.load("model_10000.pt")  # [2025-04-15, 03:59]

# 加载硬地的 encoder 和 decoder
encoder = nn.Sequential(
    nn.Linear(40, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 128)
)
decoder = nn.Sequential(
    nn.Linear(128, 256),
    nn.ReLU(),
    nn.Linear(256, 512),
    nn.ReLU(),
    nn.Linear(512, 40)
)
encoder.load_state_dict(torch.load("models_hard/encoder_hard_traj_0.pth"))
decoder.load_state_dict(torch.load("models_hard/decoder_hard_traj_0.pth"))

# 初始化 Dyna 模型
dyna_model = KoopmanDynamicsModel(encoder=encoder, decoder=decoder)

# 从状态构造 PPO 观测（参考 collect_g1_koopman_data.py，[2025-04-20, 15:35]）
def get_ppo_obs(state, action, counter):
    q = state[16:28]
    dq = state[28:40]
    omega = state[10:13] * 0.25  # 角速度缩放（[2025-04-15, 10:18]）
    target_vel = state[13:16]
    phase = (counter * SIMULATION_DT) % 1.0
    sin_phase = np.sin(2 * np.pi * phase)
    cos_phase = np.cos(2 * np.pi * phase)
    obs = np.zeros(NUM_OBS, dtype=np.float32)
    obs[:3] = omega
    obs[3:6] = state[3:6]  # 简化：用 orientation 代替 gravity_orientation
    obs[6:9] = target_vel
    obs[9:21] = q
    obs[21:33] = dq
    obs[33:45] = action
    obs[45:47] = [sin_phase, cos_phase]
    return obs

# 用 Dyna 模型生成伪数据
def generate_pseudo_data(dyna_model, state, action, num_steps=5):
    pseudo_states = []
    pseudo_actions = []
    x = state
    curr_action = action
    for _ in range(num_steps):
        x_next = dyna_model.predict(x, curr_action)
        pseudo_states.append(x_next)
        pseudo_actions.append(curr_action)
        x = x_next
        ppo_obs = get_ppo_obs(x_next, curr_action, 0)
        obs_tensor = torch.from_numpy(ppo_obs).unsqueeze(0)
        curr_action = ppo.policy.predict(obs_tensor)[0].numpy()
    return np.array(pseudo_states), np.array(pseudo_actions)

# 奖励函数（鼓励前进）
def compute_reward(state):
    lin_vel_x = state[7]
    return lin_vel_x - 0.01 * np.linalg.norm(state[28:40])

# 主循环：一边拟合 Koopman，一边调优 PPO
real_buffer = []
pseudo_buffer = []
counter = 0
traj_data = []  # 收集沙地轨迹数据

for episode in range(NUM_EPISODES):
    mujoco.mj_resetData(model, data)
    state = np.zeros(40, dtype=np.float32)
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

        # 仿真一步
        data.ctrl[:12] = action
        mujoco.mj_step(model, data)

        # 构造 full_obs（52 维，参考 collect_g1_koopman_data.py）
        init_state = np.concatenate([
            data.qpos[:3],
            data.qpos[3:7],
            data.qvel[:3],
            data.qvel[3:6]
        ])
        target_vel = state[13:16]
        q = data.qpos[7:19]
        dq = data.qvel[6:18]
        tau = action
        full_obs = np.concatenate([init_state, target_vel, q, dq, tau])
        episode_data.append(full_obs)

        # 更新状态（用真实仿真状态）
        state[:13] = init_state
        state[13:16] = target_vel
        state[16:28] = q
        state[28:40] = dq

        # 计算奖励
        reward = compute_reward(state)
        real_buffer.append((state, action, reward, state))
        counter += 1

    # 收集沙地轨迹数据
    traj_data.append(np.array(episode_data))
    print(f"Episode {episode + 1}/{NUM_EPISODES}: Collected sand data")

    # 拟合新的 Koopman 模型（每 5 个 episode 更新一次）
    if (episode + 1) % 5 == 0:
        traj_data_array = np.vstack(traj_data[-5:])
        dyna_model.fit(traj_data_array)
        print(f"Updated Koopman model with {len(traj_data[-5:])} trajectories")

        # 用新的 Koopman 模型生成伪数据
        for i in range(len(real_buffer[-STEPS_PER_EPISODE:])):
            s, a, r, _ = real_buffer[-STEPS_PER_EPISODE + i]
            pseudo_states, pseudo_actions = generate_pseudo_data(dyna_model, s, a, num_steps=5)
            for j in range(len(pseudo_states)):
                pseudo_r = compute_reward(pseudo_states[j])
                pseudo_buffer.append((pseudo_states[j], pseudo_actions[j], pseudo_r, 
                                     pseudo_states[j+1] if j+1 < len(pseudo_states) else pseudo_states[j]))

    # 微调 PPO
    for step in range(100):  # 每 episode 微调 100 步
        if np.random.rand() < 0.5 and real_buffer:
            batch = [real_buffer[np.random.randint(len(real_buffer))]]
        elif pseudo_buffer:
            batch = [pseudo_buffer[np.random.randint(len(pseudo_buffer))]]
        else:
            continue

        obs = batch[0][0]
        act = batch[0][1]
        rew = batch[0][2]
        next_obs = batch[0][3]
        ppo.learn(total_timesteps=1, reset_num_timesteps=False, 
                  tb_log_name="ppo_tune_sand", callback=None)

# 保存调优后的 PPO 和 Koopman 模型
ppo.save("ppo_tuned_sand.pt")
torch.save(dyna_model.A, "A_sand.pt")
torch.save(dyna_model.B, "B_sand.pt")
print("PPO tuning and Koopman model fitting completed.")