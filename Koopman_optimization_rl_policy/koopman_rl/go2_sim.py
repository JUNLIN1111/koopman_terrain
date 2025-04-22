import numpy as np
from gymnasium import Env, spaces
from typing import Optional
from scipy.spatial.transform import Rotation
import mujoco

class Go2Sim(Env):
    def __init__(self, render_mode: Optional[str] = None):
        # Load MuJoCo model
        model_path = "D:\\koopman_robot\\Koopman_optimization_rl_policy\\resourse\\go2\\scene_sand.xml"
        try:
            self.model = mujoco.MjModel.from_xml_path(model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load MuJoCo model from {model_path}: {e}")

        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.viewer = None

        # Simulation parameters
        self.max_episode_length = 1000  # Maximum steps per episode
        self.episode_length = 0
        self.gravity_vec = self.model.opt.gravity.copy()

        # Robot configuration
        self.n_j = self.model.nu  # Number of actuators (12)
        self.n_q = self.model.nq  # Position state dimension
        self.n_v = self.model.nv  # Velocity state dimension
        self.default_height = 0.45  # Target base height

        # Observation space (36 dimensions: joint pos, joint vel, euler, lin vel, ang vel, gravity)
        obs_dim = 36
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Action space
        if self.model.actuator_ctrlrange is not None:
            act_low, act_high = self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1]
        else:
            act_low, act_high = -np.ones(self.n_j), np.ones(self.n_j)
        self.action_space = spaces.Box(act_low, act_high, dtype=np.float32)

        # Observation scaling (inspired by LeggedRobot)
        self.obs_scales = {
            'lin_vel': 2.0,
            'ang_vel': 0.25,
            'dof_pos': 1.0,
            'dof_vel': 0.05
        }
        self.clip_obs = 100.0  # Clip observations to prevent numerical issues

        # Reward configuration
        self.reward_scales = {
            'forward': 2.0,
            'lateral': -0.5,
            'pitch': -0.3,
            'height': -0.4,
            'symmetry': -0.2,
            'termination': -10.0
        }
        self.only_positive_rewards = False

        # Buffers for tracking
        self.last_actions = np.zeros(self.n_j, dtype=np.float32)
        self.last_dof_vel = np.zeros(self.n_v, dtype=np.float32)
        self.episode_sums = {key: 0.0 for key in self.reward_scales}

        # Noise configuration
        self.add_noise = True
        self.noise_scale = 0.01

    def step(self, action):
        # Clip actions
        clip_actions = 1.0  # Similar to LeggedRobot's clip_actions
        action = np.clip(action, -clip_actions, clip_actions)
        self.last_actions[:] = action

        # Apply control and simulate
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)
        self.episode_length += 1

        # Update observations and rewards
        self._post_physics_step()

        # Clip observations
        obs = np.clip(self._get_obs(), -self.clip_obs, self.clip_obs)

        return obs, self.rew_buf, self.reset_buf, False, self._get_info()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        # Reset simulation
        mujoco.mj_resetData(self.model, self.data)

        # Initialize states
        self.data.qpos[2] = self.default_height  # Set base height
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # Neutral quaternion
        self.data.qvel[:] = 0.0  # Zero velocities
        mujoco.mj_forward(self.model, self.data)

        # Reset buffers
        self.episode_length = 0
        self.last_actions[:] = 0.0
        self.last_dof_vel[:] = 0.0
        self.episode_sums = {key: 0.0 for key in self.reward_scales}
        self.rew_buf = 0.0
        self.reset_buf = False

        return self._get_obs(), {}

    def _post_physics_step(self):
        # Update states
        self.base_pos = self.data.qpos[:3].copy()
        self.base_quat = self.data.qpos[3:7].copy()
        rotation = Rotation.from_quat([self.base_quat[1], self.base_quat[2], self.base_quat[3], self.base_quat[0]])
        self.rpy = rotation.as_euler('xyz', degrees=False)
        self.base_lin_vel = self.data.qvel[:3].copy()
        self.base_ang_vel = self.data.qvel[3:6].copy()
        self.dof_pos = self.data.qpos[7:].copy()
        self.dof_vel = self.data.qvel[6:].copy()

        # Check termination
        self._check_termination()

        # Compute rewards
        self._compute_reward()

        # Compute observations
        self._compute_observations()

    def _get_obs(self):
        # Compute projected gravity (inspired by LeggedRobot)
        # rotation = Rotation.from_quat([self.base_quat[1], self.base_quat[2], self.base_quat[3], self.base_quat[0]])
        # projected_gravity = rotation.inv().apply(self.gravity_vec)

        # Observation vector
        obs = np.concatenate([
            self.base_lin_vel * self.obs_scales['lin_vel'],  # 3
            self.base_ang_vel * self.obs_scales['ang_vel'],  # 3
            1,1,1,                               # 3
            self.dof_pos * self.obs_scales['dof_pos'],       # 12
            self.dof_vel * self.obs_scales['dof_vel'],       # 12
            
        ])

        # Add noise if enabled
        if self.add_noise:
            obs += (2 * np.random.rand(*obs.shape) - 1) * self.noise_scale

        return obs.astype(np.float32)

    def _compute_observations(self):
        self.obs_buf = self._get_obs()

    def _compute_reward(self):
        self.rew_buf = 0.0
        rewards = {}

        # Forward velocity
        rewards['forward'] = self.reward_scales['forward'] * self.base_lin_vel[0]

        # Lateral movement penalty
        rewards['lateral'] = self.reward_scales['lateral'] * (self.base_lin_vel[1] ** 2)

        # Pitch penalty
        rewards['pitch'] = self.reward_scales['pitch'] * (self.rpy[1] ** 2)

        # Height penalty
        rewards['height'] = self.reward_scales['height'] * ((self.base_pos[2] - self.default_height) ** 2)

        # Symmetry penalty (front and rear joint differences)
        front_diff = self.dof_pos[:3] - self.dof_pos[3:6]
        rear_diff = self.dof_pos[6:9] - self.dof_pos[9:12]
        rewards['symmetry'] = self.reward_scales['symmetry'] * (np.sum(front_diff ** 2) + np.sum(rear_diff ** 2))

        # Aggregate rewards
        for name, rew in rewards.items():
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # Clip rewards if needed
        if self.only_positive_rewards:
            self.rew_buf = max(0.0, self.rew_buf)

        # Termination reward
        if self.reset_buf:
            term_rew = self._reward_termination()
            self.rew_buf += term_rew
            self.episode_sums['termination'] += term_rew

    def _reward_termination(self):
        return self.reward_scales['termination'] if self.reset_buf else 0.0

    def _check_termination(self):
        # Height check
        height_condition = self.base_pos[2] < 0.15

        # Tilt check
        tilt_condition = abs(self.rpy[0]) > 0.8 or abs(self.rpy[1]) > 0.8

        # Time-out check
        time_out = self.episode_length > self.max_episode_length

        self.reset_buf = height_condition or tilt_condition or time_out

    def _get_info(self):
        info = {
            'episode_length': self.episode_length,
            'episode_rewards': {f'rew_{k}': v / max(1, self.episode_length) for k, v in self.episode_sums.items()}
        }
        if time_out := (self.episode_length > self.max_episode_length):
            info['time_out'] = time_out
        return info

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
        elif self.render_mode == "rgb_array":
            width, height = 480, 480
            scene_option = mujoco.MjvOption()
            camera = mujoco.MjvCamera()
            context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
            img = np.zeros((height, width, 3), dtype=np.uint8)
            mujoco.mjv_updateScene(self.model, self.data, scene_option, None, camera, mujoco.mjtCatBit.mjCAT_ALL, context)
            mujoco.mjr_render((0, 0, width, height), context, img)
            return img
        else:
            raise NotImplementedError("Unsupported render mode.")

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None