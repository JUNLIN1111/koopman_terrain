#--------------- Collect Data for Koopman ---------------#
#--------------- Junlin Wu 2025/4/16 ---------------#
# Import necessary libraries
import time
import mujoco  # Physics engine for simulating robots
import numpy as np
import torch
import pandas as pd

# Import utility functions and constants from custom utils module
from g1_utils import (
    get_gravity_orientation, pd_control, get_xml_path, get_data_path, get_plot_path,
    KPS, KDS, DEFAULT_ANGLES, ANG_VEL_SCALE, DOF_POS_SCALE,
    DOF_VEL_SCALE, ACTION_SCALE, CMD_SCALE, NUM_ACTIONS,
    NUM_OBS, SIMULATION_DT, CONTROL_DECIMATION, GAIT_PERIOD,
    DEFAULT_CMD, POLICY_PATH
)

# Configuration parameters
ground_type = "hard"  # Environment type: options are 'hard', 'sand', 'soft'
XML_PATH = get_xml_path(ground_type)  # Path to the corresponding MuJoCo XML model file
OUTPUT_CSV = get_data_path(ground_type)  # Output CSV file path for collected data
PLOT_PATH = get_plot_path(ground_type)  # Output path for saving trajectory plot
NUM_EPISODES = 100  # Number of trajectories to collect
STEPS_PER_EPISODE = 100  # Number of steps per trajectory

# Initialize MuJoCo simulation environment
model = mujoco.MjModel.from_xml_path(XML_PATH)  # Load model from XML
data = mujoco.MjData(model)  # Create simulation data structure
model.opt.timestep = SIMULATION_DT  # Set simulation time step

# Load the trained PPO policy (TorchScript format)
policy = torch.jit.load(POLICY_PATH)

# Observation function that returns both full state and PPO-compatible observation vector
def get_observation(data, action, counter):

    """
    Collects and formats the observation data from the simulator.
    Returns:
    - Full 52-dim vector matching structure from output9.csv:
      [body position (3), orientation (4), linear vel (3), angular vel (3),
       target velocity (3), joint positions (12), joint velocities (12), torques (12)]
    - PPO observation vector used by the neural policy (structured for agent input)
    """

    # Initial base state of the robot
    init_state = np.concatenate([
        data.qpos[:3],      # Base position (x, y, z)
        data.qpos[3:7],     # Base orientation (quaternion: w, x, y, z)
        data.qvel[:3],      # Base linear velocity
        data.qvel[3:6]      # Base angular velocity
    ])

    # Target command velocity (used as input and in saved data)
    end_state = DEFAULT_CMD * CMD_SCALE

    # Joint positions and velocities
    q = data.qpos[7:19]        # 12 DOF joint angles
    dq = data.qvel[6:18]       # 12 DOF joint velocities

    # Compute PD control torque for each joint
    tau = pd_control(target_dof_pos, q, KPS, np.zeros_like(KDS), dq, KDS)

    # Normalize joint positions and velocities
    q_scaled = (q - DEFAULT_ANGLES) * DOF_POS_SCALE
    dq_scaled = dq * DOF_VEL_SCALE

    # Compute gravity orientation from quaternion
    gravity_orientation = get_gravity_orientation(data.qpos[3:7])

    # Angular velocity scaled
    omega = data.qvel[3:6] * ANG_VEL_SCALE

    # Gait phase features (for cyclic motion representation)
    count = counter * SIMULATION_DT
    phase = count % GAIT_PERIOD / GAIT_PERIOD
    sin_phase = np.sin(2 * np.pi * phase)
    cos_phase = np.cos(2 * np.pi * phase)

    # Build PPO observation (used by neural network)
    obs = np.zeros(NUM_OBS, dtype=np.float32)
    obs[:3] = omega # angle of base 
    obs[3:6] = gravity_orientation # gravity_orientation
    obs[6:9] = DEFAULT_CMD * CMD_SCALE # # Command [x_vel, y_vel, yaw_vel]
    obs[9:9 + NUM_ACTIONS] = q_scaled 
    obs[9 + NUM_ACTIONS:9 + 2 * NUM_ACTIONS] = dq_scaled
    obs[9 + 2 * NUM_ACTIONS:9 + 3 * NUM_ACTIONS] = action
    obs[9 + 3 * NUM_ACTIONS:9 + 3 * NUM_ACTIONS + 2] = np.array([sin_phase, cos_phase])

    # Return both full observation and PPO input
    return np.concatenate([init_state, end_state, q, dq, tau]), obs

# Main data collection loop
all_data = []
action = np.zeros(NUM_ACTIONS, dtype=np.float32)  # Initial action vector (zero torque)
target_dof_pos = DEFAULT_ANGLES.copy()            # Initial target joint angles
counter = 0

for episode in range(NUM_EPISODES):
    mujoco.mj_resetData(model, data)  # Reset simulation to initial state
    episode_data = []
    counter = 0

    for step in range(STEPS_PER_EPISODE):
        # Collect observation for current step
        full_obs, ppo_obs = get_observation(data, action, counter)

        # Use PPO policy to compute new action at control interval
        if counter % CONTROL_DECIMATION == 0:
            obs_tensor = torch.from_numpy(ppo_obs).unsqueeze(0)  # Add batch dimension
            action = policy(obs_tensor).detach().numpy().squeeze()  # Predict action
            target_dof_pos = action * ACTION_SCALE + DEFAULT_ANGLES  # Rescale to joint space

        # Apply PD control torques
        tau = pd_control(target_dof_pos, data.qpos[7:19], KPS, np.zeros_like(KDS), data.qvel[6:18], KDS)
        data.ctrl[:12] = tau

        # Step the simulation forward
        mujoco.mj_step(model, data)
        counter += 1
        episode_data.append(full_obs)  # Store full observation

    # Store episode data
    all_data.append(np.array(episode_data))
    print(f"Episode {episode + 1}/{NUM_EPISODES} collected for {ground_type} ground.")

# Stack all episodes into one big array
all_data = np.vstack(all_data)

# Create DataFrame from collected data
columns = (
    [f"init_state_{i}" for i in range(13)] +
    [f"end_state_{i}" for i in range(3)] +
    [f"q_{i}" for i in range(12)] +
    [f"dq_{i}" for i in range(12)] +
    [f"tau_{i}" for i in range(12)]
)
df = pd.DataFrame(all_data, columns=columns)

# Save data to CSV file
df.to_csv(OUTPUT_CSV, index=False)
print(f"Data saved to {OUTPUT_CSV}, shape: {all_data.shape}")

# Plot and save visualization of the first trajectory (joint 0)
import matplotlib.pyplot as plt
traj = all_data[:STEPS_PER_EPISODE]
plt.plot(traj[:, 16], label="Joint 0 Position")  # Index 16 = q_0
plt.plot(traj[:, 28], label="Joint 0 Velocity")  # Index 28 = dq_0
plt.plot(traj[:, 40], label="Joint 0 Torque")    # Index 40 = tau_0
plt.legend()
plt.title(f"Sample G1 Trajectory on {ground_type} Ground")
plt.savefig(PLOT_PATH)
plt.close()
print(f"Trajectory plot saved to {PLOT_PATH}")
