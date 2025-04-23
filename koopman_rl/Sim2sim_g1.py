#--------------- Junlin Wu 2025/4/16 ---------------
#--------------- Sim to sim (deploy g1 walking legged gym to Mujoco) ---------------

import time
import mujoco
import mujoco.viewer
import numpy as np
import torch
from g1_utils import (
    get_gravity_orientation, pd_control, get_xml_path,
    KPS, KDS, DEFAULT_ANGLES, ANG_VEL_SCALE, DOF_POS_SCALE,
    DOF_VEL_SCALE, ACTION_SCALE, CMD_SCALE, NUM_ACTIONS,
    NUM_OBS, SIMULATION_DT, CONTROL_DECIMATION, GAIT_PERIOD,
    DEFAULT_CMD, POLICY_PATH
)

if __name__ == "__main__":
    # --------- Configuration ---------
    simulation_duration = 60.0  # Total simulation time (seconds)
    ground_type = "sand"        # Terrain type: 'hard', 'sand', 'soft'

    # File paths
    xml_path = get_xml_path(ground_type)

    # --------- Simulation Setup ---------
    action = np.zeros(NUM_ACTIONS, dtype=np.float32)
    target_dof_pos = DEFAULT_ANGLES.copy()
    obs = np.zeros(NUM_OBS, dtype=np.float32)
    counter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = SIMULATION_DT

    # Load policy
    policy = torch.jit.load(POLICY_PATH)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            tau = pd_control(target_dof_pos, d.qpos[7:], KPS, np.zeros_like(KDS), d.qvel[6:], KDS)
            d.ctrl[:] = tau
            mujoco.mj_step(m, d)

            counter += 1
            if counter % CONTROL_DECIMATION == 0:
                qj = d.qpos[7:]
                dqj = d.qvel[6:]
                quat = d.qpos[3:7]
                omega = d.qvel[3:6]

                qj = (qj - DEFAULT_ANGLES) * DOF_POS_SCALE
                dqj = dqj * DOF_VEL_SCALE
                gravity_orientation = get_gravity_orientation(quat)
                omega = omega * ANG_VEL_SCALE

                count = counter * SIMULATION_DT
                phase = count % GAIT_PERIOD / GAIT_PERIOD
                sin_phase = np.sin(2 * np.pi * phase)
                cos_phase = np.cos(2 * np.pi * phase)

                obs[:3] = omega
                obs[3:6] = gravity_orientation
                obs[6:9] = DEFAULT_CMD * CMD_SCALE 
                obs[9:9 + NUM_ACTIONS] = qj # position of joints
                obs[9 + NUM_ACTIONS:9 + 2 * NUM_ACTIONS] = dqj # velosity of joints
                obs[9 + 2 * NUM_ACTIONS:9 + 3 * NUM_ACTIONS] = action # last actions
                obs[9 + 3 * NUM_ACTIONS:9 + 3 * NUM_ACTIONS + 2] = np.array([sin_phase, cos_phase]) # represent the gait phase, using sin and cos of the phase

                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                action = policy(obs_tensor).detach().numpy().squeeze()
                target_dof_pos = action * ACTION_SCALE + DEFAULT_ANGLES

            viewer.sync()
            time_until_next_step = SIMULATION_DT - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)