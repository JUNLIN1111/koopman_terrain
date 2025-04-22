import numpy as np
import os

#--------------- Junlin Wu 2025/4/16 ---------------#
# ========== G1 Robot Configuration Parameters ==========

# Joint PD gains [12 joints]
KPS = np.array([100, 100, 100, 150, 40, 40, 100, 100, 100, 150, 40, 40], dtype=np.float32)
KDS = np.array([2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2], dtype=np.float32)

# Default joint angles (standing pose)
DEFAULT_ANGLES = np.array([
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,  # Front-left leg joints
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0   # Front-right leg joints
], dtype=np.float32)

# Observation normalization scales
ANG_VEL_SCALE = 0.25     # Angular velocity scale factor
DOF_POS_SCALE = 1.0      # Joint position scale factor
DOF_VEL_SCALE = 0.05     # Joint velocity scale factor
ACTION_SCALE = 0.25      # Policy output scale factor
CMD_SCALE = np.array([2.0, 2.0, 0.25], dtype=np.float32)  # Command [x_vel, y_vel, yaw_vel]

# System parameters
NUM_ACTIONS = 12         # Action space dimension (12 joints)
NUM_OBS = 47             # Observation space dimension
SIMULATION_DT = 0.002    # Physics timestep (0.002 = 500Hz)
CONTROL_DECIMATION = 10  # Policy runs at 500Hz/10 = 50Hz
GAIT_PERIOD = 0.8        # Gait period for phase encoding

# Default command (desired [forward_vel, lateral_vel, yaw_vel])
DEFAULT_CMD = np.array([0.5, 0, 0], dtype=np.float32)

# File paths
BASE_PATH = r"D:\koopman_robot\Koopman_optimization_rl_policy"
POLICY_PATH = os.path.join(BASE_PATH, r"pretrain_model\g1_motion.pt")
XML_BASE_PATH = os.path.join(BASE_PATH, r"resourse\g1")
DATA_PATH = os.path.join(BASE_PATH, r"resourse\data")

# Valid ground types
GROUND_TYPES = ["hard", "sand", "soft"]

# ========== Helper Functions ==========
def get_gravity_orientation(quaternion):
    """Convert quaternion to gravity direction vector in body frame.
    Args:
        quaternion (np.array): [qw, qx, qy, qz] orientation quaternion
    Returns:
        np.array: 3D gravity vector in z-up coordinate system
    """
    qw, qx, qy, qz = quaternion
    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)  # x-component
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)  # y-component
    gravity_orientation[2] = 1 - 2 * (qw**2 + qz**2)  # z-component
    return gravity_orientation

def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Compute PD control torque.
    Args:
        target_q: Target joint positions
        q: Current joint positions
        kp: Proportional gains
        target_dq: Target joint velocities (typically zero)
        dq: Current joint velocities
        kd: Derivative gains
    Returns:
        np.array: Computed joint torques
    """
    return (target_q - q) * kp + (target_dq - dq) * kd

def get_xml_path(ground_type):
    """Get XML file path for the specified ground type.
    Args:
        ground_type (str): 'hard', 'sand', or 'soft'
    Returns:
        str: Path to the XML file
    Raises:
        ValueError: If ground_type is invalid
        FileNotFoundError: If XML file does not exist
    """
    if ground_type not in GROUND_TYPES:
        raise ValueError(f"Invalid ground_type: {ground_type}. Choose from {GROUND_TYPES}")
    xml_path = os.path.join(XML_BASE_PATH, f"{ground_type}_ground_scene.xml")
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"XML file not found: {xml_path}")
    return xml_path

def get_data_path(ground_type, filename_prefix="g1_koopman_data"):
    """Get data file path for the specified ground type.
    Args:
        ground_type (str): 'hard', 'sand', or 'soft'
        filename_prefix (str): Prefix for the data file
    Returns:
        str: Path to the data file
    """
    if ground_type not in GROUND_TYPES:
        raise ValueError(f"Invalid ground_type: {ground_type}. Choose from {GROUND_TYPES}")
    os.makedirs(DATA_PATH, exist_ok=True)
    return os.path.join(DATA_PATH, f"{filename_prefix}_{ground_type}.csv")

def get_plot_path(ground_type, filename_prefix="g1_trajectory"):
    """Get plot file path for the specified ground type.
    Args:
        ground_type (str): 'hard', 'sand', or 'soft'
        filename_prefix (str): Prefix for the plot file
    Returns:
        str: Path to the plot file
    """
    if ground_type not in GROUND_TYPES:
        raise ValueError(f"Invalid ground_type: {ground_type}. Choose from {GROUND_TYPES}")
    os.makedirs(DATA_PATH, exist_ok=True)
    return os.path.join(DATA_PATH, f"{filename_prefix}_{ground_type}.png")