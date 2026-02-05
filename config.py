"""
Const for CARLA configuration
"""

# CARLA
CARLA_PATH = "D:\Code\Carla\CARLA_0.9.15\WindowsNoEditor"
CARLA_PORT = 2000
CARLA_MAP = "Town01"
CARLA_HOST = "localhost"

# RL
ALGO = "PPO"
TOTAL_TIMESTEPS = 100_000
SAVE_FREQ = 10_000

# Environnement
TARGET_SPEED = 30
MAX_EPISODE_STEPS = 500

# Reward
REWARDS = {
    "forward": 1.0,
    "collision": -100.0,
    "lane_invasion": -10.0,
    "timeout": -50.0,
}
