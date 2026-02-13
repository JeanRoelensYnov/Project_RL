from carla_env import CarlaEnv
from PPO import PPO

env = CarlaEnv()

try:
    env = CarlaGymEnv()
    train(env)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    agent = DQN.DQNAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=DEVICE
    )
finally:
    env.close()