from carla_env import CarlaEnv
from PPO import PPO

env = CarlaEnv()

try:
    ppo = PPO(env, lr=1e-4, n_steps=2048)
    ppo.learn(total_timesteps=1_000_000)  
finally:
    env.close()