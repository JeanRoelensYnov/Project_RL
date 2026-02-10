from carla_env import CarlaEnv
from PPO import PPO

env = CarlaEnv()

try:
    ppo = PPO(env, n_steps=512)
    ppo.learn(total_timesteps=5000)

    print("Entraînement terminé !")
finally:
    env.close()