from carla_env import CarlaEnv
import time

env = CarlaEnv()

try:
    obs, info = env.reset()
    print(f"Observation initiale: {obs}")

    for i in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"Step {i}: speed={obs[0]:.1f} km/h, reaward={reward:.1f}")

        if terminated:
            print("Collision ! Episode terminé.")
            break
        if truncated:
            print("Timeout ! Episode terminé.")
            break

finally:
    env.close()
