from carla_env import CarlaEnv
import numpy as np

env = CarlaEnv()

try:
    obs, _ = env.reset()

    total_reward = 0
    for i in range(200):
        action = np.array([0.5, 0.0])

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        
        speed, distance, angle = obs
        print(f"Step {i}: speed={speed:.1f}, angle={angle:.1f}, reward={reward:.1f}, total={total_reward:.1f}")
        
        if terminated or truncated:
            print(f"Episode terminé ! Raison: {'collision' if terminated else 'timeout'}")
            break

finally:
    env.close()