import argparse
import math
import random
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Try to import CARLA; if not available, we'll provide a dummy env for quick testing
USE_DUMMY_ENV = False
try:
    import carla
except Exception:
    carla = None
    USE_DUMMY_ENV = True

# Stable Baselines3
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
except Exception:
    PPO = None


# ----------------------------- Utilities ---------------------------------

def clamp(x, a, b):
    return max(a, min(b, x))


def vec_length(v):
    return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z) if hasattr(v, 'x') else np.linalg.norm(v)


# ----------------------------- CarlaEnv ---------------------------------
class CarlaEnv(gym.Env):
    """Gym wrapper for a simplified V0 Carla environment.
    Observation: vector [speed, heading_error, dist_ahead, ttc]
    Action: Discrete(9) => throttle/brake x steering
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, host='127.0.0.1', port=2000, town='Town03', max_episode_steps=800):
        super().__init__()
        self.host = host
        self.port = port
        self.town = town
        self.max_episode_steps = max_episode_steps

        # Observation: speed (m/s), heading_error (rad), dist_ahead (m), ttc (s)
        obs_low = np.array([0.0, -math.pi, 0.0, 0.0], dtype=np.float32)
        obs_high = np.array([50.0, math.pi, 200.0, 100.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Action space: 3 throttle (brake, coast, accel) x 3 steering (left, straight, right)
        self.action_space = spaces.Discrete(9)

        # Internal variables
        self.client = None
        self.world = None
        self.ego = None
        self.collision_sensor = None
        self._seed = None
        self.episode_step = 0
        self.collision = False

        # Connect to CARLA if available
        if carla is not None:
            try:
                self.client = carla.Client(self.host, self.port)
                self.client.set_timeout(5.0)
                self.world = self.client.load_world(self.town)
                settings = self.world.get_settings()
                settings.fixed_delta_seconds = 0.05
                settings.synchronous_mode = False
                self.world.apply_settings(settings)
            except Exception as e:
                print('Warning: could not connect to CARLA:', e)
                raise

    # -------------------- CARLA helpers --------------------
    def _spawn_ego(self):
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
        spawn_points = self.world.get_map().get_spawn_points()
        spawn_point = random.choice(spawn_points)
        vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
        if vehicle is None:
            raise RuntimeError('Failed to spawn ego vehicle')
        self.ego = vehicle
        self.ego.set_simulate_physics(True)

        # collision sensor (simple)
        collision_bp = blueprint_library.find('sensor.other.collision')
        col_sensor = self.world.try_spawn_actor(collision_bp, carla.Transform(), attach_to=self.ego)
        self.collision_sensor = col_sensor
        if col_sensor:
            col_sensor.listen(lambda event: self._on_collision(event))

    def _on_collision(self, event):
        self.collision = True

    def _destroy_actors(self):
        try:
            if self.collision_sensor is not None:
                self.collision_sensor.stop()
                self.collision_sensor.destroy()
                self.collision_sensor = None
        except Exception:
            pass
        try:
            if self.ego is not None:
                self.ego.destroy()
                self.ego = None
        except Exception:
            pass

    def _get_speed(self):
        vel = self.ego.get_velocity()
        speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
        return speed

    def _get_heading_error(self):
        # Simple heading error to next waypoint (approx) using velocity vector
        v = self.ego.get_transform().rotation.yaw
        # convert to radians
        yaw_rad = math.radians(v)
        # desired heading: follow forward vector (approx 0)
        # For simplicity, heading error = 0 here (requires waypoints for real)
        return 0.0

    def _distance_to_vehicle_ahead(self, max_dist=200.0, fov_deg=30.0):
        # Approximate: search nearby vehicles and pick the nearest in front cone
        ego_tf = self.ego.get_transform()
        ego_loc = ego_tf.location
        ego_yaw = math.radians(ego_tf.rotation.yaw)
        actors = self.world.get_actors().filter('vehicle.*')
        min_dist = max_dist
        for a in actors:
            if a.id == self.ego.id:
                continue
            vec = a.get_location() - ego_loc
            dist = math.sqrt(vec.x**2 + vec.y**2 + vec.z**2)
            if dist > max_dist:
                continue
            angle = math.degrees(math.atan2(vec.y, vec.x) - ego_yaw)
            angle = (angle + 180) % 360 - 180
            if abs(angle) <= fov_deg:
                if dist < min_dist and vec.x > 0:  # in front (approx)
                    min_dist = dist
        return float(min_dist)

    def _estimate_ttc(self, dist_ahead, rel_speed):
        if rel_speed <= 0.1:
            return 100.0
        return dist_ahead / rel_speed

    # -------------------- Gym API --------------------
    def reset(self):
        self.episode_step = 0
        self.collision = False
        # Destroy previous actors
        if carla is not None:
            try:
                self._destroy_actors()
            except Exception:
                pass
            # spawn ego
            self._spawn_ego()
        else:
            # Dummy env: init state
            self._dummy_state = np.array([0.0, 0.0, 100.0, 100.0], dtype=np.float32)

        return self._get_obs()

    def step(self, action):
        # map discrete action to throttle/brake/steer
        throttle_choice = (action // 3)  # 0,1,2 => brake/coast/accel
        steer_choice = (action % 3)  # 0,1,2 => left, straight, right

        if carla is not None:
            # apply control
            control = carla.VehicleControl()
            if throttle_choice == 0:
                control.throttle = 0.0
                control.brake = 0.6
            elif throttle_choice == 1:
                control.throttle = 0.0
                control.brake = 0.0
            else:
                control.throttle = 0.6
                control.brake = 0.0

            if steer_choice == 0:
                control.steer = -0.2
            elif steer_choice == 1:
                control.steer = 0.0
            else:
                control.steer = 0.2

            self.ego.apply_control(control)
            # step simulation small sleep (if synchronous would tick server)
            time.sleep(0.05)

        else:
            # Dummy dynamics: simple kinematics
            speed, heading, dist_ahead, ttc = self._dummy_state
            if throttle_choice == 0:
                speed = max(0.0, speed - 1.0)
            elif throttle_choice == 2:
                speed = speed + 1.0
            # steering doesn't change observation in dummy
            dist_ahead = max(0.0, dist_ahead - speed * 0.05)
            rel_speed = max(0.1, speed - 10.0)  # pretend other vehicle speed
            ttc = dist_ahead / rel_speed if rel_speed > 0.1 else 100.0
            self._dummy_state = np.array([speed, heading, dist_ahead, ttc], dtype=np.float32)

        obs = self._get_obs()
        reward, info = self._compute_reward(obs, action)

        self.episode_step += 1
        done = False
        if self.collision:
            done = True
            info['collision'] = True
        if self.episode_step >= self.max_episode_steps:
            done = True
        return obs, reward, done, info

    def _get_obs(self):
        if carla is None:
            return self._dummy_state.astype(np.float32)
        # real obs
        speed = self._get_speed()
        heading_error = self._get_heading_error()
        dist_ahead = self._distance_to_vehicle_ahead()
        # approx relative speed: assume lead vehicle speed 10 m/s for now
        rel_speed = max(0.1, speed - 10.0)
        ttc = self._estimate_ttc(dist_ahead, rel_speed)
        obs = np.array([speed, heading_error, dist_ahead, ttc], dtype=np.float32)
        return obs

    def _compute_reward(self, obs, action):
        speed, heading_error, dist_ahead, ttc = obs
        reward = 0.0
        info = {}
        # reward for moving forward: proportional to speed
        reward += 0.1 * speed
        # penalty for collision
        if self.collision:
            reward -= 100.0
        # penalty for large heading deviation
        reward -= 0.5 * abs(heading_error)
        # penalty for being too slow
        if speed < 1.0:
            reward -= 0.5
        # penalty for tailgating
        if dist_ahead < 5.0:
            reward -= 5.0
        # encourage larger TTC
        reward += clamp(ttc / 10.0, -1.0, 1.0)
        return reward, info

    def render(self, mode='human'):
        # optional: could attach pygame/carla camera display
        pass

    def close(self):
        if carla is not None:
            try:
                self._destroy_actors()
            except Exception:
                pass


# ----------------------------- DummyEnv ---------------------------------
class DummyEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=np.array([0, -math.pi, 0, 0]), high=np.array([50, math.pi, 200, 100]), dtype=np.float32)
        self.action_space = spaces.Discrete(9)
        self.state = np.array([0.0, 0.0, 100.0, 100.0], dtype=np.float32)
        self.step_count = 0

    def reset(self):
        self.state = np.array([0.0, 0.0, 100.0, 100.0], dtype=np.float32)
        self.step_count = 0
        return self.state

    def step(self, action):
        throttle_choice = (action // 3)
        if throttle_choice == 0:
            self.state[0] = max(0.0, self.state[0] - 1.0)
        elif throttle_choice == 2:
            self.state[0] = self.state[0] + 1.0
        self.state[2] = max(0.0, self.state[2] - self.state[0] * 0.05)
        self.state[3] = self.state[2] / max(0.1, self.state[0] - 10.0) if (self.state[0] - 10.0) != 0 else 100.0
        reward = 0.1 * self.state[0]
        done = False
        self.step_count += 1
        if self.step_count > 500:
            done = True
        return self.state.astype(np.float32), reward, done, {}

    def render(self, mode='human'):
        pass

    def close(self):
        pass


# ----------------------------- Training script --------------------------

def make_env(args):
    if USE_DUMMY_ENV or carla is None:
        return DummyEnv()
    else:
        return CarlaEnv(host=args.host, port=args.port, town=args.town, max_episode_steps=args.max_episode_steps)


def train(args):
    if PPO is None:
        raise RuntimeError('Stable Baselines3 not installed. Please pip install stable-baselines3')

    env_fn = lambda: make_env(args)
    vec_env = DummyVecEnv([env_fn])

    policy_kwargs = dict(
        net_arch=[dict(pi=[128, 128], vf=[128, 128])]
    )

    model = PPO('MlpPolicy', vec_env, verbose=1, learning_rate=3e-4, n_steps=2048, batch_size=64, policy_kwargs=policy_kwargs)

    print('Starting training...')
    model.learn(total_timesteps=args.train_timesteps)
    model.save(args.save_path)
    print('Training complete. Model saved to', args.save_path)


def evaluate(model_path, args, n_episodes=10):
    # load model and run evaluation
    if PPO is None:
        raise RuntimeError('Stable Baselines3 not installed.')
    env = make_env(args)
    model = PPO.load(model_path)
    stats = {'rewards': [], 'collisions': 0}
    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        total_r = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, info = env.step(int(action))
            total_r += r
            if info.get('collision', False):
                stats['collisions'] += 1
        stats['rewards'].append(total_r)
    print('Evaluation:', stats)
    env.close()


# ------------------------------- CLI -----------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--town', type=str, default='Town03')
    parser.add_argument('--train-timesteps', type=int, default=100000)
    parser.add_argument('--save-path', type=str, default='models/v0_ppo')
    parser.add_argument('--max-episode-steps', type=int, default=800)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--model-path', type=str, default='models/v0_ppo.zip')
    args = parser.parse_args()

    if args.eval:
        evaluate(args.model_path, args)
    else:
        train(args)
