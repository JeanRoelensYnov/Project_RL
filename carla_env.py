import os

import carla
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import queue
import math
import random
import time
import DQN
import torch


IM_WIDTH = 640
IM_HEIGHT = 480
RL_HEIGHT = 84
RL_WIDTH = 84
FOV = 90
LIDAR_RANGE = 50


class SensorManager:


    def __init__(self, world, vehicle):
        self.world = world
        self.vehicle = vehicle
        self.actor_list = []
        self.queues = {}
        self.bp_lib = world.get_blueprint_library()
        self.observation_space = spaces.Dict({
            "rgb": spaces.Box(0, 255, (RL_HEIGHT, RL_WIDTH, 3), dtype=np.uint8),
            "depth": spaces.Box(0, 1, (RL_HEIGHT, RL_WIDTH, 1), dtype=np.float32),
            "lidar_bev": spaces.Box(0, 255, (RL_HEIGHT, RL_WIDTH, 1), dtype=np.uint8),
            "state": spaces.Box(-np.inf, np.inf, (9,), dtype=np.float32)
        })

        cam_tf = carla.Transform(carla.Location(x=1.5, z=2.4))
        lidar_tf = carla.Transform(carla.Location(x=0, z=2.4))

        self._spawn_camera('rgb', 'sensor.camera.rgb', cam_tf)
        self._spawn_camera('depth', 'sensor.camera.depth', cam_tf)
        self._spawn_lidar('lidar', lidar_tf)
        self._spawn_sensor('collision', 'sensor.other.collision', carla.Transform())
        self._spawn_sensor('gnss', 'sensor.other.gnss', carla.Transform())

    def _spawn_camera(self, name, type_id, transform):
        bp = self.bp_lib.find(type_id)
        bp.set_attribute('image_size_x', str(IM_WIDTH))
        bp.set_attribute('image_size_y', str(IM_HEIGHT))
        bp.set_attribute('fov', str(FOV))
        self._spawn_sensor(name, bp, transform)

    def _spawn_lidar(self, name, transform):
        bp = self.bp_lib.find('sensor.lidar.ray_cast')
        bp.set_attribute('range', str(LIDAR_RANGE))
        bp.set_attribute('channels', '32')
        bp.set_attribute('points_per_second', '100000')
        bp.set_attribute('rotation_frequency', '20')
        self._spawn_sensor(name, bp, transform)

    def _spawn_sensor(self, name, bp_or_id, transform):

        if isinstance(bp_or_id, str):
            bp = self.bp_lib.find(bp_or_id)
        else:
            bp = bp_or_id

        sensor = self.world.spawn_actor(bp, transform, attach_to=self.vehicle)
        q = queue.Queue()
        sensor.listen(q.put)
        self.queues[name] = q
        self.actor_list.append(sensor)

    def get_data(self):

        data = {}


        v = self.vehicle.get_velocity()
        t = self.vehicle.get_transform()
        data['speed_kmh'] = (3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2))
        data['location'] = t.location
        data['transform'] = t



        for name, q in self.queues.items():
            if name == 'collision':
                data['collision'] = False
                while not q.empty():
                    _ = q.get()
                    data['collision'] = True
                continue

            try:
                sensor_data = q.get(timeout=2.0)

                if name == 'rgb':
                    data['rgb'] = self._parse_image(sensor_data)
                elif name == 'depth':
                    data['depth'] = self._parse_depth(sensor_data)
                elif name == 'seg':
                    data['seg'] = self._parse_seg(sensor_data)
                elif name == 'lidar':
                    data['lidar'] = self._parse_lidar(sensor_data)
                elif name == 'gnss':
                    data['gnss'] = np.array([sensor_data.latitude, sensor_data.longitude, sensor_data.altitude])

            except queue.Empty:
                print(f"Warning: Sensor {name} missing data this tick.")

        return data

    def _parse_image(self, image):
        i = np.array(image.raw_data)
        i = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
        return i[:, :, :3]
    def _parse_depth(self, image):
        i = np.array(image.raw_data).astype(np.float32)
        i = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
        depth = (i[:, :, 2] + i[:, :, 1] * 256 + i[:, :, 0] * 256 * 256) / (256 ** 3 - 1) * 1000
        return depth

    def _parse_lidar(self, lidar_data):
        points = np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        return points[:, :3]

    def cleanup(self):
        for a in self.actor_list:
            if a.is_alive: a.destroy()
        self.queues.clear()


class CarlaGymEnv(gym.Env):


    def __init__(self, host='127.0.0.1', port=2000):
        super(CarlaGymEnv, self).__init__()

        self.client = carla.Client(host, port)
        self.client.set_timeout(20.0)
        self.world = self.client.load_world('Town01')

        self.bp_lib = self.world.get_blueprint_library()
        self.map = self.world.get_map()
        self.spawn_points = self.map.get_spawn_points()
        self.npc_vehicles = []
        self.npc_controllers = []
        num_vehicles = 15
        self.num_vehicles = num_vehicles
        self.vehicle = None
        self.sensors = None
        self.immobile_steps = 0

        self.action_list = [
            [0.0, 0.6, 0.0],  # straight
            [0.0, 1.0, 0.0],  # full throttleA
            [0.0, 0.0, 0.8],  # brake
            [-0.5, 0.5, 0.0],  # left
            [0.5, 0.5, 0.0],  # right
            [-0.2, 0.6, 0.0],  # slight left
            [0.2, 0.6, 0.0],  # slight right
            [-0.8, 0.3, 0.0],  # hard left slow
            [0.8, 0.3, 0.0],  # hard right slow
        ]

        self.action_space = spaces.Discrete(len(self.action_list))


        self.observation_space = spaces.Dict({
            "rgb": spaces.Box(0, 255, (RL_HEIGHT, RL_WIDTH, 3), dtype=np.uint8),
            "depth": spaces.Box(0, 1, (RL_HEIGHT, RL_WIDTH, 1), dtype=np.float32),
            "lidar_bev": spaces.Box(0, 1, (RL_HEIGHT, RL_WIDTH, 1), dtype=np.uint8),
            "state": spaces.Box(-np.inf, np.inf, (9,), dtype=np.float32)
        })

    def reset(self, seed=None):

        super().reset(seed=seed)

        if self.sensors: self.sensors.cleanup()
        if self.vehicle: self.vehicle.destroy()
        if self.npc_vehicles : self._cleanup_npcs()

        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)
        settings.no_rendering_mode = True

        vehicle_bp = self.bp_lib.filter('vehicle.tesla.model3')[0]
        spawn_point = random.choice(self.spawn_points)
        self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
        ego_spawn = spawn_point
        available_spawn_points = [
            sp for sp in self.spawn_points if sp != ego_spawn
        ]


        while self.vehicle is None:
            spawn_point = random.choice(self.spawn_points)
            self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
            ego_spawn = spawn_point
            available_spawn_points = [
                sp for sp in self.spawn_points if sp != ego_spawn
            ]
        self._spawn_npc_vehicles(available_spawn_points)

        self.sensors = SensorManager(self.world, self.vehicle)

        for _ in range(5):
            self.world.tick()

        raw_data = self.sensors.get_data()
        obs = self._process_obs(raw_data)

        self.start_time = time.time()
        self.step_count = 0

        return obs, {}

    def step(self, action_idx):
        self.step_count += 1

        steer, throttle, brake = self.action_list[int(action_idx)]

        control = carla.VehicleControl(
            throttle=float(np.clip(throttle, 0, 1)),
            steer=float(np.clip(steer, -1, 1)),
            brake=float(np.clip(brake, 0, 1))
        )
        self.vehicle.apply_control(control)

        self.world.tick()

        raw_data = self.sensors.get_data()

        if raw_data is None or 'rgb' not in raw_data:
            return self.last_obs, 0.0, False, False, {"sensor_fail": True}

        obs = self._process_obs(raw_data)
        self.last_obs = obs  #

        reward = 0.0
        speed_kmh = raw_data['speed_kmh']

        if speed_kmh < 50:
            reward += speed_kmh / 10.0
        else:
            reward += 5.0

        if speed_kmh < 5.0:
            self.immobile_steps +=1
        else:
            self.immobile_steps = 0

        if self.immobile_steps > 20:
            reward -= (self.immobile_steps - 20) * 0.1

        reward -= abs(steer) * 0.5

        terminated = False
        if raw_data.get('collision', False):
            reward -= 300
            terminated = True

        transform = raw_data['transform']
        waypoint = self.map.get_waypoint(transform.location)
        reward -= (self.lateral_offset ** 2) * 0.5
        reward += (1.0 - abs(self.heading_error) / np.pi) * 2.0

        truncated = False
        if self.step_count > 800:
            truncated = True

        info = {
            "speed": speed_kmh,
            "location": raw_data['location'],
            "action": int(action_idx)
        }

        return obs, reward, terminated, truncated, info

    def _process_obs(self, raw):

        rgb = cv2.resize(raw['rgb'], (RL_WIDTH, RL_HEIGHT), interpolation=cv2.INTER_AREA)

        depth = cv2.resize(raw['depth'], (RL_WIDTH, RL_HEIGHT), interpolation=cv2.INTER_AREA)
        depth_norm = np.clip(depth / 50.0, 0, 1)
        depth_norm = depth_norm[..., np.newaxis]

        lidar_bev = np.zeros((RL_HEIGHT, RL_WIDTH), dtype=np.uint8)
        points = raw['lidar']

        if len(points) > 0:
            x_range, y_range = 40, 40


            points_ground = points

            if len(points_ground) > 0:
                x_img = ((points_ground[:, 1] + y_range / 2) / y_range * RL_WIDTH).astype(np.int32)
                y_img = ((-points_ground[:, 0] + x_range / 2) / x_range * RL_HEIGHT).astype(np.int32)


                distances = np.sqrt(points_ground[:, 0] ** 2 + points_ground[:, 1] ** 2)

                mask = (x_img >= 0) & (x_img < RL_WIDTH) & (y_img >= 0) & (y_img < RL_HEIGHT)

                intensity = (255 - np.clip(distances / 50.0 * 255, 0, 255)).astype(np.uint8)
                lidar_bev[y_img[mask], x_img[mask]] = intensity[mask]

        lidar_bev = lidar_bev[..., np.newaxis]

        control = self.vehicle.get_control()

        transform = raw['transform']
        waypoint = self.map.get_waypoint(transform.location)

        yaw = np.deg2rad(transform.rotation.yaw)
        vehicle_dir = np.array([np.cos(yaw), np.sin(yaw)])

        wp_yaw = np.deg2rad(waypoint.transform.rotation.yaw)
        road_dir = np.array([np.cos(wp_yaw), np.sin(wp_yaw)])

        heading_error = np.arctan2(
            vehicle_dir[0] * road_dir[1] - vehicle_dir[1] * road_dir[0],
            vehicle_dir[0] * road_dir[0] + vehicle_dir[1] * road_dir[1]
        )

        self.heading_error = heading_error

        lane_center = waypoint.transform.location
        lateral_offset = np.sqrt(
            (transform.location.x - lane_center.x) ** 2 +
            (transform.location.y - lane_center.y) ** 2
        )
        self.lateral_offset = lateral_offset

        if len(points) > 0:
            front_points = points[(points[:, 0] > 0) & (np.abs(points[:, 1]) < 2)]
            left_points = points[points[:, 1] > 2]
            right_points = points[points[:, 1] < -2]

            front_dist = np.min(np.linalg.norm(front_points[:, :2], axis=1)) if len(front_points) > 0 else 50.0
            left_dist = np.min(np.linalg.norm(left_points[:, :2], axis=1)) if len(left_points) > 0 else 50.0
            right_dist = np.min(np.linalg.norm(right_points[:, :2], axis=1)) if len(right_points) > 0 else 50.0
        else:
            front_dist = left_dist = right_dist = 50.0

        state = np.array([
            raw['speed_kmh'] / 100.0,
            control.steer,
            control.throttle,
            control.brake,
            heading_error / np.pi,
            np.clip(lateral_offset / 2.0, -1, 1),
            front_dist / 50.0,
            left_dist / 50.0,
            right_dist / 50.0,
        ], dtype=np.float32)

        return {
            "rgb": rgb,
            "depth": depth_norm,
            "lidar_bev": lidar_bev,
            "state": state
        }

    def close(self):
        if self.sensors: self.sensors.cleanup()
        if self.vehicle: self.vehicle.destroy()


        settings = self.world.get_settings()
        settings.synchronous_mode = False
        self.world.apply_settings(settings)

    def _spawn_npc_vehicles(self,spawn_points):

        blueprint_library = self.world.get_blueprint_library()
        vehicle_blueprints = blueprint_library.filter('vehicle.*')


        vehicle_blueprints = [bp for bp in vehicle_blueprints
                              if int(bp.get_attribute('number_of_wheels')) == 4]




        available_spawn_points = spawn_points[1:]
        random.shuffle(available_spawn_points)

        for i in range(min(self.num_vehicles, len(available_spawn_points))):
            blueprint = random.choice(vehicle_blueprints)


            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)


            spawn_point = available_spawn_points[i]
            npc = self.world.try_spawn_actor(blueprint, spawn_point)

            if npc is not None:
                npc.set_autopilot(True)
                self.npc_vehicles.append(npc)

        print(f"Spawned {len(self.npc_vehicles)} NPC vehicles")

    def _cleanup_npcs(self):

        for controller_id in self.npc_controllers:
            controller = self.world.get_actor(controller_id)
            if controller is not None:
                controller.stop()
                controller.destroy()

        for vehicle in self.npc_vehicles:
            if vehicle is not None and vehicle.is_alive:
                vehicle.destroy()

        self.npc_vehicles.clear()
        self.npc_controllers.clear()


def train(env):
    print("Starting DQN training...")

    NUM_EPISODES = 8000
    MAX_STEPS_PER_EPISODE = 800
    REPLAY_WARMUP = 5_000
    TRAIN_EVERY = 1

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


    agent = DQN.DQNAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=DEVICE
    )

    # CHECKPOINT_PATH = "checkpoints/dqn_episode_1980.pth"
    # agent.load(CHECKPOINT_PATH)


    global_step = 0

    try:
        for episode in range(NUM_EPISODES):
            obs, _ = env.reset()
            episode_reward = 0.0

            for step in range(MAX_STEPS_PER_EPISODE):
                global_step += 1


                action_idx = agent.select_action(obs)
                action = agent.action_from_index(action_idx)


                next_obs, reward, terminated, truncated, info = env.step(action_idx)
                done = terminated or truncated

                agent.store_transition(
                    obs,
                    action_idx,
                    reward,
                    next_obs,
                    done
                )

                obs = next_obs
                episode_reward += reward

                if len(agent.buffer) >= REPLAY_WARMUP and global_step % TRAIN_EVERY == 0:
                    loss = agent.train_step()


                rgb_disp = next_obs['rgb']

                depth_disp = (next_obs['depth'] * 255).astype(np.uint8)
                depth_disp = cv2.applyColorMap(depth_disp, cv2.COLORMAP_JET)

                lidar_disp = next_obs['lidar_bev']
                lidar_disp = cv2.cvtColor(lidar_disp, cv2.COLOR_GRAY2BGR)

                dashboard = np.hstack((rgb_disp, depth_disp, lidar_disp))
                dashboard = cv2.resize(
                    dashboard, (0, 0),
                    fx=4, fy=4,
                    interpolation=cv2.INTER_NEAREST
                )

                cv2.imshow("Agent Observation (RGB | Depth | Lidar)", dashboard)
                cv2.waitKey(1)

                print(
                    f"\rEpisode {episode} | "
                    f"Step {step} | "
                    f"Reward {reward:.2f} | "
                    f"Speed {info['speed']:.1f} km/h",
                    end=""
                )

                if done:
                    break

            print(
                f"\nEpisode {episode:04d} finished | "
                f"Total Reward: {episode_reward:8.2f} | "
                f"Epsilon: {agent.get_epsilon():.3f}"
            )

            if episode % 20 == 0:
                os.makedirs("models", exist_ok=True)
                agent.save(f"models/dqn_episode_{episode}.pth")

    finally:
        env.close()
        cv2.destroyAllWindows()


