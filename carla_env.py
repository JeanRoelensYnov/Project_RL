import gymnasium as gym
from gymnasium import spaces
import numpy as np
import carla
import logging
import random
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

class CarlaEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.current_step = 0
        self.max_steps = 500
        self.client = carla.Client(config.CARLA_HOST, config.CARLA_PORT)
        self.client.set_timeout(10.0)
        self.client.load_world(config.CARLA_MAP)
        self.world = self.client.get_world()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05 # 20 FPS
        self.world.apply_settings(settings)

        self.ego_vehicle = None
        # Sensors
        self.collision_sensor = None
        self.collision_occurred = False

        self.npc_vehicles = []
        
        self.observation_space = spaces.Box(
            low=np.array([0 ,0 , -180]),        # Speed min, distance min, angle min
            high=np.array([150, 100 , 180]),    # speed max, distance max, angle max
            dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=np.array([-1, -1]),   # [throttle/brake, steering]
            high=np.array([1, 1]),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        logger.info("Resetting environment...")
        super().reset(seed=seed)
        
        if self.collision_sensor is not None:
            self.collision_sensor.destroy()
            self.collision_sensor = None
        
        actors = self.world.get_actors()
        for vehicle in actors.filter('vehicle.*'):
            vehicle.destroy()
        
        self.ego_vehicle = None
        self.npc_vehicles = []
        logger.info("All existing vehicles cleaned")
        
        
        logger.info("ego vehicle destroyed")
        blueprint_library = self.world.get_blueprint_library()

        vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]

        spawn_points = self.world.get_map().get_spawn_points()
        spawn_point = spawn_points[0]

        self.ego_vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        logger.info("ego_vehicle_respawned")
        self.collision_occurred = False
        self._setup_collision_sensor()

        
        vehicle_blueprints = blueprint_library.filter('vehicle.*')
        vehicle_blueprints = [bp for bp in vehicle_blueprints if int(bp.get_attribute('number_of_wheels')) == 4]

        num_npcs = 0
        available_spawns = spawn_points[1:] # Exclude ego spawn points

        for i in range(min(num_npcs, len(available_spawns))):
            bp = random.choice(vehicle_blueprints)
            npc = self.world.try_spawn_actor(bp, available_spawns[i])

            if npc is not None:
                npc.set_autopilot(True) 
                self.npc_vehicles.append(npc)
        logger.info("npc vehicles spawned")

        self.current_step = 0

        logger.info("Environment ready for next episode.")
        self._update_spectator()
        return self._get_observation(), {}

    def step(self, action):
        throttle_brake = float(action[0])
        steer = float(action[1])

        if throttle_brake >= 0:
            throttle = throttle_brake
            brake = 0.0
        else:
            throttle = 0.0
            brake = -throttle_brake
        
        control = carla.VehicleControl(
            throttle=throttle,
            brake=brake,
            steer=steer
        )
        self.ego_vehicle.apply_control(control)

        self.world.tick()

        observation = self._get_observation()

        reward= 0.0
        speed = observation[0]

        if self.collision_occurred:
            reward = -50.0
        elif speed < 2.0:  
            reward = -5.0
        elif speed < 5.0:  
            reward = -2.0
        else:
            reward = 2.0 + (speed / 30.0)
        
        # Penality if not aligne with the road
        angle = abs(observation[2])
        if angle > 20:
            reward -= 1.0
        if angle > 45:
            reward -= 2.0
        

        terminated = self.collision_occurred
        self.current_step += 1
        truncated = self.current_step >= self.max_steps
        info = {}
        
        self._update_spectator()
        
        return observation, reward, terminated, truncated, info

    def close(self):
        try:
            actors = self.world.get_actors()
            for vehicle in actors.filter('vehicle.*'):
                vehicle.destroy()
            
            self.ego_vehicle = None
            self.npc_vehicles = []
            logger.info("All vehicles destroyed")
            
        except Exception as e:
            logger.error(f"Error during close: {e}")
        
        logger.info("Environment closed")
        
    def _get_speed(self):
        velocity = self.ego_vehicle.get_velocity()
        speed = 3.6 * (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5
        return speed
    
    def _get_angle_to_road(self):
        ego_transform = self.ego_vehicle.get_transform()
        ego_yaw = ego_transform.rotation.yaw
        
        # Find nearest waypoint
        map = self.world.get_map()
        waypoint = map.get_waypoint(ego_transform.location)
        road_yaw = waypoint.transform.rotation.yaw
        
        angle = ego_yaw - road_yaw
        
        if angle > 180:
            angle -= 360
        elif angle < -180:
            angle += 360
        
        return angle
    
    def _get_distance_to_vehicle_ahead(self):
        ego_location = self.ego_vehicle.get_location()
        ego_forward = self.ego_vehicle.get_transform().get_forward_vector()
        
        min_distance = 100.0
        
        for npc in self.npc_vehicles:
            npc_location = npc.get_location()
            
            dx = npc_location.x - ego_location.x
            dy = npc_location.y - ego_location.y
            
            distance = (dx**2 + dy**2)**0.5
            
            dot = dx * ego_forward.x + dy * ego_forward.y
            
            if dot > 0 and distance < min_distance:
                min_distance = distance
        
        return min_distance

    def _get_observation(self):
        speed = self._get_speed()
        distance = self._get_distance_to_vehicle_ahead()
        angle = self._get_angle_to_road()
        
        return np.array([speed, distance, angle], dtype=np.float32)

    def _setup_collision_sensor(self):
        blueprint_library = self.world.get_blueprint_library()
        collision_bp = blueprint_library.find('sensor.other.collision')

        self.collision_sensor = self.world.spawn_actor(
            collision_bp,
            carla.Transform(),
            attach_to=self.ego_vehicle
        )

        self.collision_sensor.listen(lambda event: self._on_collision(event))

    def _on_collision(self, event):
        self.collision_occurred = True
        
    def _update_spectator(self):
        spectator = self.world.get_spectator()
        transform = self.ego_vehicle.get_transform()

        forward = transform.get_forward_vector()
        spectator_location = transform.location - forward * 8 + carla.Location(z=4)

        spectator.set_transform(carla.Transform(
            spectator_location,
            carla.Rotation(pitch=-15, yaw= transform.rotation.yaw)
        ))

