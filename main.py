import glob
import os
import sys
import time
import random

try:
    sys.path.append('D:\\CARLA_0.9.15\\WindowsNoEditor\\PythonAPI\\carla\\dist\\carla-0.9.15-py3.7-win-amd64.egg')
except IndexError:
    pass

import carla
import cv2
import numpy as np


IM_WIDTH = 640
IM_HEIGHT = 480


def process_img(image):
    i = np.array(image.raw_data)
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    i3 = i2[:, :, :3]
    cv2.imshow("", i3)
    cv2.waitKey(1)
    return i3/255.0

actor_list = []
try:
    # connect to the env
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    # load the blueprint library
    world = client.get_world()
    blueprint_lib = world.get_blueprint_library()

    # select the car blueprint
    bp = blueprint_lib.filter('cybertruck')[0]

    # spawn the car
    spawn_point = random.choice(world.get_map().get_spawn_points())
    vehicle = world.spawn_actor(bp, spawn_point)

    # add the vehicle to the list of actor for future clean-up
    actor_list.append(vehicle)

    # test the car by sending him a command
    vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))

    # get the blueprint for this sensor
    blueprint = blueprint_lib.find('sensor.camera.rgb')

    # change the dimensions of the image
    blueprint.set_attribute('image_size_x', f'{IM_WIDTH}')
    blueprint.set_attribute('image_size_y', f'{IM_HEIGHT}')
    blueprint.set_attribute('fov', '110')

    # Adjust sensor relative to vehicle
    spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))

    # spawn the sensor and attach to vehicle.
    sensor = world.spawn_actor(blueprint, spawn_point, attach_to=vehicle)

    # add sensor to list of actors
    actor_list.append(sensor)

    # process the camera images
    sensor.listen(lambda data: process_img(data))

    # time to find the car
    time.sleep(20)

finally:
    print('destroying actors')
    for actor in actor_list:
        actor.destroy()
    print('done.')