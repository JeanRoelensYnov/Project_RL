import glob
import os
import sys

sys.path.append('D:\\CARLA_0.9.15\\WindowsNoEditor\\PythonAPI\\carla\\dist\\carla-0.9.15-py3.7-win-amd64.egg')
import carla

ACTOR_LIST = []
CLIENT = carla.Client('localhost', 2000)



def destroy_actor() -> None:
    for actor in ACTOR_LIST:
        actor.destroy()

