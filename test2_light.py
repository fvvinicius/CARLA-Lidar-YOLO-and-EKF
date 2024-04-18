import glob
import os
import sys

import carla

import random
import time
import numpy as np
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib import cm
import time
import xml.etree.ElementTree as ET
import tensorflow as tf
from RGB import rgb_callback, predict_light, label_dict, red_light_flag
import math

from ultralytics import YOLO




try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass





def main():
    
    actor_list = []
    OBSTACLE_LIST = [0,0]
    RED_LIGHT = []
    RED_LIGHT1 = []
    REAL_VEL = []
    BRAKES = []
    timestamps0 = []
    
    # Create lists to store sensor and real data
    camera = None
    

    


    try:
        noise_seed = '0'
        client = carla.Client('localhost', 2000)
        client.set_timeout(4.0)
        
        
        

        world = client.get_world()
        
        # Set up a fixed timestep for the simulation
        dt = 0.02
        temp = world.get_settings()
        temp.fixed_delta_seconds = dt
        world.apply_settings(temp)  

        # Startup the library
        blueprint_library = world.get_blueprint_library()

        # Pull Mustang from the library
        bp = random.choice(blueprint_library.filter('vehicle.ford.mustang'))
        
        # Set random color to the Mustang
        if bp.has_attribute('color'):
            color = bp.get_attribute('color')
            color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)

        # Pull spawn points and assign one to the vehicle then spawn
        spawn_points = world.get_map().get_spawn_points()
        # transform = spawn_points[145]
        # transform.location.x += 40
        transform = spawn_points[140]
        transform.location.y += 40
        
        vehicle = world.spawn_actor(bp, transform)
        actor_list.append(vehicle)
        print('created %s' % vehicle.type_id)

        #Set all traffic lights in the simulation to Red
        list_actor = world.get_actors()
        for actor_ in list_actor:
            if isinstance(actor_, carla.TrafficLight):
                # for any light, first set the light state, then set time. for yellow it is 
                # carla.TrafficLightState.Yellow and Red it is carla.TrafficLightState.Red
                actor_.set_state(carla.TrafficLightState.Red) 
                actor_.set_red_time(5000.0)

        
        # Move the spectator behind the vehicle to view it
        spectator = world.get_spectator() 
        transform = carla.Transform(vehicle.get_transform().transform(carla.Location(x=-4,z=2.5)),vehicle.get_transform().rotation) 
        
        spectator.set_transform(transform)
        
        actor_list.append(spectator)

        # Setup the RGB camera
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_transform = carla.Transform(carla.Location(x=1, z=2.4))
        camera_bp.set_attribute('image_size_x', '512') # Matches YOLO
        camera_bp.set_attribute('image_size_y', '512')
        
        #   Attach camera to the vehicle and spawn
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        actor_list.append(camera)
        print('created %s' % camera.type_id)
        cc = carla.ColorConverter.Raw
        
        image_w = camera_bp.get_attribute("image_size_x").as_int()
        image_h = camera_bp.get_attribute("image_size_y").as_int()   
        sensor_data = {'rgb_image': np.zeros((image_h, image_w, 3)),
               'gnss': [0,0],
               'imu': {
                    'gyro': carla.Vector3D(),
                    'accel': carla.Vector3D(),
                    'compass': 0
                }}
        
        # Start CV2 window
        # OpenCV window with initial data
        cv2.namedWindow('RGB Camera', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RGB Camera', sensor_data['rgb_image'])
        cv2.waitKey(1)
        
        camera.listen(lambda image: rgb_callback(image, RED_LIGHT, sensor_data, cc, OBSTACLE_LIST, vehicle))
        
        
        time.sleep(5)
        
        

        control = vehicle.get_control()
        control.brake = 0.
        control.throttle = 1.
        vehicle.apply_control(control)

        
        

        
        time.sleep(2)       # Let vehicle run for a couple of seconds before collecting data
        
        
        t_end = time.time() + (60 * 1)

        
        
        while time.time() < t_end:
            
            
            if cv2.waitKey(1) == ord('q'):
                break
            

            real_vel = transform.transform_vector(vehicle.get_velocity())
            vel = (real_vel - carla.Vector3D(x=0, y=0, z=real_vel.z)).length()
            REAL_VEL.append(vel)
            t = world.get_snapshot().timestamp.elapsed_seconds
            RED_LIGHT1.append(RED_LIGHT[-1])
            timestamps0.append(t)
            BRAKES.append(vehicle.get_control().brake)
            
            # print(RED_LIGHT)
            if (RED_LIGHT[-1]):
                
                
                control = vehicle.get_control()
                control.throttle = 0.
                control.brake = 1.
                vehicle.apply_control(control)
            else:
                control = vehicle.get_control()
                control.brake = 0.
                control.throttle = .4
                vehicle.apply_control(control)


            
            world.tick()   
            
            



    finally:
        camera.stop()
        camera.destroy()

        
        print('destroying actors')
        # if camera is not None:
        #     camera.destroy()
        
        for x in actor_list:
            print('destroying ', x)   
            carla.command.DestroyActor(x)
            
        print('done.')
        
        plt.figure(figsize=(10, 6))
        plt.plot(timestamps0, REAL_VEL, label='Velocity (m/s)', marker='o', markevery = 75, color='green')
        plt.plot(timestamps0, RED_LIGHT1, label='Red Light?',marker='x', markevery = 80, color='red')
        plt.plot(timestamps0, BRAKES, label = 'Brake', marker='d', markevery = 70,color='black')
        plt.title('States vs Timestamp')
        plt.xlabel('t(s)')
        plt.ylabel('State')
        plt.legend()
        plt.show()
     
        

if __name__ == '__main__':

    main()
