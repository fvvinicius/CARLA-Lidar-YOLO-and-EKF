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
from EKF import EKF
from sensor_fusion import IMUDATA, GNSSDATA, imu_callback, imu2_callback, gnss_callback, gnss_to_xyz, _get_latlon_ref
from RGB import rgb_callback, predict_light, label_dict, red_light_flag
import math
from lidar import add_open3d_axis, lidar_callback, check_obstacle_ahead, VIRIDIS, VID_RANGE

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
    
    # Create lists to store sensor and real data
    camera = None
    
    RED_LIGHT = []
    OBSTACLE_LIST = [0]
    


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
            color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)

        # Pull spawn points and assign one to the vehicle then spawn
        spawn_points = world.get_map().get_spawn_points()
        transform = random.choice(spawn_points)

        vehicle = world.spawn_actor(bp, transform)
        actor_list.append(vehicle)
        print('created %s' % vehicle.type_id)

        
        
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
        
        
        
        '''
        IMPORT THE TRAINED COMPUTER VISION MODEL
        '''
        image_w = camera_bp.get_attribute("image_size_x").as_int()
        image_h = camera_bp.get_attribute("image_size_y").as_int()      
        # Defining the structure containing all sensor data
        sensor_data = {'rgb_image': np.zeros((image_h, image_w, 3)),
               'gnss': [0,0],
               'imu': {
                    'gyro': carla.Vector3D(),
                    'accel': carla.Vector3D(),
                    'compass': 0
                }}
        
        
        
        
        
        # Start camera feed
        camera.listen(lambda image: rgb_callback(image, RED_LIGHT, sensor_data, cc, OBSTACLE_LIST, vehicle))
        
        
        
        # Start CV2 window
        # OpenCV window with initial data
        cv2.namedWindow('RGB Camera', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RGB Camera', sensor_data['rgb_image'])
        cv2.waitKey(1)

        
        
        # Add traffic  
        k = 20 # Select number of npc
        for i in range(k): 
            bp = random.choice(blueprint_library.filter('vehicle'))

            
            npc = world.try_spawn_actor(bp, random.choice(spawn_points))
            if npc is not None:
                actor_list.append(npc)
                npc.set_autopilot(True)
                print('created %s' % npc.type_id)
        time.sleep(5)
        
        
 
        


        # Update geometry and camera in game loop
        frame = 0
        
        
        time.sleep(2)       # Let vehicle run for a couple of seconds before collecting data
        
        
        t_end = time.time() + (60 * 1)

        
        i = 1
        
        
        while time.time() < t_end:
            

            
           
            
            if cv2.waitKey(1) == ord('q'):
                break

            time.sleep(0.005)
            frame += 1
            
            if i == 1:
                vehicle.set_autopilot(True)
                i = 0
            
            
            obstacle_ahead = OBSTACLE_LIST[-1]
            # print(OBSTACLE_LIST)
            if (RED_LIGHT[-1]):
                
                control = vehicle.get_control()
                control.throttle = 0.
                control.brake = 1.
                vehicle.set_autopilot(False)
                vehicle.apply_control(control)

            else:
                vehicle.set_autopilot(True)
            
            print(obstacle_ahead)
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

  

        

if __name__ == '__main__':

    main()
