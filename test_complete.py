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
    
    IMU_LIST = []
    GNSS_LIST = []
    REAL_ACC = []
    REAL_VEL = []
    POS_LIST = []
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
        


        # DEFINE SENSORS BIAS POINTS AND STDEV
        
        gnss_st_dev = 1.e-5
        imu_st_dev = 5.e-1
        
        # IMU
        noise_accel_stddev_x = imu_st_dev
        noise_accel_stddev_y = imu_st_dev
        noise_accel_stddev_z = imu_st_dev
        noise_gyro_bias_x = 0.
        noise_gyro_bias_y = 0.
        noise_gyro_bias_z = 0.
        noise_gyro_stddev_x = imu_st_dev
        noise_gyro_stddev_y = imu_st_dev
        noise_gyro_stddev_z = imu_st_dev
        
        # GNSS
        noise_lat_bias = 0
        noise_lat_stddev = gnss_st_dev
        noise_lon_bias = 0
        noise_lon_stddev = gnss_st_dev
        
        #Setup the IMU
        imu_bp = blueprint_library.find('sensor.other.imu')
        imu_transform = carla.Transform(carla.Location(x=0, z=0))
        imu_bp.set_attribute('noise_accel_stddev_x', str(noise_accel_stddev_x))          # Needs calibration
        imu_bp.set_attribute('noise_accel_stddev_y', str(noise_accel_stddev_y))          #    
        imu_bp.set_attribute('noise_accel_stddev_z', str(noise_accel_stddev_z))          #
        imu_bp.set_attribute('noise_gyro_bias_x', str(noise_gyro_bias_x))                #
        imu_bp.set_attribute('noise_gyro_bias_y', str(noise_gyro_bias_y))                #
        imu_bp.set_attribute('noise_gyro_bias_z', str(noise_gyro_bias_z))                #
        imu_bp.set_attribute('noise_gyro_stddev_x', str(noise_gyro_stddev_x))            #
        imu_bp.set_attribute('noise_gyro_stddev_y', str(noise_gyro_stddev_y))            #
        imu_bp.set_attribute('noise_gyro_stddev_z', str(noise_gyro_stddev_z))            #
        # imu_bp.set_attribute('noise_seed', noise_seed)                                 # replace with time seed for sim
        imu_bp.set_attribute('sensor_tick', str(1.0 / 200.))
        
        # Attach IMU to vehicle and spawn
        imu = world.spawn_actor(imu_bp, imu_transform, attach_to=vehicle)
        actor_list.append(imu)
        print('created %s' % imu.type_id)
        
        #Setup the IMU
        imu2_bp = blueprint_library.find('sensor.other.imu')
        imu2_transform = carla.Transform(carla.Location(x=0, z=0))
        imu2_bp.set_attribute('sensor_tick', str(1.0 / 800.))
        
        # Attach IMU to vehicle and spawn
        imu2 = world.spawn_actor(imu2_bp, imu2_transform, attach_to=vehicle)
        actor_list.append(imu2)
        print('created %s' % imu.type_id)
        
        
        
        
        # Setup GNSS
        gnss_bp = blueprint_library.find('sensor.other.gnss')
        gnss_transform = carla.Transform()
        gnss_bp.set_attribute('noise_lat_bias', str(noise_lat_bias))                     # Needs calibration
        gnss_bp.set_attribute('noise_lat_stddev', str(noise_lat_stddev))                 #     
        gnss_bp.set_attribute('noise_lon_bias', str(noise_lon_bias))                     #
        gnss_bp.set_attribute('noise_lon_stddev',str(noise_lon_stddev))                  #
        # gnss_bp.set_attribute('noise_seed', noise_seed)                                # replace with time seed for sim
        gnss_bp.set_attribute('sensor_tick', str(1.0 / 5.))
        
        # Attach GNSS to vehicle and spawn
        gnss = world.spawn_actor(gnss_bp, gnss_transform, attach_to=vehicle)
        actor_list.append(gnss)
        print('created %s' % gnss.type_id)
        
        #Set Up LIDAR
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast') 
        lidar_bp.set_attribute('range', '100.0')
        lidar_bp.set_attribute('noise_stddev', '0.1')
        lidar_bp.set_attribute('upper_fov', '15.0')
        lidar_bp.set_attribute('lower_fov', '-25.0')
        lidar_bp.set_attribute('channels', '64.0')
        lidar_bp.set_attribute('rotation_frequency', '20.0')
        lidar_bp.set_attribute('points_per_second', '500000')
        # lidar_bp.set_attribute('sensor_tick', str(1.0 / 5.))
        
        point_list = o3d.geometry.PointCloud()
        lidar_init_trans = carla.Transform(carla.Location(z=2))
        lidar = world.spawn_actor(lidar_bp, lidar_init_trans, attach_to=vehicle)
        actor_list.append(lidar)
        print('created %s' % lidar.type_id)
        
        
        
     
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
        
        
        # Start EKF and Sensors
        EKF_LIST = []
        timestamps3 = []
        timestamps4 = []
        x_in_pos = vehicle.get_location().x
        y_in_pos = vehicle.get_location().y
        initial_states = np.array([x_in_pos,y_in_pos,0,0])
        
        # setup EKF and start sensors
        ekf = EKF(gnss_st_dev, imu_st_dev, initial_states, world.get_snapshot().timestamp.elapsed_seconds )
        
        imu.listen(lambda event,: imu_callback(event, sensor_data, IMU_LIST, vehicle, REAL_VEL))
        imu2.listen(lambda event,: imu2_callback(event, REAL_ACC,timestamps4))
        gnss.listen(lambda event: gnss_callback(event, sensor_data, GNSS_LIST, vehicle, POS_LIST, world))
        lidar.listen(lambda data: lidar_callback(data, point_list, OBSTACLE_LIST))
        
        
        # Open3D visualiser for LIDAR and RADAR  # uncomment to show the lidar output, but may cause lag
        # vis = o3d.visualization.Visualizer()
        # vis.create_window(
        #     window_name='Carla Lidar',
        #     width=960,
        #     height=540,
        #     left=480,
        #     top=270)
        # vis.get_render_option().background_color = [0.05, 0.05, 0.05]
        # vis.get_render_option().point_size = 1
        # vis.get_render_option().show_coordinate_frame = True
        # add_open3d_axis(vis)
        
        # Update geometry and camera in game loop
        frame = 0
        
        
        time.sleep(2)       # Let vehicle run for a couple of seconds before collecting data
        
        
        t_end = time.time() + (60 * 1)

        
        t_old = world.get_snapshot().timestamp.elapsed_seconds
        i = 1
        
        
        while time.time() < t_end:
            
            # print('current time: ', time.time())
            # print(vehicle.get_control())
            x_meas = GNSS_LIST[-1].x
            y_meas = GNSS_LIST[-1].y
            v_gnss = GNSS_LIST[-1].speed
            v_imu = IMU_LIST[-1].speed
            a_meas = IMU_LIST[-1].acc
            theta = IMU_LIST[-1].compass
            
            
            t = world.get_snapshot().timestamp.elapsed_seconds
            
            timestamps3.append(t)
            
            dt = t - t_old
      
            
            transform = vehicle.get_transform()
            control = REAL_ACC[-1].x
            ekf_output,fused_v = ekf.new_data(x_meas, y_meas, v_gnss, v_imu, a_meas, dt, theta, control)
            t_old = t
            EKF_LIST.append(ekf_output)
            
           
            
            if cv2.waitKey(1) == ord('q'):
                break
            
            # if frame == 2:
            #     vis.add_geometry(point_list)
            #     # vis.add_geometry(radar_list)
            # vis.update_geometry(point_list)
            # vis.poll_events()
            # vis.update_renderer()
            # # This can fix Open3D jittering issues:
            time.sleep(0.005)
            frame += 1
            
            if i == 1:
                vehicle.set_autopilot(True)
                i = 0
            
            
            obstacle_ahead = OBSTACLE_LIST[-1]
            # print(OBSTACLE_LIST)
            if (obstacle_ahead==1 or RED_LIGHT[-1]):
                
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
        gnss.stop()
        gnss.destroy()
        imu.stop()
        imu.destroy()
        imu2.stop()
        imu2.destroy()
        lidar.stop()
        lidar.destroy()
        
        print('destroying actors')

        for x in actor_list:
            print('destroying ', x)   
            carla.command.DestroyActor(x)
            
        print('done.')
        
        
        timestamps = []
        timestamps2 = []
        
        imu_accs = []
        real_accs = []
        real_speeds = []
        x_gnss_pos = []
        y_gnss_pos = []
        x_positions = []
        y_positions = []
        ignore_start_index = int(0.005 * len(IMU_LIST))
        
  
    

        timestamps = [imu_data.timestamp for imu_data in IMU_LIST[:]]
        timestamps2 = [gnss_data.timestamp for gnss_data in GNSS_LIST[:]]

        
        imu_accs = [imu_data.acc for imu_data in IMU_LIST[:]]
        for real_data in REAL_ACC[:]:
            real_accs.append((real_data.x))
       
        imu_speeds = [imu_data.speed for imu_data in IMU_LIST[:]]
        
        
        for real_data in REAL_VEL[:]:
            real_speeds.append((real_data - carla.Vector3D(x=0, y=0, z=real_data.z)).length())
        

        speeds_gnss = [gnss_data.speed for gnss_data in GNSS_LIST[:]]
        
        
        x_gnss_pos = [gnss_data.x for gnss_data in GNSS_LIST[:]]

        x_positions =[real_data.x for real_data in POS_LIST[:]] 
        y_gnss_pos = [gnss_data.y for gnss_data in GNSS_LIST[:]]
        y_positions = [real_data.y for real_data in POS_LIST[:]] 
        
        EKF_LIST = np.array(EKF_LIST)
        

        # Plot States vs timestamp
        
        plt.figure(figsize=(10, 6))
        plt.plot(timestamps, real_speeds, label='real speed',  marker='o', markevery = 75, linestyle='dashed',color='black')
        plt.plot(timestamps, imu_speeds, label='imu integrated speed', marker='x', markevery = 75,  color='orange')
        plt.plot(timestamps2, speeds_gnss, label='gnss derived speeds', marker='h', markevery = 75,  color='pink')
        plt.plot(timestamps3, EKF_LIST[:,2], label = 'ekf speed', marker='d', markevery = 75,  color='green')
        plt.title('Velocity vs Timestamp')
        plt.xlabel('Time')
        plt.ylabel('v (m/s)')
        plt.legend()
        plt.show()
        

        # Plot acc vs timestamp
        plt.figure(figsize=(10, 6))
        plt.plot(timestamps4[ignore_start_index:], real_accs[ignore_start_index:], label='real acceleration', marker='o', markevery = 75,  linestyle='dashed', color='black')
        plt.plot(timestamps[ignore_start_index:], imu_accs[ignore_start_index:], label='imu accs', marker='x', markevery = 75,  color='orange')
        plt.plot(timestamps3, EKF_LIST[:,3], label = 'ekf acc',  marker='d', markevery = 75, color='green')
        plt.title('Acceleration vs Timestamp')
        plt.xlabel('Time (s)')
        plt.ylabel('Acceleration (m/s^2)')
        plt.legend()
        plt.show()
        

        
        plt.figure(figsize=(10, 6))
        plt.plot(timestamps2, x_gnss_pos, label='gnss positions',  marker='x', markevery = 200, color='orange')
        plt.plot(timestamps2, x_positions, label='real positions', linestyle='dashed', marker='o', markevery = 200,  color='black')
        plt.plot(timestamps3, EKF_LIST[:,0], label = 'ekf pos', marker='d', markevery = 200,  color='green')
        plt.title('x vs t')
        plt.xlabel('t (s)')
        plt.ylabel('x (m)')
        plt.legend()
        # plt.show()
        
        
        plt.figure(figsize=(10, 6))
        plt.plot(timestamps2, y_gnss_pos, label='gnss positions', marker='x', markevery = 75,  color='orange')
        plt.plot(timestamps2, y_positions, label='real positions', linestyle='dashed', marker='o', markevery = 75,  color='black')
        plt.plot(timestamps3, EKF_LIST[:,1], label = 'ekf pos', marker='d', markevery = 75,  color='green')
        plt.title('y vs t')
        plt.xlabel('t (s)')
        plt.ylabel('y (m)')
        plt.legend()
        plt.show()
        
        plt.figure(figsize=(10, 6))
        plt.plot(EKF_LIST[:,0], EKF_LIST[:,1], label='ekf positions', marker='x', markevery = 75,  color='green')
        plt.plot(x_gnss_pos, y_gnss_pos, label='gnss positions',  marker='o', markevery = 75, color='red')
        plt.plot(x_positions, y_positions, label='real positions', linestyle='dashed', marker='d', markevery = 75,  color='black')
        plt.title('y vs x')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.legend()
        plt.show()
        

if __name__ == '__main__':

    main()
