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
from EKF import EKF



import math

class IMUDATA:
    
    def __init__(self, imu: carla.IMUMeasurement, real_vel, imu_old = None) -> None:
        self.x_acc = imu.accelerometer.x 
        self.y_acc = imu.accelerometer.y
        self.timestamp = imu.timestamp
        # self.control_status = control_status
        # self.acc = (imu.accelerometer - carla.Vector3D(x=0, y=0, z=imu.accelerometer.z)).length()
        self.acc = self.x_acc
        self.imu_old = imu_old
        self.compass = imu.compass
        self.is_same_direction = np.dot([self.x_acc, self.y_acc], [real_vel.x, real_vel.y]) > 0
        
        
        self.control_status = 1.
       
        if ((imu_old is not None) and (abs(self.acc) < 100)):
            self.dt = imu.timestamp - imu_old.timestamp
            self.speed = imu_old.speed + (imu_old.control_status * self.acc * self.dt)
           
        else:
            self.dt = 0.02
            
            self.speed = 0
    def get_speed(self):
        return self.speed, self.dt
            
        
def gnss_to_xyz(latitude, longitude, z, lat_ref, lon_ref):
    """Creates Location from GPS (latitude, longitude, altitude)."""
    EARTH_RADIUS_EQUA = 6378137.0

    scale = math.cos(lat_ref * math.pi / 180.0)
    basex = scale * math.pi * EARTH_RADIUS_EQUA / 180.0 * lon_ref 
    basey = scale * EARTH_RADIUS_EQUA * math.log(math.tan((90.0 + lat_ref) * math.pi / 360.0))

    x = scale * math.pi * EARTH_RADIUS_EQUA / 180.0 * longitude - basex
    y = (scale * EARTH_RADIUS_EQUA * math.log(math.tan((90.0 + latitude) * math.pi / 360.0))) - basey

    # This wasn't in the original method, but seems to be necessary.
    y *= -1

    return (x, y, z)

class GNSSDATA:
    def __init__(self, gnss: carla.GnssMeasurement, gnss_old=None, lat_ref=42., lon_ref=2.) -> None:
        self.x, self.y, self.z = gnss_to_xyz(gnss.latitude, gnss.longitude, gnss.altitude, lat_ref, lon_ref)
        self.timestamp = gnss.timestamp
        self.lat_ref = lat_ref
        self.lon_ref = lon_ref

        if gnss_old:
            self.dt = gnss.timestamp - gnss_old.timestamp
            self.x_vel = (self.x - gnss_old.x) / self.dt
            self.y_vel = (self.y - gnss_old.y) / self.dt
            self.speed = math.sqrt((self.x_vel ** 2) + (self.y_vel ** 2))
        else:
            self.dt = .2
            self.x_vel = 0
            self.y_vel = 0
            self.speed = 0


try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass




def gnss_callback(data, data_dict, GNSS_LIST, vehicle, POS_LIST, world):
    
    POS_LIST.append(vehicle.get_location())
    data_dict['gnss'] = [data.latitude, data.longitude]
    lat_ref, lon_ref = _get_latlon_ref(world)
    gnss_new = GNSSDATA(data, GNSS_LIST[-1] if GNSS_LIST else None, lat_ref = lat_ref, lon_ref = lon_ref)
    GNSS_LIST.append(gnss_new)

    
            
def imu_callback(data, data_dict, IMU_LIST, vehicle, REAL_VEL):
   
    control = vehicle.get_control()  
    transform = vehicle.get_transform()
    real_vel = transform.transform_vector(vehicle.get_velocity())
    
    REAL_VEL.append(real_vel)

        
    

    imu_new = IMUDATA(data, real_vel, IMU_LIST[-1] if IMU_LIST else None, )
    IMU_LIST.append(imu_new)
    

    
def imu2_callback(data, REAL_ACC,timestamps4):
   

    
    timestamps4.append(data.timestamp)
    REAL_ACC.append(data.accelerometer)

    

    

    
def _get_latlon_ref(world):
    """
    Convert from waypoints world coordinates to CARLA GPS coordinates
    :return: tuple with lat and lon coordinates
    """
    xodr = world.get_map().to_opendrive()
    tree = ET.ElementTree(ET.fromstring(xodr))

    # default reference
    lat_ref = 42.0
    lon_ref = 2.0

    for opendrive in tree.iter("OpenDRIVE"):
        for header in opendrive.iter("header"):
            for georef in header.iter("geoReference"):
                if georef.text:
                    str_list = georef.text.split(' ')
                    for item in str_list:
                        if '+lat_0' in item:
                            lat_ref = float(item.split('=')[1])
                        if '+lon_0' in item:
                            lon_ref = float(item.split('=')[1])
    return lat_ref, lon_ref


def main():
    actor_list = []
    camera = None
    IMU_LIST = []
    GNSS_LIST = []
    REAL_ACC = []
    REAL_VEL = []
    POS_LIST = []
    throttle_list = []
    

    # In this tutorial script, we are going to add a vehicle to the simulation
    # and let it drive in autopilot. We will also create a camera attached to
    # that vehicle, and save all the images generated by the camera to disk.

    try:
        noise_seed = '0'
        client = carla.Client('localhost', 2000)
        client.set_timeout(4.0)
        
        
        

        world = client.get_world()
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

        # START WITH AUTOPILOT OFF (Needs to be changed)
        
        
        # Move the spectator behind the vehicle to view it
        spectator = world.get_spectator() 
        transform = carla.Transform(vehicle.get_transform().transform(carla.Location(x=-4,z=2.5)),vehicle.get_transform().rotation) 
        spectator.set_transform(transform)
        
        actor_list.append(spectator)



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
        
        
        '''
        SETUP LIDAR AND ADD TO VEHICLE
        .
        .
        .
        .
        .
        .
        .
        .
        .
        .
        .
        .
        '''
        
        
        
        
        '''
        IMPORT THE TRAINED COMPUTER VISION MODEL
        '''
        # image_w = camera_bp.get_attribute("image_size_x").as_int()
        # image_h = camera_bp.get_attribute("image_size_y").as_int()      
        # Defining the structure containing all sensor data
        sensor_data = {
            # 'rgb_image': np.zeros((image_h, image_w, 4)),
               'gnss': [0,0],
               'imu': {
                    'gyro': carla.Vector3D(),
                    'accel': carla.Vector3D(),
                    'compass': 0
                }}
        
        for i in range(0): 
            bp = random.choice(blueprint_library.filter('vehicle'))

            # This time we are using try_spawn_actor. If the spot is already
            # occupied by another object, the function will return None.
            npc = world.try_spawn_actor(bp, random.choice(spawn_points))
            if npc is not None:
                actor_list.append(npc)
                npc.set_autopilot(True)
                print('created %s' % npc.type_id)
        
        

        time.sleep(5)
        
        
        
        ''' 
        CHANGE HOW CAMERA IS CALLED SO THAT IT DISPLAYS AND SAVES TO DISK
        '''       
        # camera.listen(lambda image: rgb_callback(image, sensor_data, cc))
        
        # x_position = 0
        # y_position = 0
        # count = 0
        
        # for datapoint in GNSS_LIST
        EKF_LIST = []
        timestamps3 = []
        timestamps4 = []
        x_in_pos = vehicle.get_location().x
        y_in_pos = vehicle.get_location().y
        initial_states = np.array([x_in_pos,y_in_pos,0,0])
        
        # setup EKF
        ekf = EKF(gnss_st_dev, imu_st_dev, initial_states, world.get_snapshot().timestamp.elapsed_seconds )
        imu.listen(lambda event,: imu_callback(event, sensor_data, IMU_LIST, vehicle, REAL_VEL))
        gnss.listen(lambda event: gnss_callback(event, sensor_data, GNSS_LIST, vehicle, POS_LIST, world))
        imu2.listen(lambda event,: imu2_callback(event, REAL_ACC,timestamps4))
        
        REAL_ACC2 = []
        # Start CV2 window
        # # OpenCV window with initial data
        # cv2.namedWindow('RGB Camera', cv2.WINDOW_AUTOSIZE)
        # cv2.imshow('RGB Camera', sensor_data['rgb_image'])
        # cv2.waitKey(1)
        time.sleep(2)
        
     

        

        
        t_end = time.time() + (60 * .5)
        
        
        print("Start time:", time.time())
        print("t_end:", t_end)
        t_old = world.get_snapshot().timestamp.elapsed_seconds
        i = 1
        while time.time() < t_end:
            
            
            spectator = world.get_spectator() 
            transform = carla.Transform(vehicle.get_transform().transform(carla.Location(x=-4,z=2.5)),vehicle.get_transform().rotation) 
            spectator.set_transform(transform)
            
            
            # print('current time: ', time.time())
            # print(vehicle.get_control())
            control = vehicle.get_control()
            x_meas = GNSS_LIST[-1].x
            y_meas = GNSS_LIST[-1].y
            v_gnss = GNSS_LIST[-1].speed
            v_imu = IMU_LIST[-1].speed
            a_meas = IMU_LIST[-1].acc
            theta = IMU_LIST[-1].compass
            control_status = IMU_LIST[-1].control_status
            if (control.throttle > 0):
                control_status = 1
            elif (control.brake > 0):
                control_status = -1
            else:
                control_status = 0
            
            t = world.get_snapshot().timestamp.elapsed_seconds
            
            timestamps3.append(t)
            
            dt = t - t_old
            
            # ((real_data - carla.Vector3D(x=0, y=0, z=real_data.z)).length())
            # control = control_status*((REAL_ACC[-1] - carla.Vector3D(x=0, y=0, z=REAL_ACC[-1].z)).length())
            
            
            transform = vehicle.get_transform()
            acc = transform.transform_vector(vehicle.get_acceleration())
            control = REAL_ACC[-1].x
            # control = IMU_LIST[-1].acc
            ekf_output,fused_v = ekf.new_data(x_meas, y_meas, v_gnss, v_imu, a_meas, dt, theta, control)
            t_old = t
            EKF_LIST.append(ekf_output)
            REAL_ACC2.append(control)
            
            
            if i == 1:
                vehicle.set_autopilot(True)
                i = 0
            world.tick()   
        # time.sleep(10)



        
    finally:
        print("End time:", time.time())
        gnss.stop()
        imu.stop()
        imu2.stop()
        
        
        print('destroying actors')
        # if camera is not None:
        #     camera.destroy()
        
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
        

        # Plot acc vs timestamp
        
        plt.figure(figsize=(10, 6))
        plt.plot(timestamps, real_speeds, label='real speed',  linestyle='dashed',color='black')
        plt.plot(timestamps, imu_speeds, label='imu integrated speed', color='orange')
        plt.plot(timestamps2, speeds_gnss, label='gnss derived speeds', color='pink')
        plt.plot(timestamps3, EKF_LIST[:,2], label = 'ekf speed', color='green')
        plt.title('Velocity vs Timestamp')
        plt.xlabel('Timestamp')
        plt.ylabel('v')
        plt.legend()
        plt.show()
        

        # Plot acc vs timestamp
        plt.figure(figsize=(10, 6))
        plt.plot(timestamps4[ignore_start_index:], real_accs[ignore_start_index:], label='real acceleration', linestyle='dashed', color='black')
        plt.plot(timestamps[ignore_start_index:], imu_accs[ignore_start_index:], label='imu accs', color='orange')
        plt.plot(timestamps3, EKF_LIST[:,3], label = 'ekf acc', color='green')
        plt.title('acc vs Timestamp')
        plt.xlabel('Timestamp')
        plt.ylabel('acc')
        plt.legend()
        plt.show()
        

        
        plt.figure(figsize=(10, 6))
        plt.plot(timestamps2, x_gnss_pos, label='gnss positions', color='red')
        plt.plot(timestamps2, x_positions, label='real positions', linestyle='dashed', color='black')
        plt.plot(timestamps3, EKF_LIST[:,0], label = 'ekf pos', color='green')
        plt.title('x vs t')
        plt.xlabel('t')
        plt.ylabel('x')
        plt.legend()
        # plt.show()
        
        
        plt.figure(figsize=(10, 6))
        plt.plot(timestamps2, y_gnss_pos, label='gnss positions', color='red')
        plt.plot(timestamps2, y_positions, label='real positions', linestyle='dashed', color='black')
        plt.plot(timestamps3, EKF_LIST[:,1], label = 'ekf pos', color='green')
        plt.title('y vs t')
        plt.xlabel('t')
        plt.ylabel('y')
        plt.legend()
        plt.show()
        
        plt.figure(figsize=(10, 6))
        plt.plot(EKF_LIST[:,0], EKF_LIST[:,1], label='ekf positions', color='green')
        plt.plot(x_gnss_pos, y_gnss_pos, label='gnss positions', color='red')
        plt.plot(x_positions, y_positions, label='real positions', linestyle='dashed', color='black')
        plt.title('y vs x')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.show()
        


if __name__ == '__main__':

    main()
