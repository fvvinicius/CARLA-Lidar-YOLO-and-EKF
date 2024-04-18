import carla 
import math 
import random 
import time 
import numpy as np
import cv2
import open3d as o3d
from matplotlib import cm

# global obstacle_ahead

# Auxilliary code for colormaps and axes
VIRIDIS = np.array(cm.get_cmap('plasma').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])

COOL_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])
COOL = np.array(cm.get_cmap('winter')(COOL_RANGE))
COOL = COOL[:,:3]

def add_open3d_axis(vis):
    """Add a small 3D axis on Open3D Visualizer"""
    axis = o3d.geometry.LineSet()
    axis.points = o3d.utility.Vector3dVector(np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]]))
    axis.lines = o3d.utility.Vector2iVector(np.array([
        [0, 1],
        [0, 2],
        [0, 3]]))
    axis.colors = o3d.utility.Vector3dVector(np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]]))
    vis.add_geometry(axis)


# LIDAR and RADAR callbacks

def lidar_callback(point_cloud, point_list, OBSTACLE_LIST):
    """Prepares a point cloud with intensity
    colors ready to be consumed by Open3D"""
    data = np.copy(np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4')))
    data = np.reshape(data, (int(data.shape[0] / 4), 4))

    # Isolate the intensity and compute a color for it
    intensity = data[:, -1]
    # intensity_col = 1. - np.log(intensity) / np.log(np.exp(-0.004 * 100))
    intensity_col = intensity
    int_color = np.c_[
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 0]),
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 1]),
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 2])] 
    # int_color = [intensity, intensity, intensity]

    points = data[:, :-1]
    # points[:, :1] = -points[:, :1]
    
    # Check for obstacles using the updated check_obstacle_ahead function
    obstacle_indices = check_obstacle_ahead(data, OBSTACLE_LIST, threshold_distance=10.0, angle_range=(-15, 15))

    
    # valid_indices = obstacle_indices[obstacle_indices < len(int_color)]
   
    
    # Set the color of detected obstacles to red
    for index in obstacle_indices:
        if index != 0:
            int_color[index] = [1.0, 0.0, 0.0]
    # int_color[valid_indices] = [1.0, 0.0, 0.0]
    
    point_list.points = o3d.utility.Vector3dVector(points)
    point_list.colors = o3d.utility.Vector3dVector(int_color)
    # points = data[:, :-1]

    # points[:, :1] = -points[:, :1]

    # point_list.points = o3d.utility.Vector3dVector(points)
    # point_list.colors = o3d.utility.Vector3dVector(int_color)

# Define a function to check for obstacles in front of the vehicle
def check_obstacle_ahead(points , OBSTACLE_LIST, threshold_distance=1.0, angle_range=(-30, 30)):
    global obstacle_ahead
    if isinstance(points, o3d.cpu.pybind.geometry.PointCloud):
        point_data = np.asanyarray(points.points)
    else:
        point_data = points
        

    x_coords = point_data[:, 0]
    y_coords = point_data[:, 1]
    z_coords = point_data[:, 2]

    # Calculate distances from the origin for each point
    distances = np.sqrt(x_coords**2 + y_coords**2)
    
    # Calculate angles of the points
    angles = np.degrees(np.arctan2(y_coords, x_coords))
    
    # Filter points within the specified angle range and distance threshold
    i = int(0)
    valid_obstacles = []
    

    
    #  and (angle_range[0] <= angles[i] <= angle_range[1])
    for distance in distances:
        
        if ((distance < threshold_distance)and (angle_range[0] <= angles[i] <= angle_range[1]) and (z_coords[i] > -1.5) ):
        # if (intensity[i] ):
            valid_obstacles.append(i)
            # print(points[i])
            print("Obstacle Ahead!")
            
            # print(intensity[i])
        else:
            valid_obstacles.append(0)
            
        i += 1
    # valid_points = np.where(angles >= angle_range[0]) & (angles <= angle_range[1])
    o = 0
    for point in valid_obstacles:
        if point != 0:
            o = 1
    OBSTACLE_LIST.append(o) 
    
    
    return valid_obstacles
    # # Check if there's any point within the threshold distance in front of the vehicle
    # if np.any((distances[valid_points]) < threshold_distance):
    #     return True
    # else:
    #     return False




def radar_callback(data, point_list):
    
    radar_data = np.zeros((len(data), 4))
    
    for i, detection in enumerate(data):
        x = detection.depth * math.cos(detection.altitude) * math.cos(detection.azimuth)
        y = detection.depth * math.cos(detection.altitude) * math.sin(detection.azimuth)
        z = detection.depth * math.sin(detection.altitude)
        
        radar_data[i, :] = [x, y, z, detection.velocity]
        
    intensity = np.abs(radar_data[:, -1])
    intensity_col = 1.0 - np.log(intensity) / np.log(np.exp(-0.004 * 100))
    int_color = np.c_[
        np.interp(intensity_col, COOL_RANGE, COOL[:, 0]),
        np.interp(intensity_col, COOL_RANGE, COOL[:, 1]),
        np.interp(intensity_col, COOL_RANGE, COOL[:, 2])]
    
    points = radar_data[:, :-1]
    points[:, :1] = -points[:, :1]
    point_list.points = o3d.utility.Vector3dVector(points)
    point_list.colors = o3d.utility.Vector3dVector(int_color)
    
# Camera callback
def camera_callback(image, data_dict):
    data_dict['image'] = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4)) 



def main():
    actor_list = []
    
    try:
        noise_seed = '0'
        client = carla.Client('localhost', 2000)
        client.set_timeout(4.0)
        
        
        

        world = client.get_world()
        dt = 0.005
        world.get_settings().fixed_delta_seconds = dt
        world.apply_settings(world.get_settings())  

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
        
        
        
        
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast') 
        lidar_bp.set_attribute('range', '100.0')
        lidar_bp.set_attribute('noise_stddev', '0.1')
        lidar_bp.set_attribute('upper_fov', '15.0')
        lidar_bp.set_attribute('lower_fov', '-25.0')
        lidar_bp.set_attribute('channels', '64.0')
        lidar_bp.set_attribute('rotation_frequency', '20.0')
        lidar_bp.set_attribute('points_per_second', '500000')
        # lidar_bp.set_attribute('sensor_tick', str(1.0 / 5.))
        lidar_init_trans = carla.Transform(carla.Location(z=2))
        lidar = world.spawn_actor(lidar_bp, lidar_init_trans, attach_to=vehicle)

        
        # Spawn camera
        camera_bp = blueprint_library.find('sensor.camera.rgb') 
        camera_init_trans = carla.Transform(carla.Location(z=2.5, x=-3), carla.Rotation())
        camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)

        # Add auxilliary data structures
        point_list = o3d.geometry.PointCloud()

        # Set up dictionary for camera data
        image_w = camera_bp.get_attribute("image_size_x").as_int()
        image_h = camera_bp.get_attribute("image_size_y").as_int()
        camera_data = {'image': np.zeros((image_h, image_w, 4))} 

        
        camera.listen(lambda image: camera_callback(image, camera_data))
        
        # OpenCV window for camera
        cv2.namedWindow('RGB Camera', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RGB Camera', camera_data['image'])
        cv2.waitKey(1)


        # Open3D visualiser for LIDAR and RADAR
        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name='Carla Lidar',
            width=960,
            height=540,
            left=480,
            top=270)
        vis.get_render_option().background_color = [0.05, 0.05, 0.05]
        vis.get_render_option().point_size = 1
        vis.get_render_option().show_coordinate_frame = True
        add_open3d_axis(vis)

        # Update geometry and camera in game loop
        frame = 0
        
        
        # Move the spectator behind the vehicle to view it
        spectator = world.get_spectator() 
        transform = carla.Transform(vehicle.get_transform().transform(carla.Location(x=-4,z=2.5)),vehicle.get_transform().rotation) 
        spectator.set_transform(transform)
        
        actor_list.append(spectator)


        time.sleep(2)
        
        for i in range(20): 
            bp = random.choice(blueprint_library.filter('vehicle'))

            # This time we are using try_spawn_actor. If the spot is already
            # occupied by another object, the function will return None.
            npc = world.try_spawn_actor(bp, random.choice(spawn_points))
            if npc is not None:
                actor_list.append(npc)
                npc.set_autopilot(True)
                print('created %s' % npc.type_id)
        
        time.sleep(3)
        lidar.listen(lambda data: lidar_callback(data, point_list))
        
        time.sleep(2)
        
        
        vehicle.set_autopilot(True)
        


        
        t_end = time.time() + (60 * 2)
        
        while time.time() < t_end:
            if frame == 2:
                vis.add_geometry(point_list)
            vis.update_geometry(point_list)
            
            vis.poll_events()
            vis.update_renderer()
            
            # # This can fix Open3D jittering issues:
            time.sleep(0.005)
            frame += 1

            cv2.imshow('RGB Camera', camera_data['image'])
            
            # Break if user presses 'q'
            if cv2.waitKey(1) == ord('q'):
                break
            
            world.tick()

        
    finally:
        print("End time:", time.time())
        
        lidar.stop()
        lidar.destroy()
        camera.stop()
        camera.destroy()
        vis.destroy_window()
        
        print('destroying actors')
       
        
        for x in actor_list:
            print('destroying ', x)   
            carla.command.DestroyActor(x)
            
        print('done.')
        
        
if __name__ == '__main__':
    main()