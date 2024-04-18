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
import math
from lidar import add_open3d_axis, lidar_callback, check_obstacle_ahead, VIRIDIS, VID_RANGE
from typing import Dict, Tuple


from ultralytics import YOLO

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (50,50)
fontScale              = 0.5
fontColor              = (0,0,255)
thickness              = 1
lineType               = 2
tf_model = tf.keras.models.load_model('model_and_weights.h5')
model = YOLO('yolov8n.pt')
red_light_flag = False


label_dict = {0: 'backside', 1: 'GREEN',2: 'RED', 3: 'YELLOW'}


def resize_image(images: np.ndarray, new_shape: Tuple[int, int]):
    """Resize the image to the input shape of the TinyResNet."""
    images = tf.cast(images, tf.float32)
    images = tf.math.subtract(tf.math.divide(images, 127.5), 1.0)
    images = tf.image.resize(images, size=new_shape)
    return images

def predict_light(box,image, model):
    
            box = box.xyxy.numpy()
            # image = cv2.imread(img_path)
            # print(box)
            
            x1, y1, x2, y2 = map(int, box[0])
            
            
            
            # Crop the image using the coordinates
            cropped_image = image[y1:y2, x1:x2]
            cropped_image = cropped_image[:, :, ::-1]
            processed_image = tf.keras.preprocessing.image.array_to_img(cropped_image)
            print(type(processed_image))
            
            resized_image = resize_image(processed_image, (32, 32))
            print(type(resized_image))

            # Expand dimensions to add batch size dimension (assuming batch size is 1)
            resized_image_with_batch = np.expand_dims(resized_image, axis=0)
            print(type(resized_image_with_batch))
            
            # Now predict using the resized input data
            prediction = model.predict(resized_image_with_batch)
            predicted_index = np.argmax(prediction)

            # Map the index to the corresponding label using the dictionary
            predicted_label = label_dict[predicted_index]
            print(predicted_label)
            return predicted_label
            


def rgb_callback(image, RED_LIGHT, data_dict, cc, OBSTACLE_LIST, vehicle):
    
    # print(image.raw_data)
    resized_image = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
    yolo_pred = model(resized_image[:,:,:3])
    annotated_image= yolo_pred[0].plot()
    data_dict['rgb_image'] = annotated_image
    
    obstacle_ahead = OBSTACLE_LIST[-1]
    control = vehicle.get_control()
    red_light_detected = False
    
    for r in yolo_pred:
        boxes = r.boxes
        for box in boxes:
            if (box.cls == 9):
                predicted_light = predict_light(box,resized_image[:,:,:3], tf_model)
                if(predicted_light == 'RED'):
                    red_light_detected = True
                    cv2.putText(annotated_image, 
                    'RED LIGHT WARNING',
                    (50, 100),
                    font, 
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)
                    
    red_light_flag = red_light_detected
    RED_LIGHT.append(red_light_detected)
    
    if(obstacle_ahead == 1):
        cv2.putText(annotated_image, 
                    'OBSTACLE WARNING!!',
                    bottomLeftCornerOfText,
                    font, 
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)
        

    cv2.putText(annotated_image, 
                'Brakes: ' + str(control.brake),
                (50, 75),
                font, 
                fontScale,
                fontColor,
                thickness,
                lineType)
        
    cv2.imshow('RGB Camera', annotated_image)
    # annotated_image.save_to_disk('_out/%06d.png' % annotated_image.frame, cc)  #uncomment this to save the feed to the disk
    