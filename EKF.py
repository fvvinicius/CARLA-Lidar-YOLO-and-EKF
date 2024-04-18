import glob
import os
import sys

import carla

import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
import time
import xml.etree.ElementTree as ET


import math

class EKF:
    
    def __init__(self, gnss_stdev, imu_stdev, initial_states, timestamp):
        
        
        self.timestamp = timestamp
        self.v_imu = 0
        self.a_meas = 0
        self.theta = 0
        self.control = 0
        self.v_gnss = 0
        self.x_meas = 0
        self.y_meas = 0
        
        
        #define the covariance terms
        self.gnss_stdev = gnss_stdev*1000
        self.imu_stdev = imu_stdev
        
        self.x_var = gnss_stdev
        self.y_var = gnss_stdev
        self.var_acc = 2*imu_stdev
        self.var_pos = 2*gnss_stdev
        self.gnss_vel_var = 2*(gnss_stdev)
        self.imu_vel_var = 2*(imu_stdev)
        self.vel_var = 2*((self.gnss_vel_var*self.imu_vel_var)**2)/((self.gnss_vel_var**2)+(self.imu_vel_var**2))
        # self.vel_var = 0
        self.k = 0
        self.Q = 0.1*np.diag([.1, .1, .1, .1])

        self.R = np.diag([5000,5000,5000,5000])

        self.x_hat = np.array(initial_states)
        
        
        
        self.P = np.diag([1,1,1,1])
        self.H = np.diag([1,1,1,1])
        print('ekf initialized')
        
    def predict(self, theta, dt, control):
        

        self.F = np.array([[1, 0, dt*math.sin(theta), 0],
                            [0, 1, -1*dt*math.cos(theta), 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 0]]) 

        self.B = np.array([[(dt**2)*math.sin(theta)], 
                           [(dt**2)*math.cos(theta)], 
                           [dt], 
                           [1]])
        
        
        self.U = np.array([control])
        
        # predict x_hat using state at t-1 and control at t
        self.x_hat = np.dot(self.F, self.x_hat) + np.dot(self.B, self.U)
        
        # predict P given t-1
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        
        return self.x_hat
        
    def update(self, x_meas, y_meas, v_fused, a_meas, dt, theta):
        
        
        self.z = np.array([x_meas, y_meas, v_fused, a_meas])

        
        self.A = np.array([[1, 0, dt*math.sin(theta), 0],
                           [0, 1, -1*dt*math.cos(theta), 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 0]])
        
       
        
        self.temp = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        
        self.K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(self.temp))
        
        
        # update the state using kalman gain: x_hat 
        self.x_hat = self.x_hat + (np.dot(self.K, (self.z - np.dot(self.H, self.x_hat))))
        
        #covariance update
        self.P = np.dot(np.eye(4) - np.dot(self.K, self.H), self.P)
        
        return self.x_hat
    
    def sensor_fusion(self, v_gnss, v_imu):

        denominator = (self.gnss_vel_var) + (self.imu_vel_var)
        v = ((self.gnss_vel_var * v_imu) + (self.imu_vel_var * v_gnss)) / denominator
        self.v_fused = v
        return v
        
    def new_data(self, x_meas, y_meas, v_gnss, v_imu, a_meas, dt, theta, control):
        # self.sensor_fusion(v_gnss, v_imu)
        self.v_fused = self.v_imu       
        self.predict(theta, dt, control)
        self.update(x_meas, y_meas, self.v_fused, a_meas, dt, theta)
        return [self.x_hat, self.v_fused]
    
    def new_data_gnss(self, x_meas, y_meas, v_gnss, t):
        
        self.dt = self.timestamp - t
        self.timestamp = t
        self.v_gnss = v_gnss
        self.x_meas = x_meas
        self.y_meas = y_meas
        self.sensor_fusion(self.v_gnss, self.v_imu)

        self.predict(self.theta, self.dt, self.control)
        self.update(x_meas, y_meas, self.v_fused, self.a_meas, self.dt, self.theta)
        return [self.x_hat, self.v_fused]

    def new_data_imu(self, v_imu, a_meas, t, theta, control):
        
        self.dt = self.timestamp - t
        self.timestamp = t
        self.v_imu = v_imu
        self.a_meas = a_meas
        self.theta = theta
        self.control = control
        # self.sensor_fusion(self.v_gnss, self.v_imu)
        self.v_fused = self.v_imu
        self.predict(self.theta, self.dt, self.control)
        self.update(self.x_meas, self.y_meas, self.v_fused, a_meas, self.dt, theta)
        return [self.x_hat, self.v_fused]    
    
    
    
    