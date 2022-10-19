
from collections import OrderedDict
import math
from typing import Dict, List, Tuple
import numpy as np

from util.landmark import Landmark

class Simulation:

    def __init__(self, sim_robot, map_data, obstacle_radius = 0.2, target_list = []):
        # State components
        self.robot: SimRobot = sim_robot
        self.obstacle_radius = obstacle_radius
        self.landmarks = np.zeros((2, 0))
        self.taglist = {}
        self.target_list = target_list
        # Covariance matrix
        self.P = np.zeros((self.__get_state_length(), self.__get_state_length()))
        self.init_lm_cov = 1e3
        self.__parse_map_data(map_data)
    
    def __parse_map_data(self, map_data):
        try:
            taglist = {}
            landmarks = np.zeros((2, 0))
            for key in map_data:
                taglist[key] = len(taglist)
                position = np.reshape(np.array([map_data[key]["x"],map_data[key]["y"]]),(-1,1), order='F')
                landmarks = np.concatenate((landmarks, position), axis=1)
            self.taglist = taglist
            self.landmarks = landmarks
            self.P = np.zeros((self.__get_state_length(), self.__get_state_length()))
        except:
            self.taglist = map_data["taglist"]
            self.landmarks = np.array(map_data["landmarks"])
            self.P = np.array(map_data["covariance"])

    def __get_state_length(self):
        return len(self.robot.state) + 2 * self.__get_landmarks_length()

    def __get_landmarks_length(self):
        return int(self.landmarks.shape[1])

    def __get_state_vector(self):
        state = np.concatenate(
            (self.robot.state, np.reshape(self.landmarks, (-1,1), order='F')), axis=0)
        return state
    
    def __set_state_vector(self, state):
        self.robot.state = state[0:5,:]
        self.landmarks = np.reshape(state[5:,:], (2,-1), order='F')

    def __state_transition(self, dt):
        n = self.__get_state_length()
        F = np.eye(n)
        F[0:5,0:5] = self.robot.drive_derivative(dt)
        return F
    
    def __predict_covariance(self, dt):
        n = self.__get_state_length()
        Q = np.zeros((n,n))
        E = 0.001 * np.eye(5)
        E[3:5,3:5] = 0
        Q[0:5,0:5] = self.robot.drive_covariance(dt) + E
        return Q

    def __find_fruit(self, type, position):
        min_dist = 10000
        matching_fruits = [tag for tag in self.taglist if type in tag]
        matching_fruit = None
        for fruit in matching_fruits:
            index = self.taglist[fruit]
            dx = self.landmarks[0, index] - position[0]
            dy = self.landmarks[1, index] - position[1]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < min_dist:
                matching_fruit = fruit
                min_dist = dist
        return matching_fruit, min_dist, len(matching_fruits)
    
    def __remove_outliers(self, measurements):
        filtered_measurements = []
        th = self.robot.state[2]
        robot_xy = self.robot.state[0:2,:]
        R_theta = np.block([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
        for lm in measurements:

            if "aruco" in lm.tag:
                filtered_measurements.append(lm)
                continue
                
            if abs(lm.position[0]) > 1.5 or abs(lm.position[1] > 1.5):
                continue
            
            fruit_type = lm.tag
            fruit_pos = robot_xy + R_theta @ lm.position

            matching_fruit, dist, num_of_matching_fruit = self.__find_fruit(fruit_type, fruit_pos)

            if num_of_matching_fruit == 0:
                lm.tag += "_0"
                filtered_measurements.append(lm)
                continue
            
            is_target_fruit = any(target_fruit in lm.tag for target_fruit in self.target_list)

            if is_target_fruit:
                lm.tag = matching_fruit
            elif dist <= 0.4 or num_of_matching_fruit >= 2:
                lm.tag = matching_fruit
            else:
                lm.tag += f"_{num_of_matching_fruit}"
            filtered_measurements.append(lm)
        return filtered_measurements

    def __add_landmarks(self, measurements):
        th = self.robot.state[2]
        robot_xy = self.robot.state[0:2,:]
        R_theta = np.block([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
        # Add new landmarks to the state
        for lm in measurements:
            if lm.tag in self.taglist:
                # ignore known tags
                continue
            lm_bff = lm.position
            lm_inertial = robot_xy + R_theta @ lm_bff
            self.taglist[lm.tag] = len(self.taglist)
            self.landmarks = np.concatenate((self.landmarks, lm_inertial), axis=1)
            # Create a simple, large covariance to be fixed by the update step
            self.P = np.concatenate((self.P, np.zeros((2, self.P.shape[1]))), axis=0)
            self.P = np.concatenate((self.P, np.zeros((self.P.shape[0], 2))), axis=1)
            self.P[-2,-2] = self.init_lm_cov**2
            self.P[-1,-1] = self.init_lm_cov**2
    
    def find_fruit_index(self, fruit_name):
        for key in self.taglist:
            if fruit_name in key:
                return self.taglist[key]
        return -1

    def check_collision(self, x, y, dx = 0, dy = 0, radius_multiplier = 1):
        if dx == 0 and dy == 0:
            # Simple collision check
            for i in range(self.landmarks.shape[1]):
                dist = np.sqrt((x - self.landmarks[0,i])**2 + (y - self.landmarks[1,i])**2)
                if dist <= self.obstacle_radius * radius_multiplier:
                    return True # Collision
            return False # Safe
        else:
            # Direction based collision check
            for i in range(self.landmarks.shape[1]):
                dx2 = self.landmarks[0,i] - x
                dy2 = self.landmarks[1,i] - y
                dist = np.sqrt(dx2*dx2 + dy2*dy2)
                dx2 /= dist
                dy2 /= dist
                length = np.sqrt(dx*dx + dy*dy)
                dx /= length
                dy /= length
                dot_product = dx * dx2 + dy * dy2
                if dist <= (self.obstacle_radius * radius_multiplier) and dot_product > 0:
                    return True # Collision
            return False # Safe

    def get_position(self):
        return self.robot.get_position()
    
    def get_angle(self):
        return self.robot.get_angle()
        
    def predict(self, left_vel, right_vel, dt):
        self.robot.drive(left_vel, right_vel, dt)
        A = self.__state_transition(dt)
        Q = self.__predict_covariance(dt)
        self.P = A @ self.P @ A.T + Q

    def update(self, measurements):
        #self.__find_fruit(measurements)
        measurements = self.__remove_outliers(measurements)
        if not measurements:
            return
        self.__add_landmarks(measurements)
        # Construct measurement index list
        tags = [lm.tag for lm in measurements]
        idx_list = [self.taglist[tag] for tag in tags]
        # Stack measurements and set covariance
        z = np.concatenate([lm.position.reshape(-1,1) for lm in measurements], axis=0)
        R = np.zeros((2*len(measurements),2*len(measurements)))
        for i in range(len(measurements)):
            R[2*i:2*i+2,2*i:2*i+2] = measurements[i].covariance
        # Compute own measurements
        z_hat = self.robot.measure(self.landmarks, idx_list)
        z_hat = z_hat.reshape((-1,1),order="F")
        C = self.robot.measure_derivative(self.landmarks, idx_list)
        x = self.__get_state_vector()
        temp = C @ self.P @ C.T + R
        K = self.P @ C.T @ np.linalg.inv(temp)
        x = x + K @ (z - z_hat)
        self.P = (np.eye(x.shape[0]) - K @ C) @ self.P
        self.__set_state_vector(x)
    
    def get_slam_output(self):
        tags = [tag for tag in self.taglist if "aruco" in tag]
        taglist = [int(tag[5:-2]) for tag in tags]
        indices = [self.taglist[tag] for tag in tags]
        markers = self.landmarks[:, indices].tolist()
        covariance = self.landmarks[:, indices].tolist()
        slam = {
            "taglist": taglist,
            "map": markers,
            "covariance": covariance
        }
        return slam

    def get_targets_output(self):
        targets = {}
        fruits = [tag for tag in self.taglist if "aruco" not in tag]
        for fruit in fruits:
            index = self.taglist[fruit]
            targets[fruit] = {
                "y": self.landmarks[1, index],
                "x": self.landmarks[0, index]
            }
        return targets
    
    def get_save_data(self):
        json_data = {
            "taglist": self.taglist,
            "landmarks": self.landmarks.tolist(),
            "covariance": self.P.tolist()
        }
        return json_data

class SimRobot:
    def __init__(self, wheels_width, wheels_scale):
        # Initialise state
        self.state = np.zeros((5,1))
        # Setup parameters
        self.wheels_width = wheels_width
        self.wheels_scale = wheels_scale
    
    def __convert_wheel_speeds(self, left_vel, right_vel):
        # Convert to m/s
        left_speed_m = left_vel * self.wheels_scale
        right_speed_m = right_vel * self.wheels_scale
        # Compute the linear and angular velocity
        linear_velocity = (left_speed_m + right_speed_m) / 2.0
        angular_velocity = (right_speed_m - left_speed_m) / self.wheels_width
        return linear_velocity, angular_velocity
    
    def get_position(self):
        return self.state[0:2,0]
    
    def get_angle(self):
        return self.state[2,0]

    def drive(self, left_vel, right_vel, dt):
        linear_velocity, angular_velocity = self.__convert_wheel_speeds(left_vel, right_vel)  
        # Remove float error where values are close to 0
        # if (abs(linear_velocity) < 0.001):
        #     linear_velocity = 0
        # if (abs(angular_velocity) < 0.001):
        #     angular_velocity = 0
        self.state[3] = linear_velocity
        self.state[4] = angular_velocity
        # Apply the velocities
        th = self.state[2]
        if angular_velocity == 0:
            self.state[0] += np.cos(th) * linear_velocity * dt
            self.state[1] += np.sin(th) * linear_velocity * dt
        else:
            self.state[0] += linear_velocity / angular_velocity * (np.sin(th+dt*angular_velocity) - np.sin(th))
            self.state[1] += -linear_velocity / angular_velocity * (np.cos(th+dt*angular_velocity) - np.cos(th))
            self.state[2] += dt*angular_velocity
    
    def drive_derivative(self, dt):
        derivative = np.eye(5)
        th = self.state[2]
        lin_vel = self.state[3]
        ang_vel = self.state[4]
        th2 = th + dt * ang_vel
        if ang_vel == 0:
            derivative[0,3] = dt*np.cos(th)
            derivative[1,3] = dt*np.sin(th)
        else:
            derivative[0,3] = 1/ang_vel * (np.sin(th2) - np.sin(th))
            derivative[0,4] = -lin_vel/(ang_vel**2) * (np.sin(th2) - np.sin(th)) + \
                            lin_vel / ang_vel * (dt * np.cos(th2))
            derivative[1,3] = -1/ang_vel * (np.cos(th2) - np.cos(th))
            derivative[1,4] = lin_vel/(ang_vel**2) * (np.cos(th2) - np.cos(th)) + \
                            -lin_vel / ang_vel * (-dt * np.sin(th2))
            derivative[2,4] = dt
        return derivative

    def drive_covariance(self, dt):
        # Derivative of lin_vel, ang_vel w.r.t. left_speed, right_speed
        Jac1 = np.array([[self.wheels_scale/2, self.wheels_scale/2],
                [-self.wheels_scale/self.wheels_width, self.wheels_scale/self.wheels_width]])
        #lin_vel, ang_vel = self.convert_wheel_speeds(drive_meas.left_speed, drive_meas.right_speed)
        th = self.state[2]
        lin_vel = self.state[3]
        ang_vel = self.state[4]
        th2 = th + dt*ang_vel
        # Derivative of x,y,theta, lin_vel, ang_vel w.r.t. lin_vel, ang_vel
        Jac2 = np.zeros((5,2))
        Jac2[3, 0] = 1
        Jac2[4, 1] = 1
        if (ang_vel < 0.0001):
            ang_vel = 0
        if ang_vel == 0:
            Jac2[0,0] = dt*np.cos(th)
            Jac2[1,0] = dt*np.sin(th)
        else:
            Jac2[0,0] = 1/ang_vel * (np.sin(th2) - np.sin(th))
            Jac2[0,1] = -lin_vel/(ang_vel**2) * (np.sin(th2) - np.sin(th)) + \
                            lin_vel / ang_vel * (dt * np.cos(th2))

            Jac2[1,0] = -1/ang_vel * (np.cos(th2) - np.cos(th))
            Jac2[1,1] = lin_vel/(ang_vel**2) * (np.cos(th2) - np.cos(th)) + \
                            -lin_vel / ang_vel * (-dt * np.sin(th2))
            Jac2[2,1] = dt
        # Derivative of x,y,theta, lin_vel, ang_vel w.r.t. left_speed, right_speed
        Jac = Jac2 @ Jac1
        # Compute covariance
        cov = np.diag((1, 1)) # Left wheel covariance, right wheel covariance
        cov = Jac @ cov @ Jac.T
        return cov
    
    def measure(self, markers, idx_list):
        # Markers are 2d landmarks in a 2xn structure where there are n landmarks.
        # The index list tells the function which landmarks to measure in order.
        # Construct a 2x2 rotation matrix from the robot angle
        th = self.state[2]
        Rot_theta = np.block([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
        robot_xy = self.state[0:2,:]
        measurements = []
        for idx in idx_list:
            marker = markers[:,idx:idx+1]
            marker_bff = Rot_theta.T @ (marker - robot_xy)
            measurements.append(marker_bff)
        # Stack the measurements in a 2xm structure.
        markers_bff = np.concatenate(measurements, axis=1)
        return markers_bff
    
    def measure_derivative(self, markers, idx_list):
        # Compute the derivative of the markers in the order given by idx_list w.r.t. robot and markers
        robot_state_length = self.state.shape[0]
        n = 2*len(idx_list)
        m = robot_state_length + 2*markers.shape[1]
        DH = np.zeros((n,m))
        robot_xy = self.state[0:2,:]
        th = self.state[2]        
        Rot_theta = np.block([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
        DRot_theta = np.block([[-np.sin(th), -np.cos(th)],[np.cos(th), -np.sin(th)]])
        for i in range(n//2):
            j = idx_list[i]
            # i identifies which measurement to differentiate.
            # j identifies the marker that i corresponds to.
            lmj_inertial = markers[:,j:j+1]
            # lmj_bff = Rot_theta.T @ (lmj_inertial - robot_xy)
            # robot xy DH
            DH[2*i:2*i+2, 0:2] = - Rot_theta.T
            # robot theta DH
            DH[2*i:2*i+2, 2:3] = DRot_theta.T @ (lmj_inertial - robot_xy)
            # lm xy DH
            DH[2*i:2*i+2, robot_state_length+2*j:robot_state_length+2*j+2] = Rot_theta.T
            # print(DH[i:i+2,:])
        return DH
        