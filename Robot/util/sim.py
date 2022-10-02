
from typing import Dict, List, Tuple
import numpy as np

from util.landmark import Landmark

class Simulation:
    def __init__(self, map_data, sim_robot):
        self.robot: SimRobot = sim_robot
        self.robot_P = np.zeros((len(self.robot.state),))
        self.markers: Dict[str, np.ndarray] = self.__parse_markers(map_data)
        self.fruits: Dict[str, np.ndarray] = self.__parse_fruit(map_data)
        # Minimum distance to determine whether or not two fruit are the same
        self.fruit_distance: float = 1.0
        self.dict = {}

    def __get_state_length(self):
        return len(self.robot.state) + 2 * (len(self.markers) + len(self.fruits))

    def __get_state_vector(self):
        # Create markers array
        self.dict = {}
        i = len(self.robot.state)
        markers = []
        for id in self.markers:
            markers.append(self.markers[id][0:2])
            self.dict[id] = i
            i += 2
        markers = np.array(markers).flatten()
        # Create fruit array
        fruits = []
        for id in self.fruits:
            fruits.append(self.fruits[id][0:2])
            self.dict[id] = i
            i += 2
        fruits = np.array(fruits).flatten()
        # Create state array
        state = np.concatenate((self.robot.state, markers, fruits), axis=0)
        return state
    
    def __set_state_vector(self, state):
        index = len(self.robot.state)
        self.robot.state = state[0:index]        
        # Update each marker
        for id in self.markers:
            self.markers[id][0:2] = state[index:index+2]
            index += 2
        # Update each fruit
        for id in self.fruits:
            self.fruits[id][0:2] = state[index:index+2]
            index += 2
    
    def __get_P_matrix(self):
        # Create markers array
        markers = []
        for id in self.markers:
            markers.append(self.markers[id][2:4])
        markers = np.array(markers).flatten()
        # Create fruit array
        fruits = []
        for id in self.fruits:
            fruits.append(self.fruits[id][2:4])
        fruits = np.array(fruits).flatten()
        # Create state array
        vector = np.concatenate((self.robot_P, markers, fruits), axis=0)
        matrix = np.diag(vector)
        return matrix

    def __set_P_matrix(self, P):
        vector = np.diag(P)
        index = len(self.robot_P)
        self.robot_P = vector[0:index]        
        # Update each marker
        for id in self.markers:
            self.markers[id][2:4] = vector[index:index+2]
            index += 2
        # Update each fruit
        for id in self.fruits:
            self.fruits[id][2:4] = vector[index:index+2]
            index += 2
    
    def __parse_markers(self, map_data):
        markers = {}
        for key in map_data:
            if "aruco" in key:
                print(map_data[key])
                markers[key] = np.array([map_data[key]["x"],map_data[key]["y"], 0, 0])
        return markers

    def __parse_fruit(self, map_data):
        fruits = {}
        for key in map_data:
            if "aruco" not in key:
                fruits[key] = np.array([map_data[key]["x"],map_data[key]["y"], 0, 0])
        return fruits

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

    def __transform_landmarks(self, landmarks: List[Landmark]):
        th = self.robot.state[2]
        robot_xy = self.robot.state[0:2]
        R_theta = np.block([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
        for key in landmarks:
            landmarks[key].position = robot_xy + R_theta @ landmarks[key].position
    
    def __add_landmarks(self, landmarks, dictionary):
        for key in landmarks:
            if key not in dictionary:
                dictionary[key] = np.array([landmarks[key].position[0], landmarks[key].position[1], 10000, 10000])
    
    def __find_fruits(self, fruits: Dict[str, Landmark]):
        pass

    def __add_fruits(self, fruits: Dict[str, Landmark]):
        self.__add_landmarks(fruits, self.fruits)
    
    def __add_markers(self, markers: Dict[str, Landmark]):
        self.__add_landmarks(markers, self.markers)
    
    # def __get_marker_index(self, key):
    #     i = 0
    #     if key in self.markers:
    #         for marker in self.markers:
    #             if key == marker:
    #                 return i
    #             i += 1
    #     i = -1

    # def __get_fruit_index(self, key):
    #     i = 0
    #     if key in self.markers:
    #         for marker in self.markers:
    #             if key == marker:
    #                 return i
    #             i += 1
    #     i = -1
                

    def predict(self, left_vel, right_vel, dt):
        self.robot.drive(left_vel, right_vel, dt)
        A = self.__state_transition(dt)
        Q = self.__predict_covariance(dt)
        self.__set_P_matrix(A @ self.__get_P_matrix() @ A.T + Q)
    
    def update(self, markers: Dict[str, Landmark] = {}, fruits: Dict[str, Landmark] = {}):
        # Markers
        self.__transform_landmarks(markers)
        self.__add_markers(markers)
        # Fruit
        self.__transform_landmarks(fruits)
        self.__find_fruits(fruits)
        self.__add_fruits(fruits)
        
        marker_keys = markers.keys()
        fruit_keys = fruits.keys()

        robot_xy = self.robot.state[0:2]
        th = self.robot.state[2]        
        Rot_theta = np.block([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
        DRot_theta = np.block([[-np.sin(th), -np.cos(th)],[np.cos(th), -np.sin(th)]])

        z = np.zeros((0,))
        R = np.zeros((0,))
        z_hat = np.zeros((0,))

        if len(markers) > 0:
            positions = [markers[key].position for key in marker_keys]
            z = np.concatenate((z, *positions), axis = 0)
            R = np.concatenate((R, *[markers[key].covariance for key in marker_keys]), axis = 0)
            z_hat = np.concatenate((z_hat, *[self.markers[key][0:2] for key in marker_keys]), axis = 0)
        
        if len(fruits) > 0:
            positions = [fruits[key].position for key in fruit_keys]
            z = np.concatenate((z, *positions), axis = 0)
            R = np.concatenate((R, *[fruits[key].covariance for key in marker_keys]), axis = 0)
            z_hat = np.concatenate((z_hat, *[self.fruits[key][0:2] for key in fruit_keys]), axis = 0)
        
        #print(C)
        #print(C.shape)

        #C = np.eye(len(z), self.__get_state_length())
        #C[0:5, 0:5] = self.robot.drive_derivative(1)

        R = np.diag(R)
        P = self.__get_P_matrix()
        x = self.__get_state_vector()

        C = np.zeros((len(z), self.__get_state_length()))
        i = 0
        for key in marker_keys:
            j = self.dict[key]
            C[i, j] = 1
            C[i + 1, j + 1] = 1
            i += 2
            print(j)
        for key in fruit_keys:
            j = self.dict[key]
            C[i, j] = 1
            C[i + 1, j + 1] = 1
            i += 2
            
        print(C)
        

        temp = C @ P @ C.T + R
        K = P @ C.T @ np.linalg.inv(temp)

        x = x + K @ (z - z_hat)

        P = (np.eye(x.shape[0]) - K @ C) @ P

        self.__set_P_matrix(P)
        self.__set_state_vector(x)

class SimRobot:
    def __init__(self, wheels_width, wheels_scale):
        # Initialise state
        self.state = np.zeros((5,))
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

    def drive(self, left_vel, right_vel, dt):
        linear_velocity, angular_velocity = self.__convert_wheel_speeds(left_vel, right_vel)  
        # Remove float error where values are close to 0
        if (abs(linear_velocity) < 0.001):
            linear_velocity = 0
        if (abs(angular_velocity) < 0.001):
            angular_velocity = 0
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

if __name__ == "__main__":
    sim = Simulation(
        map_data = {},
        sim_robot = SimRobot(
            wheels_width = 0.154, 
            wheels_scale = 0.005
        )
    )

    sim.markers = {
        "0": np.array([0.33, 0.555]),
        "2": np.array([2, 1])
    }

    sim.fruits = {
        "apple1": np.array([-0.3412, 0.32113])
    }

    x = sim._Simulation__get_state_vector()
    print(x)
    x[4] = 2
    x[10] = -3
    sim._Simulation__set_state_vector(x)
    x = sim._Simulation__get_state_vector()
    print(x)
    print(sim.fruits)
    