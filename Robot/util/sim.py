
from typing import Dict
import numpy as np

class Simulation:
    def __init__(self, map_data, sim_robot):
        self.robot: SimRobot = sim_robot
        self.markers: Dict[str, np.ndarray] = self.__parse_markers(map_data)
        self.fruits: Dict[str, np.ndarray] = self.__parse_fruit(map_data)
        state_length = self.__get_state_length()
        self.P = np.zeros((state_length, state_length))

    def __get_state_length(self):
        return len(self.robot.state) + len(self.markers) + len(self.fruits)

    def __get_state_vector(self):
        # Create markers array
        markers = []
        for id in self.markers:
            markers.append(self.markers[id])
        markers = np.array(markers).flatten()
        # Create fruit array
        fruits = []
        for id in self.fruits:
            fruits.append(self.fruits[id])
        fruits = np.array(fruits).flatten()
        # Create state array
        state = np.concatenate((self.robot.state, markers, fruits), axis=0)
        return state
    
    def __set_state_vector(self, state):
        index = len(self.robot.state)
        self.robot.state = state[0:index]        
        # Update each marker
        for id in self.markers:
            self.markers[id] = state[index:index+2]
            index += 2
        # Update each fruit
        for id in self.fruits:
            self.fruits[id] = state[index:index+2]
            index += 2
    
    def __parse_markers(self, map_data):
        return {}

    def __parse_fruit(self, map_data):
        return {}

    def __state_transition(self, dt):
        n = self.__get_state_length()
        F = np.eye((n,n))
        F[0:5,0:5] = self.robot.drive_derivative(dt)
        return F
    
    def __predict_covariance(self, dt):
        n = self.__get_state_length()
        Q = np.zeros((n,n))
        E = 0.001 * np.eye(5)
        E[3,3] = 0
        E[4,4] = 0
        Q[0:5,0:5] = self.robot.drive_covariance(dt) + E
        return Q

    def predict(self, left_vel, right_vel, dt):
        self.robot.drive(left_vel, right_vel, dt)
        A = self.__state_transition(dt)
        Q = self.__predict_covariance(dt)
        self.P = A @ self.P @ A.T + Q
    
    def update(self, landmarks = np.zeros((0, 1)), fruits = np.zeros((0, 1))):
        pass
        #fruits = __get_or_add_fruits(fruits)

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
        cov = np.diag((1, 1))
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
    