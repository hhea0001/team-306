
import numpy as np

class RobotSim:
    def __init__(self, map_data, wheels_width, wheels_scale, camera_matrix):
        # Initialise state
        self.state = np.zeros((5,1))
        #self.markers = self.__parse_markers(map_data)
        #self.fruits = self.__parse_fruit(map_data)
        # Setup parameters
        self.wheels_width = wheels_width
        self.wheels_scale = wheels_scale
        self.camera_matrix = camera_matrix
    
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
    
    def update(self, measurements):
        pass
    
    def drive_dt(self, dt):
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