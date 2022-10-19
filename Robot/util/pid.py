
import math
import numpy as np
from util.sim import SimRobot, Simulation

LOOK_AT_GOAL = 0
MOVE_TO_GOAL = 1
ORIENT_TO_GOAL = 2
FINISHED = 3

class RobotPID:
    def __init__(self, sim_robot: SimRobot, distance_criteria = 0.01, angle_criteria = 0.01, goal = np.array([0, 0, 0]), K = np.array([10, 20, 10]), margin_of_error = 50):
        self.robot = sim_robot
        self.distance_criteria = distance_criteria
        self.angle_criteria = angle_criteria
        self.goal = goal.astype(float)
        self.Kg = K.astype(float)
        self.K = np.zeros_like(self.Kg)
        self.error = np.array([0.0, 0.0, 0.0])
        self.must_face = True
        self.state = FINISHED
        self.margin = margin_of_error
        self.update_error()
    
    def set_goal(self, goal: np.ndarray, must_face = True):
        self.state = LOOK_AT_GOAL
        self.goal = goal.astype(float)
        self.must_face = must_face
        self.update_error()

    def __look_at_goal(self):
        self.K[0] = 0
        self.K[1] = self.Kg[1]
        self.K[2] = 0
        if self.looking_at_goal():
            self.state = MOVE_TO_GOAL
        if self.at_goal():
            self.state = ORIENT_TO_GOAL

    def __move_to_goal(self):
        self.K[0] = self.Kg[0]
        self.K[1] = self.Kg[1]
        self.K[2] = 0
        # if not self.is_looking_at_goal(self.margin * self.angle_criteria):
        #     self.state = LOOK_AT_GOAL
        if self.at_goal():
            self.state = ORIENT_TO_GOAL

    def __orient_to_goal(self):
        self.K[0] = 0
        self.K[1] = 0
        self.K[2] = self.Kg[2]
        if not self.at_goal(self.margin * self.distance_criteria):
            self.state = LOOK_AT_GOAL
        if self.oriented_to_goal():
            self.state = FINISHED

    def __clamp_angle2(self, rad_angle=0, min_value=-np.pi/2, max_value=np.pi/2):
        if min_value > 0:
            min_value *= -1
        angle = (rad_angle + max_value) % (np.pi) + min_value
        return angle  
    
    def __clamp_angle(self, rad_angle=0, min_value=-np.pi, max_value=np.pi):
        if min_value > 0:
            min_value *= -1
        angle = (rad_angle + max_value) % (2 * np.pi) + min_value
        return angle
    
    def __get_forward_distance_to_goal(self):
        pos = self.robot.get_position()
        angle = self.robot.get_angle()
        norm = np.array([np.cos(angle), np.sin(angle)])
        dist = norm[0] * (self.goal[0] - pos[0]) + norm[1] * (self.goal[1] - pos[1])
        return dist + 0.05 if dist > 0 else - 0.05
    
    def __get_distance_to_goal(self):
        pos = self.robot.get_position()
        return np.sqrt((self.goal[0] - pos[0])**2 + (self.goal[1] - pos[1])**2)
    
    def __get_angle_to_goal(self):
        pos = self.robot.get_position()
        x_diff = self.goal[0] - pos[0]
        y_diff = self.goal[1] - pos[1]
        return self.__clamp_angle2(np.arctan2(y_diff, x_diff) - self.robot.get_angle()) \
            if self.state == MOVE_TO_GOAL else \
                self.__clamp_angle(np.arctan2(y_diff, x_diff) - self.robot.get_angle())

    def update_error(self):
        self.error[0] = self.__get_forward_distance_to_goal()
        self.error[1] = self.__get_angle_to_goal()
        orient_angle = self.goal[2] - self.robot.get_angle()
        self.error[2] = self.__clamp_angle(orient_angle) #if self.must_face else self.__clamp_angle2(orient_angle)

    def looking_at_goal(self, criteria = 0):
        if criteria == 0:
            criteria = self.angle_criteria
        return abs(self.error[1]) < criteria

    def at_goal(self, criteria = 0):
        if criteria == 0:
            criteria = self.distance_criteria
        return self.__get_distance_to_goal() < criteria
    
    def oriented_to_goal(self, criteria = 0):
        if criteria == 0:
            criteria = self.angle_criteria
        return abs(self.error[2]) < criteria
    
    def has_finished(self):
        return self.state == FINISHED
    
    def solve_velocities(self):
        if self.state == FINISHED:
            return 0, 0
        self.update_error()
        if self.state == LOOK_AT_GOAL:
            self.__look_at_goal()
        if self.state == MOVE_TO_GOAL:
            self.__move_to_goal()
        if self.state == ORIENT_TO_GOAL:
            self.__orient_to_goal()
        v_k = self.K[0] * self.error[0]
        w_k = self.K[1] * self.error[1] + self.K[2] * self.error[2]
        return max(min(round(20 * v_k), 20), -20), max(min(round(5 * w_k), 5), -5)
