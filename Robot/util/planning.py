
import math
import random
import sys
import pathlib

import numpy as np
sys.path.append(str(pathlib.Path(__file__).parent))

from sim import Simulation

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.path_x = []
        self.path_y = []
        self.parent = None

class Bounds:
    def __init__(self, xmin, xmax, ymin, ymax):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

class Planner:
    def __init__(self, simulation: Simulation, bounds = Bounds(-1, 1, -1, 1), robot_radius = 0.1, obstacle_radius = 0.1, max_iter = 5000, expand_dis = 0.5, path_resolution = 0.5, goal_sample_rate=50):
        # Parameters
        self.simulation = simulation
        self.bounds = bounds
        self.robot_radius = robot_radius
        self.obstacle_radius = obstacle_radius
        self.max_iter = max_iter
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        # Init node stuff
        self.node_list = []
        self.start = None
        self.goal = None
        # Generated plan stuff
        self.i = 0
        self.current_plan = []
        self.angle = 0
    
    def has_next_goal(self):
        return (self.i + 1) < len(self.current_plan)
    
    def get_next_goal(self):
        self.i += 1
        if self.i >= len(self.current_plan):
            return None
        x, y = self.current_plan[self.i]
        if self.i + 1 < len(self.current_plan):
            x2, y2 = self.current_plan[self.i + 1]
            dx, dy = x2 - x, y2 - y
            theta = math.atan2(dy, dx)
        else:
            theta = self.angle
        return np.array([x, y, theta])
    
    def plan(self, start, goal):
        self.goal = self.ensure_safe_goal(goal, self.simulation.landmarks)
        if self.goal == None:
            return False
        self.start = self.ensure_safe_start(start, self.simulation.landmarks)
        self.current_plan = []
        self.angle = goal[2]
        self.i = 0
        self.node_list = [self.start]
        for i in range(self.max_iter):
            rnd_node = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]
            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)
            if self.check_if_inside_bounds(new_node) and self.check_collision(new_node, self.simulation.landmarks, self.robot_radius, self.obstacle_radius):
                self.node_list.append(new_node)
            if self.calc_dist_to_goal(self.node_list[-1].x, self.node_list[-1].y) <= self.expand_dis:
                final_node = self.steer(self.node_list[-1], self.goal, self.expand_dis)
                if self.check_collision(final_node, self.simulation.landmarks, self.robot_radius, self.obstacle_radius):
                    self.current_plan = self.generate_final_course(len(self.node_list) - 1)
                    print(f"Plan found after {i} iterations.")
                    for i in range(len(self.current_plan)):
                        print(f"Point {i}: {self.current_plan[i][0]:.2f}, {self.current_plan[i][1]:.2f}")
                    return True
        print(f"Could not find valid plan after {i} iterations...")
        return False
    
    def ensure_safe_start(self, start, obstacles):
        x, y = start[0], start[1]
        safe_dist = self.obstacle_radius + self.robot_radius
        for i in range(obstacles.shape[1]):
            dx = x - obstacles[0,i]
            dy = y - obstacles[1,i]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist <= safe_dist:
                dist_back = safe_dist - dist + 0.1
                angle = math.atan2(dy, dx)
                x += dist_back * math.cos(angle)
                y += dist_back * math.sin(angle)
        return Node(x, y)
    
    def ensure_safe_goal(self, goal, obstacles):
        x, y = goal[0], goal[1]
        safe_dist = self.obstacle_radius + self.robot_radius
        for i in range(obstacles.shape[1]):
            dx = x - obstacles[0,i]
            dy = y - obstacles[1,i]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist <= safe_dist:
                return None
        goal_node = Node(x, y)
        if not self.check_if_inside_bounds(goal_node):
            return None
        return goal_node

    def steer(self, from_node, to_node, extend_length=float("inf")):
        new_node = Node(from_node.x, from_node.y)
        d, theta = self.calc_distance_and_angle(new_node, to_node)
        new_node.path_x = [new_node.x]
        new_node.path_y = [new_node.y]
        if extend_length > d:
            extend_length = d
        n_expand = math.floor(extend_length / self.path_resolution)
        for _ in range(n_expand):
            new_node.x += self.path_resolution * math.cos(theta)
            new_node.y += self.path_resolution * math.sin(theta)
            new_node.path_x.append(new_node.x)
            new_node.path_y.append(new_node.y)
        d, _ = self.calc_distance_and_angle(new_node, to_node)
        if d <= self.path_resolution:
            new_node.path_x.append(to_node.x)
            new_node.path_y.append(to_node.y)
            new_node.x = to_node.x
            new_node.y = to_node.y
        new_node.parent = from_node
        return new_node

    def generate_final_course(self, goal_ind):
        path = [[self.goal.x, self.goal.y]]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])
        path.reverse()
        if self.calc_dist_to_goal(path[-2][0], path[-2][1]) < 0.01:
            path.pop()
        return path

    def calc_dist_to_goal(self, x, y):
        dx = x - self.goal.x
        dy = y - self.goal.y
        return math.hypot(dx, dy)
    
    def get_random_node(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            rnd = Node(
                random.uniform(self.bounds.xmin, self.bounds.xmax),
                random.uniform(self.bounds.ymin, self.bounds.ymax))
        else:  # goal point sampling
            rnd = Node(self.goal.x, self.goal.y)
        return rnd
    
    def check_if_inside_bounds(self, node):
        if node.x < self.bounds.xmin or node.x > self.bounds.xmax or \
           node.y < self.bounds.ymin or node.y > self.bounds.ymax:
            return False  # outside - bad
        else:
            return True  # inside - ok
    
    @staticmethod
    def get_nearest_node_index(node_list, rnd_node):
        dlist = [(node.x - rnd_node.x)**2 + (node.y - rnd_node.y)**2 for node in node_list]
        minind = dlist.index(min(dlist))
        return minind
    
    @staticmethod
    def check_collision(node, obstacles, robot_radius, obstacle_radius):
        if node is None:
            return False
        for i in range(obstacles.shape[1]):
            dx_list = [obstacles[0,i] - x for x in node.path_x]
            dy_list = [obstacles[1,i] - y for y in node.path_y]
            d_list = [dx * dx + dy * dy for (dx, dy) in zip(dx_list, dy_list)]
            if min(d_list) <= (obstacle_radius+robot_radius)**2:
                return False  # collision
        return True  # safe

    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        return d, theta
