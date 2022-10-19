
import math
import random

import numpy as np
from util.sim import Simulation

class Node:
    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta
        self.path_x = []
        self.path_y = []
        self.path_theta = []
        self.parent = None

class Bounds:
    def __init__(self, xmin, xmax, ymin, ymax):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

class Plan:
    def __init__(self, simulation: Simulation, path):
        self.simulation = simulation
        self.path = path
        self.index = 0
    
    def length(self):
        length = 0
        for i in range(len(self.path) - 1):
            dx = self.path[i][0] - self.path[i + 1][0]
            dy = self.path[i][1] - self.path[i + 1][1]
            dist = np.sqrt(dx*dx + dy*dy)
            length += dist
        return length

    def has_next_goal(self):
        return (self.index + 1) < len(self.path)
    
    def get_next_goal(self):
        if self.has_next_goal():
            self.index += 1
            x, y, theta = self.path[self.index]
            return np.array([x, y, theta])
        return None
    
    def has_possible_collision(self):
        if len(self.path) < 2: # or self.index > len(self.path) - 1:
            return False
        start_index = random.randint(self.index - 1, len(self.path) - 2)
        random_float = random.random()
        start_pos = self.path[start_index] if (start_index > self.index - 1) else self.simulation.get_position()
        end_pos = self.path[start_index + 1]
        x = random_float * start_pos[0] + (1 - random_float) * end_pos[0]
        y = random_float * start_pos[1] + (1 - random_float) * end_pos[1]
        dx, dy = end_pos[0] - start_pos[0], end_pos[1] - start_pos[1]
        if self.simulation.check_collision(x, y, dx, dy):
            print("Future possible collision detected...")
            return True
        else:
            return False

class Planner:
    def __init__(self, simulation: Simulation, bounds = Bounds(-1, 1, -1, 1), max_iter = 100, expand_dis = 0.5, path_resolution = 0.1):
        # Parameters
        self.simulation = simulation
        self.bounds = bounds
        self.max_iter = max_iter
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
    
    def create_good_plan(self, start, goal, iterations = 10):
        plan = None
        min_length = 100000000
        for i in range(iterations):
            new_plan = self.create_plan(start, goal)
            if new_plan != None:
                new_plan_length = new_plan.length()
                if new_plan_length < min_length:
                    plan = new_plan
                    min_length = new_plan_length
        # if plan != None:
        #     print(f"Plan found.")
        #     for i in range(len(plan.path)):
        #         print(f"Point {i}: {plan.path[i][0]:.2f}, {plan.path[i][1]:.2f}")
        # else:
        #     print("Failed to find a good plan.")
        return plan
    
    def create_plan(self, start, goal):
        if not self.is_valid_goal(goal):
            return None
        start = Node(start[0], start[1], start[2])
        goal = Node(goal[0], goal[1], goal[2])
        node_list = [start]
        for i in range(self.max_iter):
            steer_node = self.get_random_node() if i % 2 == 0 else goal
            nearest_ind = self.get_nearest_node_index(node_list, steer_node)
            nearest_node = node_list[nearest_ind]
            new_node = self.steer(nearest_node, steer_node, self.expand_dis)
            if self.check_if_inside_bounds(new_node) and self.check_collision(new_node):
                node_list.append(new_node)
                dist_to_goal, _ = self.calc_distance_and_angle(new_node, goal)
                if dist_to_goal <= self.expand_dis:
                    final_node = self.steer(node_list[-1], goal, self.expand_dis)
                    if self.check_collision(final_node):
                        goal.parent = node_list[-1]
                        plan = self.generate_plan(goal)
                        return plan
        #print(f"Could not find valid plan after {i} iterations...")
        return None

    def steer(self, from_node, to_node, extend_length=float("inf")):
        new_node = Node(from_node.x, from_node.y, 0)
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

    def check_collision(self, node):
        for i in range(len(node.path_x) - 1):
            x, y = node.path_x[i], node.path_y[i]
            dx, dy = node.path_x[i + 1] - x, node.path_y[i + 1] - y 
            if self.simulation.check_collision(x, y, dx, dy, 1.2):
                return False
        return True

    def generate_plan(self, goal):
        path = [[goal.x, goal.y, goal.theta]]
        node = goal
        while node.parent is not None:
            previous_node = node
            node = node.parent
            dist, theta = self.calc_distance_and_angle(node, previous_node)
            if dist > 0.01:
                path.append([node.x, node.y, theta])
        path.reverse()
        return Plan(self.simulation, path)

    def is_valid_goal(self, goal):
        return True

    def check_if_inside_bounds(self, node):
        if node.x < self.bounds.xmin or node.x > self.bounds.xmax or \
           node.y < self.bounds.ymin or node.y > self.bounds.ymax:
            return False # Is outside
        else:
            return True # Is inside

    def get_random_node(self):
        return Node(
            random.uniform(self.bounds.xmin, self.bounds.xmax),
            random.uniform(self.bounds.ymin, self.bounds.ymax),
            0 # Theta, unused
        )
    
    @staticmethod
    def get_nearest_node_index(node_list, rnd_node):
        dlist = [(node.x - rnd_node.x)**2 + (node.y - rnd_node.y)**2 for node in node_list]
        minind = dlist.index(min(dlist))
        return minind

    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        return d, theta