
import json
import math
import os
import random
import sys
import time
from typing import Dict, List, Tuple
import numpy as np
import pygame
from util.fruit import FRUIT_TYPES, FruitDetector
from util.landmark import Landmark
from util.pibot import PenguinPi as Robot
from util.pid import RobotPID
from util.planning2 import Bounds, Planner, Plan
from util.sim import Simulation, SimRobot
from util.window import Window
from util.aruco import ArucoDetector
import util.window as win

class Team306:
    def __init__(self, ip, port, map_data, search_list, camera_matrix, scale, baseline, fruit_model, obstacle_radius, speed, confidence, max_dist, max_target_dist):
        # Setup robot
        self.robot = Robot(
            ip = ip, 
            port = port
        )
        self.speed = speed
        self.left_vel, self.right_vel = 0, 0
        # Setup simulated robot
        self.sim = Simulation(
            map_data = map_data,
            sim_robot = SimRobot(
                wheels_width = baseline, 
                wheels_scale = scale
            ),
            target_list = search_list,
            obstacle_radius = obstacle_radius,
            max_fruit_dist = max_dist,
            max_target_dist = max_target_dist
        )
        # Setup PID controller
        self.pid = RobotPID(
            sim_robot = self.sim.robot,
            distance_criteria = 0.03,
            margin_of_error = 15
        )
        # Setup RRT planner
        self.rrt = Planner(
            simulation = self.sim,
            bounds = Bounds(
                -1.2, 1.2, -1.2, 1.2
            )
        )
        self.current_plan: Plan = None
        self.start_time = time.time()
        # Setup searching
        self.search_list = search_list
        self.search_index = -1
        self.search_fruit_name = "NONE"
        self.search_fruit_index = 0
        # Setup aruco detector
        self.aruco_detector = ArucoDetector(camera_matrix)
        # Setup fruit detector
        self.fruit_detector = FruitDetector(
            model_name = fruit_model, 
            camera_matrix = camera_matrix, 
            confidence = confidence
        )
        # Initialise time
        self.previous_time = time.time()
        # Initialise image
        self.image = self.robot.get_image()
        self.marked_image = np.zeros([480,640,3], dtype=np.uint8)
        # Quit
        self.quit = False
        self.manual_driving = True
        self.wasd = [False, False, False, False]
    
    def __try_get_new_image(self):
        image = self.robot.get_image()
        # Check if same image
        if self.image.shape == image.shape:
            diff = ((self.image - image)**2).mean()
            if diff < 0.01:
                # Return false if not a new image
                return False
        # Update image and return true if it is
        self.image = image
        return True
    
    def __solve_velocity(self):
        if not self.manual_driving:
            return self.pid.solve_velocities()
        else:
            linear_vel = 0
            angular_vel = 0
            linear_vel += 20 if self.wasd[0] else 0
            angular_vel += 5 if self.wasd[1] else 0
            linear_vel -= 20 if self.wasd[2] else 0
            angular_vel -= 5 if self.wasd[3] else 0
            return linear_vel, angular_vel

    def __set_velocity(self, linear_vel, angular_vel):
        self.left_vel, self.right_vel = self.robot.set_velocity([linear_vel, angular_vel], self.speed, self.speed)
    
    def __detect_aruco_markers(self) -> List[Landmark]:
        markers, self.marked_image = self.aruco_detector.detect_marker_positions(self.image)
        return markers

    def __detect_fruits(self) -> List[Landmark]:
        fruits, self.marked_image = self.fruit_detector.detect_fruit_positions(self.image, self.marked_image)
        return fruits
    
    def __create_new_plan(self):
        # This may take a while, so make sure the robot has stopped
        self.__set_velocity(0, 0)
        self.current_plan = None
        # Find the fruit in the simulation
        self.search_fruit_index = self.sim.find_fruit_index(self.search_fruit_name)
        # If it doesn't exist, move around randomly
        if self.search_fruit_index == -1:
            while self.current_plan == None:
                # Set the goal to a random point in the arena
                x = random.random() * (self.rrt.bounds.xmax - self.rrt.bounds.xmin) + self.rrt.bounds.xmin
                y = random.random() * (self.rrt.bounds.ymax - self.rrt.bounds.ymin) + self.rrt.bounds.ymin
                angle = random.random() * math.pi * 2
                # Try making a plan with this estimated goal
                current_pos, current_angle = self.sim.get_position(), self.sim.get_angle()
                self.current_plan = self.rrt.create_good_plan([current_pos[0], current_pos[1], current_angle], [x, y, angle])
        # If it does exist, attempt to move to the fruit
        else:
            fruit_location = self.sim.landmarks[:,self.search_fruit_index]
            # Loop until a valid path has been found
            while self.current_plan == None:
                min_dist = 10000
                for i in range(10):
                    # Set the goal to a random point in a circle around the fruit
                    radius = random.random() * 0.1 + (self.sim.obstacle_radius)
                    angle = random.random() * math.pi*2
                    x = fruit_location[0] - radius * math.cos(angle)
                    y = fruit_location[1] - radius * math.sin(angle)
                    # Try making a plan with this estimated goal
                    current_pos, current_angle = self.sim.get_position(), self.sim.get_angle()
                    new_plan = self.rrt.create_good_plan([current_pos[0], current_pos[1], current_angle], [x, y, angle])
                    if new_plan != None:
                        new_plan_length = new_plan.length()
                        if new_plan_length < min_dist:
                            min_dist = new_plan_length
                            self.current_plan = new_plan
        # Update the PID
        self.pid.set_goal(self.current_plan.get_next_goal())
    
    def __goto_next_fruit(self):
        # If this is the last fruit, then just exit function
        if self.search_index + 1 >= len(self.search_list):
            return
        time.sleep(3)
        # Otherwise if found the previous fruit, increment current fruit index
        if self.search_fruit_index != -1:
            self.search_index += 1
        self.search_fruit_name = self.search_list[self.search_index]
        # Create a plan to get there
        self.__create_new_plan()
    
    def __at_fruit(self):
        current_pos = self.sim.get_position()
        fruit_pos = self.sim.landmarks[:,self.search_fruit_index]
        x = current_pos[0] - fruit_pos[0]
        y = current_pos[1] - fruit_pos[1]
        dist = np.sqrt(x*x + y*y)
        if dist >= 0.45:
            return False
        else:
            return True
        
    def __save(self):
        print("Saving...")
        export_type = "sim" if self.robot.ip == 'localhost' else "robot"
        slam_filename_format = f"slam_{export_type}_{{0}}_306.txt"
        targets_filename_format = f"targets_{export_type}_{{0}}_306.txt"
        raw_filename_format = f"r_{export_type[0]}_{{0}}.txt"
        it = 1
        while os.path.exists(slam_filename_format.format(it)):
            it += 1
        slam_filename = slam_filename_format.format(it)
        targets_filename = targets_filename_format.format(it)
        raw_filename = raw_filename_format.format(it)
        with open(slam_filename, 'w') as f:
            json.dump(self.sim.get_slam_output(), f)
            print(f"Saved slam as {slam_filename}")
        with open(targets_filename, 'w') as f:
            json.dump(self.sim.get_targets_output(), f)
            print(f"Saved targets as {targets_filename}")
        with open(raw_filename, 'w') as f:
            json.dump(self.sim.get_save_data(), f)
            print(f"Saved targets as {raw_filename}")
    
    def handle_input(self):
        # Quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                self.__save()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                self.manual_driving = not self.manual_driving
            elif event.type == pygame.KEYDOWN or event.type == pygame.KEYUP:
                down = event.type == pygame.KEYDOWN
                if event.key == pygame.K_UP:
                    self.wasd[0] = down
                elif event.key == pygame.K_DOWN:
                    self.wasd[2] = down
                elif event.key == pygame.K_LEFT:
                    self.wasd[1] = down
                elif event.key == pygame.K_RIGHT:
                    self.wasd[3] = down
    
    def plan(self):
        if time.time() < self.start_time + 10:
            return

        if self.manual_driving:
            return

        if self.current_plan != None:
            if self.current_plan.has_possible_collision():
                self.__create_new_plan()
                return
            if self.pid.has_finished():
                if self.current_plan.has_next_goal():
                    next_goal = self.current_plan.get_next_goal()
                    self.pid.set_goal(next_goal)
                elif self.__at_fruit():
                    self.__goto_next_fruit()
                else:
                    self.__create_new_plan()
        else:
            self.__goto_next_fruit()

    def drive(self):
        # Update the time parameters
        current_time = time.time()
        dt = current_time - self.previous_time
        self.previous_time = current_time
        self.sim.predict(self.left_vel, self.right_vel, dt)

        # Update the real robot velocity
        linear_vel, angular_vel = self.__solve_velocity()
        self.__set_velocity(linear_vel, angular_vel)
        # Predict the changes in simulation
        
    
    def view(self):
        # If we have recieved a new image
        if self.__try_get_new_image():
            # Detect aruco landmarks
            markers = self.__detect_aruco_markers()
            # Detect fruits
            fruits = self.__detect_fruits()
            # Update sim
            self.sim.update([*markers, *fruits])
        

if __name__ == "__main__":
    import argparse

    # Read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", metavar='', type=str, default='localhost')
    parser.add_argument("--port", metavar='', type=int, default=40000)
    parser.add_argument("--param_dir", type=str, default="param/")
    parser.add_argument("--map", type=str, default='')
    parser.add_argument("--search", type=str, default='')
    parser.add_argument("--stop", action='store_true')
    parser.add_argument("--speed", type=int, default=2)
    parser.add_argument("--radius", type=float, default=0.20)
    parser.add_argument("--confidence", type=float, default=0.75)
    parser.add_argument("--max_dist", type=float, default=0.4)
    parser.add_argument("--max_target_dist", type=float, default=1)
    args = parser.parse_args()

    if args.stop:
        stop_robot = Robot(
            args.ip,
            args.port
        )
        stop_robot.set_velocity([0, 0])
        sys.exit()

    # Load calibration parameters
    if args.ip == 'localhost':
        camera_matrix = np.loadtxt(args.param_dir + "intrinsic_sim.txt", delimiter=',')
        scale = np.loadtxt(args.param_dir + "scale_sim.txt", delimiter=',') / 2
        baseline = np.loadtxt(args.param_dir + "baseline_sim.txt", delimiter=',')
    else:
        camera_matrix = np.loadtxt(args.param_dir + "intrinsic.txt", delimiter=',')
        scale = np.loadtxt(args.param_dir + "scale.txt", delimiter=',')
        baseline = np.loadtxt(args.param_dir + "baseline.txt", delimiter=',')
    
    # Load map
    if args.map == '':
        map_data = {}
    else:
        with open(args.map, 'r') as f:
            map_data = json.load(f)
    
    if args.search == '':
        search_list = []
    else:
        with open(args.search, 'r') as f:
            search_list = f.read().split("\n")
            search_list = [x for x in search_list if x]
    
    fruit_model = args.param_dir + ("net_sim.pt" if args.ip == 'localhost' else "net.pt")

    # Setup robot & robot simulation
    team306 = Team306(
        ip = args.ip, 
        port = args.port, 
        map_data = map_data, 
        search_list = search_list,
        camera_matrix = camera_matrix, 
        scale = scale, 
        baseline = baseline,
        fruit_model = fruit_model,
        speed = args.speed,
        obstacle_radius = args.radius,
        confidence = args.confidence,
        max_dist = args.max_dist,
        max_target_dist = args.max_target_dist
    )

    # Create preview window
    window = Window()

    # Run operation loop
    while True:
        team306.handle_input()
        team306.plan()
        team306.drive()
        team306.view()
        
        window.draw(team306.sim, team306.marked_image, team306.current_plan)
        window.update()

        if team306.quit:
            window.quit()
            break
