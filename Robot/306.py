
import json
import math
import random
import time
from typing import Dict, List, Tuple
import numpy as np
from util.fruit import FRUIT_TYPES, FruitDetector
from util.landmark import Landmark
from util.pibot import PenguinPi as Robot
from util.pid import RobotPID
from util.planning import Bounds, Node, Planner
from util.sim import Simulation, SimRobot
from util.window import Window
from util.aruco import ArucoDetector
import util.window as win

class Team306:
    def __init__(self, ip, port, map_data, search_list, camera_matrix, scale, baseline, fruit_model):
        # Setup robot
        self.robot = Robot(
            ip = ip, 
            port = port
        )
        # Setup simulated robot
        self.sim = Simulation(
            map_data = map_data,
            sim_robot = SimRobot(
                wheels_width = baseline, 
                wheels_scale = scale
            )
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
            robot_radius = 0.1,
            obstacle_radius = 0.25,
            bounds = Bounds(
                -1.3, 1.3, -1.3, 1.3
            )
        )
        # Setup searching
        self.search_list = search_list
        self.search_index = -1
        self.search_fruit_name = "NONE"
        self.search_fruit_index = 0
        # Setup aruco detector
        self.aruco_detector = ArucoDetector(camera_matrix)
        # Setup fruit detector
        self.fruit_detector = FruitDetector(fruit_model, camera_matrix)
        # Initialise time
        self.previous_time = time.time()
        # Initialise image
        self.image = np.zeros([480,640,3], dtype=np.uint8)
        self.marked_image = np.zeros([480,640,3], dtype=np.uint8)
    
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
        return self.pid.solve_velocities()
    
    def __detect_aruco_markers(self) -> List[Landmark]:
        markers, self.marked_image = self.aruco_detector.detect_marker_positions(self.image)
        return markers

    def __detect_fruits(self) -> List[Landmark]:
        fruits, self.marked_image = self.fruit_detector.detect_fruit_positions(self.image, self.marked_image)
        return fruits
    
    def __create_new_plan(self):
        # This may take a while, so make sure the robot has stopped
        self.robot.set_velocity([0, 0])
        found_path = False
        # Find the fruit in the simulation
        self.search_fruit_index = self.sim.find_fruit_index(self.search_fruit_name)
        # If it doesn't exist, move around randomly
        if self.search_fruit_index == -1:
            while not found_path:
                # Set the goal to a random point in the arena
                x = random.random() * 2.6 - 1.3
                y = random.random() * 2.6 - 1.3
                angle = random.random() * math.pi * 2
                # Try making a plan with this estimated goal
                found_path = self.rrt.plan(self.sim.get_position(), [x, y, angle])
        # If it does exist, attempt to move to the fruit
        else:
            fruit_location = self.sim.landmarks[:,self.search_fruit_index]
            # Loop until a valid path has been found
            while not found_path:
                # Set the goal to a random point in a circle around the fruit
                radius = random.random() * 0.05 + (self.rrt.robot_radius + self.rrt.obstacle_radius)
                angle = random.random() * math.pi*2
                x = fruit_location[0] - radius * math.cos(angle)
                y = fruit_location[1] - radius * math.sin(angle)
                # Try making a plan with this estimated goal
                found_path = self.rrt.plan(self.sim.get_position(), [x, y, angle])
        # Update the PID
        self.pid.set_goal(self.rrt.get_next_goal())
    
    def __goto_next_fruit(self):
        # If this is the last fruit, then just exit function
        if self.search_index + 1 >= len(self.search_list):
            return
        # Otherwise if found the previous fruit, increment current fruit index
        if self.search_fruit_index != -1:
            self.search_index += 1
        self.search_fruit_name = self.search_list[self.search_index]
        # Create a plan to get there
        self.__create_new_plan()
    
    def plan(self):
        # Check if robot is about to crash into an obstacle
        # and if so create a new plan to the fruit
        if self.sim.detect_imminent_collision():
            self.__create_new_plan()
            return
        # If the robot has finished moving
        if self.pid.is_finished():
            # And there is another step in the plan towards a fruit
            if self.rrt.has_next_goal():
                # Go to the next position
                next_goal = self.rrt.get_next_goal()
                self.pid.set_goal(next_goal)
            # If we have arrived at the fruit
            else:
                # Wait 2 seconds, then move onto the next fruit
                if self.search_index != -1:
                    time.sleep(2)
                self.__goto_next_fruit()
        
    def drive(self):
        # Update the time parameters
        current_time = time.time()
        dt = current_time - self.previous_time
        self.previous_time = current_time
        # Update the real robot velocity
        linear_vel, angular_vel = self.__solve_velocity()
        left_vel, right_vel = self.robot.set_velocity([linear_vel, angular_vel])
        # Predict the changes in simulation
        self.sim.predict(left_vel, right_vel, dt)
    
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
    parser.add_argument("--fruit", action='store_true')
    args = parser.parse_args()

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
    
    if args.fruit:
        fruit_model = args.param_dir + ("net_sim.pt" if args.ip == 'localhost' else "net.pt")
    else:
        fruit_model = ''


    # Setup robot & robot simulation
    team306 = Team306(
        ip = args.ip, 
        port = args.port, 
        map_data = map_data, 
        search_list = search_list,
        camera_matrix = camera_matrix, 
        scale = scale, 
        baseline = baseline,
        fruit_model = fruit_model
    )

    # Create preview window
    window = Window()

    # Run operation loop
    while True:
        team306.plan()
        team306.drive()
        team306.view()

        #window.draw_image(team306.marked_image)
        
        window.draw(team306.sim, team306.marked_image, team306.rrt)
        window.update()

        if window.quit:
            break
