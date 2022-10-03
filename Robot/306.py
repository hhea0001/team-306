
import json
import time
from typing import Dict, List, Tuple
import numpy as np
from util.fruit import FRUIT_TYPES
from util.landmark import Landmark
from util.pibot import PenguinPi as Robot
from util.pid import RobotPID
from util.planning import Bounds, Node, Planner
from util.sim import Simulation, SimRobot
from util.window import Window
from util.aruco import ArucoDetector
import util.window as win

class Team306:
    def __init__(self, ip, port, map_data, camera_matrix, scale, baseline):
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
            sim_robot = self.sim.robot
        )
        # Setup RRT planner
        self.rrt = Planner(
            simulation = self.sim,
            robot_radius = 0.2,
            obstacle_radius = 0.1,
            bounds = Bounds(
                -1.3, 1.3, -1.3, 1.3
            )
        )
        # Setup aruco detector
        self.aruco_detector = ArucoDetector(camera_matrix)
        # Initialise time
        self.previous_time = time.time()
        # Initialise image
        self.image = np.zeros([480,640,3], dtype=np.uint8)
        self.marked_image = np.zeros([480,640,3], dtype=np.uint8)
    
    def move_to(self, goal):
        self.robot.set_velocity([0, 0])
        found_path = self.rrt.plan(self.sim.get_position(), goal)
    
    def __check_plan(self):
        if self.pid.is_finished() and self.rrt.has_next_goal():
            next_goal = self.rrt.get_next_goal()
            self.pid.set_goal(next_goal)
    
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
        return []
    
    def drive(self):
        # Update time
        current_time = time.time()
        dt = current_time - self.previous_time
        self.previous_time = current_time
        # Update robot velocity
        self.__check_plan()
        linear_vel, angular_vel = self.__solve_velocity()
        left_vel, right_vel = self.robot.set_velocity([linear_vel, angular_vel])
        # Predict in simulation
        self.sim.predict(left_vel, right_vel, dt)
    
    def view(self):
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

    # Setup robot & robot simulation
    team306 = Team306(
        ip = args.ip, 
        port = args.port, 
        map_data = map_data, 
        camera_matrix = camera_matrix, 
        scale = scale, 
        baseline = baseline
    )

    #team306.pid.set_goal(np.array([0.300, -0.415, 0.0]))
    team306.move_to([1.2, 0, 3.14])

    # Create preview window
    window = Window()

    # Run operation loop
    while True:
        team306.drive()
        team306.view()

        window.draw_image(team306.marked_image)
        window.draw_text(str(team306.sim.robot.state), (0, 0))
        window.update()

        if window.quit:
            break
