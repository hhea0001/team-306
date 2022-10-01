
import time
from typing import List, Tuple
import numpy as np
from util.fruit import FRUIT_TYPES

from util.pibot import PenguinPi as Robot
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
        # Setup aruco detector
        self.aruco_detector = ArucoDetector(camera_matrix)
        # Initialise time
        self.previous_time = time.time()
        # Initialise image
        self.image = np.zeros([480,640,3], dtype=np.uint8)

    def __get_position(self):
        return self.sim.state[0:2]
    
    def __get_angle(self):
        return self.sim.state[2]
    
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
        linear_vel = 0
        angular_vel = 0
        return linear_vel, angular_vel
    
    def __detect_aruco_landmarks(self) -> List[Tuple[str, np.ndarray]]:
        landmarks, marked_image = self.aruco_detector.detect_marker_positions(self.image)
        return landmarks

    def __detect_fruits(self) -> List[Tuple[str, np.ndarray]]:
        return []
    
    def drive(self):
        # Update time
        current_time = time.time()
        dt = current_time - self.previous_time
        self.previous_time = current_time
        # Update robot velocity
        linear_vel, angular_vel = self.__solve_velocity()
        left_vel, right_vel = self.robot.set_velocity([linear_vel, angular_vel])
        # Predict in simulation
        self.sim.predict(left_vel, right_vel, dt)
    
    def view(self):
        if self.__try_get_new_image():
            # Detect aruco landmarks
            landmarks = self.__detect_aruco_landmarks()
            # Detect fruits
            fruits = self.__detect_fruits()
            # Update sim
            self.sim.update(landmarks = landmarks, fruits = fruits)
        

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
        map_data = {} # TODO

    # Setup robot & robot simulation
    team306 = Team306(
        ip = args.ip, 
        port = args.port, 
        map_data = map_data, 
        camera_matrix = camera_matrix, 
        scale = scale, 
        baseline = baseline
    )

    # Create preview window
    window = Window()

    # Run operation loop
    while True:
        team306.drive()
        team306.view()

        window.draw_text(str(team306.sim.state), (0, 0))
        window.update()

        if window.quit:
            break
