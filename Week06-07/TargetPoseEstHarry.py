from typing import List
import numpy as np
import json
import os
from pathlib import Path
import ast
import PIL
import math
import torch

model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')
FRUIT_SIZES = {
    "apple": [0.075448, 0.074871, 0.071889],
    "lemon": [0.060588, 0.059299, 0.053017],
    "pear": [0.0946, 0.0948, 0.135],
    "orange": [0.0721, 0.0771, 0.0739],
    "strawberry": [0.052, 0.0346, 0.0376],
    "person": [1, 1, 1]
    
    
    hello
    
}

class BoundingBox:
    def __init__(self, name, confidence, xmin, xmax, ymin, ymax):
        self.name = name
        self.confidence = confidence
        self.xmin = xmin
        self.xman = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.width = xmax - xmin
        self.height = ymax - ymin
        self.x = (xmax + xmin) / 2
        self.y = (ymax + ymin) / 2

    def __str__(self):
        return f"{self.name:12} x:{self.x:.4f} y:{self.y:.4f}  width:{self.width:.4f} height:{self.height:.4f}"

class Pose:
    def __init__(self, name, confidence, x, y):
        self.name = name
        self.confidence = confidence
        self.x = x
        self.y = y
        self.count = 1

    def is_same_pose(self, pose):
        dist = math.sqrt((pose.x - self.x)**2 + (pose.y - self.y)**2)
        # If the distance between two poses is relatively close
        if dist < 0.5:
            # Average with previous poses that were found in the same whereabouts
            self.x = self.x * self.count / (self.count + 1) + pose.x / (self.count + 1)
            self.y = self.y * self.count / (self.count + 1) + pose.y / (self.count + 1)
            self.count += 1
            # Update confidence
            self.confidence = max(self.confidence, pose.confidence)
            return True
        return False

    def __str__(self):
        return f"{self.name:12} x:{self.x:.4f} y:{self.y:.4f}"

class Image:
    def __init__(self, filename, robot_pose):
        self.filename = str(filename)
        self.robot_pose = [robot_pose[0][0], robot_pose[1][0], robot_pose[2][0]]
        self.bounds: List[BoundingBox] = []
        self.poses: List[Pose] = []

    def find_target_bounds(self):
        self.bounds = []
        image = PIL.Image.open(self.filename).resize((640,480), PIL.Image.NEAREST)
        results = model(image)
        labels, coords, confidences, names = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1], results.xyxyn[0][:, 4], results.names

        for i, label in enumerate(labels):

            if (confidences[i] < 0.4):
                continue

            xmin, ymin, xmax, ymax = coords[i][0].item(), coords[i][1].item(), coords[i][2].item(), coords[i][3].item()
            name = names[label.item()]
            confidence = confidences[i].item()
            self.bounds.append(BoundingBox(name, confidence, xmin, xmax, ymin, ymax))
    
    def calculate_target_poses(self, fov_x, fov_y):
        angle_vertical = self.robot_pose[2]
        c = math.cos(angle_vertical)
        s = math.sin(angle_vertical)

        self.poses = []
        for box in self.bounds:

            angle_vertical = fov_y * box.height
            forward_distance = (FRUIT_SIZES[box.name][2]/2) / math.atan(angle_vertical/2)
            angle_horizontal = -fov_x * (box.x - 0.5)
            side_distance = forward_distance * math.tan(angle_horizontal)

            x = self.robot_pose[0] + c * forward_distance - s * side_distance
            y = self.robot_pose[1] + c * side_distance + s * forward_distance

            self.poses.append(Pose(box.name, box.confidence, x, y))

def combine_all_poses(images: List[Image]):
    targets = {}

    # For every image
    for image in images:
        # For every target found in that image
        for pose in image.poses:

            # Not doing humans this year
            if pose.name == "person":
                continue

            # If this is the first targeet of certain fruit, add it without checking
            if pose.name not in targets:
                targets[pose.name] = [pose]
                continue
            
            # Run through all previous targets of the same fruit
            already_exists = False
            for potentially_same_pose in targets[pose.name]:
                # Check if they are the same target, i.e in the same position in world space
                if potentially_same_pose.is_same_pose(pose):
                    already_exists = True
                    break
            
            # If it is not close to any others, add it to the list as a new target
            if not already_exists:
                targets[pose.name].append(pose)
    
    #Make it only the three most confident of each type
    for target_name in targets:
        if len(targets[target_name]) > 3:
            targets[target_name] = sorted(targets[target_name], key=lambda x: x.confidence, reverse=True)[0:3]

    return targets

def save_target_estimates(targets):
    counts = {}
    save_obj = {}
    for target_name in targets:
        for target in targets[target_name]:

            if target_name not in counts:
                counts[target_name] = 0
            else:
                counts[target_name] += 1

            name = target_name + f"_{counts[target_name]}"

            save_obj[name] = {"y": target.y, "x": target.x}
    
    with open(base_dir/'lab_output/targets.txt', 'w') as fo:
        json.dump(save_obj, fo)

# main
if __name__ == "__main__":

    # Calculate FOV
    fileK = "{}intrinsic.txt".format('./calibration/param/')
    camera_matrix = np.loadtxt(fileK, delimiter=',')
    fov_x = 2 * np.arctan2(camera_matrix[0][2], camera_matrix[0][0])
    fov_y = 2 * np.arctan2(camera_matrix[1][2], camera_matrix[1][1])
    
    base_dir = Path('./')

    # List of images
    images = []
    with open(base_dir/'lab_output/images.txt') as fp:
        for line in fp.readlines():

            # Read images.txt
            data = ast.literal_eval(line)
            filename = data['imgfname']
            pose = data['pose']

            # Load image and find target bounding boxes
            new_image = Image(base_dir / filename, pose)
            new_image.find_target_bounds()

            # Calculate each targets position, requires camera FOV
            new_image.calculate_target_poses(fov_x, fov_y)

            # Add to list
            images.append(new_image)

    # Combine all detected targets
    target_est = combine_all_poses(images)
    
    # Print results
    for pose in target_est:
        for target in target_est[pose]:
            print(f"{target.name}: ({target.x}, {target.y})")
    
    save_target_estimates(target_est)