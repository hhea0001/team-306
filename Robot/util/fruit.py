
import math
from typing import List
import PIL
import numpy as np
import torch

from util.landmark import Landmark
from util.settings import Settings

FRUIT_TYPES = ["apple", "orange", "pear", "lemon", "strawberry"]
FRUIT_SIZES = {
    "apple": [0.075448, 0.074871, 0.071889],
    "lemon": [0.060588, 0.059299, 0.053017],
    "pear": [0.0946, 0.0948, 0.135],
    "orange": [0.0721, 0.0771, 0.0739],
    "strawberry": [0.052, 0.0346, 0.0376]
}

class FruitDetector:
    def __init__(self, model_name, settings: Settings):
        if model_name == '':
            self.model = None
        else:
            self.model = torch.hub.load('./yolov5', 'custom', path=model_name, source='local')
        self.settings = settings
        self.i00 = settings.camera_matrix.value[0][0]
        self.fov_x = 2 * np.arctan2(self.settings.camera_matrix.value[0][2], self.settings.camera_matrix.value[0][0])
        self.fov_y = 2 * np.arctan2(self.settings.camera_matrix.value[1][2], self.settings.camera_matrix.value[1][1])
    
    def detect_fruit_positions(self, image_array, marked_image):
        if self.model == None:
            return [], marked_image
        image = PIL.Image.fromarray(image_array).resize((640,480), PIL.Image.NEAREST)
        # Run image through model
        results = self.model(image)
        labels, coords, confidences, names = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1], results.xyxyn[0][:, 4], results.names
        # Setup results array
        landmarks: List[Landmark] = []

        if len(labels) > 0 and self.i00 != self.settings.camera_matrix.value[0][0]:
            self.i00 = self.settings.camera_matrix.value[0][0]
            self.fov_x = 2 * np.arctan2(self.settings.camera_matrix.value[0][2], self.settings.camera_matrix.value[0][0])
            self.fov_y = 2 * np.arctan2(self.settings.camera_matrix.value[1][2], self.settings.camera_matrix.value[1][1])

        for i, label in enumerate(labels):
            if (confidences[i] < self.settings.confidence.value):
                continue
            xmin, ymin, xmax, ymax = coords[i][0].item(), coords[i][1].item(), coords[i][2].item(), coords[i][3].item()
            name = names[label.item()]
            if name == 'person':
                continue
            confidence = confidences[i].item()
            x = (xmax + xmin) / 2
            height = ymax - ymin
            angle_vertical = self.fov_y * height
            forward_distance = (FRUIT_SIZES[name][2]/2) / math.tan(angle_vertical/2)
            angle_horizontal = -self.fov_x * (x - 0.5)
            side_distance = forward_distance * math.tan(angle_horizontal)
            landmarks.append(Landmark(np.array([forward_distance, side_distance]).reshape(-1,1), name, 1/confidence * np.eye(2)))
        return landmarks, marked_image