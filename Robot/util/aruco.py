
from typing import List, Tuple
import numpy as np
import cv2
import os, sys

class ArucoDetector:
    def __init__(self, camera_matrix, marker_length=0.07, distortion_params = np.zeros((1, 5))):
        self.camera_matrix = camera_matrix
        self.distortion_params = distortion_params
        self.marker_length = marker_length
        self.aruco_params = cv2.aruco.DetectorParameters_create()
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_100)
    
    def detect_marker_positions(self, image) -> Tuple[List[Tuple(str, np.ndarray)], np.ndarray]:
        # Perform detection
        corners, ids, rejected = cv2.aruco.detectMarkers(image, self.aruco_dict, parameters = self.aruco_params)
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, self.marker_length, self.camera_matrix, self.distortion_params)
        # rvecs, tvecs = cv2.aruco.estimatePoseSingleMarkers(corners, self.marker_length, self.camera_matrix, self.distortion_params) # use this instead if you got a value error
        if ids is None:
            return [], image
        # Compute the marker positions
        measurements = []
        seen_ids = []
        for i in range(len(ids)):
            idi = ids[i,0]
            # Some markers appear multiple times but should only be handled once.
            if idi in seen_ids:
                continue
            else:
                seen_ids.append(idi)
            lm_tvecs = tvecs[ids==idi].T
            lm_bff2d = np.block([[lm_tvecs[2,:]],[-lm_tvecs[0,:]]])
            lm_bff2d = np.mean(lm_bff2d, axis=1).reshape(-1,1)
            #lm_measurement = Landmark(lm_bff2d, idi, covariance=0)
            measurements.append((idi, lm_bff2d))
        # Draw markers on image copy
        img_marked = image.copy()
        cv2.aruco.drawDetectedMarkers(img_marked, corners, ids)
        return measurements, img_marked
