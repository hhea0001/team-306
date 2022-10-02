
import numpy as np

class Landmark:
    def __init__(self, position, covariance = np.array([1.0, 1.0])):
        self.position = position
        self.covariance = covariance