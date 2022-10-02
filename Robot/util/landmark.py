
import numpy as np

class Landmark:
    # Measurements are of landmarks in 2D and have a position as well as tag id.
    def __init__(self, position, tag, covariance = (0.1*np.eye(2))):
        self.position = position
        self.tag = tag
        self.covariance = covariance