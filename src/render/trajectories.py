import numpy as np


class FixedPointTrajectory:
    def __init__(self, eye):
        self.eye = eye

    def __call__(self, percent, camera):
        camera.set_eye(self.eye)


class CircularTrajectory:
    def __init__(self, dist):
        self.dist = dist

    def __call__(self, percent, camera):
        xVal = self.dist * np.sin(2 * np.pi * percent)
        yVal = self.dist * np.cos(2 * np.pi * percent)
        camera.set_eye([xVal, yVal, 0])
