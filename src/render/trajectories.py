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


class HelixTrajectory:
    def __init__(self, dist):
        self.dist = dist

    def __call__(self, percent, camera):
        theta_max = 2 * np.pi
        theta_min = 0
        phi_max = np.pi / 3
        phi_min = 0
        xVal = self.dist * np.sin(theta_max * percent + theta_min)
        yVal = self.dist * np.cos(theta_max * percent + theta_min)
        zVal = self.dist * np.sin(phi_max * percent + phi_min)
        camera.set_eye([xVal, yVal, zVal], ref_up=[0, 0, 1])
