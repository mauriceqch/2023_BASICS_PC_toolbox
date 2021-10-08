import numpy as np
from pyrr import matrix33
from pyrr.matrix44 import create_from_translation, create_from_scale, create_from_eulers, create_look_at, \
    create_perspective_projection
from pyrr.vector import normalize


def normalize_camera(xyz):
    xyz_min = xyz.min(axis=0)
    xyz_max = xyz.max(axis=0)
    xyz_mid = (xyz_max - xyz_min) / 2.0
    translate = -xyz_mid
    scale = [1 / np.abs(xyz + translate).max()] * 3
    return translate, scale


def build_mvp(camera):
    # Normalize voxelized point cloud to [-1, 1]^3 range
    translate = create_from_translation(camera.translate, dtype=np.float32)
    scale = create_from_scale(camera.scale, dtype=np.float32)
    rotate = create_from_eulers(camera.rotate, dtype=np.float32)
    # View and projection matrices
    view = create_look_at(camera.eye, camera.center, camera.up, dtype=np.float32)
    proj = create_perspective_projection(camera.fovy, camera.aspect_ratio, camera.z_near, camera.z_far, dtype=np.float32)
    return translate @ scale @ rotate @ view @ proj


def compute_right(front, up):
    return normalize(np.cross(front, up))


class BaseCamera:
    def __init__(self, xyz, aspect_ratio):
        self.translate, self.scale = normalize_camera(xyz)
        self.rotate = [-np.pi / 2, 0, 0]
        self.eye = np.array([3, 0, 0], dtype=np.float32)
        self.center = np.array([0, 0, 0], dtype=np.float32)
        self.up = np.array([0, 0, 1], dtype=np.float32)
        self.fovy = 45.0
        self.aspect_ratio = aspect_ratio
        self.z_near = 0.1
        self.z_far = 10.0

    def compute_front(self):
        return normalize(self.center - self.eye)

    def build_mvp(self):
        return build_mvp(self)

    def update_up(self, ref_up=None):
        front = self.compute_front()
        up = self.up if ref_up is None else ref_up
        right = compute_right(front, up)
        self.up = normalize(np.cross(right, front))


class CenteredCamera(BaseCamera):
    def set_eye(self, new_eye, ref_up=None):
        self.eye = np.asarray(new_eye, dtype=np.float32)
        self.update_up(ref_up)


class FreeViewpointCamera(BaseCamera):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rotate = [0, 0, 0]
        self.roll = 0
        self.pitch = 0
        self.yaw = 0
        # self.update_rotation()

    def update_rotation(self, dyaw, dpitch, droll):
        # yaw
        rotate = matrix33.create_from_axis_rotation(self.up, dyaw, dtype=np.float32)
        front = self.compute_front()
        self.center = self.eye + normalize(rotate @ front)

        # pitch
        front = self.compute_front()
        right = compute_right(front, self.up)
        rotate = matrix33.create_from_axis_rotation(right, dpitch, dtype=np.float32)
        self.center = self.eye + normalize(rotate @ front)
        self.up = normalize(rotate @ self.up)

        # roll
        front = self.compute_front()
        rotate = matrix33.create_from_axis_rotation(front, droll, dtype=np.float32)
        self.up = normalize(rotate @ self.up)

    def rotate_view(self, droll, dyaw, dpitch):
        self.update_rotation(dyaw, dpitch, droll)

    def move_forward(self, delta):
        front = self.compute_front()
        fwd = front * delta
        self.move(fwd)

    def move_side(self, delta):
        front = self.compute_front()
        right = compute_right(front, self.up)
        side = right * delta
        self.move(side)

    def move_up(self, delta):
        self.move(self.up * delta)

    def move(self, fwd):
        self.eye += fwd
        self.center += fwd
