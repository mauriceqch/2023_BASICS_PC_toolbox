import argparse

import moderngl_window as mglw

from render.utils import read_pc
from render.renderers import InteractiveRenderer
from render.camera import FreeViewpointCamera

LSHIFT = 65505

class CameraController:
    def __init__(self):
        self.move_speed = 2
        self.rotate_speed = 0.005
        self.forward = 0
        self.side = 0
        self.up = 0
        self.roll = 0

    def on_render(self, camera, frametime):
        if self.forward != 0:
            delta = self.forward * frametime * self.move_speed
            camera.move_forward(delta)
        if self.side != 0:
            delta = self.side * frametime * self.move_speed
            camera.move_side(delta)
        if self.up != 0:
            delta = self.up * frametime * self.move_speed
            camera.move_up(delta)
        if self.roll != 0:
            delta = self.roll * frametime * self.move_speed
            camera.rotate_view(delta, 0, 0)

    def key_event(self, wnd, key, action, modifiers):
        if action == wnd.keys.ACTION_PRESS:
            if key == wnd.keys.Z or key == wnd.keys.W:
                self.forward = 1
            if key == wnd.keys.S:
                self.forward = -1
            if key == wnd.keys.D:
                self.side = 1
            if key == wnd.keys.Q or key == wnd.keys.A:
                self.side = -1
            if key == wnd.keys.SPACE:
                self.up = 1
            if key == LSHIFT:
                self.up = -1
            if key == wnd.keys.E:
                self.roll = 1
            if key == wnd.keys.R:
                self.roll = -1
            # Key releases
        elif action == wnd.keys.ACTION_RELEASE:
            if key == wnd.keys.Z or key == wnd.keys.W or key == wnd.keys.S:
                self.forward = 0
            if key == wnd.keys.D or key == wnd.keys.Q or key == wnd.keys.A:
                self.side = 0
            if key == wnd.keys.SPACE or key == LSHIFT:
                self.up = 0
            if key == wnd.keys.R or key == wnd.keys.E:
                self.roll = 0

    def mouse_drag_event(self, camera, x, y, dx, dy):
        camera.rotate_view(0, dx  * self.rotate_speed, dy * self.rotate_speed)


class InteractiveWindow(mglw.WindowConfig):
    gl_version = (3, 3)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ctx.enable_only(self.ctx.DEPTH_TEST)
        xyz, rgb = read_pc(self.argv.input_path)
        self.camera = FreeViewpointCamera(xyz, self.wnd.aspect_ratio)
        self.camera_controller = CameraController()
        self.renderer = InteractiveRenderer(self.ctx, xyz, rgb)

    def render(self, time, frametime):
        self.camera_controller.on_render(self.camera, frametime)
        self.ctx.clear(0.0, 0.0, 0.0, 0.0)
        self.renderer(self.camera)

    def key_event(self, key, action, modifiers):
        self.camera_controller.key_event(self.wnd, key, action, modifiers)

    def mouse_drag_event(self, x, y, dx, dy):
        self.camera_controller.mouse_drag_event(self.camera, x, y, dx, dy)

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument('input_path', type=str)


if __name__ == '__main__':
    InteractiveWindow.run()
