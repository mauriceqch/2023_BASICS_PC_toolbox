import argparse

import moderngl_window as mglw

from render.utils import read_pc, InteractiveRenderer, FreeViewpointCamera

LSHIFT = 65505


class InteractiveWindow(mglw.WindowConfig):
    gl_version = (3, 3)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ctx.enable_only(self.ctx.DEPTH_TEST)
        xyz, rgb = read_pc(self.argv.input_path)
        self.camera = FreeViewpointCamera(xyz, self.wnd.aspect_ratio)
        self.renderer = InteractiveRenderer(self.ctx, xyz, rgb)

        self.move_speed = 2
        self.rotate_speed = 0.005
        self.forward = 0
        self.side = 0
        self.up = 0
        self.roll = 0

    def render(self, time, frametime):
        if self.forward != 0:
            delta = self.forward * frametime * self.move_speed
            self.camera.move_forward(delta)
        if self.side != 0:
            delta = self.side * frametime * self.move_speed
            self.camera.move_side(delta)
        if self.up != 0:
            delta = self.up * frametime * self.move_speed
            self.camera.move_up(delta)
        if self.roll != 0:
            delta = self.roll * frametime * self.move_speed
            self.camera.rotate_view(delta, 0, 0)

        self.ctx.clear(0.0, 0.0, 0.0, 0.0)
        self.renderer(self.camera)

    def key_event(self, key, action, modifiers):
        print(key, action, modifiers)

        # Key presses
        if action == self.wnd.keys.ACTION_PRESS:
            if key == self.wnd.keys.Z or key == self.wnd.keys.W:
                self.forward = 1
            if key == self.wnd.keys.S:
                self.forward = -1
            if key == self.wnd.keys.D:
                self.side = 1
            if key == self.wnd.keys.Q or key == self.wnd.keys.A:
                self.side = -1
            if key == self.wnd.keys.SPACE:
                self.up = 1
            if key == LSHIFT:
                self.up = -1
            if key == self.wnd.keys.E:
                self.roll = 1
            if key == self.wnd.keys.R:
                self.roll = -1
        # Key releases
        elif action == self.wnd.keys.ACTION_RELEASE:
            if key == self.wnd.keys.Z or key == self.wnd.keys.W or key == self.wnd.keys.S:
                self.forward = 0
            if key == self.wnd.keys.D or key == self.wnd.keys.Q or key == self.wnd.keys.A:
                self.side = 0
            if key == self.wnd.keys.SPACE or key == LSHIFT:
                self.up = 0
            if key == self.wnd.keys.R or key == self.wnd.keys.E:
                self.roll = 0

    def mouse_position_event(self, x, y, dx, dy):
        pass

    def mouse_drag_event(self, x, y, dx, dy):
        self.camera.rotate_view(0, dx  * self.rotate_speed, dy * self.rotate_speed)

    def mouse_scroll_event(self, x_offset: float, y_offset: float):
        pass

    def mouse_press_event(self, x, y, button):
        pass

    def mouse_release_event(self, x: int, y: int, button: int):
        pass

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument('input_path', type=str)


if __name__ == '__main__':
    InteractiveWindow.run()
