import itertools

import moderngl_window as mglw
import numpy as np

from render.colors import z_color, d1_color
from render.utils import read_pc
from render.renderers import InteractiveRenderer
from render.camera import FreeViewpointCamera
from render.text import TextRenderer, render_multiline_text

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
        camera.rotate_view(0, dx * self.rotate_speed, dy * self.rotate_speed)


class InteractiveWindow(mglw.WindowConfig):
    gl_version = (3, 3)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ctx.enable_only(self.ctx.DEPTH_TEST)
        input_paths = self.argv.input_paths
        self.wnd.title = ', '.join(input_paths)

        self.data = []
        for input_path in input_paths:
            xyz, attrs = read_pc(input_path)
            self.data.append({'path': input_path, 'xyz': xyz, 'attrs': attrs, 'cube_size': 1})

        self.build_colors()

        for pc in self.data:
            pc['renderer'] = InteractiveRenderer(self.ctx, pc['xyz'], pc['colors'][self.sel_color])

        self.camera = FreeViewpointCamera(self.data[0]['xyz'], self.wnd.aspect_ratio / len(self.data))
        self.camera_controller = CameraController()
        self.text_renderer_vp = TextRenderer(self.ctx, self.wnd.width // len(self.data), self.wnd.height)
        self.text_renderer = TextRenderer(self.ctx, self.wnd.width, self.wnd.height)
        self.display_help = False
        self.selected_pc = 0

    def build_colors(self):
        zmin, zmax = np.mean([np.quantile(pc['xyz'][:, 2], [0.01, 0.99]) for pc in self.data], axis=0)
        for pc in self.data:
            colors = {}
            xyz = pc['xyz']
            attrs = pc['attrs']
            if 'rgb' in attrs:
                rgb = attrs['rgb']
                colors['rgb'] = rgb
            z_rgb = z_color(xyz, zmin, zmax)
            colors['z'] = z_rgb
            pc['colors'] = colors

        if len(self.data) == 2:
            colors1, colors2 = d1_color(self.data[0]['xyz'], self.data[1]['xyz'])
            self.data[0]['colors']['D1'] = colors1
            self.data[1]['colors']['D1'] = colors2

        colorset = set.intersection(*[set(x['colors'].keys()) for x in self.data])
        self.sel_color = 'rgb' if 'rgb' in colorset else 'z'
        self.avail_colors = itertools.cycle(colorset)

    def render(self, time, frametime):
        self.camera_controller.on_render(self.camera, frametime)
        self.ctx.clear(0.0, 0.0, 0.0, 0.0)

        n_viewports = len(self.data)
        w = self.wnd.width // n_viewports
        for vp in range(n_viewports):
            self.ctx.viewport = (w * vp, 0, w, self.wnd.height)
            cur_data = self.data[vp]
            cur_data['renderer'](self.camera)
            text = [self.data[vp]['path'], f'Cube size: {cur_data["cube_size"]}']
            if self.selected_pc == vp:
                text.insert(0, 'Selected')
            render_multiline_text(self.text_renderer_vp, text, 20, 20, 1)

        self.ctx.viewport = (0, 0, self.wnd.width, self.wnd.height)
        render_multiline_text(self.text_renderer, [f'Color: {self.sel_color}', 'Press h for help'],
                              20, self.wnd.height - 30, -1)
        if self.display_help:
            render_multiline_text(self.text_renderer, ['ZQSD/WASD: move', 'ER: roll', 'Mouse drag: rotate',
                                                       'Space/LSHIFT: rise/fall', 'V/B: Change cube size'],
                                  20, self.wnd.height - 120, -1)

    def resize(self, width: int, height: int):
        n_viewports = len(self.data)
        self.camera.aspect_ratio = width / (height * n_viewports)
        self.text_renderer_vp.set_resolution(width // n_viewports, height)
        self.text_renderer.set_resolution(width, height)

    def key_event(self, key, action, modifiers):
        self.camera_controller.key_event(self.wnd, key, action, modifiers)
        if action == self.wnd.keys.ACTION_PRESS:
            if key == self.wnd.keys.H:
                self.display_help = not self.display_help
            n_viewports = len(self.data)
            if key == self.wnd.keys.B:
                cur_data = self.data[self.selected_pc]
                cur_data['cube_size'] += 1
                cur_data['renderer'].set_cube_size(cur_data['cube_size'])
            if key == self.wnd.keys.V:
                cur_data = self.data[self.selected_pc]
                cur_data['cube_size'] = max(1, cur_data['cube_size'] - 1)
                cur_data['renderer'].set_cube_size(cur_data['cube_size'])
            if key == self.wnd.keys.C:
                self.sel_color = next(self.avail_colors)
                for vp in range(n_viewports):
                    self.data[vp]['renderer'].set_rgb(self.data[vp]['colors'][self.sel_color])
            if key == self.wnd.keys.N:
                self.selected_pc = (self.selected_pc + 1) % len(self.data)

    def mouse_drag_event(self, x, y, dx, dy):
        self.camera_controller.mouse_drag_event(self.camera, x, y, dx, dy)

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument('input_paths', type=str, nargs='+')


if __name__ == '__main__':
    InteractiveWindow.run()
