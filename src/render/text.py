import pathlib
import moderngl
import freetype
import numpy as np
from pyrr.matrix44 import create_orthogonal_projection

fontfile = str(pathlib.Path(__file__).parent / 'Vera.ttf')


class CharacterSlot:
    def __init__(self, texture, glyph):
        self.texture = texture
        self.textureSize = (glyph.bitmap.width, glyph.bitmap.rows)

        if isinstance(glyph, freetype.GlyphSlot):
            self.bearing = (glyph.bitmap_left, glyph.bitmap_top)
            self.advance = glyph.advance.x
        elif isinstance(glyph, freetype.BitmapGlyph):
            self.bearing = (glyph.left, glyph.top)
            self.advance = None
        else:
            raise RuntimeError('unknown glyph type')


VERTEX_SHADER = """
        #version 330 core
        layout (location = 0) in vec4 vertex; // <vec2 pos, vec2 tex>
        out vec2 tex_coords;

        uniform mat4 projection;

        void main()
        {
            gl_Position = projection * vec4(vertex.xy, 0.0, 1.0);
            tex_coords = vertex.zw;
        }
       """


FRAGMENT_SHADER = """
        #version 330 core
        in vec2 tex_coords;
        out vec4 color;

        uniform sampler2D text;
        uniform vec3 textColor = vec3(1.0, 1.0, 1.0);

        void main()
        {    
            vec4 sampled = vec4(1.0, 1.0, 1.0, texture(text, tex_coords).r);
            color = vec4(textColor, 1.0) * sampled;
        }
        """


def load_chars(ctx):
    chars = {}
    # disable byte-alignment restriction
    face = freetype.Face(fontfile)
    face.set_char_size(24 * 64)
    # load first 128 characters of ASCII set
    for i in range(0, 128):
        face.load_char(chr(i))
        glyph = face.glyph
        bitmap = glyph.bitmap
        texture = ctx.texture((bitmap.width, bitmap.rows), 1, data=bytes(bitmap.buffer), samples=0, alignment=1, dtype='f1')
        texture.repeat_x = False
        texture.repeat_y = False
        texture.filter = (moderngl.LINEAR, moderngl.LINEAR)

        # now store character for later use
        chars[chr(i)] = CharacterSlot(texture, glyph)
    return chars


class TextRenderer:
    def __init__(self, ctx, w, h):
        self.ctx = ctx
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.prog = ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)
        self.set_resolution(w, h)
        self.texture_loc = 0
        self.prog['text'] = self.texture_loc
        self.vbo = ctx.buffer(reserve=6 * 4 * 4, dynamic=True)
        self.vao = ctx.vertex_array(self.prog, [(self.vbo, '4f4', 'vertex')])
        self.chars = load_chars(ctx)

    def set_resolution(self, w, h):
        projection = create_orthogonal_projection(0, w, 0, h, 0, 100, dtype=np.float32)
        self.prog['projection'].write(projection)

    def render_text(self, text, x, y, scale):
        self.ctx.enable(self.ctx.BLEND)
        for c in text:
            ch = self.chars[c]
            w, h = ch.textureSize
            w = w * scale
            h = h * scale

            xpos = x + ch.bearing[0] * scale
            ypos = y - (ch.textureSize[1] - ch.bearing[1]) * scale

            vertices = np.asarray([
                xpos, ypos + h, 0, 0,
                xpos, ypos, 0, 1,
                xpos + w, ypos, 1, 1,
                xpos, ypos + h, 0, 0,
                xpos + w, ypos, 1, 1,
                xpos + w, ypos + h, 1, 0
            ], dtype=np.float32)
            # render glyph texture over quad, update content of VBO memory
            self.vbo.write(bytes(vertices))
            # render quad
            ch.texture.use(location=self.texture_loc)
            self.vao.render(vertices=6)
            # now advance cursors for next glyph (note that advance is number of 1/64 pixels)
            x += (ch.advance >> 6) * scale
        self.ctx.disable(self.ctx.BLEND)


def render_multiline_text(text_renderer, text, x, y, direction):
    assert direction in [1, -1]
    line_height = 25
    scale = 1.0
    if direction == 1:
        text = reversed(text)
    for i, line in enumerate(text):
        text_renderer.render_text(line, x, y + direction * i * line_height, scale)