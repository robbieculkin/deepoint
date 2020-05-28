from src.ImageObject import ImageObject

import random
import numpy as np

from OpenGL.GL import *


class Quad(ImageObject):

    def __init__(self, base_color, screen_size):
        ImageObject.__init__(self, base_color, screen_size)
        self.vertices = [
            [-0.2, 0.2, 0.0],
            [0.2, 0.2, 0.0],
            [0.2, -0.2, 0.0],
            [-0.2, -0.2, 0.0]
        ]
        self.tex_coords = [
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0)
        ]

        # tweak image shape
        self.vertices[0][0] += (-1*random.randint(0, 1)) * random.uniform(0.0, 0.15)
        self.vertices[1][1] += (-1*random.randint(0, 1)) * random.uniform(0.0, 0.15)
        self.vertices[2][0] += (-1*random.randint(0, 1)) * random.uniform(0.0, 0.15)
        self.vertices[3][0] += (-1*random.randint(0, 1)) * random.uniform(0.0, 0.15)

    def render(self, vertex_highlighting=False):
        """
        Renders the object created from the generate shape method and highlights the 
        vertices if there are any. Store the piels from the frame buffer with and 
        without the highlighted vertices in case they are needed for saving an image.
        """   
        if self.has_texture:
            glBindTexture(GL_TEXTURE_2D, self.texture_id)

        self.transform_rot_pos()

        glColor(*self.color)
        glBegin(GL_QUADS)
        for i in range(0, len(self.vertices)):
            if self.has_texture:
                glTexCoord(*self.tex_coords[i])
            glVertex(*self.vertices[i])
        glEnd()

        self.render_vertices(vertex_highlighting=vertex_highlighting)
