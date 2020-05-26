from src.ImageObject import ImageObject

import random
import numpy as np

from OpenGL.GL import *


class Background(ImageObject):

    def __init__(self, base_color, screen_size):
        ImageObject.__init__(self, base_color, screen_size)
        self.vertex_pixels_calculated = True
        position = -10
        coord = 10
        self.vertices = [
            (coord, coord, position),
            (coord, -coord, position),
            (-coord, -coord, position),
            (-coord, coord, position)
        ]
        self.tex_coords = [
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0),
        ]
        self.normals = [
            (0, 0, 1.0)
        ]

        # apply a blur to the texture image

    def render(self, vertex_highlighting=False):
        """
        Renders the object created from the generate shape method and highlights the 
        vertices if there are any. Store the piels from the frame buffer with and 
        without the highlighted vertices in case they are needed for saving an image.
        """   
        if self.has_texture:
            glBindTexture(GL_TEXTURE_2D, self.texture_id)

        # glScale(5, 5, 1) 

        glColor(self.base_color, self.base_color, self.base_color)
        glBegin(GL_QUADS)
        for i in range(0, len(self.vertices)):
            glNormal3d(*self.normals[0])
            if self.has_texture:
                glTexCoord(*self.tex_coords[i])
            glVertex(*self.vertices[i])
        glEnd()

