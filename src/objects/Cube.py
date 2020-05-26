from src.ImageObject import ImageObject

import random
import numpy as np

from OpenGL.GL import *


class Cube(ImageObject):

    def __init__(self, base_color, screen_size):
        ImageObject.__init__(self, base_color, screen_size)
        self.vertices = [
            [0.0, 0.0, 0.25],
            [0.25, 0.0, 0.25],
            [0.25, 0.25, 0.25],
            [0.0, 0.25, 0.25],
            [0.0, 0.0, 0.0],
            [0.25, 0.0, 0.0],
            [0.25, 0.25, 0.0],
            [0.0, 0.25, 0.0]
        ]

        self.normals = [
            (0.0, 0.0, 1.0), # front
            (0.0, 0.0, -1.0), # back
            (1.0, 0.0, 0.0), # right
            (-1.0, 0.0, 0.0), # left
            (0.0, 1.0, 0.0), # top
            (0.0, -1.0, 1.0)  # bottom
        ]

        self.vertex_indices = [
            (0, 1, 2, 3), # front
            (4, 5, 6, 7), # back
            (1, 5, 6, 2), # right
            (0, 4, 7, 3), # left
            (3, 2, 6, 7), # top
            (0, 1, 5, 4)  # bottom
        ]

        self.tex_coords = [
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0)
        ]

        # tweak object shape
        height_adjust = (-1*random.randint(0, 1)) * random.uniform(0.0, 0.20)
        width_adjust = (-1*random.randint(0, 1)) * random.uniform(0.0, 0.20)
        depth_adjust = (-1*random.randint(0, 1)) * random.uniform(0.0, 0.20)

        # height
        self.vertices[2][1] += height_adjust
        self.vertices[3][1] += height_adjust
        self.vertices[6][1] += height_adjust
        self.vertices[7][1] += height_adjust

        # width 
        self.vertices[1][0] += width_adjust
        self.vertices[2][0] += width_adjust
        self.vertices[5][0] += width_adjust
        self.vertices[6][0] += width_adjust

        # depth 
        self.vertices[0][2] += depth_adjust
        self.vertices[1][2] += depth_adjust
        self.vertices[2][2] += depth_adjust
        self.vertices[3][2] += depth_adjust

        # Color
        self.line_color = (0.0, 0.0, 0.0)
        if self.color[0] > 0.5:
            diff = (1.0 - self.color[0] )
            self.line_color = (diff, diff, diff)
        else:
            diff = self.color[0]
            color = 1.0 - diff
            self.line_color = (color, color, color)
            

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
        for face_no in range(0, len(self.vertex_indices)):
            glNormal3d(*self.normals[face_no])

            v_idxs = self.vertex_indices[face_no]

            i = 0
            for v in v_idxs:
                if self.has_texture:
                    glTexCoord(*self.tex_coords[i])

                glVertex(*self.vertices[v])
                i += 1
        glEnd()

        glLineWidth(1)
        glColor(*self.line_color)
        glBegin(GL_LINES)
        for face_no in range(0, len(self.vertex_indices)):

            v1, v2, v3, v4 = self.vertex_indices[face_no]

            glVertex(*self.vertices[v1])
            glVertex(*self.vertices[v2])

            glVertex(*self.vertices[v2])
            glVertex(*self.vertices[v3])

            glVertex(*self.vertices[v3])
            glVertex(*self.vertices[v4])

            glVertex(*self.vertices[v4])
            glVertex(*self.vertices[v1])

        glEnd()
        
        self.render_vertices(vertex_highlighting=vertex_highlighting)
