from ImageObject import ImageObject

import random
import numpy as np

from OpenGL.GL import *


class CheckerSquare(object):

    def __init__(self, color, width, pos_x, pos_y):
        self.vertices = [
            [pos_x, pos_y, 0.0],
            [pos_x + width, pos_y, 0.0],
            [pos_x + width, pos_y + width, 0.0],
            [pos_x, pos_y + width, 0.0]
        ]

        self.color = [color, color, color]


class Checkerboard(ImageObject):

    def __init__(self, base_color):
        ImageObject.__init__(self, base_color)

        self.square_width = 0.25
        self.face_count = random.randint(2, 5)

        self.vertices = []
        self.squares = []

        lcorner_pos = [0.0, 0.0]
        for r in range(0, self.face_count+1):
            for c in range(0, self.face_count):
                self.vertices.append([lcorner_pos[0], lcorner_pos[1], 0.0])

                if r != self.face_count:
                    if self.color[0] <= 0.5:
                        color = self.color[0] + random.uniform(0.2, 1-self.color[0])
                    else:
                        color = self.color[0] - random.uniform(0.2, self.color[0])

                    self.squares.append(CheckerSquare(color=color, width=self.square_width, pos_x=lcorner_pos[0], pos_y=lcorner_pos[1]))

                lcorner_pos[0] += self.square_width

            self.vertices.append([lcorner_pos[0], lcorner_pos[1], 0.0])
            lcorner_pos[1] += self.square_width
            lcorner_pos[0] = 0.0

    def render(self, vertex_highlighting=False):
        """
        Renders the object created from the generate shape method and highlights the 
        vertices if there are any. Store the piels from the frame buffer with and 
        without the highlighted vertices in case they are needed for saving an image.
        """   
        self.transform_rot_pos()

        glColor(*self.color)
        glBegin(GL_QUADS)
        for square in self.squares:
            glColor(*square.color)
            for vertex in square.vertices:
                glVertex(*vertex)
        glEnd()

        # glColor(*self.color)
        # glBegin(GL_LINES)
        # for square in self.squares:
            
        #     v1, v2, v3, v4 = square.vertices

        #     glVertex(*v1)
        #     glVertex(*v2)

        #     glVertex(*v2)
        #     glVertex(*v3)

        #     glVertex(*v3)
        #     glVertex(*v4)

        #     glVertex(*v4)
        #     glVertex(*v1)

        # glEnd()

        if vertex_highlighting:
            glColor(*ImageObject.VERTEX_COLOR)
            glBegin(GL_POINTS)
            for vertex in self.vertices:
                glVertex(*vertex)
            glEnd()
