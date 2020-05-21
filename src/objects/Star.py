from src.ImageObject import ImageObject

import random
import numpy as np

from OpenGL.GL import *


class Star(ImageObject):

    def __init__(self, base_color):
        ImageObject.__init__(self, base_color)
        self.vertices = [
            [0.0, 0.0, 0.0], # star center
            [0.0, 0.5, 0.0], # top
            [-0.5, 0.0, 0.0], # left arm
            [0.5, 0.0, 0.0], # right arm
            [-0.5, -0.5, 0.0], # left leg
            [0.5, -0.5, 0.0] # right leg
        ]

        self.width = random.randint(1, 300) 
        # tweak image shape
        for i in range(1, len(self.vertices)):
            v1 = (-1*random.randint(0, 1)) * random.uniform(0.0, 0.25)
            v2 = (-1*random.randint(0, 1)) * random.uniform(0.0, 0.25)
            v3 = (-1*random.randint(0, 1)) * random.uniform(0.0, 0.25)
            self.vertices[i] = [self.vertices[i][0]+v1, self.vertices[i][1]+v2, self.vertices[i][2]+v3]

    def render(self, vertex_highlighting=False):
        """
        Renders the object created from the generate shape method and highlights the 
        vertices if there are any. Store the piels from the frame buffer with and 
        without the highlighted vertices in case they are needed for saving an image.
        """   
        self.transform_rot_pos()
        glLineWidth(self.width)


        glColor(*self.color)
        glBegin(GL_LINES)
        for i in range(1, len(self.vertices)):
            glVertex(*self.vertices[0])
            glVertex(*self.vertices[i])
        glEnd()

        self.render_vertices(vertex_highlighting=vertex_highlighting)