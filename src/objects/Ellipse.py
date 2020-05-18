from src.ImageObject import ImageObject

import math
import random
import numpy as np

from OpenGL.GL import *


class Ellipse(ImageObject):

    def __init__(self, base_color):
        ImageObject.__init__(self, base_color)
        self.a = 0.5
        self.b = 0.25

        self.a = random.uniform(0.1, 0.5)
        self.b = random.uniform(0.1, 0.5)

        # x^2/a^2 + y^2/b^2 = 1
        # x, y = a*cos(t), bsin(t) for 0<t<2pi
        self.vertices = []

        for t in np.arange(0.0, 2.0 * math.pi, 0.1):
            x, y = self.a * math.cos(t), self.b * math.sin(t)
            self.vertices.append([x, y, 0.0])

    def render(self, vertex_highlighting=False):
        """
        Renders the object created from the generate shape method and highlights the 
        vertices if there are any. Store the piels from the frame buffer with and 
        without the highlighted vertices in case they are needed for saving an image.
        """   
        self.transform_rot_pos()

        glColor(*self.color)
        glBegin(GL_POLYGON)
        for i in range(0, len(self.vertices)):
            if i > 0:
                glVertex(*self.vertices[i-1])
            
            glVertex(*self.vertices[i])

        glVertex(*self.vertices[len(self.vertices)-1])
        glVertex(*self.vertices[0])

        glEnd()
