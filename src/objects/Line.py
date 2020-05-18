from src.ImageObject import ImageObject

import random
import numpy as np

from OpenGL.GL import *


class Line(ImageObject):

    def __init__(self, base_color):
        ImageObject.__init__(self, base_color)
        self.vertices = [
            [-0.5, 0.0, 0.0],
            [0.5, 0.0, 0.0]
        ]

        self.width = random.randint(1, 300) 
        # tweak image shape
        self.vertices[0][0] += (-1*random.randint(0, 1)) * random.uniform(0.0, 0.45)
        self.vertices[1][0] += (-1*random.randint(0, 1)) * random.uniform(0.0, 0.45)

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
        for i in range(0, len(self.vertices)):
            glVertex(*self.vertices[i])
        glEnd()

        if vertex_highlighting:
            glColor(*ImageObject.VERTEX_COLOR)
            glBegin(GL_POINTS)
            for v in self.vertices:
                glVertex(*v)
            glEnd()
