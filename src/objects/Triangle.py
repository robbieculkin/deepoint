#
# file:         Triangle.py
#
# description:  Generates a basic triangle
#

import random
import numpy as np

from OpenGL.GL import *
from ImageObject import ImageObject


class Triangle(ImageObject):

    def __init__(self):
        super().__init__()
        self.vertices = []
        self.pos = []
        self.rot = []
        self.generate_shape()

    def generate_shape(self, seed=None):
        if seed is None:
            random.seed()
        else:
            random.seed(seed)

        # generate color
        # random number between 0 and 1
        self.color = round(random.random(), 2)
        if self.color < 0.25:
            self.color = 0.25

        # generating shape
        self.vertices = []
        for v in range(0, 3):
            negate = [-1*random.choice([-1, 1]), -1*random.choice([-1, 1])]
            vert = (negate[0] * round(random.random()*0.5,2), negate[1] * round(random.random()*0.5,2))
            self.vertices.append(vert)

        # generating rotation
        rot_speed = random.randint(0, 360)
        negate = [-1*random.choice([-1, 1]), -1*random.choice([-1, 1])]
        self.rot = [rot_speed, random.random(), random.random(), random.random()]
        self.pos = [negate[0] * round(random.random()*0.5, 2), negate[1] * round(random.random()*0.5, 2), 0]

    def render(self, vertex_highlighting=False):
        self.render_position()
        
        if vertex_highlighting:
            self.render_vertices()

        glBegin(GL_TRIANGLES)
        glColor3f(self.color, self.color, self.color)
        for v in self.vertices:
            glVertex2f(v[0], v[1])
        glEnd()

    def render_vertices(self):
        glBegin(GL_POINTS)
        glColor3f(*ImageObject.VERTEX_COLOR)
        for v in self.vertices:
            glVertex2f(v[0], v[1])
        glEnd()  

    def render_position(self):
        glRotatef(self.rot[0], self.rot[1], self.rot[2], self.rot[3])
        glTranslatef(self.pos[0], self.pos[1], self.pos[2])
