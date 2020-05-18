#
# file:         ImageObject.py
#
# description:  Abstract Base Class for objects to render onto images, that can 
#               be randomly varied.
#

from texture import salt_and_pepper, random_uniform, AdvancedTextures

import random
import numpy as np
import os

from abc import ABC, abstractmethod
from OpenGL.GL import *

import pygame


class ImageObject(ABC):
    """
    An object that gets rendered onto an image. The object will have a general shape, but will
    have randomly generated qualities on each construction OR when commanded. Keeps track 
    and highlights the verticies for corner detection.
    """
    VERTEX_COLOR = (0, 1, 0)
    POS_WIDTH_LIMIT = 1
    POS_HEIGHT_LIMIT = 0.5
    ROT_LIMIT = 30  # 360

    def __init__(self, base_color):
        # color information
        self.base_color = base_color
        self.color = None
        # make texture
        self.texture = None
        self.texture_data = None
        self.texture_id = None
        self.has_texture = False
        self.tex_coords = []
        # object shape 
        self.vertices = []
        self.normals = []
        # object orientation
        self.pos = [] # x y z
        self.rot = [] # angle_x angle_y angle_z
        # setting object properties
        self.generate_color()
        self.generate_position()
        self.generate_texture()

    def __del__(self):
        if self.texture_id is not None:
            glDeleteTextures(self.texture_id)

    def generate_texture(self, wantTexture=False):
        # 50% change of texture
        if not wantTexture:
            if random.random() > 0.5:
                self.has_texture = False
                return
        
        self.has_texture = True
        # roll the dice to see what texture to use
        texture_idx = random.randint(1, 5)
        # loading texture image
        texture_path = os.path.join(os.path.dirname(__file__), "textures/texture{}.png".format(texture_idx))
        self.texture = pygame.image.load(texture_path)
        self.texture_data = pygame.image.tostring(self.texture, 'RGB', True)
        # loading texture id
        self.texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        # setting the texture parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        # unpacking pixel elements
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        width, height = self.texture.get_rect().size
        glTexImage2D( GL_TEXTURE_2D,
                        0,
                        3,
                        width,
                        height,
                        0,
                        GL_RGB,
                        GL_UNSIGNED_BYTE,
                        self.texture_data)

    def generate_color(self):
        if self.base_color <= 0.5:
            level = random.uniform(self.base_color+0.05, 1.0)
        else:
            level = random.uniform(0.0, self.base_color-0.05)

        self.color = (level, level, level)

    def generate_position(self):
        self.pos.append( random.uniform(-ImageObject.POS_WIDTH_LIMIT,ImageObject.POS_WIDTH_LIMIT) )
        self.pos.append( random.uniform(-ImageObject.POS_HEIGHT_LIMIT,ImageObject.POS_HEIGHT_LIMIT) )
        self.pos.append( -1 * random.uniform(0, 2)) # adding depth

        self.rot = [ random.randint(-ImageObject.ROT_LIMIT, ImageObject.ROT_LIMIT) for i in range(0, 3) ]

    def transform_rot_pos(self):
        glTranslatef(self.pos[0], self.pos[1], self.pos[2])
        glRotatef(self.rot[0], 1, 0, 0)
        glRotatef(self.rot[1], 0, 1, 0)
        glRotatef(self.rot[2], 0, 0, 1)        

    @abstractmethod
    def render(self, vertex_highlighting=False):
        """
        Renders the object created from the generate shape method and highlights the 
        vertices if there are any. Store the piels from the frame buffer with and 
        without the highlighted vertices in case they are needed for saving an image.
        """
        pass

