#
# file:         ImageObject.py
#
# description:  Abstract Base Class for objects to render onto images, that can 
#               be randomly varied.
#

import random
import numpy as np

from abc import ABC, abstractmethod
from OpenGL.GL import *


class ImageObject(ABC):
    """
    An object that gets rendered onto an image. The object will have a general shape, but will
    have randomly generated qualities on each construction OR when commanded. Keeps track 
    and highlights the verticies for corner detection.
    """
    VERTEX_COLOR = (0, 1, 0)

    def __init__(self):
        self.color = None
        self.vertices = None
        self.pos = None
        self.rot = None

    @abstractmethod
    def generate_shape(self, seed=None):
        """
        Randomly picks out the verticies to produce the shape in question. This function
        will fill in the vertex arrays, which will be used to render in the render step.

        Inputs:
            seed (int): the seed for the random number generator
        """
        pass

    @abstractmethod
    def render(self, vertex_highlighting=False):
        """
        Renders the object created from the generate shape method and highlights the 
        vertices if there are any. Store the piels from the frame buffer with and 
        without the highlighted vertices in case they are needed for saving an image.
        """
        pass

    @abstractmethod
    def render_vertices(self):
        """
        Responsible for rendering the vertex highlights for each shape. 
        """
        pass

    @abstractmethod
    def render_position(self):
        """
        Renders the object position. This should happen only once.
        """
        pass
