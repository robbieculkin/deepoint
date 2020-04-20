#
# file:         ImageGenerator.py
#
# description:  Main driver for creating new images
#

import argparse
import random
import numpy as np
import cv2
import time
import threading

import pygame
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *
from PIL import Image

from objects.Triangle import Triangle


''' Core Functions '''
def init():
    """
    Initialize the OpenGL environment as well as any other components before 
    generating the image dataset.
    """
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LESS)

    glClearColor(0, 0, 0, 0.0)

    glMatrixMode(GL_MODELVIEW)

    glShadeModel(GL_FLAT)
    glEnable(GL_COLOR_MATERIAL)

    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glLight(GL_LIGHT0, GL_POSITION, (0, 1, 1, 0))

    glPointSize(5.0)

def resize(width, height):
    """
    Re-calculate portions of the viewport such that rendering looks proper
    """
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(60.0, float(width)/float(height), 0.1, 1000.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

def run(output_size, render_queue, generate_image=False):
    """
    Generate all of the objects to render through different random scenarious
    created by the objects up to a certian limit.
    """
    # initializing pygame engine
    pygame.init()
    screen_dimensions = (640, 480)
    screen = pygame.display.set_mode(screen_dimensions, HWSURFACE|OPENGL|DOUBLEBUF)
    
    # initializing the opengl environment
    # resize(*screen_dimensions)
    init()

    # initialize more
    frame = 0
    objectlist = []
    image_count = 0
    random_shape_trigger = 50
    frame_total = output_size * 2 * random_shape_trigger
    render_vertices = False

    # iterate through all the different scenarious that you would like to generate
    # randomly pick a different scenario on each iteration and render it
    while True:
        # check all events
        for event in pygame.event.get():
            if event.type == QUIT:
                return

        # frame == 0 means we finished all the different variations of the last
        # image object list and it is time to go onto the next one
        if frame == 0:
            if len(render_queue) > 0:
                objectlist = render_queue.pop(0)
                render_vertices = False
            else:
                # out of objects to render!
                print('out of objects to render!')
                return

        # check to see if shapes should be re-generated
        if (frame%(2*random_shape_trigger)) == 0:
            for obj in objectlist:
                obj.generate_shape()
            render_vertices = False

        if (frame%(2*random_shape_trigger)) > random_shape_trigger:
            render_vertices = True

        # clear the screen and the z-buffer
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # RENDER        
        for obj in objectlist:
            glPushMatrix()
            if render_vertices == True:
                obj.render(vertex_highlighting=True)
            else:
                obj.render(vertex_highlighting=False)
            glPopMatrix()
        # END RENDER

        if generate_image == True:
            if (frame%random_shape_trigger) == 0:
                glReadBuffer(GL_FRONT)
                pixels = glReadPixels(0, 0, screen_dimensions[0], screen_dimensions[1], GL_RGB, GL_UNSIGNED_BYTE)
                
                image = Image.frombytes("RGB", screen_dimensions, pixels)
                image = image.transpose( Image.FLIP_TOP_BOTTOM )
                # print(image.size)
                image.save('./images/output{}.png'.format(image_count))
                image_count += 1
        
        pygame.display.flip()
        frame = (frame+1)%frame_total


''' Main Driver '''
if __name__ == '__main__':

    # argument management
    ap = argparse.ArgumentParser()
    args = vars(ap.parse_args())

    RenderQueue = [
        [ Triangle() ],
        [ Triangle(), Triangle(), Triangle()]
    ]

    # Start Generating different settings of objects
    # starting the main driver for generating the dataset
    run(output_size=5, render_queue=RenderQueue, generate_image=False)

