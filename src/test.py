
import random
import numpy as np
import cv2

import pygame
from pygame.locals import *
from OpenGL.GL import *
from PIL import Image

from objects.Triangle import Triangle

def init():

    glEnable(GL_DEPTH_TEST)
    glClearColor(0, 0, 0, 0.0)

    glMatrixMode(GL_PROJECTION)

    glShadeModel(GL_FLAT)
    glEnable(GL_COLOR_MATERIAL)

    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glLight(GL_LIGHT0, GL_POSITION, (0, 1, 1, 0))

    glPointSize(5.0)

def run():
    
    pygame.init()
    screen_dimensions = (640, 480)
    screen = pygame.display.set_mode(screen_dimensions, HWSURFACE|OPENGL|DOUBLEBUF)
    pixelArray = None 

    TriangleList = []
    for triangle in range(0, 1):
        TriangleList.append(Triangle())

    # initializing opengl 
    init()

    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                return
        
        pressed = pygame.key.get_pressed()

        # clear the screen and the z-buffer
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # RENDERING THE TRIANGLE AND POINTS
        for triangle in TriangleList:
            glPushMatrix()
            triangle.render()
            glPopMatrix()

        if pressed[K_UP]:
            glReadBuffer(GL_FRONT)
            pixels = glReadPixels(0, 0, screen_dimensions[0], screen_dimensions[1], GL_RGB, GL_UNSIGNED_BYTE)
            
            image = Image.frombytes("RGB", screen_dimensions, pixels)
            image = image.transpose( Image.FLIP_TOP_BOTTOM )
            print(image.size)
            image.save('output.png')

        pygame.display.flip()

if __name__ == '__main__':
    run()

