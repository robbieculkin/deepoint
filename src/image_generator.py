#
# file:         ImageGenerator.py
#
# description:  Main driver for creating new images
#
# Define something  that is independent of the object
# that you are warping
#
# inputs to the network should be a
# 640 x 480 numy array
#
# should be a generator that keeps yielding new
# images
#
# *** just focus on generating one image ***
# try doing on-the-fly generation
#

import argparse
import random
import numpy as np
import cv2
import time
import threading
import os

import pygame
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *
from PIL import Image

from scipy.ndimage import gaussian_filter
from src.texture import salt_and_pepper, random_uniform, AdvancedTextures
from matplotlib import pyplot as plt

from src.objects.Ellipse import Ellipse
from src.objects.Checkerboard import Checkerboard
from src.objects.Cube import Cube
from src.objects.Quad import Quad
from src.objects.Triangle import Triangle
from src.objects.Line import Line
from src.objects.Star import Star
from src.objects.Background import Background

# SCREEN_SIZE = (640, 480)
SCREEN_SIZE = (200, 200)
TEXTURE_DIM = (128, 128)
# if 0, treat it as if it were a mac display, where the screen buffer only reads one
# quarter of the information
# if 1, treat it as if it were a bigger dislpay, where the screen can buffer the entire
# thing with the appropriate indexing
DISPLAY_MODE = 0

NONE_PROB = 0.25
OBJECT_DEFS = {
    'ellipse': Ellipse,
    'checkerboard': Checkerboard,
    'cube': Cube,
    'quad': Quad,
    'star': Star,
    'line': Line,
    'triangle': Triangle,
    'none' : None
}

''' Rendering Functions '''
def renderTextures():
    # generate all the textures
    noise = AdvancedTextures(*TEXTURE_DIM)
    img_random = random_uniform(*TEXTURE_DIM)
    img_snp = salt_and_pepper(*TEXTURE_DIM)
    img_cloud = noise('cloud')
    img_wood = noise('wood')
    img_marble = noise('marble')

    # originally 0-2
    sigmas = [ random.randint(0, 3) for i in range(5) ]

    # apply a gaussian blur
    img_random = gaussian_filter(img_random, sigma=sigmas[0])
    img_snp = gaussian_filter(img_snp, sigma=sigmas[1])
    img_cloud = gaussian_filter(img_cloud, sigma=sigmas[2])
    img_wood = gaussian_filter(img_wood, sigma=sigmas[3])
    img_marble = gaussian_filter(img_marble, sigma=sigmas[4])

    # save the images
    plt.imsave( os.path.join(os.path.dirname(__file__), 'textures/texture1.png'), img_random, cmap='gray', vmin=0, vmax=255)
    plt.imsave( os.path.join(os.path.dirname(__file__), 'textures/texture2.png'), img_snp, cmap='gray', vmin=0, vmax=255)
    plt.imsave( os.path.join(os.path.dirname(__file__), 'textures/texture3.png'), img_cloud, cmap='gray', vmin=0, vmax=255)
    plt.imsave( os.path.join(os.path.dirname(__file__), 'textures/texture4.png'), img_wood, cmap='gray', vmin=0, vmax=255)
    plt.imsave( os.path.join(os.path.dirname(__file__), 'textures/texture5.png'), img_marble, cmap='gray', vmin=0, vmax=255)

def renderObjects(ObjectList, render_vertices):

    for object in ObjectList:
        glPushMatrix()
        object.render(render_vertices)
        glPopMatrix()

''' Helper Functions '''
#!!! Might have to change
def screen_capture(display_mode, screen_size):
    glReadBuffer(GL_BACK)

    # reading pixels from the rendered screen
    if display_mode == 0:
        pxLoLeft = glReadPixels(0, 0, screen_size[0], screen_size[1], GL_RGB, GL_UNSIGNED_BYTE)
        pxLoRight = glReadPixels(screen_size[0], 0, screen_size[0], screen_size[1], GL_RGB, GL_UNSIGNED_BYTE)
        pxHiLeft = glReadPixels(0, screen_size[1], screen_size[0], screen_size[1], GL_RGB, GL_UNSIGNED_BYTE)
        pxHiRight = glReadPixels(screen_size[0], screen_size[1], screen_size[0], screen_size[1], GL_RGB, GL_UNSIGNED_BYTE)
        pixels = [pxLoLeft, pxLoRight, pxHiLeft, pxHiRight]
        # saving pixels to an image
        resultImage = Image.new("RGB", (screen_size[0]*2, screen_size[1]*2))

        image = Image.frombytes("RGB", screen_size, pixels[2])
        image = image.transpose( Image.FLIP_TOP_BOTTOM )
        resultImage.paste(image, None)

        image = Image.frombytes("RGB", screen_size, pixels[3])
        image = image.transpose( Image.FLIP_TOP_BOTTOM )
        resultImage.paste(image, (screen_size[0], 0))

        image = Image.frombytes("RGB", screen_size, pixels[0])
        image = image.transpose( Image.FLIP_TOP_BOTTOM )
        resultImage.paste(image, (0, screen_size[1]))

        image = Image.frombytes("RGB", screen_size, pixels[1])
        image = image.transpose( Image.FLIP_TOP_BOTTOM )
        resultImage.paste(image, (screen_size[0], screen_size[1]))

        # resize the image
        resizedImage = resultImage.resize(screen_size)

    elif display_mode == 1:
        pixels = glReadPixels(0, 0, screen_size[0], screen_size[1], GL_RGB, GL_UNSIGNED_BYTE)
        image = Image.frombytes("RGB", screen_size, pixels)
        image = image.transpose( Image.FLIP_TOP_BOTTOM )
        resizedImage = image

    return resizedImage

def highlight_vertices(output, screen_size):
    # performance is generally slow, but hopefully should be fast enouch
    # to not be the slowest link in data generation
    images = []
    masks = []
    for i in range(0, len(output)):
        img, mask, pixels = output[i]
        img = np.array(img)
        mask = np.array(mask)
        mask_out = np.zeros(mask.shape)
        
        images.append(img)

        for pixel in pixels:
            if pixel[0] < 0 or pixel[0] > screen_size[0]-1:
                continue
            elif pixel[1] < 0 or pixel[1] > screen_size[1]-1:
                continue
            
            current_pixel = mask[pixel[1], pixel[0]]
            if current_pixel[1] > current_pixel[0] and current_pixel[1] > current_pixel[2]:
                mask_out[pixel[1], pixel[0]] = 255, 255, 255
            else:
                # see if there are any neighboring green pixels 
                # for one off errors
                hasGreenNeighbor = False
                for row in (-1, 2):
                    for col in (-1, 2):
                        if row == 0 and col == 0:
                            continue

                        if (0 <= current_pixel[1]+col < screen_size[1] ) and (0 <= current_pixel[0]+row < screen_size[0] ):
                            px = mask[current_pixel[1]+col, current_pixel[0]+row]
                            if px[1] > px[0] and px[1] > px[2]:
                                hasGreenNeighbor = True

                if hasGreenNeighbor == True:
                    mask_out[pixel[1], pixel[0]] = 255, 255, 255

        masks.append(mask_out)

    return images, masks

''' Core Functions '''
def resize(display_mode, width, height):
    """
    Re-calculate portions of the viewport such that rendering looks proper
    """
    if display_mode == 0:
        glViewport(0, 0, width*2, height*2) #!!! Might Have to Change
    elif display_mode == 1:
        glViewport(0, 0, width, height) #!!! Might Have to Change

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(60.0, float(width)/float(height), 0.1, 1000.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

def init():
    """
    Initialize the OpenGL environment as well as any other components before
    generating the image dataset.
    """
    glEnable(GL_DEPTH_TEST)

    glEnable(GL_TEXTURE_2D)
    glEnable(GL_LINE_SMOOTH)
    glClearColor(1.0, 1.0, 1.0, 0.0)

    glPointSize(10.0)

def render_images(display_mode=0, screen_size=(200, 200), object_types=[], count=1, object_count=1, frames_per_count=100, test=False):
    outputRenders = []

    if count < 1:
        count = 1

    # render textures
    renderTextures()

    try:
        # render screen
        pygame.init()
        screen = pygame.display.set_mode(screen_size, HWSURFACE|OPENGL|DOUBLEBUF)

        # initializing opengl
        resize(display_mode, screen_size[0], screen_size[1])
        init()

        # initializing render sim
        frame = 0
        frames_per_count = frames_per_count
        frames_per_render = int(frames_per_count/2)
        total_generated = 0
        render_vertices = False

        # objects to render
        BaseColor = round(random.random(), 2)
        ObjectList = [Background(BaseColor, screen_size), Triangle(BaseColor, screen_size)]

        while True:

            for event in pygame.event.get():
                if event.type == QUIT:
                    return outputRenders
                if event.type == KEYUP and event.key == K_ESCAPE:
                    return outputRenders

            pressed = pygame.key.get_pressed()

            if test == True:
                if pressed[K_SPACE]:
                    resizedImage = screen_capture(display_mode=display_mode, screen_size=screen_size)
                    resizedImage.save("output.png")

            # frame management
            if test:
                render_vertices = True
            else:
                if total_generated == count:
                    # print('[INFO] rendering image {}/{}'.format(total_generated, count))
                    return outputRenders

                elif frame == 0:

                    # if (total_generated%(int(count/4)+1)) == 0: #+1 fixes mod by zero bug
                    #     print('[INFO] rendering image {}/{}'.format(total_generated, count))
                    # generate a new image
                    render_vertices = False

                    # generate new clear color
                    BaseColor = round(random.random(), 2)

                    # pick the objects to create
                    ObjectList = []
                    for object in object_types:
                        # remove all other objects if n
                        if object == 'none':
                            if random.random() <= NONE_PROB:
                                ObjectList = [Background(BaseColor, screen_size)]
                                break

                        # for all other objects
                        if object in OBJECT_DEFS.keys():
                            num_objects = random.randint(1, object_count)
                            for i in range(0, num_objects):
                                obj = OBJECT_DEFS[object]
                                if obj is not None:
                                    ObjectList.append(OBJECT_DEFS[object](BaseColor, screen_size))

                    # add the background last so it gets rendered last
                    ObjectList.append(Background(BaseColor, screen_size))

                elif frame == frames_per_render:
                    output = screen_capture(display_mode=display_mode, screen_size=screen_size)
                    outputRenders.append([output])
                    render_vertices = True

                elif frame == frames_per_count-1:
                    output = screen_capture(display_mode=display_mode, screen_size=screen_size)

                    pixels = []
                    for obj in ObjectList:
                        pixels.extend(obj.pixels)

                    outputRenders[-1].append(output)
                    outputRenders[-1].append(pixels)
                    total_generated += 1

            # clear the screen and the z-buffer
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glLoadIdentity()
            glTranslatef(0.0, 0.0, -1)

            # RENDER START
            renderObjects(ObjectList, render_vertices)
            # RENDER END

            pygame.display.flip()

            if not test:
                frame = (frame + 1) % frames_per_count
    finally:
        pygame.quit()

def render(display_mode=0, screen_size=(200, 200), object_types=[], count=1, object_count=1, frames_per_count=100, test=False):
    result = render_images(display_mode=display_mode, screen_size=screen_size, object_types=object_types, count=count, object_count=object_count, frames_per_count=frames_per_count, test=test)
    return result

def generate_images(
    object_types=[
        'ellipse',
        'checkerboard',
        'cube',
        'quad',
        'star',
        'line',
        'triangle',
        'none'], 
    batch_size=1, 
    object_count=1, 
    display_mode=0,
    shape = (200,200)):

    while(1):
        #generate
        image = render(display_mode=display_mode, screen_size=shape, object_types=object_types, count=batch_size, object_count=object_count, frames_per_count=5, test=False)
        image, vertex_mask = highlight_vertices(image, shape)
        # only need one color channel bc grayscale
        image = [i[:,:,0] for i in image]
        vertex_mask = [m[:,:,0] for m in vertex_mask]
        yield image, vertex_mask
