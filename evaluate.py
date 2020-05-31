#
# file:         evaluate.py
#
# description:  Code to test the ability for magic point to detect corners
#               versus the classical ORB detector
#

import cv2
import time
import os
import argparse
import glob

import tensorflow as tf
import keras
from tensorflow.keras.models import load_model

import numpy as np
import matplotlib.pyplot as plt

def info(msg):
    print('[INFO]: {}'.format(msg))

def error(msg):
    print('[ERROR]: {}'.format(msg))
    exit()

''' Main Driver '''
if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True, help="path to pre-trained model")
    ap.add_argument("-d", "--data", required=True, help="path to the dataset")
    args = vars(ap.parse_args())

    # header
    print('*** magic point network versus classical ORB detector ***')
    screen_size = (160, 120)

    ####################
    # Load Both Models #
    ####################
    # load the model
    mp_model = None
    model_path = os.path.join(os.path.dirname(__file__), args["model"])

    if os.path.isfile(model_path) == True:
        mp_model = load_model(model_path, custom_objects={'tf':tf})

    if mp_model is None:
        error('unable to find model ...')

    # set up ORB detector
    orb_model = cv2.ORB_create()

    ####################
    # Load the Dataset #
    ####################
    path_to_data = os.path.join(os.path.dirname(__file__), args['data'])
    if not os.path.isdir(path_to_data):
        error('unable to find data path')

    dataset = [file for file in glob.glob(path_to_data + "*.png", recursive=False)]

    dataset = [ cv2.imread(i, cv2.IMREAD_GRAYSCALE) for i in dataset ]
    dataset = [ cv2.resize(i, screen_size, interpolation=cv2.INTER_CUBIC) for i in dataset ]

    # # normalize the data
    # # dataset = [ i.astype("float")/255.0 for i in dataset ]
    # # dataset = [ i.astype("float")-i.mean() for i in dataset ]

    ###################
    # Show The Images # 
    ###################
    print('{} Images To Compare'.format(len(dataset)))
    for image in dataset:
        # use magic point
        
        # use orb
        kp = orb_model.detect(image)
        # compute descriptors
        kp, des = orb_model.compute(image, kp)
        # draw the keypoints
        orbImage = cv2.drawKeypoints(image,kp,np.array([]), color=(0,255,0), flags=0)

        # compare

        # display
        plt.imshow(orbImage)
        plt.title("ORB Detector")
        plt.show()


