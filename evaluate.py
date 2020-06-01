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

''' Messaging Components '''
def info(msg):
    print('[INFO]: {}'.format(msg))

def error(msg):
    print('[ERROR]: {}'.format(msg))
    exit()

''' Evaluation Measures '''
def correct(pixel, truth_image, epsilon=4):
    """
    Determine if the current pixel is a correct corner 

    @param: pixel ( (x_coord, y_coord) ): x and y coordinates of the pixel
    @param: truth_image (np.array): image with the result of what actually is a corner or not
    @param: epsilon (int): neighborhood threshold for corner-ness
    """

    x_min, y_min = np.array(pixel) - epsilon
    x_max, y_max = np.array(pixel) + epsilon
    eps_window = truth_image[x_min:x_max, y_min:y_max]

    return eps_window.sum() > 0

def average_precision():
    pass

def localization_error():
    pass

def repeatability():
    pass


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
    path_to_datax = os.path.join(os.path.dirname(__file__), args['data'], 'images/')
    path_to_datay = os.path.join(os.path.dirname(__file__), args['data'], 'masks/')
    if not os.path.isdir(path_to_datax) or not os.path.isdir(path_to_datay):
        error('unable to find data paths')

    dataset_x = [file for file in glob.glob(path_to_datax + "*.png", recursive=False)]
    dataset_y = [file for file in glob.glob(path_to_datay + "*.png", recursive=False)]

    dataset_x = np.array([ plt.imread(i) for i in dataset_x ])
    dataset_y = np.array([ plt.imread(i) for i in dataset_y ])

    mp_dataset = dataset_x.copy()
    mp_dataset = np.array([m[:,:,0] for m in mp_dataset]).reshape((80,160,120,1))
    mp_dataset = mp_dataset.astype("float") / 255.0
    for i in range(0, len(mp_dataset)):
        mp_dataset[i] = mp_dataset[i].astype("float") - mp_dataset[i].mean()

    y_test_hat = mp_model.predict(mp_dataset)
 
    for i in range(0, len(dataset_x)):

        if i > 5:
            break

        _, ax = plt.subplots(nrows=1, ncols=4,figsize=(15,10))
        ax[0].imshow(dataset_x[i])
        ax[1].imshow(y_test_hat[i].reshape(120,160))
        plt.show()
