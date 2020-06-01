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
from src.postprocess import non_mamima_suppression

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
    total_tests = len(dataset_x)

    # actual results
    dataset_y = np.array([ cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) for img_path in dataset_y ])

    # using magic point
    mp_dataset_x = [ cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) for img_path in dataset_x ]
    mp_dataset_x = np.array( [ i.reshape(screen_size[0], screen_size[1], 1) for i in mp_dataset_x ] )

    mp_dataset_x = mp_dataset_x.astype("float") / 255.0
    for i in range(0, len(mp_dataset_x)):
        mp_dataset_x[i] = mp_dataset_x[i].astype("float") - mp_dataset_x[i].mean()

    mp_yhat = mp_model.predict(mp_dataset_x)
    window = 10
    thresh = 0.15
    mp_yhat = np.array([non_mamima_suppression(yh, window, thresh) for yh in mp_yhat])

    # using orb
    orb_dataset_x = np.array([ cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) for img_path in dataset_x ])
    
    for i in range(0, total_tests):
        if i > 4:
            break

        # compute orb keypoints
        kp = orb_model.detect(orb_dataset_x[i], None)
        # draw keypoints
        orb_result = cv2.drawKeypoints(orb_dataset_x[i], kp, None, color=(0, 255, 0), flags=0)

        # compute AP of points
        actual_points = 0
        for row in range(0, len(dataset_y[i])):
            for col in range(0, len(dataset_y[i][0])):
                if dataset_y[i][row][col] > 0:
                    actual_points += 1

        print('Actual Points: {}'.format(actual_points))

        # magic point
        mp_total_px = 0
        mp_correct_px = 0

        image = mp_yhat[i].reshape(screen_size[1], screen_size[0], 1)
        for row in range(0, image.shape[0]):
            for col in range(0, image.shape[1]):

                # filter for response
                if image[row][col] > 0:
                    mp_total_px += 1

                    if correct((row, col), dataset_y[i]) == True:
                        mp_correct_px += 1

        print('Magic Point: {}/{}'.format(mp_correct_px, mp_total_px))

        # orb detector
        orb_total_px = len(kp)
        orb_correct_px = 0

        for point in kp:
            
            if correct((int(point.pt[1]), int(point.pt[0])), dataset_y[i]) == True:
                orb_correct_px += 1

        print('ORB: {}/{}'.format(orb_correct_px, orb_total_px))

        # plot the two results
        _, ax = plt.subplots(nrows=1, ncols=4,figsize=(15,10))
        plt.title("{}".format(dataset_x[i]))
        ax[0].imshow(mp_dataset_x[i].reshape(screen_size[1],screen_size[0]), cmap='gray')
        ax[1].imshow(dataset_y[i], cmap='gray')
        ax[2].imshow(mp_dataset_x[i].reshape(screen_size[1],screen_size[0]), cmap='gray')
        ax[2].imshow(mp_yhat[i].reshape(screen_size[1], screen_size[0]), cmap='jet', alpha=0.5)
        ax[3].imshow(orb_result)
        plt.show()

    # plot all of the actual results
