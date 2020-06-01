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
from sklearn import  metrics

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
        # if i > 10:
        #     break

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

        # print('Actual Points: {}'.format(actual_points))

        # PRECISION RECALL CURVE
        step_size = 0.1
        precision_recall_x  = np.arange(0, 1.0 + step_size, step_size)
        mp_ap_list = []
        orb_ap_list = []

        # magic point
        mp_precision_recall_y = np.zeros(len(precision_recall_x))

        for i_conf in range(1, len(precision_recall_x)):
            mp_tp = 0
            mp_fp = 0
            
            image = mp_yhat[i].reshape(screen_size[1], screen_size[0], 1)
            for row in range(0, image.shape[0]):
                for col in range(0, image.shape[1]):

                    # filter for response
                    if image[row][col] > 0 and (precision_recall_x[i_conf-1] < image[row][col] <= precision_recall_x[i_conf]):
                        if correct((row, col), dataset_y[i]) == True:
                            mp_tp += 1
                        else:
                            mp_fp += 1

            if mp_tp + mp_fp > 0: 
                mp_precision_recall_y[i_conf] = mp_tp / (mp_tp + mp_fp)
            else:
                mp_precision_recall_y[i_conf] = 0

        mp_ap = mp_precision_recall_y.sum()
        # print(mp_precision_recall_y)
        # print('Magic Point: {}'.format(mp_ap))
        mp_ap_list.append(mp_ap)


        # orb detector
        orb_precision_rexall_y = np.zeros(len(precision_recall_x))

        for i_conf in range(1, len(precision_recall_x)):
            orb_tp = 0
            orb_fp = 0

            for point in kp:
                
                if correct((int(point.pt[1]), int(point.pt[0])), dataset_y[i]) == True and (precision_recall_x[i_conf-1] < point.response <= precision_recall_x[i_conf]):
                    orb_tp += 1
                else:
                    orb_fp += 1

            if orb_tp + orb_fp > 0: 
                orb_precision_rexall_y[i_conf] = orb_tp / (orb_tp + orb_fp)
            else:
                orb_precision_rexall_y[i_conf] = 0

        orb_ap = orb_precision_rexall_y.sum()
        # print(orb_precision_rexall_y)
        # print('ORB Detector: {}'.format(orb_ap))
        orb_ap_list.append(orb_ap)

        # plot the two results
        # f, ax = plt.subplots(nrows=1, ncols=4,figsize=(15,10))
        # ax[0].imshow(mp_dataset_x[i].reshape(screen_size[1],screen_size[0]), cmap='gray')
        # ax[1].imshow(dataset_y[i], cmap='gray')
        # ax[2].imshow(mp_dataset_x[i].reshape(screen_size[1],screen_size[0]), cmap='gray')
        # ax[2].imshow(mp_yhat[i].reshape(screen_size[1], screen_size[0]), cmap='jet', alpha=0.5)
        # ax[3].imshow(orb_result)
        # plt.show()

    # plot all of the actual results
    mp_ap_mean = np.array(mp_ap_list).mean()
    orb_ap_mean = np.array(orb_ap_list).mean()

    print(mp_ap_mean)
    print(orb_ap_mean)
    plt.style.use("ggplot")
    plt.figure()
    bar_list = plt.bar(('Magic Point Network', 'ORB Feature Detector'), [mp_ap_mean, orb_ap_mean])
    bar_list[0].set_color('gray')
    bar_list[1].set_color('gray')
    plt.ylabel('Average Precision Mean')
    plt.yticks(precision_recall_x)
    plt.title('Mean of the Average Precision over the Synthetic Shapes Dataset')
    plt.show()
