import cv2
import time
import os
import argparse
from src.image_generator import render, highlight_vertices, generate_images
from src.postprocess import non_mamima_suppression

import keras
from keras.optimizers import Adam, SGD
from keras.layers import Activation, Conv2D, Lambda, BatchNormalization, ZeroPadding2D, MaxPooling2D, Dropout, Softmax, Reshape
from tensorflow.keras.models import load_model
from keras.activations import softmax
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

''' Main Driver '''
if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True, help="path to pre-trained model")
    args = vars(ap.parse_args())

    screen_size = (160, 120)
    # screen_size = (640, 480)

    # data_generator = generate_images(object_types=[
    #     # 'ellipse',
    #     # 'checkerboard',
    #     # 'cube',
    #     # 'quad',
    #     # 'star',
    #     # 'line',
    #     # 'triangle',
    #     # 'none'
    # ],
    # batch_size=1000, 
    # object_count=50, 
    # display_mode=1,
    # shape = screen_size,
    # single_channel=True)

    # TRAINING GENERATORS
    tri_data_generator = generate_images(object_types=['triangle'], batch_size=1000, object_count=50, display_mode=1, shape = screen_size, single_channel=True)
    quad_data_generator = generate_images(object_types=['quad'], batch_size=1000, object_count=50, display_mode=1, shape = screen_size, single_channel=True)
    quadtri_data_generator = generate_images(object_types=['quad','triangle'], batch_size=1000, object_count=5, display_mode=1, shape = screen_size, single_channel=True)
    check_data_generator = generate_images(object_types=['checkerboard'], batch_size=1000, object_count=1, display_mode=1, shape = screen_size, single_channel=True)
    line_data_generator = generate_images(object_types=['line'], batch_size=1000, object_count=50, display_mode=1, shape = screen_size, single_channel=True)
    star_data_generator = generate_images(object_types=['star'], batch_size=1000, object_count=1, display_mode=1, shape = screen_size, single_channel=True)
    ellipse_data_generator = generate_images(object_types=['ellipse'], batch_size=1000, object_count=20, display_mode=1, shape = screen_size, single_channel=True)
    cube_data_generator = generate_images(object_types=['cube'], batch_size=1000, object_count=1, display_mode=1, shape = screen_size, single_channel=True)

    # make a list of generators to use for this instance
    used_data_generators = [
        tri_data_generator,
        quad_data_generator,
        quadtri_data_generator,
        check_data_generator,
        line_data_generator,
        star_data_generator,
        ellipse_data_generator,
        # cube_data_generator
    ]

    model = None
    model_path = os.path.join(os.path.dirname(__file__), args["model"])

    if os.path.isfile(model_path) == True:
        response = input("continue training/using from current weights of {}? (y/n) ".format(args['model']))
        if response.lower() == 'y' or response.lower() == 'yes':
            model = load_model(model_path, custom_objects={'tf':tf})
        else:
            print('Training model from scratch ...')

    onlyTesting = False
    if model is not None:
        response = input("skip to testing {} ? (y/n) ".format(args['model']))
        if response.lower() == 'y' or response.lower() == 'yes':
            onlyTesting = True

    if model is None:
        # building the model
        model = keras.models.Sequential()

        # VGG - Like Encoder: VGG19
        # C1 
        model.add(Conv2D(64,3, padding='same', activation='relu', input_shape=(screen_size[0],screen_size[1],1)))
        model.add(Conv2D(64,3, padding='same', activation='relu'))
        model.add(MaxPooling2D((2,2), strides=2))
        model.add(BatchNormalization())
        # C2 
        model.add(Conv2D(128,3, padding='same', activation='relu'))
        model.add(Conv2D(128,3, padding='same', activation='relu'))
        model.add(MaxPooling2D((2,2), strides=2))
        model.add(BatchNormalization())
        # C3 
        model.add(Conv2D(256,3, padding='same', activation='relu'))
        model.add(Conv2D(256,3, padding='same', activation='relu'))
        model.add(MaxPooling2D((2,2), strides=2))
        model.add(BatchNormalization())
        # C4
        model.add(Conv2D(256,3, padding='same', activation='relu'))
        model.add(Conv2D(256,3, padding='same', activation='relu'))
        model.add(BatchNormalization())

        # Explicit Decoder
        model.add(Conv2D(65,1, activation='softmax'))
        model.add(Lambda(lambda x: x[:,:,:,:-1]))
        model.add(Lambda(lambda x: tf.nn.depth_to_space(x, block_size=8)))

    # compile
    opt = SGD(lr=0.01, decay=0.01/40, momentum=0.9, nesterov=True)
    # try using MSE loss and MSE metrics
    model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=[keras.metrics.MeanIoU(num_classes=2), 'accuracy'])

    model.summary()

    # TRAINING START
    if not onlyTesting:
        print('*** START TRAINING ***')
        try:

            x_train, y_train = next(used_data_generators[0])

            for generator in used_data_generators[1:]:
                x, y = next(generator)
                x_train = np.concatenate((x_train, x))
                y_train = np.concatenate((y_train, y))

            model.fit(x_train, y_train, batch_size=32, epochs=32)

        except KeyboardInterrupt as e:
            pass
        finally:
            model.save(args["model"])
    # TRAINING END

    # TESTING START 
    print('*** START TESTING ***')
    test_data_generator = generate_images(object_types=[
            # 'ellipse',
            # 'checkerboard',
            # 'cube',
            # 'quad',
            # 'star',
            # 'line',
            'triangle',
            # 'none'
        ],
        batch_size=10, 
        object_count=1, 
        display_mode=1,
        shape = screen_size,
        single_channel=True)

    x_test, y_test = next(test_data_generator)

    y_test_hat = model.predict(x_test)

    #non-maxima suppression
    window = 10
    thresh = 0.15
    y_test_hat = np.array([non_mamima_suppression(yh, window, thresh) for yh in y_test_hat])

    for x,y,y_hat in zip(x_test,y_test,y_test_hat):

        f, ax = plt.subplots(nrows=1, ncols=4,figsize=(15,10))
        ax[0].imshow(x.reshape(screen_size[1],screen_size[0]))
        ax[1].imshow(y.reshape(screen_size[1],screen_size[0]))
        ax[2].imshow(y_hat.reshape(screen_size[1],screen_size[0]))

        # resulting figure
        ax[3].imshow(x.reshape(120,160), cmap='gray')
        ax[3].imshow(y_hat.reshape(120,160), cmap='jet', alpha=0.5)

        plt.show()
    # TESTING END
