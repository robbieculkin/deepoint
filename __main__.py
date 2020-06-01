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
    quadtri_data_generator = generate_images(object_types=['quad','triangle'], batch_size=1000, object_count=25, display_mode=1, shape = screen_size, single_channel=True)
    check_data_generator = generate_images(object_types=['checkerboard'], batch_size=500, object_count=1, display_mode=1, shape = screen_size, single_channel=True)
    line_data_generator = generate_images(object_types=['line'], batch_size=1000, object_count=50, display_mode=1, shape = screen_size, single_channel=True)
    star_data_generator = generate_images(object_types=['star'], batch_size=1000, object_count=50, display_mode=1, shape = screen_size, single_channel=True)
    ellipse_data_generator = generate_images(object_types=['ellipse'], batch_size=500, object_count=20, display_mode=1, shape = screen_size, single_channel=True)
    cube_data_generator = generate_images(object_types=['cube'], batch_size=1000, object_count=50, display_mode=1, shape = screen_size, single_channel=True)

    # VALIDATION GENERATORS
    v_tri_data_generator = generate_images(object_types=['triangle'], batch_size=200, object_count=5, display_mode=1, shape = screen_size, single_channel=True)
    v_quad_data_generator = generate_images(object_types=['quad'], batch_size=200, object_count=5, display_mode=1, shape = screen_size, single_channel=True)
    v_check_data_generator = generate_images(object_types=['checkerboard'], batch_size=200, object_count=1, display_mode=1, shape = screen_size, single_channel=True)
    v_line_data_generator = generate_images(object_types=['line'], batch_size=200, object_count=5, display_mode=1, shape = screen_size, single_channel=True)
    v_star_data_generator = generate_images(object_types=['star'], batch_size=200, object_count=5, display_mode=1, shape = screen_size, single_channel=True)
    v_ellipse_data_generator = generate_images(object_types=['ellipse'], batch_size=200, object_count=5, display_mode=1, shape = screen_size, single_channel=True)
    v_cube_data_generator = generate_images(object_types=['cube'], batch_size=200, object_count=5, display_mode=1, shape = screen_size, single_channel=True)


    # make a list of generators to use for this instance
    used_data_generators = [
        tri_data_generator,
        quad_data_generator,
        quadtri_data_generator,
        check_data_generator,
        line_data_generator,
        star_data_generator,
        ellipse_data_generator,
        cube_data_generator
    ]

    used_v_data_generators = [
        v_tri_data_generator,
        v_quad_data_generator,
        v_check_data_generator,
        v_line_data_generator,
        v_star_data_generator,
        v_ellipse_data_generator,
        v_cube_data_generator
    ]

    H = None
    model = None
    model_path = os.path.join(os.path.dirname(__file__), args["model"])

    if os.path.isfile(model_path) == True:
        response = input("continue training/using from current weights of {}? (y/n) ".format(args['model']))
        if 'y' in response.lower() or 'yes' in response.lower():
            model = load_model(model_path, custom_objects={'tf':tf})
        else:
            print('Training model from scratch ...')

    onlyTesting = False
    if model is not None:
        response = input("skip to testing {} ? (y/n) ".format(args['model']))
        if 'y' in response.lower() or 'yes' in response.lower():
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

            # Training Dataset
            print('Generating Training data...')
            start = time.time()
            x_train, y_train = next(used_data_generators[0])

            for generator in used_data_generators[1:]:
                x, y = next(generator)
                x_train = np.concatenate((x_train, x))
                y_train = np.concatenate((y_train, y))

            end = time.time()
            diff = end - start
            print("Time To Complete Rendering: {} seconds".format(diff))

            # Validation Dataset
            print('Generating Validation data...')
            x_valid, y_valid = next(used_v_data_generators[0])

            for generator in used_v_data_generators[1:]:
                x, y = next(generator)
                x_valid = np.concatenate((x_valid, x))
                y_valid = np.concatenate((y_valid, y))

            # do the training with train/validation
            H = model.fit(x_train, y_train, validation_data=(x_valid, y_valid), batch_size=32, epochs=64)

        except KeyboardInterrupt as e:
            pass
        finally:
            model.save(args["model"])
    # TRAINING END

    # TESTING START 
    print('*** START TESTING ***')
    test_data_generator = generate_images(object_types=[
            # 'ellipse',
            'checkerboard',
            # 'cube',
            # 'quad',
            # 'star',
            # 'line',
            # 'triangle',
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
        ax[0].imshow(x.reshape(screen_size[1],screen_size[0]), cmap='gray')
        ax[1].imshow(y.reshape(screen_size[1],screen_size[0]), cmap='gray')
        ax[2].imshow(y_hat.reshape(screen_size[1],screen_size[0]), cmap='gray')

        # resulting figure
        ax[3].imshow(x.reshape(120,160), cmap='gray')
        ax[3].imshow(y_hat.reshape(120,160), cmap='jet', alpha=0.5)

        plt.show()
    # TESTING END

    # Plot the training loss and accuracy
    if H is not None:
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(np.arange(0, 64), H.history["loss"], label="train_loss")
        plt.plot(np.arange(0, 64), H.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, 64), H.history["accuracy"], label="train_acc")
        plt.plot(np.arange(0, 64), H.history["val_accuracy"], label="val_acc")
        plt.title("Training Loss and Accuracy on Synthetic Dataset")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend()
        plt.show()
