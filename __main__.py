import cv2
import time
import argparse
from src.image_generator import render, highlight_vertices, generate_images

import numpy as np

''' Main Driver '''
if __name__ == '__main__':
    # for showcase: frames_per_count = 50
    # for performance: frames_per_count = 5
    # output = render(display_mode=1, screen_size=(640, 480), object_types=['triangle', 'quad'], count=10, object_count=1, frames_per_count=5, test=False)
    # images, masks = highlight_vertices(output,screen_size=(640, 480))

    data_generator = generate_images(
        object_types=['cube'],
        batch_size=32,
        object_count=1,
        display_mode=1,
        shape=(640, 480)
    )

    images, masks = next(data_generator)

    print()
    print(len(images))
    for i in range(0, len(images)):
        cv2.imshow("Image", images[i])
        cv2.imshow("Mask", masks[i])
        cv2.waitKey(0)
