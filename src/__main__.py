import cv2
from image_generator import render, highlight_vertices

''' Main Driver '''
if __name__ == '__main__':

    # Start Generating different settings of objects
    # starting the main driver for generating the dataset
    # wrap these two functions

    # for showcase: frames_per_count = 50
    # for performance: frames_per_count = 5
    output = render(display_mode=0, screen_size=(200, 200), object_types=['checkerboard'], count=10, object_count=1, frames_per_count=25, test=False)
    # makes green pixels brighter
    output = highlight_vertices(output) # output images

    print()
    print(len(output))

    for out in output:
        cv2.imshow("out", out)
        cv2.waitKey(0)
