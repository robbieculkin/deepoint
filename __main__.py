import cv2
import time
import argparse
from src.image_generator import render, highlight_vertices

''' Main Driver '''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--displaymode', type=int, default=0, choices=[0,1])
    args = parser.parse_args()

    # for showcase: frames_per_count = 50
    # for performance: frames_per_count = 5
    start = time.time()
    output = render(display_mode=args.displaymode, screen_size=(200, 200), object_types=['star'], count=10, object_count=1, frames_per_count=100, test=False)
    # makes green pixels brighter
    end = time.time()
    diff = end - start
    print("Time To Complete: {} seconds".format(diff))

    print()
    print(len(output))