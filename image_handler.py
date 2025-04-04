import os
import cv2
import matplotlib.pyplot as plt
from path import path
from database import *
from data_extract import *
from POD import *


def get_image_path(image_number):
    """
    Get the path to the image
    :return: path to the image
    """
    images = os.listdir(path + '/images')
    if image_number in images:
        print('image found')
        image_path = os.path.join(path + '/images', image_number)
    else:
        print('image not found')
        image_path = 'image not found'
    return image_path

def show_image(image_number):
    """
    Show the image
    :param image_path: path to the image
    :return: None
    """
    # get the image path
    image_path = get_image_path(image_number + '.jpg')

    # Read the image using OpenCV
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if input("crop image? (y/n) ") == 'y':
        # Crop the image
        x1, y1, x2, y2 = get_image_data(image_number)
        image = image[y1 + 1:y2 + 1, x1 +1:x2 +1]
        print("cropped image")
    if input("resize image? (y/n) ") == 'y':
        # Resize the image to 64x64
        size = int(input("size of the image (x, x): "))
        image = cv2.resize(image, (size, size))
    plt.title(f'Image number = {int(image_number)}')
    plt.imshow(image)

    plt.show()