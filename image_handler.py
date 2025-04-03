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
    print(image_path)

    # Read the image using OpenCV
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    plt.title(f'Image number = {int(image_number)}')
    plt.imshow(image)

    plt.show()