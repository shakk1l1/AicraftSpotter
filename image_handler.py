import os
import cv2
import matplotlib.pyplot as plt
from path import path
from database import *
from data_extract import *
from PCA_SVC import *


def get_image_path(image_number):
    """
    Get the path to the image
    :return: path to the image
    """
    images = os.listdir(path + '/images')   # get all the images in the images folder in a list
    if image_number in images:      # check if the image is in the list
        print('image found')
        image_path = os.path.join(path + '/images', image_number)
    else:       # if the image is not in the list
        print('image not found')
        image_path = 'image not found'
    return image_path

def show_image(image_number):
    """
    Show the image using matplotlib and OpenCV
    :param image_path: path to the image
    :return: None
    """
    # get the image path
    image_path = get_image_path(image_number + '.jpg')

    # Read the image using OpenCV
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if input("crop image? (y/n) ").lower() == 'y':
        # Crop the image
        x1, y1, x2, y2 = get_image_data(image_number)
        image = image[y1 + 1:y2 + 1, x1 +1:x2 +1]
        print("cropped image")
    if input("resize image? (y/n) ").lower() == 'y':
        size = int(input("size of the image (x, x): "))
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)

    plt.title(f'Image number = {int(image_number)}')
    plt.imshow(image)
    plt.show()