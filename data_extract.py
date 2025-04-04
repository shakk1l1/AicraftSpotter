import os

import cv2

from path import *

from database import get_image_data

import regex as re

def data_extraction(set):
    match set:
        case "train":
            print("extracting family training data...")

            f_data_path = os.path.join(path, "images_family_train.txt")
            f_images = []
            f_labels = []
            with open(f_data_path, 'r') as file:
                for line in file:
                    # Process each line as needed
                    parts = re.split(' ', line, maxsplit=1)
                    # Assuming the first part is the image name and the rest are labels
                    image_name = parts[0]
                    family = parts[1].removesuffix('\n')
                    img = cv2.imread(os.path.join(path + '/images', image_name + '.jpg'), cv2.IMREAD_GRAYSCALE)
                    values = get_image_data(image_name)
                    x1, y1, x2, y2 = values[0], values[1], values[2], values[3]
                    croped_img = img[x1+1:x2+1, y1+1:y2+1]
                    f_images.append(croped_img.flatten())
                    f_labels.append(family)

            print("family training data extracted")
            print("images: " + str(len(f_images)))
            print("labels: " + str(len(f_labels)))
            print("extracting manufacturer training data...")

            m_data_path = os.path.join(path, "images_manufacturer_train.txt")
            m_images = []
            m_labels = []
            with open(m_data_path, 'r') as file:
                for line in file:
                    # Process each line as needed
                    parts = re.split(' ', line, maxsplit=1)
                    # Assuming the first part is the image name and the rest are labels
                    image_name = parts[0]
                    family = parts[1].removesuffix('\n')
                    img = cv2.imread(os.path.join(path + '/images', image_name + '.jpg'))
                    values = get_image_data(image_name)
                    x1, y1, x2, y2 = values[0], values[1], values[2], values[3]
                    croped_img = img[x1+1:x2+1, y1+1:y2+1]

                    m_images.append(croped_img.flatten())
                    m_labels.append(family)
            print("manufacturer training data extracted")
            print("images: " + str(len(m_images)))
            print("labels: " + str(len(m_labels)))
            return f_images, f_labels, m_images, m_labels

        case "test":
            print("extracting family test data...")

            print("extracting manufacturer training data...")
        case _:
            print("Invalid set")
            return None
    return None