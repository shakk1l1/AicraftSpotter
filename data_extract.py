import os

import cv2

from path import *

from database import get_image_data

import regex as re

def data_extraction(set):
    match set:
        case "train":
            print("loading trainning data path...")
            f_data_path = os.path.join(path, "images_family_train.txt")
            m_data_path = os.path.join(path, "images_manufacturer_train.txt")
        case "test":
            print("loading trainning data path...")
            f_data_path = os.path.join(path, "images_family_test.txt")
            m_data_path = os.path.join(path, "images_manufacturer_test.txt")
        case _:
            print("Invalid set")
            return None

    print("extracting family data...")
    f_images = []
    f_labels = []
    with open(f_data_path, "rb") as f:
        num_lines = sum(1 for _ in f)
    with open(f_data_path, 'r') as file:
        i = 0
        for line in file:
            if (round(i / num_lines, 1) * 100) % 2 == 0:
                print('\r' + ' ' * 30, end='', flush=True)  # Clear the line
                print('\r', end='', flush=True)  # Move the cursor back to the
                print("exracting family data... " + str((i / num_lines) * 100) + "%", end='', flush=True)
            # Process each line as needed
            parts = re.split(' ', line, maxsplit=1)
            # Assuming the first part is the image name and the rest are labels
            image_name = parts[0]
            family = parts[1].removesuffix('\n')
            img = cv2.imread(os.path.join(path + '/images', image_name + '.jpg'), cv2.IMREAD_GRAYSCALE)
            values = get_image_data(image_name)
            x1, y1, x2, y2 = values[0], values[1], values[2], values[3]
            croped_img = img[x1 + 1:x2 + 1, y1 + 1:y2 + 1]
            f_images.append(croped_img.flatten())
            f_labels.append(family)
            i += 1

    print("\nfamily data extracted")
    print("images: " + str(len(f_images)))
    print("labels: " + str(len(f_labels)))

    print("extracting manufacturer data...")
    m_images = []
    m_labels = []
    with open(m_data_path, "rb") as f:
        num_lines = sum(1 for _ in f)
    with open(m_data_path, 'r') as file:
        i = 0
        for line in file:
            if (round(i / num_lines, 1) * 100) % 2 == 0:
                print('\r' + ' ' * 30, end='', flush=True)  # Clear the line
                print('\r', end='', flush=True)  # Move the cursor back to the
                print("exracting manufacturer data... " + str((i / num_lines) * 100) + "%", end='', flush=True)
            # Process each line as needed
            parts = re.split(' ', line, maxsplit=1)
            # Assuming the first part is the image name and the rest are labels
            image_name = parts[0]
            family = parts[1].removesuffix('\n')
            img = cv2.imread(os.path.join(path + '/images', image_name + '.jpg'))
            values = get_image_data(image_name)
            x1, y1, x2, y2 = values[0], values[1], values[2], values[3]
            croped_img = img[x1 + 1:x2 + 1, y1 + 1:y2 + 1]

            m_images.append(croped_img.flatten())
            m_labels.append(family)
            i += 1
    print("\nmanufacturer data extracted")
    print("images: " + str(len(m_images)))
    print("labels: " + str(len(m_labels)))
    return f_images, f_labels, m_images, m_labels