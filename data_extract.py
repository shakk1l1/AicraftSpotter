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
    if input("resize image (for squared final images (POD needed))? (y/n) ") == 'y':
        size = int(input("define size of the image (x, x): "))
        resizing = True
    else:
        resizing = False
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
            if img is None:
                print(f"Error loading image: {image_name}")
                continue
            values = get_image_data(image_name)
            x1, y1, x2, y2 = values[0], values[1], values[2], values[3]
            if x2 - x1 < size or y2 - y1 < size:
                print("\nimage too small (" + image_name + ")")
                print("adjusting size...")
                # Adjust the size of the image
                # Calculate the new size
                if x2 - x1 < size:
                    x1 = int(x1 - ((size - (x2 - x1)) / 2))
                    x2 = int(x2 + ((size - (x2 - x1)) / 2))
                elif y2 - y1 < size:
                    y1 = int(y1 - ((size - (y2 - y1)) / 2))
                    y2 = int(y2 + ((size - (y2 - y1)) / 2))
            croped_img = img[y1 + 1:y2 + 1, x1 +1:x2 +1]
            if resizing:
                croped_img = cv2.resize(croped_img, (size, size))
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
            if x2 - x1 < size or y2 - y1 < size:
                print("\nimage too small (" + image_name + ")")
                print(image_name)
                print("adjusting size...")
                # Adjust the size of the image
                # Calculate the new size
                if x2 - x1 < size:
                    x1 = int(x1 - ((size - (x2 - x1)) / 2))
                    x2 = int(x2 + ((size - (x2 - x1)) / 2))
                if y2 - y1 < size:
                    y1 = int(y1 - ((size - (y2 - y1)) / 2))
                    y2 = int(y2 + ((size - (y2 - y1)) / 2))
            croped_img = img[y1 + 1:y2 + 1, x1 +1:x2 +1]
            if resizing:
                croped_img = cv2.resize(croped_img, (size, size))
            m_images.append(croped_img.flatten())
            m_labels.append(family)
            i += 1
    print("\nmanufacturer data extracted")
    print("images: " + str(len(m_images)))
    print("labels: " + str(len(m_labels)))
    return f_images, f_labels, m_images, m_labels, size