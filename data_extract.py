import os
import cv2
from PCA_SVC import load_svc_models
from path import *
from database import get_image_data
import regex as re
import numpy as np
import joblib
import time

size = None
gray = None

def load_models(model):
    if "svc" in model:
        temporary_model = "svc"
    else:
        temporary_model = model
    match temporary_model:
        case "svc":
            print("Finding SVC models...")
            load_svc_models(model)
            trained = 'trained'
            
        case _:
            print("Invalid model")
            trained = 'not trained'
    return trained
            

def data_extraction(set):
    start_time = time.time()
    global size, gray
    match set:
        case "train":
            print("loading trainning data path...")
            f_data_path = os.path.join(path, "images_family_trainval.txt")
            m_data_path = os.path.join(path, "images_manufacturer_trainval.txt")
        case "test":
            print("loading trainning data path...")
            f_data_path = os.path.join(path, "images_family_test.txt")
            m_data_path = os.path.join(path, "images_manufacturer_test.txt")
        case _:
            print("Invalid set")
            return None
    if size is None:
        if input("resize image (for squared final images (needed for PCA/SVC))? (y/n) ") == 'y':
            size = int(input("define size of the image (x, x) (0 for not resizing): "))
            resizing = True
        else:
            resizing = False
    else:
        print("size already defined (" + str(size) + ')')
        if input("is it correct? (y/n) ") == 'y':
            resizing = True
        else:
            size = int(input("define size of the image (x, x) (0 for not resizing): "))
            if size == 0:
                resizing = False
            else:
                resizing = True
    if gray is None:
        if input("gray scale the image? (y/n) ") == 'y':
            gray = True
        else:
            gray = False
    else:
        print("gray scaling already defined (" + str(gray) + ')')
        if input("is it correct? (y/n) ") == 'y':
            gray = gray
        else:
            if input("gray scale the image? (y/n) ") == 'y':
                gray = True
            else:
                gray = False

    start_file_time_1 = time.time()
    f_images, f_labels = file_extractor(f_data_path, "family", resizing, gray)
    end_file_time_1 = time.time()

    print("\nfamily data extracted")
    print("number of images and labels: " + str(len(f_images)))

    start_file_time_2 = time.time()
    m_images, m_labels = file_extractor(m_data_path, "manufacturer", resizing, gray)
    print("\nmanufacturer data extracted")
    print("number of images and labels: " + str(len(m_images)))

    end_file_time_2 = time.time()
    print(f"Time taken to extract family data: {end_file_time_1 - start_file_time_1:.2f} seconds")
    print(f"Time taken to extract manufacturer data: {end_file_time_2 - start_file_time_2:.2f} seconds")
    print(f"Total time taken: {end_file_time_2 - start_time:.2f} seconds")
    return f_images, f_labels, m_images, m_labels

def file_extractor(data_path, set, resizing, gray):
    m_images = []
    m_labels = []
    with open(data_path, "rb") as f:
        num_lines = sum(1 for _ in f)
    with open(data_path, 'r') as file:
        i = 0
        for line in file:
            start = time.time()
            # Process each line as needed
            parts = re.split(' ', line, maxsplit=1)
            # Assuming the first part is the image name and the rest are labels
            image_name = parts[0]
            manufacturer = parts[1].removesuffix('\n')
            if gray:
                img = cv2.imread(os.path.join(path + '/images', image_name + '.jpg'), cv2.IMREAD_GRAYSCALE)
            else:
                img = cv2.imread(os.path.join(path + '/images', image_name + '.jpg'))
            values = get_image_data(image_name)
            x1, y1, x2, y2 = values[0], values[1], values[2], values[3]
            croped_img = img[y1 + 1:y2 + 1, x1 +1:x2 +1]
            if resizing:
                # Adjust the size of the image
                croped_img = cv2.resize(croped_img, (size, size), interpolation=cv2.INTER_AREA)
            if gray:
                g = 1
            else:
                g = 3
            if np.array(croped_img.flatten()).shape != (size * size * g,):
                print(f"Error cropping image: {image_name}")
                print(f"Expected shape: {(1, size * size * 3)}, but got: {np.array(croped_img.flatten()).shape}")
                return None
            if np.array(croped_img.flatten()).shape != (size * size * 3,):
                print(f"Error cropping image: {image_name}")
                print(f"Expected shape: {(1, size * size * 3)}, but got: {np.array(croped_img.flatten()).shape}")
                return None
            m_images.append(croped_img.flatten())
            m_labels.append(manufacturer)
            if (round(i / num_lines, 1) * 100) % 2 == 0:
                print('\r' + ' ' * 50, end='', flush=True)  # Clear the line
                print('\r', end='', flush=True)  # Move the cursor back to the
                print("exracting " + set + " data... " + str((i / num_lines) * 100) + "%" + f"  time per file: {time.time() - start:.2f} seconds", end='', flush=True)
            i += 1
    return m_images, m_labels
def change_size(model):
    """
    Change the size of the images
    """
    global size, gray
    size = joblib.load(os.path.join(path, 'models/' + model, 'size.pkl'))
    gray = joblib.load(os.path.join(path, 'models/' + model, 'gray.pkl'))
    print("Size changed to:", size)
    print("Gray scale changed to:", gray)
    return size, gray