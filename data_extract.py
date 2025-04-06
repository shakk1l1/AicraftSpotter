import os
import cv2
from PCA_SVC import load_svc_models
from path import *
from database import get_image_data
import regex as re
import joblib

size = None

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
    global size
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
    f_images = []
    f_labels = []
    with open(f_data_path, "rb") as f:
        num_lines = sum(1 for _ in f)
    with open(f_data_path, 'r') as file:
        i = 0
        for line in file:
            if (round(i / num_lines, 1) * 100) % 2 == 0:
                print('\r' + ' ' * 50, end='', flush=True)  # Clear the line
                print('\r', end='', flush=True)  # Move the cursor back to the
                print("exracting family data... " + str((i / num_lines) * 100) + "%", end='', flush=True)
            # Process each line as needed
            parts = re.split(' ', line, maxsplit=1)
            # Assuming the first part is the image name and the rest are labels
            image_name = parts[0]
            family = parts[1].removesuffix('\n')
            img = cv2.imread(os.path.join(path + '/images', image_name + '.jpg'))
            if img is None:
                print(f"Error loading image: {image_name}")
                continue
            values = get_image_data(image_name)
            x1, y1, x2, y2 = values[0], values[1], values[2], values[3]
            croped_img = img[y1 + 1:y2 + 1, x1 +1:x2 +1]
            if resizing:
                # Adjust the size of the image
                croped_img = cv2.resize(croped_img, (size, size), interpolation=cv2.INTER_AREA)
            f_images.append(croped_img.flatten())
            f_labels.append(family)
            i += 1

    print("\nfamily data extracted")
    print("images: " + str(len(f_images)))
    print("labels: " + str(len(f_labels)))

    m_images = []
    m_labels = []
    with open(m_data_path, "rb") as f:
        num_lines = sum(1 for _ in f)
    with open(m_data_path, 'r') as file:
        i = 0
        for line in file:
            if (round(i / num_lines, 1) * 100) % 2 == 0:
                print('\r' + ' ' * 50, end='', flush=True)  # Clear the line
                print('\r', end='', flush=True)  # Move the cursor back to the
                print("exracting manufacturer data... " + str((i / num_lines) * 100) + "%", end='', flush=True)
            # Process each line as needed
            parts = re.split(' ', line, maxsplit=1)
            # Assuming the first part is the image name and the rest are labels
            image_name = parts[0]
            manufacturer = parts[1].removesuffix('\n')
            img = cv2.imread(os.path.join(path + '/images', image_name + '.jpg'))
            values = get_image_data(image_name)
            x1, y1, x2, y2 = values[0], values[1], values[2], values[3]
            croped_img = img[y1 + 1:y2 + 1, x1 +1:x2 +1]
            if resizing:
                croped_img = cv2.resize(croped_img, (size, size), interpolation=cv2.INTER_AREA)
            m_images.append(croped_img.flatten())
            m_labels.append(manufacturer)
            i += 1
    print("\nmanufacturer data extracted")
    print("images: " + str(len(m_images)))
    print("labels: " + str(len(m_labels)))
    return f_images, f_labels, m_images, m_labels

def change_size():
    """
    Change the size of the images
    """
    global size
    size = joblib.load(os.path.join(path, 'models/svc', 'size.pkl'))
    print("Size changed to:", size)
    return size