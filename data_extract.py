## imports
import os
import cv2
from alive_progress import alive_bar
from PCA_SVC import load_svc_models
from lreg import load_lreg_models
from cv import load_cv_models
from nn import load_nn_models
from path import *
from database import get_image_data
import regex as re
import joblib
import time

## Global variables
size = None
gray = None

def load_models(model):
    """
    Load the right saves depending on the model selected
    :param model: model to load
    :return: trained: if the model was well loaded or not
    """
    print("\nFinfing models...")
    #check if the file exists
    if os.path.exists(os.path.join(path, 'models/' + model)):
        print("model found\n")
    else:
        print("Model not found")
        print("Please train the model first\n")
        return 'not trained'

    # as some models have the same function and file, we can goup them
    if "svc" in model:
        temporary_model = "svc"
    elif "lreg" in model:
        temporary_model = "lreg"
    elif "cv" in model:
        temporary_model = "cv"
    elif "nn" in model:
        temporary_model = "nn"
    else:
        temporary_model = model

    match temporary_model:
        case "svc":
            print("Finding SVC models...")
            load_svc_models(model)
            trained = 'trained'
        case "lreg":
            print("Finding LREG models...")
            load_lreg_models(model)
            trained = 'trained'
        case "cv":
            print("Finding CV models...")
            load_cv_models(model)
            trained = 'trained'
        case "nn":
            print("Finding NN models...")
            load_nn_models(model)
            trained = 'trained'
        case _:
            print("Invalid model")
            trained = 'not trained'
    return trained
            

def data_extraction(data_set):
    """
    Extract the data from the files
    :param data_set: the data_set to extract (train or test)
    :return: f_images, f_labels, m_images, m_labels: the images and labels
    """

    # start the timer
    start_time = time.time()

    # global variables as they will be modified
    global size, gray

    # which data_set to extract
    # and get the path
    match data_set:
        case "train":
            print("loading trainning data path...")
            f_data_path = os.path.join(path, "images_family_trainval.txt")
            m_data_path = os.path.join(path, "images_manufacturer_trainval.txt")
        case "test":
            print("loading trainning data path...")
            f_data_path = os.path.join(path, "images_family_test.txt")
            m_data_path = os.path.join(path, "images_manufacturer_test.txt")
        case _:
            print("Invalid data_set")
            return None, None, None, None

    # check if size already defined
    # else ask the user
    print(" ")
    if size is None:
        size = int(input("define size of the image (x, x): "))
    else:
        print("size already defined (" + str(size) + ')')
        if input("is it correct? (y/n) ").lower() == 'n':
            size = int(input("define size of the image (x, x): "))
            if size == 0:
                size = None

    # check if gray already defined
    if gray is None:
        if input("gray scale the image? (y/n) ").lower() == 'y':
            gray = True
        else:
            gray = False
    else:
        print("gray scaling already defined (" + str(gray) + ')')
        if input("is it correct? (y/n) ").lower() == 'y':
            gray = gray
        else:
            if input("gray scale the image? (y/n) ").lower() == 'y':
                gray = True
            else:
                gray = False

    # start the extraction of the first data set
    start_file_time_1 = time.time()
    f_images, f_labels = file_extractor(f_data_path, "family", gray)
    end_file_time_1 = time.time()

    # get all the family names

    print("\nfamily data extracted")
    print("number of images and labels: " + str(len(f_images)))
    print(f"number of different family: {len(set(f_labels))}")
    print(f"different family: {set(f_labels)}")

    # start the extraction of the second data set
    start_file_time_2 = time.time()
    m_images, m_labels = file_extractor(m_data_path, "manufacturer", gray)
    print("\nmanufacturer data extracted")
    print("number of images and labels: " + str(len(m_images)))
    print("number of different manufacturer: " + str(len(set(m_labels))))
    print("different manufacturer: " + str(set(m_labels)))
    end_file_time_2 = time.time()

    # print the time taken to extract the data
    print(f"\nTotal time taken: {end_file_time_2 - start_time:.2f} seconds")
    return f_images, f_labels, m_images, m_labels

def file_extractor(data_path, data_set, gray):
    """
    Extract the data from the files
    :param data_path: path to the data
    :param data_set: which data_set to extract (family or manufacturer)
    :param gray: if the image should be gray scaled
    :return: m_images, m_labels: the images and labels
    """

    # initializing arrays
    m_images = []
    m_labels = []

    print(" ")

    # get the number of lines in the file
    with open(data_path, "rb") as f:
        num_lines = sum(1 for _ in f)

    # open the file and extract the data
    file = open(data_path, "r")
    lines = file.readlines()
    file.close()
    print(f"Extracting {data_set} data...")
    with alive_bar(num_lines, force_tty=True, max_cols=270) as bar:
        for i in range(num_lines):
            # get the i th line
            line = lines[i]
            start = time.time()
            # Process each line as needed and split the line in parts
            # Assuming the first part is the image name and the rest are labels
            # split the line at the first space
            parts = re.split(' ', line, maxsplit=1)
            image_name = parts[0]
            manufacturer = parts[1].removesuffix('\n')

            # read the image and put it like a list
            if gray:
                img = cv2.imread(os.path.join(path + '/images', image_name + '.jpg'), cv2.IMREAD_GRAYSCALE)
            else:
                img = cv2.imread(os.path.join(path + '/images', image_name + '.jpg'))

            # get the crop ratio from the database
            values = get_image_data(image_name)
            x1, y1, x2, y2 = values[0], values[1], values[2], values[3]

            # croping the image
            croped_img = img[y1 + 1:y2 + 1, x1 +1:x2 +1]

            # resize the image if needed
            if size is not None:
                # Adjust the size of the image
                croped_img = cv2.resize(croped_img, (size, size), interpolation=cv2.INTER_AREA)

            # check if the image was well cropped and resized
            # if gray:
            #     g = 1
            # else:
            #     g = 3
            # if np.array(croped_img.flatten()).shape != (size * size * g,):
            #     print(f"Error cropping image: {image_name}")
            #     print(f"Expected shape: {(1, size * size * 3)}, but got: {np.array(croped_img.flatten()).shape}")
            #     return None

            # add the image and label to the arrays
            #TODO: add possibility to not flatten the image for the nn
            #   or use torchvision.io.read_image
            m_images.append(croped_img.flatten())
            m_labels.append(manufacturer)
            bar()

    return m_images, m_labels

def change_size(model):
    """
    Change the size of the images. Needed because global variables must be changed in the right folders
    :param model: model to load
    :return: size, gray: the size and gray scale status of the images
    """
    # global variables will be modified
    global size, gray

    # load values from the model
    size = joblib.load(os.path.join(path, 'models/' + model, 'size.pkl'))
    gray = joblib.load(os.path.join(path, 'models/' + model, 'gray.pkl'))
    print("Size changed to:", size)
    print("Gray scale changed to:", gray)
    return size, gray