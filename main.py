import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from command_path import *
import joblib
from path import *
from database import *
from data_extract import *
from POD import *


def command(method):
    """
    Handle user commands for posting images to Instagram.
    """
    c = input(">> ")
    match c:
        case 'help':
            print("List of commands")
            print('     esc: escape')
            print('     path: open path commands')
            print('     close: close image')
            print('     show: show image')
            print('     train: train the model')
            print('     test: test the model')
            print('     method: change the method')
            command(method)
        case "esc":
            return None
        case "show":
            image_number = input("image number: ")
            show_image(image_number)
            command(method)
        case "close":
            plt.close()
            print("Image closed")
            command(method)
        case "path":
            Commandpath()
            command(method)
        case "train":
            print("extracting training data...")
            # Load the training data
            f_train_data, m_train_data = data_extraction("train")
            match method:
                case "pod":
                    print("Training with POD")
                    # Add your training code here
                    pod(f_train_data, m_train_data)
            command(method)
        case "test":
            print("extracting test data...")
            # Load the test data
            match method:
                case "pod":
                    print("Testing with POD")
                    # Add your testing code here
                    print("WIP")
            command(method)

        case "method":
            print("select the AI method: ")
            print("     >POD")
            method = input(">> ")
            print("method selected: " + method)
            command(method)
        case _:
            print("unknown command")
            print("use help for commands list")
            command(method)

def create_table_values():
    with open(os.path.join(path, "images_box.txt"), 'r') as box_file:
        for box_line in box_file:
            # Process each line as needed
            parts = box_line.split()
            # Assuming the first part is the image name and the rest are bounding box coordinates
            image_name = parts[0]
            value1 = parts[1]
            value2 = parts[2]
            value3 = parts[3]
            value4 = parts[4]
            add_image_data(image_name, value1, value2, value3, value4)
# create a path to the image

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

## Main function
def main():
    print("Welcome to the AI Craft Spotter")
    try:
        db.get_connection()
        print("Database found")
        if verify_database_count():
            print("Database verified")
        else:
            print("Database not complete")
            delete_database()
            print("recreating database...")
            create_table()
            print("Database recreated")
            create_table_values()
    except Exception as e:
        print("Database not found")
        print("Creating database...")
        create_table()
        create_table_values()
        print("Database created")

    try:
        pathtest = path
    except Exception as e:
        print("path file not found")
        print("Creating path file...")
        new_path = input('path to data folder (finish with .../AicraftSpotter/data):')
        with open('command_path.py', 'w') as f:
            f.write('path = ' + "'" + new_path + "'")
        print("Path created")

    print("select the AI method: ")
    print("     >POD")
    method = input(">> ".lower())
    print("method selected: " + method)
    print("to begin and see the list of commands, type 'help'")
    command(method)


if __name__ == '__main__':
    plt.close()
    main()