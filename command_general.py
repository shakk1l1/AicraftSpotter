import matplotlib.pyplot as plt
import os
from path import *
from database import *
from data_extract import *
from POD import *
from image_handler import *

def commandpath():
    """
    Handle path-related commands for setting and displaying the folder path.
    """
    c_p = input('/path >> ')
    match c_p:
        case 'help':
            print("List of commands")
            print('     esc: escape')
            print('     show: show existing path')
            print('     new: create new path')
            commandpath()
        case "esc":
            return None
        case "show":
            try:
                f = open("path.py", "r")
            except FileNotFoundError:
                print("No path found, please create a new one")
                commandpath()
                return
            print(f.read())
            f.close()
            commandpath()
        case "new":
            new_path = input('new path:')
            with open('path.py', 'w') as f:
                f.write('path = ' + "'" + new_path + "'")
            print("Path saved!")
            commandpath()
        case _:
            print("unknown command")
            print("use help for commands list")
            commandpath()

    return None

def command(method, train_status):
    """
    Handle user commands for posting images to Instagram.
    """
    c = input('('+ train_status +')' + '>> ')
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
            pass
        case "esc":
            return False
        case "show":
            image_number = input("image number: ")
            show_image(image_number)
            pass
        case "close":
            plt.close()
            print("Image closed")
            pass
        case "path":
            commandpath()
            pass
        case "train":
            print("extracting training data...")
            # Load the training data
            f_train_data, f_train_label, m_train_data, m_train_label = data_extraction("train")
            match method:
                case "pod":
                    print("Training with POD")
                    # Add your training code here
                    pod_train(f_train_data, f_train_label)
            pass
        case "test":
            print("extracting test data...")
            # Load the test data
            match method:
                case "pod":
                    print("Testing with POD")
                    # Add your testing code here
                    print("WIP")
            pass

        case "method":
            print("select the AI method: ")
            print("     >POD")
            method = input(">> ")
            print("method selected: " + method)
            pass
        case _:
            print("unknown command")
            print("use help for commands list")
            pass
    return True