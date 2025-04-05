import matplotlib.pyplot as plt
import os
from path import *
from database import *
from data_extract import *
from POD import *
from image_handler import *
from main import image_list

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
                f.write('path = ' + "r'" + new_path + "'")
            print("Path saved!")
            commandpath()
        case _:
            print("unknown command")
            print("use help for commands list")
            commandpath()

    return None

def command(model, train_status):
    """
    Handle user commands for posting images to Instagram.
    """
    c = input('('+ model + '/' + train_status +')' + '>> ')
    on = True
    match c:
        case 'help':
            print("List of commands")
            print('     help: list of commands')
            print('     esc: escape')
            print('     path: open path commands')
            print('     close: close image')
            print('     show: show image')
            print('     train: train the model')
            print('     load: Load the models')
            print('     test: test the model')
            print('     model: change the model')
            print('     predict: predict specific image')
            pass
        case "esc":
            on = False
        case "show":
            image_number = input("image number: ")
            if image_number == 'image not found':
                print('maybe an typo error')
                print('try again')
            else:
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
            if model is None:
                print("No model selected, please select a model using model command")
            else:
                print("extracting training data...")
                # Load the training data
                f_train_data, f_train_label, m_train_data, m_train_label = data_extraction("train")
                # update size variable
                match model:
                    case "pod":
                        print("Training with POD")
                        # Add your training code here
                        pod_train(f_train_data, f_train_label, m_train_data, m_train_label)
                        train_status = "trained"
            pass
        case "test":
            print("extracting test data...")
            # Load the test data
            match model:
                case "pod":
                    print("Testing with POD")
                    # Add your testing code here
                    print("WIP")
            pass
        case "model":
            print("select the AI model: ")
            print("     > pod")
            model = input("=> ")
            print("model selected: " + model)
            print("checking if model is valid and already trained...")
            match model:
                case "pod":
                    if f_D_m is None or m_D_m is None:
                        print("model not trained")
                        train_status = "not trained"
                    else:
                        print("model already trained")
                        print("you can already use it")
                        train_status = "trained"
                case _:
                    print("unknown model")
                    print("use help for commands list")
            pass
        case "predict":
            if train_status == "not trained":
                print("No training done, please train the model first")
            else:
                image_number = input("image number: ")
                if (image_number + '.jpg') in image_list:
                    pod_predict(image_number)
                else:
                    print("image not found")
                    print("maybe an typo error")
                    print("try again")
            pass
        case "load":
            train_status = load_models(model) 
            pass
        case "backdoor":
            pass
        case _:
            print("unknown command")
            print("use help for commands list")
            pass
    return on, model, train_status