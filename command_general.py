## imports
import matplotlib.pyplot as plt
import os
from path import *
from database import *
from data_extract import data_extraction
from PCA_SVC import *
from lreg import *
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

def command_model(actual_model=None, actual_train_status=None):
    # Change the model
    print("CAREFUL : this will reset the training")
    print("select the AI model: ")
    print("-----------regression based-----------")
    print("-------(PCA possible beforehand)-------")
    print("     > svc => svc")
    print("     > linear svc => lsvc")
    print("     > polynomial svc => psvc")
    print("     > lasso => lreg-lasso (WIP)")
    print("     > ridge => lreg-ridge (WIP)")
    print("     > Least-Squares regression => lreg-lsr (WIP)")
    print("------------neural networks------------")
    print("     > WIP")
    model = input("=> ")
    print("model selected: " + model)
    print("checking if model is valid and already trained...")
    if "svc" in model:
        temporary_model = "svc"
    elif "lreg" in model:
        temporary_model = "lreg"
    else:
        temporary_model = model

    match temporary_model:
        case "svc":
            from PCA_SVC import f_D_m, m_D_m
            if f_D_m is None or m_D_m is None:
                print("model not trained")
                train_status = "not trained"
            else:
                print("model already trained")
                print("you can already use it")
                train_status = "trained"
        case "lreg":
            from lreg import f_D_m, m_D_m
            if f_D_m is None or m_D_m is None:
                print("model not trained")
                train_status = "not trained"
            else:
                print("model already trained")
                print("you can already use it")
                train_status = "trained"
        case _:
            print("unknown model")
            model = actual_model
            train_status = actual_train_status
            print("use help for commands list")
    return model, train_status

def command(model, train_status):
    """
    Handle user commands for the command line interface.
    """
    # Get the command from the user
    c = input('('+ model + '/' + train_status +')' + '>> ')
    on = True

    match c:
        case 'help':
            # Display the list of commands
            print("List of commands")
            print('     help: list of commands')
            print('     esc: escape')
            print('     path: open path commands')
            print('     close: close image')
            print('     show: show image')
            print('     train: train the model')
            print('     load: Load the training of the selected model (if already trained)')
            print('     test: test the model')
            print('     model: change the model')
            print('     predict: predict specific image')
            pass
        case "esc":
            # Escape the program
            print("Exiting...")
            on = False
            pass

        case "show":
            # Show the image
            image_number = input("image number: ")
            if image_number == 'image not found':
                print('maybe an typo error')
                print('try again')
            else:
                show_image(image_number)
            pass

        case "close":
            # Close the image
            plt.close()
            print("Image closed")
            pass

        case "path":
            # Handle path-related commands
            commandpath()
            pass

        case "train":
            # Train the model
            if model is None:
                print("No model selected, please select a model using model command")
            else:
                print("extracting training data...")
                # Load the training data
                f_train_data, f_train_label, m_train_data, m_train_label = data_extraction("train")
                # update size variable
                match model:
                    case "svc":
                        print("Training with svc")
                        # Add your training code here
                        svc_train(f_train_data, f_train_label, m_train_data, m_train_label, model)
                        train_status = "trained"
                    case "lsvc":
                        print("Training with lsvc")
                        svc_train(f_train_data, f_train_label, m_train_data, m_train_label, model)
                        train_status = "trained"
                    case "psvc":
                        print("Training with psvc")
                        svc_train(f_train_data, f_train_label, m_train_data, m_train_label, model)
                        train_status = "trained"
            pass

        case "test":
            # Test the model
            print("extracting test data...")
            # Load the test data
            f_test_data, f_test_label, m_test_data, m_test_label = data_extraction("test")
            if "svc" in model:
                temporary_model = "svc"
            else:
                temporary_model = model
            match temporary_model:
                case "svc":
                    print("Testing with svc...")
                    svc_test(f_test_data, f_test_label, m_test_data, m_test_label)
            pass

        case "model":
            model, train_status = command_model(model, train_status)
            pass

        case "predict":

            # Predict a specific image
            if train_status == "not trained" or model is None:
                print("No model selected or training done, please select and train a model")
            else:
                image_number = input("image number: ")
                if (image_number + '.jpg') in image_list:
                    svc_predict(image_number)
                else:
                    print("image not found")
                    print("maybe an typo error")
                    print("try again")
            pass
        case "load":
            # Load the model
            train_status = load_models(model) 
            pass
        case "backdoor":
            # Backdoor command (for testing purposes)
            print("what is going on")
            pass
        case _:
            # Handle unknown commands
            print("unknown command")
            print("use help for commands list")
            pass
    return on, model, train_status