## imports
import matplotlib.pyplot as plt
import os
from path import *
from database import *
from data_extract import data_extraction
from PCA_SVC import *
from lreg import *
from cv import *
from image_handler import *
from nn import *
from main import image_list

model_list = ['svc', 'lsvc', 'psvc', 'lreg-lasso', 'lreg-lsr', 'lreg-ridge', 'cv-ridge', 'cv-lasso', 'cl_nn']      # list of models available

def commandpath():
    """
    Handle path-related commands for setting and displaying the folder path.
    It handle path.py that is just a file that contains a variable definition named path
    :return: None
    """
    c_p = input('/path >> ')        # get command from user
    match c_p:      # command possibilities
        case 'help':
            print("List of commands")
            print('     esc: escape')
            print('     show: show existing path')
            print('     new: create new path')      # no need to create a new path beause it is automatically created
            commandpath()                               # see main.py
        case "esc":     # Quit the command path section
            return None
        case "show":
            try:
                f = open("path.py", "r")        # open the path.py
            except FileNotFoundError:       # if path.py not found, error
                print("No path found, please create a new one")
                commandpath()
                return
            print(f.read())
            f.close()
            commandpath()
        case "new":     # change the path
            new_path = input('new path:')
            with open('path.py', 'w') as f:
                f.write('path = ' + "r'" + new_path + "'")      # write the new path to the path.py
            print("Path saved!")
            commandpath()
        case _:     # unknown command
            print("unknown command")
            print("use help for commands list")
            commandpath()
    return None

def command_model(actual_model=None, actual_train_status=None):
    """
    Handle the model selection and check training status.
    :param actual_model: the model that is already selected
    :param actual_train_status: the status of the model that is already selected
    :return: model: the model selected, train_status: the status of the model selected
    """
    print("\nCAREFUL : this will reset the training if you select a new model\n")
    print("select the AI model: ")
    print("-----------regression based-----------")
    print("-------(PCA possible beforehand)-------")
    print("     > svc => svc")
    print("     > linear svc => lsvc")
    print("     > polynomial svc => psvc")
    print("     > lasso => lreg-lasso (x)")
    print("     > ridge classifier=> lreg-ridge")
    print("     > Least-Squares regression => lreg-lsr (x)")
    print("----------(Cross Validation)----------")
    print("     > cross validation ridge => cv-ridge")
    print("     > cross validation lasso => cv-lasso (x)")
    print("-----------Neural Network-------------")
    print("     > Conventional Linear Neural Network => cl_nn")
    print("WIP = Work In Progress")
    print("x = not working well as it use continuous data prediction, i.e. it is not a classification model\n")
    model = input("     => ")       # get the new wanted model from user

    print("model selected: " + model)
    print("checking if model is valid and already trained...")

    # check if the model is in the list of models
    if not check_model_existance(model):       # check if the model is in the list of models
        print("model not found")
        print("use help for commands list")
        return actual_model, actual_train_status

    # as some models have the same load function and files, we can just check family
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
            from PCA_SVC import f_D_m, m_D_m        # import global variables from PCA_SVC that are modified in the training
            if f_D_m is None or m_D_m is None:      # check if the model is trained by checking if one value is None
                print("model not trained")          # when trained, those values can't be None
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
        case "cv":
            from cv import f_D_m, m_D_m
            if f_D_m is None or m_D_m is None:
                print("model not trained")
                train_status = "not trained"
            else:
                print("model already trained")
                print("you can already use it")
                train_status = "trained"
        case "nn":
            from nn import f_hidden_size, m_hidden_size
            if f_hidden_size is None or m_hidden_size is None:
                print("model not trained")
                train_status = "not trained"
            else:
                print("model already trained")
                print("you can already use it")
                train_status = "trained"
        case _:     # unknown model
            print("unknown model")
            model = actual_model        # keep the previous model
            train_status = actual_train_status      # keep the previous training status
            print("use help for commands list")
    print(" ")
    return model, train_status

def command(model, train_status):
    """
    Handle user commands for the command line interface.
    :param model: the model selected
    :param train_status: the training status of the model selected
    :return: on: if the program should continue running
    :return: model: the model selected
    :return: train_status: the training status of the model selected
    """
    # Get the command from the user
    c = input('('+ model + '/' + train_status +')' + '>> ')
    on = True       # keep the program running

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
            on = False      # exit the program (see main.py loop)
            pass

        case "show":
            # Show the image
            image_number = input("image number: ")
            if image_number + '.jpg' not in image_list:        # check if the image exists in the folder
                print("image not found")
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
            if model == 'None':     # check if a model is selected
                print("\nNo model selected, please select a model using model command")
            else:
                print("\nextracting training data...")
                # Load the training data
                f_train_data, f_train_label, m_train_data, m_train_label = data_extraction("train")
                if f_train_data is None or f_train_label is None or m_train_data is None or m_train_label is None:
                    print("Error extracting training data")
                    print("Please verify the data")
                    return model, train_status

                print(" ")

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
                    case "lreg-lasso":
                        print("Training with lasso")
                        lreg_train(f_train_data, f_train_label, m_train_data, m_train_label, model)
                        train_status = "trained"
                    case "lreg-lsr":
                        print("training with Least-Squares regression")
                        lreg_train(f_train_data, f_train_label, m_train_data, m_train_label, model)
                        train_status = "trained"
                    case "lreg-ridge":
                        print("training with ridge")
                        lreg_train(f_train_data, f_train_label, m_train_data, m_train_label, model)
                        train_status = "trained"
                    case "cv-ridge":
                        print("training with cross validation ridge")
                        cv_train(f_train_data, f_train_label, m_train_data, m_train_label, model)
                        train_status = "trained"
                    case "cv-lasso":
                        print("training with cross validation lasso")
                        cv_train(f_train_data, f_train_label, m_train_data, m_train_label, model)
                        train_status = "trained"
                    case "cl_nn":
                        print("training with conventional linear neural network")
                        nn_train(f_train_data, f_train_label, m_train_data, m_train_label, model)
                        train_status = "trained"
                    case _:
                        print("unknown model")
            pass

        case "test":
            print(" ")
            if train_status == "not trained" or model == 'None':        # check if a model is selected and trained
                print("No model selected or training done, please select and train a model")
            else:
                # Test the model
                print("extracting test data...")
                # Load the test data
                f_test_data, f_test_label, m_test_data, m_test_label = data_extraction("test")
                if f_test_data is None or f_test_label is None or m_test_data is None or m_test_label is None:
                    print("Error extracting training data")
                    print("Please verify the data")
                    return model, train_status

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

                print(" ")

                match temporary_model:
                    case "svc":
                        print("Testing with svc...")
                        svc_test(f_test_data, f_test_label, m_test_data, m_test_label)
                    case "lreg":
                        print("Testing with linear regression...")
                        lreg_test(f_test_data, f_test_label, m_test_data, m_test_label)
                    case "cv":
                        print("Testing with cross validation...")
                        cv_test(f_test_data, f_test_label, m_test_data, m_test_label)
                    case "nn":
                        print("Testing with neural network...")
                        nn_test(f_test_data, f_test_label, m_test_data, m_test_label, model)
            pass

        case "model":
            # Change the model
            model, train_status = command_model(model, train_status)
            pass

        case "predict":
            # Predict a specific image
            if train_status == "not trained" or model == 'None':       # check if a model is selected and trained
                print("No model selected or training done, please select and train a model")
            else:
                image_number = input("image number: ")
                if (image_number + '.jpg') in image_list:       # check if the image exists in the folder
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
                            print("Predicting with svc...")
                            svc_predict(image_number)
                        case "lreg":
                            print("Predicting with linear regression...")
                            lreg_predict(image_number)
                        case "cv":
                            print("Predicting with cross validation...")
                            cv_predict(image_number)
                        case "nn":
                            print("Predicting with neural network...")
                            nn_predict(image_number)
                        case _:
                            print("unknown model")
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
            show_image("0034309")
            pass

        case _:
            # Handle unknown commands
            print("unknown command")
            print("use help for commands list")
            pass
    print(" ")
    return on, model, train_status

def check_model_existance(model):
    """
    Check if the model exists in the models folder
    :param model: model to check
    :return: True if the model exists, False otherwise
    """
    if model in model_list:
        return True
    else:
        print("model not exising")
        return False