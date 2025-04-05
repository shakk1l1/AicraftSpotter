import os
print("Welcome to the AI Craft Spotter")
try:
    from path import path

    pathtest = path
except Exception as e:
    print("path file not found")
    print("Creating path file...")
    new_path = input('path to data folder (finish with .../AicraftSpotter/data):')
    with open('path.py', 'w') as f:
        f.write('path = ' + "r'" + new_path + "'")
    print("Path created")
    from path import path
image_list = os.listdir(path + '/images')
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from command_general import *
import joblib
from database import *
from data_extract import *
from POD import *

## Main function
def main():
    try:
        print("Connecting to database...")
        db.get_connection()
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
        print("\nDatabase created")
    print("this is a command line interface")
    print("the main interface is in the command line")
    print("it looks like this: (xxxxx/xxxxx)>>")
    print("                       ^     ^")
    print("                       |     |")
    print("                    method train_status")
    print("to begin and see the list of commands, type 'help'")
    method = "None"
    on = True
    train_status = "not trained"
    while on:
        on, method, train_status = command(method, train_status)


if __name__ == '__main__':
    plt.close()
    main()