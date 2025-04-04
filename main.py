import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from command_general import *
import joblib
from database import *
from data_extract import *
from POD import *
try:
    from path import path

    pathtest = path
except Exception as e:
    print("path file not found")
    print("Creating path file...")
    new_path = input('path to data folder (finish with .../AicraftSpotter/data):')
    with open('command_general.py', 'w') as f:
        f.write('path = ' + "r'" + new_path + "'")
    print("Path created")
    from path import path

## Main function
def main():
    print("Welcome to the AI Craft Spotter")
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

    print("select the AI method: ")
    print("     >POD")
    method = input(">> ".lower())
    print("method selected: " + method)
    print("to begin and see the list of commands, type 'help'")
    on = True
    train_status = "not trained"
    while on:
        on = command(method, train_status)


if __name__ == '__main__':
    plt.close()
    main()