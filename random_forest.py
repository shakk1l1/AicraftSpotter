import os
import cv2
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier
from database import get_image_data
from path import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import joblib
import time
from alive_progress import alive_bar

# Global variables
f_D_m = None
f_pca = None
f_clf = None
f_le = None
f_encoded_labels = None

m_D_m = None
m_pca = None
m_clf = None
m_le = None
m_encoded_labels = None

def load_rf_models(model):
    """
    Load the Random Forest models
    :param model: model name
    :return: None
    """
    # import the functions from data_extract
    # to avoid circular imports
    from data_extract import change_size

    # global variables as they will be modified
    global f_D_m, f_pca, f_clf, f_le
    global m_D_m, m_pca, m_clf, m_le

    print("Loading models...")
    f_clf = joblib.load(os.path.join(path, 'models/' + model, 'family_model.pkl'))
    f_le = joblib.load(os.path.join(path, 'models/' + model, 'family_label_encoder.pkl'))
    m_clf = joblib.load(os.path.join(path, 'models/' + model, 'manufacturer_model.pkl'))
    m_le = joblib.load(os.path.join(path, 'models/' + model, 'manufacturer_label_encoder.pkl'))
    f_D_m = joblib.load(os.path.join(path, 'models/' + model, 'family_mean.pkl'))
    m_D_m = joblib.load(os.path.join(path, 'models/' + model, 'manufacturer_mean.pkl'))
    f_pca = joblib.load(os.path.join(path, 'models/' + model, 'family_pca.pkl'))
    m_pca = joblib.load(os.path.join(path, 'models/' + model, 'manufacturer_pca.pkl'))

    # call the function to change size and gray
    change_size(model)
    print("Models loaded")

def rf_train(data_family, label_family, data_manufacturer, label_manufacturer, model):
    """
    Train the two Random Forest models
    :param data_family: family data
    :param label_family: family labels
    :param data_manufacturer: manufacturer data
    :param label_manufacturer: manufacturer labels
    :param model: model name
    :return: None
    """
    # global variables as they will be modified
    global f_D_m, f_pca, f_clf, f_le
    global m_D_m, m_pca, m_clf, m_le

    print("Random Forest training")

    if input("use PCA? (y/n) ").lower() == 'y':
        if input("apply the same PCA on the two data sets? (y/n) ").lower() == 'y':
            spca = n_components_selecter()
        else:
            spca = None
    else:
        spca = False

    # Get hyperparameters for Random Forest
    n_estimators = int(input("Number of trees in the forest (default: 100): ") or "100")
    max_depth = input("Maximum depth of the trees (default: None): ")
    max_depth = None if max_depth == "" else int(max_depth)
    min_samples_split = int(input("Minimum samples required to split a node (default: 2): ") or "2")
    min_samples_leaf = int(input("Minimum samples required at a leaf node (default: 1): ") or "1")

    print("Training family model...")
    f_D_m, f_pca, f_le, f_clf, f_encoded_labels = rf_train_s(
        data_family, label_family, model, spca, 
        n_estimators, max_depth, min_samples_split, min_samples_leaf
    )
    
    print("Training manufacturer model...")
    m_D_m, m_pca, m_le, m_clf, m_encoded_labels = rf_train_s(
        data_manufacturer, label_manufacturer, model, spca,
        n_estimators, max_depth, min_samples_split, min_samples_leaf
    )
    
    # Save the models
    newpath = os.path.join(path, 'models/' + model)

    # create the directory if it does not exist
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    joblib.dump(f_clf, os.path.join(path, 'models/' + model, 'family_model.pkl'))
    joblib.dump(f_le, os.path.join(path, 'models/' + model, 'family_label_encoder.pkl'))
    joblib.dump(m_clf, os.path.join(path, 'models/' + model, 'manufacturer_model.pkl'))
    joblib.dump(m_le, os.path.join(path, 'models/' + model, 'manufacturer_label_encoder.pkl'))
    joblib.dump(f_D_m, os.path.join(path, 'models/' + model, 'family_mean.pkl'))
    joblib.dump(m_D_m, os.path.join(path, 'models/' + model, 'manufacturer_mean.pkl'))
    joblib.dump(f_pca, os.path.join(path, 'models/' + model, 'family_pca.pkl'))
    joblib.dump(m_pca, os.path.join(path, 'models/' + model, 'manufacturer_pca.pkl'))

    # Save hyperparameters
    hyperparams = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf
    }
    joblib.dump(hyperparams, os.path.join(path, 'models/' + model, 'hyperparams.pkl'))

    # import last version of size and gray
    from data_extract import size, gray
    joblib.dump(size, os.path.join(path, 'models/' + model, 'size.pkl'))
    joblib.dump(gray, os.path.join(path, 'models/' + model, 'gray.pkl'))

    print("Models saved")
    print("Training complete")

def n_components_selecter():
    """
    Select the number of components for PCA
    :return: number of components
    """
    n_components = float(input("Number of components for PCA (0 for no PCA): "))
    if n_components > 1: # if n_components is an integer it must be the number of components
        try:
            n_components = int(n_components)
        except:
            print("Invalid number of components, please enter an integer over 1 or a float under 1")
            n_components_selecter()
    # else it must be a % of the image data to keep
    return n_components

def rf_train_s(data, label, model, spca, n_estimators, max_depth, min_samples_split, min_samples_leaf):
    """
    Train a single Random Forest model
    :param data: training data
    :param label: training labels
    :param model: model name
    :param spca: PCA components
    :param n_estimators: number of trees
    :param max_depth: maximum depth of trees
    :param min_samples_split: minimum samples to split
    :param min_samples_leaf: minimum samples at leaf
    :return: D_m, pca, le, clf, encoded_labels
    """
    # start the timer
    start_training = time.time()

    # Encode labels for the plots
    le = LabelEncoder()
    encoded_labels = le.fit_transform(label)

    # convert data to numpy array
    data_train = np.array(data)

    # normalize data
    print("\nNormalizing data...")
    data_train = data_train / 255.0

    # centering data
    print("Calculating mean...")
    start_mean = time.time()
    D_m = np.mean(data_train, axis=0)
    end_mean = time.time()
    print("Centering data...")
    start_center = time.time()
    D0_train = data_train - D_m
    end_center = time.time()

    # training PCA
    if type(spca) is float or type(spca) is int:
        n_components = spca
    elif spca is False:
        print("No PCA")
    else:
        # pca
        n_components = n_components_selecter()
    
    if spca is not False:
        print("Calculating PCA...")
        pca = PCA(n_components=n_components)
        start_pca = time.time()
        A = pca.fit_transform(D0_train)
        end_pca = time.time()
    else:
        A = D0_train
        pca = None

    # print the explained variance ratio
    if not os.path.exists(os.path.join(path, 'models/' + model)):
        os.makedirs(os.path.join(path, 'models/' + model))

    if pca is not None and input("Plot eigenvalues distribution? (y/n) ").lower() == 'y':
        # Plot the distribution of the eigenvalues
        plt.title("Eigenvalues distribution")
        plt.xlabel("Eigenvalue index")
        plt.ylabel("Eigenvalue")
        plt.grid()
        plt.scatter(np.arange(len(pca.explained_variance_)), pca.explained_variance_)
        plt.savefig(os.path.join(path, 'models/' + model, 'eigenvalues.png'))
        plt.show()

    if pca is not None and input("Plot the distribution of the PCA coefficients? (y/n) ").lower() == 'y':
        plt.title(f"PCA coefficients distribution\nNumber of components: {pca.n_components_}")
        plt.xlabel("PCA coefficient 1")
        plt.ylabel("PCA coefficient 2")
        im = plt.scatter(A[:, 0], A[:, 1], c=encoded_labels, cmap='Accent', alpha=0.6)
        plt.colorbar(im)
        plt.savefig(os.path.join(path, 'models/' + model, 'pca_coefficients.png'))
        plt.show()

    # training Random Forest
    print("Training Random Forest...")
    start_rf = time.time()
    
    # Create and train the Random Forest classifier
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        n_jobs=-1  # Use all available cores
    )
    
    # Train with progress bar
    print(f"Training with {len(A)} samples...")
    clf.fit(A, label)

    end_rf = time.time()

    # print the time taken for each step
    print("\nTraining complete")
    print(f"\nCalculating mean time: {end_mean - start_mean:.2f} seconds")
    print(f"Centering time: {end_center - start_center:.2f} seconds")
    if spca is not False:
        print(f"PCA time: {end_pca - start_pca:.2f} seconds")
    print(f"Random Forest training time: {end_rf - start_rf:.2f} seconds")
    print(f"\nTotal training time: {end_rf - start_training:.2f} seconds")
    
    # Plot feature importance if not using PCA
    if pca is None and input("Plot feature importance? (y/n) ").lower() == 'y':
        feature_importance = clf.feature_importances_
        # Get the indices of the top 20 features
        top_indices = np.argsort(feature_importance)[-20:]
        plt.figure(figsize=(10, 6))
        plt.title("Top 20 Feature Importances")
        plt.barh(range(20), feature_importance[top_indices])
        plt.yticks(range(20), [f"Feature {i}" for i in top_indices])
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(os.path.join(path, 'models/' + model, 'feature_importance.png'))
        plt.show()
    
    return D_m, pca, le, clf, encoded_labels

def rf_predict(image_name):
    """
    Predict the family and manufacturer of an image
    :param image_name: name of the image
    :return: None
    """
    # start the timer
    start_predict = time.time()
    # get the size and gray scale from the model to be in accordance with the training
    from data_extract import size, gray
    start_extract = time.time()

    # extract the image
    if gray:
        img = cv2.imread(os.path.join(path + '/images', image_name + '.jpg'), cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(os.path.join(path + '/images', image_name + '.jpg'))
    if img is None:
        print(f"Error loading image: {image_name}")
        return None

    # get crop values
    values = get_image_data(image_name)
    x1, y1, x2, y2 = values[0], values[1], values[2], values[3]
    #crop and resize the image
    croped_img = img[y1 + 1:y2 + 1, x1 + 1:x2 + 1]
    croped_img = cv2.resize(croped_img, (size, size), interpolation=cv2.INTER_AREA)
    d_img = croped_img.flatten()
    d_img_array = []
    d_img_array.append(d_img)
    end_extract = time.time()

    print("\nPredicting family...")

    print("Normalizing data...")
    d_img_array = np.array(d_img_array) / 255.0
    # centering
    print("Centering data...")
    start_center_f = time.time()
    d_img_c = d_img_array - f_D_m
    end_center_f = time.time()

    # pca transformation
    if f_pca is None:
        print("No PCA used")
        A_img = d_img_c
    else:
        print("Calculating PCA...")
        start_pca_f = time.time()
        A_img = f_pca.transform(d_img_c)
        end_pca_f = time.time()

    # Predicting
    print("\nPredicting...")
    start_predict_f = time.time()
    pred = f_clf.predict(A_img)
    
    # Get prediction probabilities
    pred_proba = f_clf.predict_proba(A_img)
    end_predict_f = time.time()
    pred_percentage = np.max(pred_proba) * 100
    print(f"Predicted family: {pred[0]} ({pred_percentage:.2f}%)")

    print("\nPredicting manufacturer...")
    # centering
    start_center_m = time.time()
    d_img_c = d_img_array - m_D_m
    end_center_m = time.time()
    # pca
    if m_pca is None:
        print("No PCA used")
        A_img = d_img_c
    else:
        print("Calculating PCA...")
        start_pca_m = time.time()
        A_img = m_pca.transform(d_img_c)
        end_pca_m = time.time()
    # Predicting
    print("\nPredicting...")
    start_predict_m = time.time()
    pred = m_clf.predict(A_img)
    pred_proba = m_clf.predict_proba(A_img)
    end_predict_m = time.time()
    pred_percentage = np.max(pred_proba) * 100

    # print the time taken for each step
    print(f"Predicted manufacturer: {pred[0]} ({pred_percentage:.2f}%)")
    print(f"\nExtracting time: {end_extract - start_extract:.2f} seconds")
    print(f"\nFamily centering time: {end_center_f - start_center_f:.2f} seconds")
    if f_pca is not None:
        print(f"Family PCA time: {end_pca_f - start_pca_f:.2f} seconds")
    print(f"Family predicting time: {end_predict_f - start_predict_f:.2f} seconds")
    print(f"\nManufacturer centering time: {end_center_m - start_center_m:.2f} seconds")
    if m_pca is not None:
        print(f"Manufacturer PCA time: {end_pca_m - start_pca_m:.2f} seconds")
    print(f"Manufacturer predicting time: {end_predict_m - start_predict_m:.2f} seconds")
    print(f"\nTotal predicting time: {end_predict_m - start_predict:.2f} seconds")
    return None

def rf_test(data_family, label_family, data_manufacturer, label_manufacturer):
    """
    Test the Random Forest models for family and manufacturer
    :param data_family: family data
    :param label_family: family labels
    :param data_manufacturer: manufacturer data
    :param label_manufacturer: manufacturer labels
    :return: None
    """
    print("\nTesting family model...")
    rf_test_s(data_family, label_family, f_D_m, f_clf, f_pca)
    print("\nTesting manufacturer model...")
    rf_test_s(data_manufacturer, label_manufacturer, m_D_m, m_clf, m_pca)

def rf_test_s(data, label, D_m, clf, pca):
    """
    Test a single Random Forest model
    :param data: test data
    :param label: test labels
    :param D_m: mean
    :param clf: classifier
    :param pca: PCA
    :return: None
    """
    # start the timer
    start = time.time()

    # convert data to numpy array
    data = np.array(data)
    print("Normalizing data...")
    data = data / 255.0

    print("Centering data...")
    start_center = time.time()
    D0_test = data - D_m
    end_center = time.time()

    if pca is None:
        print("No PCA used")
        A_test = D0_test
    else:
        print("Calculating PCA...")
        start_pca = time.time()
        A_test = pca.transform(D0_test)
        end_pca = time.time()

    print("\nPredicting and calculating score...")
    # Calculate accuracy
    start_predict_1 = time.time()
    scores = clf.score(A_test, label)
    end_predict_1 = time.time()

    # Calculate balanced accuracy
    start_predict_2 = time.time()
    predictions = clf.predict(A_test)
    accuracy = balanced_accuracy_score(label, predictions)
    end_predict_2 = time.time()

    print(f"Accuracy: {scores *100:.2f}%")
    print(f"\nTime for accuracy score: {end_predict_1 - start_predict_1:.2f} seconds")
    print(f'Balanced accuracy (mean accuracy of each label): {accuracy * 100:.2f}%')
    print(f"Time for balanced accuracy score: {end_predict_2 - start_predict_2:.2f} seconds")
    print(f"Centering time: {end_center - start_center:.2f} seconds")
    if pca is not None:
        print(f"PCA time: {end_pca - start_pca:.2f} seconds")
    print(f"Total testing time: {end_predict_2 - start:.2f} seconds")

def rf_param(model):
    """
    Show the parameters of the Random Forest model
    :param model: model name
    :return: None
    """
    from data_extract import size, gray
    
    # Load hyperparameters
    hyperparams = joblib.load(os.path.join(path, 'models/' + model, 'hyperparams.pkl'))
    
    print(" ")
    print("Showing model parameters")
    print("Size of the image: " + str(size))
    print("Gray scale: " + str(gray))

    print(" ")
    print("Family model parameters")
    if f_pca is None:
        print("No PCA used")
    else:
        chosen_components = f_pca.n_components_
        print(f"Number of components chosen for PCA: {chosen_components}")
        explained_variance = sum(f_pca.explained_variance_ratio_) * 100
        print(f"Percentage of variance explained by the chosen components: {explained_variance:.2f}%")
    
    print(f"Number of trees: {hyperparams['n_estimators']}")
    print(f"Maximum depth: {hyperparams['max_depth']}")
    print(f"Minimum samples split: {hyperparams['min_samples_split']}")
    print(f"Minimum samples leaf: {hyperparams['min_samples_leaf']}")
    
    # Print number of features used
    print(f"Number of features used: {f_clf.n_features_in_}")
    
    print(" ")
    print("Manufacturer model parameters")
    if m_pca is None:
        print("No PCA used")
    else:
        chosen_components = m_pca.n_components_
        print(f"Number of components chosen for PCA: {chosen_components}")
        explained_variance = sum(m_pca.explained_variance_ratio_) * 100
        print(f"Percentage of variance explained by the chosen components: {explained_variance:.2f}%")
    
    print(f"Number of trees: {hyperparams['n_estimators']}")
    print(f"Maximum depth: {hyperparams['max_depth']}")
    print(f"Minimum samples split: {hyperparams['min_samples_split']}")
    print(f"Minimum samples leaf: {hyperparams['min_samples_leaf']}")
    
    # Print number of features used
    print(f"Number of features used: {m_clf.n_features_in_}")