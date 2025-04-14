from scipy import sparse
import os
import cv2
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from database import get_image_data
from sklearn.linear_model import Lasso, LinearRegression, RidgeClassifier, RidgeClassifierCV, LassoCV
from path import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import joblib
import time

# Global variables
# f_ for family
f_D_m = None
f_pca = None
f_clf = None
f_le = None
f_encoded_labels = None

# m_ for manufacturer
m_D_m = None
m_pca = None
m_clf = None
m_le = None
m_encoded_labels = None

def load_cv_models(model):
    """
    Load the cv models saved in the models folder by changing global variables
    :param model: cv model to load
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

def cv_train(data_family, label_family, data_manufacturer, label_manufacturer, model):
    """
    Train the two cv models
    :param data_family: image data in numpy array (#images, size*size)
    :param label_family: labels of each image
    :param data_manufacturer:
    :param label_manufacturer:
    :param model: model to use for training
    """

    # global variables as they will be modified
    global f_D_m, f_pca, f_clf, f_le
    global m_D_m, m_pca, m_clf, m_le

    print("cv training")

    if input("use PCA? (y/n) ").lower() == 'y':     # ask if PCA is wanted
        if input("apply the same PCA on the two data sets? (y/n) ").lower() == 'y':
            spca = n_components_selecter()      # select the number of components that will be used for PCA
        else:
            spca = None
    else:
        spca = False

    # select the number of folds for cross validation
    cv = int(input("selcect number of fold for cross validation: "))

    print("training family model...")
    f_D_m, f_pca, f_le, f_clf, f_encoded_labels = cv_train_s(data_family, label_family, model, spca, cv)
    print("training manufacturer model...")
    m_D_m, m_pca, m_le, m_clf, m_encoded_labels = cv_train_s(data_manufacturer, label_manufacturer, model, spca, cv)
    
    # create the path to the models folder
    newpath = os.path.join(path, 'models/' + model)

    # create the directory if it does not exist
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    # save the models
    joblib.dump(f_clf, os.path.join(path, 'models/' + model, 'family_model.pkl'))
    joblib.dump(f_le, os.path.join(path, 'models/' + model, 'family_label_encoder.pkl'))
    joblib.dump(m_clf, os.path.join(path, 'models/' + model, 'manufacturer_model.pkl'))
    joblib.dump(m_le, os.path.join(path, 'models/' + model, 'manufacturer_label_encoder.pkl'))
    joblib.dump(f_D_m, os.path.join(path, 'models/' + model, 'family_mean.pkl'))
    joblib.dump(m_D_m, os.path.join(path, 'models/' + model, 'manufacturer_mean.pkl'))
    joblib.dump(f_pca, os.path.join(path, 'models/' + model, 'family_pca.pkl'))
    joblib.dump(m_pca, os.path.join(path, 'models/' + model, 'manufacturer_pca.pkl'))

    # import last version of size and gray
    from data_extract import size, gray
    # save the size and gray variables
    joblib.dump(size, os.path.join(path, 'models/' + model, 'size.pkl'))
    joblib.dump(gray, os.path.join(path, 'models/' + model, 'gray.pkl'))

    print("models saved")
    print("training complete")


def n_components_selecter():
    """
    select the number of components for PCA
    :return: number of components
    """
    try:
        n_components = float(input("number of components for pca (0 for no pca): "))
        if n_components > 1: # if n_components is an integer it must be the number of components
            n_components = int(n_components)
    except:
        print("Invalid number of components, please enter an integer over 1 or a float under 1")
        n_components_selecter()
    # else it must be a % of the image data to keep
    return n_components

def cv_train_s(data, label, model, spca, coefficient=None):
    """
    train the cv model
    :param data: data for the model training
    :param label: labels for the model training
    :param model: model selected for training
    :param spca: number of components for PCA
    :param coefficient: number of folds for cross validation
    :return: model variables trained
    """
    # start the timer
    start_training = time.time()

    # Encode labels for the plots and the non-classifier models
    le = LabelEncoder()
    encoded_labels = le.fit_transform(label)        # transform class labels to numbers

    # convert data to numpy array
    data_train = np.array(data)

    # normalize data
    print("\nnormalizing data...")
    data_train = data_train / 255.0

    # centering data
    print("calculating mean...")
    start_mean = time.time()
    D_m = np.mean(data_train, axis=0)
    end_mean = time.time()

    print("centering data...")
    start_center = time.time()
    D0_train = data_train - D_m
    end_center = time.time()

    # check the 3 cases for PCA
    # 1st case: spca is a number of components
    # 2nd case: spca is False (no PCA)
    # 3rd case: spca is None (ask for the number of components for each data set)
    if type(spca) is float or type(spca) is int:
        n_components  = spca
    elif spca is False:
        print("no pca")
    else:       # normally spca is None here
        n_components = n_components_selecter()

    # pca transformation
    if spca is not False:
        print("calculating pca...")
        pca = PCA(n_components=n_components)
        start_pca = time.time()
        A = pca.fit_transform(D0_train)
        end_pca = time.time()
    else:       # no pca. Data is unchanged
        pca = None
        A = D0_train

    # non-obligatory plots
    if input("plot eigenvalues distribution? (y/n) ").lower() == 'y':
        # Plot the distribution of the eigenvalues
        plt.title("Eigenvalues distribution")
        plt.xlabel("Eigenvalue index")
        plt.ylabel("Eigenvalue")
        plt.grid()
        plt.scatter(np.arange(len(pca.explained_variance_)), pca.explained_variance_)
        plt.show()


    if input("plot the distribution of the pca coefficients ? (y/n) ").lower() == 'y':
        plt.title("pca coefficients distribution")
        plt.xlabel("pca coefficient index")
        plt.ylabel("pca coefficient")
        im = plt.scatter(A[:, 0], A[:, 1], c=encoded_labels, cmap='Accent', alpha=0.6)
        plt.colorbar(im)
        plt.show()

    # training cv models
    print("training cv...")
    # variance of cv possible models
    match model:
        case "cv-lasso":
            clf = LassoCV(cv=coefficient, random_state=0)
            label = le.transform(label)     # transform class labels to numbers as lassoCV does not accept strings
                                            # LassoCV is not a Classifier. It will predict a float value
        case "cv-ridge":
            clf = RidgeClassifierCV(cv=coefficient)
            le = None       # set le to None to know afterwards that the model is a classifier
                            # and does not need to have prediction transformed

    start_cv = time.time()
    # train the cv
    clf.fit(A, label)
    end_cv = time.time()

    # print the optimal alpha found for lasso
    if model == "cv-lasso":
        print(f"Optimal alpha found for lasso: {clf.alpha_}")

    # print the time taken for each step
    print("\ntraining complete")
    print(f"\ncalculating mean time: {end_mean - start_mean:.2f} seconds")
    print(f"centering time: {end_center - start_center:.2f} seconds")
    if spca is not False:
        print(f"pca time: {end_pca - start_pca:.2f} seconds")
    print(f"cv time: {end_cv - start_cv:.2f} seconds")
    print(f"\ntotal training time: {end_cv - start_training:.2f} seconds")

    return D_m, pca, le, clf, encoded_labels

def cv_predict(image_name):
    """
    Predict the family and manufacturer of an image
    :param image_name: the name of the image to predict
    :return: None
    """

    # start the timer
    start_predict = time.time()

    # get the size and gray scale from the model to be in accordance with the training
    from data_extract import size, gray
    start_extract = time.time()
    # extract the image
    if gray:        # if the model was trained with gray scale images
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

    # flatten the image to a semi-1D array
    d_img = croped_img.flatten()
    d_img_array = []
    d_img_array.append(d_img)       # transform the image to a 2D array (1, size*size)
    end_extract = time.time()

    print("\npredicting family...")

    print("normalizing data...")
    d_img_array = np.array(d_img_array) / 255.0

    ## Family model prediction
    # centering
    print("centering data...")
    start_center_f = time.time()
    d_img_c = d_img_array - f_D_m
    end_center_f = time.time()

    # pca transformation
    if f_pca is None:
        print("no pca used")
        A_img = d_img_c
    else:
        print("calculating pca...")
        start_pca_f = time.time()
        A_img = f_pca.transform(d_img_c)
        end_pca_f = time.time()

    # Predicting
    print("\npredicting...")
    start_predict_f = time.time()
    pred = f_clf.predict(A_img)
    end_predict_f = time.time()
    if f_le is not None:        # transform the prediction to the original labels if needed
        pred = f_le.inverse_transform(pred.astype(int))     # transform float to the nearest int
    print(f"Predicted family: {pred[0]}")


    print("\npredicting manufacturer...")
    ## Manufacturer model prediction
    # centering
    start_center_m = time.time()
    d_img_c = d_img_array - m_D_m
    end_center_m = time.time()
    # pca
    if m_pca is None:
        print("no pca used")
        A_img = d_img_c
    else:
        print("calculating pca...")
        start_pca_m = time.time()
        A_img = m_pca.transform(d_img_c)
        end_pca_m = time.time()
    # Predicting
    print("\npredicting...")
    start_predict_m = time.time()
    pred = m_clf.predict(A_img)
    end_predict_m = time.time()
    if m_le is not None:
        pred = m_le.inverse_transform(pred.astype(int))
    print(f"Predicted manufacturer: {pred[0]}")
    # print the time taken for each step
    print(f"\nextracting time: {end_extract - start_extract:.2f} seconds")
    print(f"\nfamily centering time: {end_center_f - start_center_f:.2f} seconds")
    print(f"family pca time: {end_pca_f - start_pca_f:.2f} seconds")
    print(f"family predicting time: {end_predict_f - start_predict_f:.2f} seconds")
    print(f"\nmanufacturer centering time: {end_center_m - start_center_m:.2f} seconds")
    print(f"manufacturer pca time: {end_pca_m - start_pca_m:.2f} seconds")
    print(f"manufacturer predicting time: {end_predict_m - start_predict_m:.2f} seconds")
    print(f"\ntotal predicting time: {end_predict_m - start_predict:.2f} seconds")
    return None

def cv_test(data_family, label_family, data_manufacturer, label_manufacturer):
    """
    Test the cv models for family and manufacturer
    :param data_family:
    :param label_family:
    :param data_manufacturer:
    :param label_manufacturer:
    """
    print("\ntesting family model...")
    cv_test_s(data_family, label_family, f_D_m, f_clf, f_pca, f_le)
    print("\ntesting manufacturer model...")
    cv_test_s(data_manufacturer, label_manufacturer, m_D_m, m_clf, m_pca, m_le)

def cv_test_s(data, label, D_m, clf, pca, le):
    """
    Test the cv model
    :param data: data for the model testing
    :param label: labels for the model testing
    :param D_m: mean of the data used for training
    :param clf: model used
    :param pca: pca model used if any
    """

    # start the timer
    start = time.time()

    # convert data to numpy array
    data = np.array(data)
    print("normalizing data...")
    data = data / 255.0

    print("centering data...")
    start_center = time.time()
    D0_test = data - D_m
    end_center = time.time()

    if pca is None:     # check if pca was used
        print("no pca used")
        A_test = D0_test
    else:
        print("calculating pca...")
        start_pca = time.time()
        A_test = pca.transform(D0_test)
        end_pca = time.time()

    if le is not None:      # transform the labels to numbers if needed
        print("label encoding...")
        label = le.transform(label)

    print("\npredicting and calculating score...")
    # 3 methods to calculate the score
    # 1st method
    # start_predict_1 = time.time()
    # scores = clf.score(A_test, label)
    # end_predict_1 = time.time()

    # 2nd method
    start_predict_2 = time.time()
    predictions = clf.predict(A_test)
    if le is not None:
        predictions = le.inverse_transform(predictions.astype(int))
    accuracy = accuracy_score(label, predictions)

    end_predict_2 = time.time()

    # 3d method

    start_predict_3 = time.time()
    accuracy_b = balanced_accuracy_score(label, predictions)
    end_predict_3 = time.time()

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"\ntime for accuracy score: {end_predict_2 - start_predict_2:.2f} seconds")
    print(f'Balanced accuracy (mean accuracy of each label): {accuracy_b * 100:.2f}%')
    print(f"time for balanced accuracy score: {end_predict_3 - start_predict_3:.2f} seconds")
    print(f"centering time: {end_center - start_center:.2f} seconds")
    if pca is not None:
        print(f"pca time: {end_pca - start_pca:.2f} seconds")
    print(f"total testing time: {end_predict_3 - start:.2f} seconds")