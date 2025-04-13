from scipy import sparse
import os
import cv2
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from database import get_image_data
from sklearn.linear_model import Lasso, LinearRegression, RidgeClassifier
from path import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import joblib
import time

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

def load_lreg_models(model):
    """
    Load the lreg models
    :param model:
    :return:
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

def lreg_train(data_family, label_family, data_manufacturer, label_manufacturer, model):
    """
    Train the two lreg models
    :param data_family:
    :param label_family:
    :param data_manufacturer:
    :param label_manufacturer:
    :param model:
    :return:
    """

    # global variables as they will be modified
    global f_D_m, f_pca, f_clf, f_le
    global m_D_m, m_pca, m_clf, m_le

    print("lreg training")

    if input("use PCA? (y/n) ") == 'y':
        if input("apply the same PCA on the two data sets? (y/n) ") == 'y':
            spca = n_components_selecter()
        else:
            spca = None
    else:
        spca = False

    if "lsr" in model:
        alpha = 0
    else:
        alpha = float(input("choose alpha coefficient for the training method: "))

    print("training family model...")
    f_D_m, f_pca, f_le, f_clf, f_encoded_labels = lreg_train_s(data_family, label_family, model, spca, alpha)
    print("training manufacturer model...")
    m_D_m, m_pca, m_le, m_clf, m_encoded_labels = lreg_train_s(data_manufacturer, label_manufacturer, model, spca, alpha)
    
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
    joblib.dump(alpha, os.path.join(path, 'models/' + model, 'alpha.pkl'))

    # import last version of size and gray
    from data_extract import size, gray
    joblib.dump(size, os.path.join(path, 'models/' + model, 'size.pkl'))
    joblib.dump(gray, os.path.join(path, 'models/' + model, 'gray.pkl'))

    print("models saved")
    print("training complete")


def n_components_selecter():
    """
    select the number of components for PCA
    :return:
    """
    n_components = float(input("number of components for pca (0 for no pca): "))
    if n_components > 1: # if n_components is an integer it must be the number of components
        try:
            n_components = int(n_components)
        except:
            print("Invalid number of components, please enter an integer over 1 or a float under 1")
            n_components_selecter()
    # else it must be a % of the image data to keep
    return n_components

def lreg_train_s(data, label, model, spca, coefficient=None):
    """
    train the lreg model
    :param data:
    :param label:
    :param model:
    :param spca:
    :return:
    """
    # start the timer
    start_training = time.time()

    # Encode labels for the plots
    le = LabelEncoder()
    encoded_labels = le.fit_transform(label)

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

    # training PCA
    if type(spca) is float or type(spca) is int:
        n_components  = spca
    elif spca is False:
        print("no pca")
    else:
    # pca
        n_components = n_components_selecter()
    if spca is not False:
        print("calculating pca...")
        pca = PCA(n_components=n_components)
        start_pca = time.time()
        A = pca.fit_transform(D0_train)
        end_pca = time.time()
    else:
        A = D0_train
        pca = None

    if input("plot eigenvalues distribution? (y/n) ") == 'y':
        # Plot the distribution of the eigenvalues
        plt.title("Eigenvalues distribution")
        plt.xlabel("Eigenvalue index")
        plt.ylabel("Eigenvalue")
        plt.grid()
        plt.scatter(np.arange(len(pca.explained_variance_)), pca.explained_variance_)
        plt.show()


    if input("plot the distribution of the pca coefficients ? (y/n) ") == 'y':
        plt.title("pca coefficients distribution")
        plt.xlabel("pca coefficient index")
        plt.ylabel("pca coefficient")
        im = plt.scatter(A[:, 0], A[:, 1], c=encoded_labels, cmap='Accent', alpha=0.6)
        plt.colorbar(im)
        plt.show()

    # training lreg
    print("training lreg...")
    # variance of lreg possible models
    # probability=True is needed for the predict_proba method and have the probability of the prediction
    match model:
        case "lreg-lasso":
            clf = Lasso(alpha=coefficient)
            label = le.fit_transform(label)
        case "lreg-ridge":
            clf = RidgeClassifier(alpha=coefficient)
            le = None
        case "lreg-lsr":
            clf = LinearRegression()
            label = le.fit_transform(label)
    start_lreg = time.time()
    # train the lreg
    clf.fit(A, label)
    end_lreg = time.time()

    # print the time taken for each step
    print("\ntraining complete")
    print(f"\ncalculating mean time: {end_mean - start_mean:.2f} seconds")
    print(f"centering time: {end_center - start_center:.2f} seconds")
    if spca is not False:
        print(f"pca time: {end_pca - start_pca:.2f} seconds")
    print(f"lreg time: {end_lreg - start_lreg:.2f} seconds")
    print(f"\ntotal training time: {end_lreg - start_training:.2f} seconds")
    return D_m, pca, le, clf, encoded_labels

def lreg_predict(image_name):
    """
    Predict the family and manufacturer of an image
    :param image_name:
    :return:
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

    print("\npredicting family...")

    print("normalizing data...")
    d_img_array = np.array(d_img_array) / 255.0
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
    if f_le is not None:
        pred = f_le.inverse_transform(pred.astype(int))
    print(f"Predicted family: {pred[0]}")

    print("\npredicting manufacturer...")
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

def lreg_test(data_family, label_family, data_manufacturer, label_manufacturer):
    """
    Test the lreg models for family and manufacturer
    :param data_family:
    :param label_family:
    :param data_manufacturer:
    :param label_manufacturer:
    :return:
    """
    print("\ntesting family model...")
    lreg_test_s(data_family, label_family, f_D_m, f_clf, f_pca, f_le)
    print("\ntesting manufacturer model...")
    lreg_test_s(data_manufacturer, label_manufacturer, m_D_m, m_clf, m_pca, m_le)

def lreg_test_s(data, label, D_m, clf, pca, le):
    """
    Test the lreg model
    :param data:
    :param label:
    :param D_m:
    :param clf:
    :param pca:
    :return:
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

    if pca is None:
        print("no pca used")
        A_test = D0_test
    else:
        print("calculating pca...")
        start_pca = time.time()
        A_test = pca.transform(D0_test)
        end_pca = time.time()

    print("\npredicting and calculating score...")
    # two methods to calculate the score
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

    print(f"Accuracy: {accuracy *100:.2f}%")
    print(f"\ntime for accuracy score: {end_predict_2 - start_predict_2:.2f} seconds")
    print(f'Balanced accuracy (mean accuracy of each label): {accuracy_b * 100:.2f}%')
    print(f"time for balanced accuracy score: {end_predict_3 - start_predict_3:.2f} seconds")
    print(f"centering time: {end_center - start_center:.2f} seconds")
    if pca is not None:
        print(f"pca time: {end_pca - start_pca:.2f} seconds")
    print(f"total testing time: {end_predict_3 - start:.2f} seconds")