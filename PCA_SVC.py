from scipy import sparse
import os
import cv2
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from database import get_image_data
from path import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import joblib
import time

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

def load_svc_models(model):
    from data_extract import change_size
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
    change_size(model)
    print("Models loaded")

def svc_train(data_family, label_family, data_manufacturer, label_manufacturer, model):
    global f_D_m, f_pca, f_clf, f_le
    global m_D_m, m_pca, m_clf, m_le
    print("svc training")
    print("training family model...")
    f_D_m, f_pca, f_le, f_clf, f_encoded_labels = svc_train_s(data_family, label_family, model)
    print("training manufacturer model...")
    m_D_m, m_pca, m_le, m_clf, m_encoded_labels = svc_train_s(data_manufacturer, label_manufacturer, model)
    
    # Save the models
    newpath = os.path.join(path, 'models/' + model)
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
    from data_extract import size, gray
    joblib.dump(size, os.path.join(path, 'models/' + model, 'size.pkl'))
    joblib.dump(gray, os.path.join(path, 'models/' + model, 'gray.pkl'))
    print("models saved")
    print("training complete")


def n_components_selecter():
    n_components = float(input("number of components for pca: "))
    if n_components > 1:
        try:
            n_components = int(n_components)
        except:
            print("Invalid number of components, please enter an integer over 1 or a float under 1")
            n_components_selecter()
    return n_components

def svc_train_s(data, label, model):
    start_training = time.time()
    # Encode labels
    le = LabelEncoder()
    encoded_labels = le.fit_transform(label)
    # centering
    data_train = np.array(data)
    print("normalizing data...")
    data_train = data_train / 255.0
    print("calculating mean...")
    start_mean = time.time()
    D_m = np.mean(data_train, axis=0)
    end_mean = time.time()
    print("centering data...")
    start_center = time.time()
    D0_train = data_train - D_m
    end_center = time.time()

    # svd
    print("calculating pca...")
    # pca
    n_components = n_components_selecter()
    pca = PCA(n_components=n_components)
    start_pca = time.time()
    A = pca.fit_transform(D0_train)
    end_pca = time.time()

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

    print("training SVC...")
    match model:
        case "svc":
            clf = SVC(probability=True)
        case "lsvc":
            clf = SVC(kernel='linear', probability=True)
        case "psvc":
            degree = int(input("degree of the polynomial kernel: "))
            clf = SVC(kernel='poly', degree=degree, probability=True)
    start_svc = time.time()
    clf.fit(A, label)
    end_svc = time.time()
    print("training complete")
    print(f"calculating mean time: {end_mean - start_mean:.2f} seconds")
    print(f"centering time: {end_center - start_center:.2f} seconds")
    print(f"pca time: {end_pca - start_pca:.2f} seconds")
    print(f"svc time: {end_svc - start_svc:.2f} seconds")
    print(f"total training time: {end_svc - start_training:.2f} seconds")
    return D_m, pca, le, clf, encoded_labels

def svc_predict(image_name):
    start_predict = time.time()
    from data_extract import size, gray
    start_extract = time.time()
    if gray:
        img = cv2.imread(os.path.join(path + '/images', image_name + '.jpg'), cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(os.path.join(path + '/images', image_name + '.jpg'))
    if img is None:
        print(f"Error loading image: {image_name}")
        return None
    values = get_image_data(image_name)
    x1, y1, x2, y2 = values[0], values[1], values[2], values[3]
    croped_img = img[y1 + 1:y2 + 1, x1 + 1:x2 + 1]
    croped_img = cv2.resize(croped_img, (size, size), interpolation=cv2.INTER_AREA)
    d_img = croped_img.flatten()
    d_img_array = []
    d_img_array.append(d_img)
    end_extract = time.time()

    print("predicting family...")
    print("normalizing data...")
    d_img_array = np.array(d_img_array) / 255.0
    # centering
    print("centering data...")
    start_center_f = time.time()
    d_img_c = d_img_array - f_D_m
    end_center_f = time.time()

    # # pca
    print("calculating pca...")
    start_pca_f = time.time()
    A_img = f_pca.transform(d_img_c)
    end_pca_f = time.time()

    # Predicting
    print("predicting...")
    start_predict_f = time.time()
    pred = f_clf.predict(A_img)
    pred_proba = f_clf.predict_proba(A_img)
    end_predict_f = time.time()
    pred_percentage = np.max(pred_proba) * 100
    print(f"Predicted family: {pred[0]} ({pred_percentage:.2f}%)")

    print("predicting manufacturer...")
    # centering
    start_center_m = time.time()
    d_img_c = d_img_array - m_D_m
    end_center_m = time.time()
    # # pca
    print("calculating pca...")
    start_pca_m = time.time()
    A_img = m_pca.transform(d_img_c)
    end_pca_m = time.time()
    # Predicting
    print("predicting...")
    start_predict_m = time.time()
    pred = m_clf.predict(A_img)
    pred_proba = m_clf.predict_proba(A_img)
    end_predict_m = time.time()
    pred_percentage = np.max(pred_proba) * 100
    print(f"Predicted manufacturer: {pred[0]} ({pred_percentage:.2f}%)")
    print(f"extracting time: {end_extract - start_extract:.2f} seconds")
    print(f"family centering time: {end_center_f - start_center_f:.2f} seconds")
    print(f"family pca time: {end_pca_f - start_pca_f:.2f} seconds")
    print(f"family predicting time: {end_predict_f - start_predict_f:.2f} seconds")
    print(f"manufacturer centering time: {end_center_m - start_center_m:.2f} seconds")
    print(f"manufacturer pca time: {end_pca_m - start_pca_m:.2f} seconds")
    print(f"manufacturer predicting time: {end_predict_m - start_predict_m:.2f} seconds")
    print(f"total predicting time: {end_predict_m - start_predict:.2f} seconds")
    return None

def svc_test(data_family, label_family, data_manufacturer, label_manufacturer):
    print("testing family model...")
    svc_test_s(data_family, label_family, f_D_m, f_clf, f_pca)
    print("testing manufacturer model...")
    svc_test_s(data_manufacturer, label_manufacturer, m_D_m, m_clf, m_pca)

def svc_test_s(data, label, D_m, clf, pca):
    start = time.time()
    print("centering data...")
    data = np.array(data)
    print("normalizing data...")
    data = data / 255.0
    print("centering data...")
    start_center = time.time()
    D0_test = data - D_m
    end_center = time.time()
    print("calculating pca...")
    start_pca = time.time()
    A_test = pca.transform(D0_test)
    end_pca = time.time()
    print("predicting and calculating score...")
    start_predict_1 = time.time()
    scores = clf.score(A_test, label)
    end_predict_1 = time.time()
    start_predict_2 = time.time()
    predictions = clf.predict(A_test)
    accuracy = accuracy_score(label, predictions)
    end_predict_2 = time.time()
    print(f"Accuracy (score method): {scores *100:.2f}%")
    print(f"time for accuracy score: {end_predict_1 - start_predict_1:.2f} seconds")
    print(f'Accuracy (accuracy_score method): {accuracy * 100:.2f}%')
    print(f"time for accuracy score: {end_predict_2 - start_predict_2:.2f} seconds")
    print(f"centering time: {end_center - start_center:.2f} seconds")
    print(f"pca time: {end_pca - start_pca:.2f} seconds")
    print(f"total testing time: {end_predict_2 - start:.2f} seconds")