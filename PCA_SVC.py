
from scipy import sparse
import os
import cv2
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from database import get_image_data
from path import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import joblib

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
    change_size()
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
    from data_extract import size
    joblib.dump(size, os.path.join(path, 'models/' + model, 'size.pkl'))
    print("models saved")
    print("training complete")

def svc_train_s(data, label, model):
    # Encode labels
    le = LabelEncoder()
    encoded_labels = le.fit_transform(label)
    # centering
    data_train = np.array(data)
    print("normalizing data...")
    data_train = data_train / 255.0
    print("transposing data...")
    D_train = data_train.T
    print("calculating mean...")
    D_m = np.mean(D_train, axis=1)[:, np.newaxis]
    print("centering data...")
    D0_train = D_train - D_m

    # svd
    print("calculating pca...")
    # pca
    n_components = int(input("number of components for pca: "))
    pca = PCA(n_components=n_components)
    A = pca.fit_transform(D0_train.T)

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
    clf.fit(A, encoded_labels)
    return D_m, pca, le, clf, encoded_labels

def svc_predict(image_name):
    from data_extract import size
    
    img = cv2.imread(os.path.join(path + '/images', image_name + '.jpg'))
    if img is None:
        print(f"Error loading image: {image_name}")
        return None
    values = get_image_data(image_name)
    x1, y1, x2, y2 = values[0], values[1], values[2], values[3]
    croped_img = img[y1 + 1:y2 + 1, x1 + 1:x2 + 1]
    croped_img = cv2.resize(croped_img, (size, size), interpolation=cv2.INTER_AREA)
    d_img = croped_img.flatten()
    print("d_img shape: ", d_img.shape)

    print("predicting family...")
    print("normalizing data...")
    d_img = d_img / 255.0
    # centering
    print("centering data...")
    d_img_c = d_img.T - f_D_m

    # # pca
    print("calculating pca...")
    A_img = f_pca.transform(d_img_c)

    # Predicting
    print("predicting...")
    raw_pred = f_clf.predict(A_img)
    pred = f_le.inverse_transform(raw_pred)
    pred_proba = f_clf.predict_proba(A_img)
    pred_percentage = np.max(pred_proba) * 100
    print(f"Predicted family: {pred[0]} ({pred_percentage:.2f}%)")

    print("predicting manufacturer...")
    # centering
    d_img_c = d_img.T - m_D_m
    # # pca
    print("calculating pca...")
    A_img = m_pca.transform(d_img_c)
    # Predicting
    print("predicting...")
    raw_pred = m_clf.predict(A_img)
    pred = m_le.inverse_transform(raw_pred)
    pred_proba = m_clf.predict_proba(A_img)
    pred_percentage = np.max(pred_proba) * 100
    print(f"Predicted manufacturer: {pred[0]} ({pred_percentage:.2f}%)")
    return None

def svc_test(data_family, label_family, data_manufacturer, label_manufacturer):
    print("svc testing")
    print("testing family model...")
    svc_test_s(data_family, label_family, f_D_m, f_clf, f_pca)
    print("testing manufacturer model...")
    svc_test_s(data_manufacturer, label_manufacturer, m_D_m, m_clf, m_pca)

def svc_test_s(data, label, D_m, clf, pca):
    print("centering data...")
    data = np.array(data)
    print("normalizing data...")
    data = data / 255.0
    print("centering data...")
    D0_test = data.T - D_m
    print("calculating pca...")
    A_test = pca.transform(D0_test.T)
    print("predicting and calculating score...")
    scores = clf.score(A_test, label)
    print(f"Accuracy: {scores:.2f}")