
from scipy import sparse
import os
import cv2
from sklearn.svm import SVC
from database import get_image_data
from path import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from data_extract import change_size
import joblib

f_D_m = None
f_U = None
f_s = None
f_Vt = None
f_clf = None
f_le = None
f_encoded_labels = None

m_D_m = None
m_U = None
m_s = None
m_Vt = None
m_clf = None
m_le = None
m_encoded_labels = None

def load_POD_models():
    global f_D_m, f_U, f_s, f_Vt, f_clf, f_le
    global m_D_m, m_U, m_s, m_Vt, m_clf, m_le
    print("Loading models...")
    f_clf = joblib.load(os.path.join(path, 'models/POD', 'family_model.pkl'))
    f_le = joblib.load(os.path.join(path, 'models/POD', 'family_label_encoder.pkl'))
    m_clf = joblib.load(os.path.join(path, 'models/POD', 'manufacturer_model.pkl'))
    m_le = joblib.load(os.path.join(path, 'models/POD', 'manufacturer_label_encoder.pkl'))
    f_D_m = joblib.load(os.path.join(path, 'models/POD', 'family_mean.pkl'))
    f_U = joblib.load(os.path.join(path, 'models/POD', 'family_U.pkl'))
    f_s = joblib.load(os.path.join(path, 'models/POD', 'family_s.pkl'))
    f_Vt = joblib.load(os.path.join(path, 'models/POD', 'family_Vt.pkl'))
    m_D_m = joblib.load(os.path.join(path, 'models/POD', 'manufacturer_mean.pkl'))
    m_U = joblib.load(os.path.join(path, 'models/POD', 'manufacturer_U.pkl'))
    m_s = joblib.load(os.path.join(path, 'models/POD', 'manufacturer_s.pkl'))
    m_Vt = joblib.load(os.path.join(path, 'models', 'manufacturer_Vt.pkl'))
    change_size()
    print("Models loaded")

def pod_train(data_family, label_family, data_manufacturer, label_manufacturer):
    global f_D_m, f_U, f_s, f_Vt, f_clf, f_le
    global m_D_m, m_U, m_s, m_Vt, m_clf, m_le
    print("POD training")
    print("training family model...")
    f_D_m, f_U, f_s, f_Vt, f_le, f_clf, f_encoded_labels = pod_train_s(data_family, label_family)
    print("training manufacturer model...")
    m_D_m, m_U, m_s, m_Vt, m_le, m_clf, m_encoded_labels = pod_train_s(data_manufacturer, label_manufacturer)
    
    # Save the models
    joblib.dump(f_clf, os.path.join(path, 'models', 'family_model.pkl'))
    joblib.dump(f_le, os.path.join(path, 'models', 'family_label_encoder.pkl'))
    joblib.dump(m_clf, os.path.join(path, 'models', 'manufacturer_model.pkl'))
    joblib.dump(m_le, os.path.join(path, 'models', 'manufacturer_label_encoder.pkl'))
    joblib.dump(f_D_m, os.path.join(path, 'models', 'family_mean.pkl'))
    joblib.dump(f_U, os.path.join(path, 'models', 'family_U.pkl'))
    joblib.dump(f_s, os.path.join(path, 'models', 'family_s.pkl'))
    joblib.dump(f_Vt, os.path.join(path, 'models', 'family_Vt.pkl'))
    joblib.dump(m_D_m, os.path.join(path, 'models', 'manufacturer_mean.pkl'))
    joblib.dump(m_U, os.path.join(path, 'models', 'manufacturer_U.pkl'))
    joblib.dump(m_s, os.path.join(path, 'models', 'manufacturer_s.pkl'))
    joblib.dump(m_Vt, os.path.join(path, 'models', 'manufacturer_Vt.pkl'))
    from data_extract import size
    joblib.dump(size, os.path.join(path, 'models', 'size.pkl'))
    print("models saved")
    print("training complete")

def pod_train_s(data, label):
    # Encode labels
    le = LabelEncoder()
    encoded_labels = le.fit_transform(label)
    # centering
    #print("resizing data...")
    #data_train = pad_sequences(data)
    data_train = np.array(data)
    print("transposing data...")
    D_train = data_train.T
    print("calculating mean...")
    D_m = np.mean(D_train, axis=1)[:, np.newaxis]
    print("centering data...")
    D0_train = D_train - D_m

    # svd
    print("calculating SVD...")
    U, s, Vt = np.linalg.svd(D0_train, full_matrices=False)
    A = np.dot(np.diag(s), Vt).T

    if input("plot eigenvalues distribution? (y/n) ") == 'y':
        # Plot the distribution of the eigenvalues
        print("plotting eigenvalues...")
        plt.title("Eigenvalues distribution")
        plt.xlabel("Eigenvalue index")
        plt.ylabel("Eigenvalue")
        plt.grid()
        plt.scatter(np.arange(len(s)), s **2)
        print(s.shape)
        plt.show()

    if input("plot the distribution of the POD coefficients ? (y/n) ") == 'y':
        print("plotting POD coefficients...")
        plt.title("POD coefficients distribution")
        plt.xlabel("POD coefficient index")
        plt.ylabel("POD coefficient")
        im = plt.scatter(A[:, 0], A[:, 1], c=encoded_labels, cmap='Accent', alpha=0.6)
        plt.colorbar(im)
        plt.show()

    print("training SVC...")
    clf = SVC(probability=True)
    clf.fit(A, encoded_labels)
    return D_m, U, s, Vt, le, clf, encoded_labels

def pod_predict(image_name):
    from data_extract import size
    
    img = cv2.imread(os.path.join(path + '/images', image_name + '.jpg'), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error loading image: {image_name}")
        return None
    values = get_image_data(image_name)
    x1, y1, x2, y2 = values[0], values[1], values[2], values[3]
    if x2 - x1 < size or y2 - y1 < size:
        print("\nimage too small (" + image_name + ")")
        print("adjusting size...")
        # Adjust the size of the image
        # Calculate the new size
        if x2 - x1 < size:
            x1 = int(x1 - ((size - (x2 - x1)) / 2))
            x2 = int(x2 + ((size - (x2 - x1)) / 2))
        elif y2 - y1 < size:
            y1 = int(y1 - ((size - (y2 - y1)) / 2))
            y2 = int(y2 + ((size - (y2 - y1)) / 2))
    croped_img = img[y1 + 1:y2 + 1, x1 + 1:x2 + 1]
    croped_img = cv2.resize(croped_img, (size, size))
    d_img = croped_img.flatten()
    print("d_img shape: ", d_img.shape)

    print("predicting family...")
    # centering
    d_img_c = d_img.T - f_D_m.T

    # POD
    A_img = np.dot(d_img_c, f_U)

    # Predicting
    raw_pred = f_clf.predict(A_img)
    pred = f_le.inverse_transform(raw_pred)
    pred_proba = f_clf.predict_proba(A_img)
    pred_percentage = np.max(pred_proba) * 100
    print(f"Predicted family: {pred[0]} ({pred_percentage:.2f}%)")

    if input("show plot of the POD coefficients? (y/n) ") == 'y':
        print("plotting POD coefficients...")
        plt.title("POD coefficients distribution")
        plt.xlabel("POD coefficient index")
        plt.ylabel("POD coefficient")
        A = np.dot(np.diag(f_s), f_Vt).T
        im = plt.scatter(A[:, 0], A[:, 1], c=f_encoded_labels, cmap='Accent', alpha=0.6)
        plt.scatter(A_img[:, 0], A_img[:, 1], c='red', s = 100, marker='x', label='Image')
        plt.colorbar(im)
        plt.show()

    print("predicting manufacturer...")
    # centering
    d_img_c = d_img.T - m_D_m.T
    # POD
    A_img = np.dot(d_img_c, m_U)
    # Predicting
    raw_pred = m_clf.predict(A_img)
    pred = m_le.inverse_transform(raw_pred)
    pred_proba = m_clf.predict_proba(A_img)
    pred_percentage = np.max(pred_proba) * 100
    print(f"Predicted manufacturer: {pred[0]} ({pred_percentage:.2f}%)")

    if input("show plot of the POD coefficients? (y/n) ") == 'y':
        print("plotting POD coefficients...")
        plt.title("POD coefficients distribution")
        plt.xlabel("POD coefficient index")
        plt.ylabel("POD coefficient")
        A = np.dot(np.diag(m_s), m_Vt).T
        im = plt.scatter(A[:, 0], A[:, 1], c=m_encoded_labels, cmap='Accent', alpha=0.6)
        plt.scatter(A_img[:, 0], A_img[:, 1], c='red', s = 100, marker='x', label='Image')
        plt.colorbar(im)
        plt.show()
    return None

# def pad_sequences(sequences, maxlen=None, dtype='float32', padding='post', truncating='post', value=0.):
#     lengths = [len(s) for s in sequences]
#     if maxlen is None:
#         maxlen = max(lengths)
#
#     sample_shape = tuple()
#     for s in sequences:
#         if len(s) > 0:
#             sample_shape = np.asarray(s).shape[1:]
#             break
#
#     x = (np.ones((len(sequences), maxlen) + sample_shape) * value).astype(dtype)
#     for idx, s in enumerate(sequences):
#         if len(s) == 0:
#             continue  # empty list/sequence
#         if truncating == 'pre':
#             trunc = s[-maxlen:]
#         else:
#             trunc = s[:maxlen]
#
#         trunc = np.asarray(trunc, dtype=dtype)
#         if trunc.shape[1:] != sample_shape:
#             raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
#                              (trunc.shape[1:], idx, sample_shape))
#
#         if padding == 'post':
#             x[idx, :len(trunc)] = trunc
#         else:
#             x[idx, -len(trunc):] = trunc
#
#     return x