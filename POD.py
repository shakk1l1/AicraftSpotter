from fontTools.subset import prune_post_subset
from scipy import sparse
import os
import cv2
from sklearn.svm import SVC
from database import get_image_data
from path import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

f_D_m = None
f_U = None
f_s = None
f_Vt = None
f_clf = None
f_le = None

m_D_m = None
m_U = None
m_s = None
m_Vt = None
m_clf = None
m_le = None

def pod_train(data_family, label_family, data_manufacturer, label_manufacturer):
    global f_D_m, f_U, f_s, f_Vt, f_clf, f_le
    global m_D_m, m_U, m_s, m_Vt, m_clf, m_le
    print("POD training")
    print("training family model...")
    f_D_m, f_U, f_s, f_Vt, f_le, f_clf = pod_train_s(data_family, label_family)
    print("training manufacturer model...")
    m_D_m, m_U, m_s, m_Vt, m_le, m_clf = pod_train_s(data_manufacturer, label_manufacturer)

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
    return D_m, U, s, Vt, le, clf

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

    print("predicting family...")
    # centering
    d_img_c = d_img.T - f_D_m

    # POD
    A_img = np.dot(d_img_c.T, f_U)

    # Predicting
    raw_pred = f_clf.predict(A_img)
    pred = f_le.inverse_transform(raw_pred)
    pred_proba = f_clf.predict_proba(A_img)
    pred_percentage = np.max(pred_proba) * 100
    print(f"Predicted family: {pred[0]} ({pred_percentage:.2f}%)")

    print("predicting manufacturer...")
    # centering
    d_img_c = d_img.T - m_D_m
    # POD
    A_img = np.dot(d_img_c.T, m_U)
    # Predicting
    raw_pred = m_clf.predict(A_img)
    pred = m_le.inverse_transform(raw_pred)
    pred_proba = m_clf.predict_proba(A_img)
    pred_percentage = np.max(pred_proba) * 100
    print(f"Predicted manufacturer: {pred[0]} ({pred_percentage:.2f}%)")
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