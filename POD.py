from fontTools.subset import prune_post_subset
from scipy import sparse

from path import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

def pod_train(data, label, size):

    # Encode labels
    le = LabelEncoder()
    encoded_labels = le.fit_transform(label)
    print(encoded_labels)
    print(len(encoded_labels))
    # centering
    #print("resizing data...")
    #data_train = pad_sequences(data)
    data_train = np.array(data)
    print("data shape: ", data_train.shape)
    print("transposing data...")
    D_train = data_train.T
    print("calculating mean...")
    D_m = np.mean(D_train, axis=1)[:, np.newaxis]
    print("centering data...")
    D0_train = D_train - D_m
    print("centered data shape: ", D0_train.shape)

    # svd
    print("calculating SVD...")
    U, s, Vt = np.linalg.svd(D0_train, full_matrices=False)
    A = np.dot(np.diag(s), Vt).T

    # Plot the distribution of the eigenvalues
    print("plotting eigenvalues...")
    plt.title("Eigenvalues distribution")
    plt.xlabel("Eigenvalue index")
    plt.ylabel("Eigenvalue")
    plt.grid()
    plt.scatter(np.arange(len(s)), s **2)
    print(s.shape)
    plt.show()

    # Plot the distribution of the POD coefficients
    print("plotting POD coefficients...")
    plt.title("POD coefficients distribution")
    plt.xlabel("POD coefficient index")
    plt.ylabel("POD coefficient")
    im = plt.scatter(A[:, 0], A[:, 1], c=encoded_labels, cmap='Accent', alpha=0.6)
    plt.colorbar(im)
    plt.show()
    return None

def pad_sequences(sequences, maxlen=None, dtype='float32', padding='post', truncating='post', value=0.):
    lengths = [len(s) for s in sequences]
    if maxlen is None:
        maxlen = max(lengths)

    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((len(sequences), maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list/sequence
        if truncating == 'pre':
            trunc = s[-maxlen:]
        else:
            trunc = s[:maxlen]

        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        else:
            x[idx, -len(trunc):] = trunc

    return x