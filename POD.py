from path import *
import numpy as np
import matplotlib.pyplot as plt

def pod(data, label):
    # centering
    print("resizing data...")
    data_train = pad_sequences(data)
    print("data shape: ", data_train.shape)
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

    # Plot the distribution of the eigenvalues
    print("plotting eigenvalues...")
    plt.scatter(np.arange(64), s ** 2)
    plt.show()

    # Plot the distribution of the POD coefficients

    im = plt.scatter(A[:, 0], A[:, 1], c=label, alpha=0.6)
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
