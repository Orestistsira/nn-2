import numpy as np


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')

    x = np.array(data[b'data'])
    y = np.array(data[b'labels'])

    x = x.astype('float32')  # this is necessary for the division below
    x /= 255

    return x, y


def filter_samples(X, y, classes=(0, 1)):
    mask = np.isin(y, classes)
    X_filtered = X[mask]
    y_filtered = y[mask]
    return X_filtered, y_filtered
