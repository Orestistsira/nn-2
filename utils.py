import numpy as np
from matplotlib import pyplot as plt


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


def show_image(x, y, prediction):
    if y == -1:
        y = 0
    if prediction == -1:
        prediction = 0

    classes = ["airplane", "automobile"]
    plt.figure()
    im_r = x[0:1024].reshape(32, 32)
    im_g = x[1024:2048].reshape(32, 32)
    im_b = x[2048:].reshape(32, 32)

    img = np.dstack((im_r, im_g, im_b))
    plt.imshow(img)
    plt.title(f"Label: {classes[y]} Prediction: {classes[prediction]}")
    plt.axis('off')
    plt.show()
