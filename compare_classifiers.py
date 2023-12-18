import numpy as np
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.metrics import accuracy_score

import utils

# Load cifar-10 dataset
x_train_1, y_train_1 = utils.unpickle("cifar-10/data_batch_1")
x_train_2, y_train_2 = utils.unpickle("cifar-10/data_batch_2")
x_train_3, y_train_3 = utils.unpickle("cifar-10/data_batch_3")
x_train_4, y_train_4 = utils.unpickle("cifar-10/data_batch_4")
x_train_5, y_train_5 = utils.unpickle("cifar-10/data_batch_5")

x_train = np.concatenate([x_train_1, x_train_2, x_train_3, x_train_4, x_train_5])
y_train = np.concatenate([y_train_1, y_train_2, y_train_3, y_train_4, y_train_5])

x_test, y_test = utils.unpickle("cifar-10/test_batch")

# Extract a subset of the dataset for simplicity
x_train, y_train = utils.filter_samples(x_train, y_train)
x_test, y_test = utils.filter_samples(x_test, y_test)

# Define and train 1-NN classifier
knn_1 = KNeighborsClassifier(n_neighbors=1)
knn_1.fit(x_train, y_train)

# Predict with 1-NN
y_pred_knn_1 = knn_1.predict(x_test)

# Calculate accuracy for 1-NN
accuracy_knn_1 = accuracy_score(y_test, y_pred_knn_1)
print("1-NN Classifier Accuracy:", accuracy_knn_1)

# Define and train 3-NN classifier
knn_3 = KNeighborsClassifier(n_neighbors=3)
knn_3.fit(x_train, y_train)

# Predict with 3-NN
y_pred_knn_3 = knn_3.predict(x_test)

# Calculate accuracy for 3-NN
accuracy_knn_3 = accuracy_score(y_test, y_pred_knn_3)
print("3-NN Classifier Accuracy:", accuracy_knn_3)

# Define and train the NearestCentroid classifier
nearest_centroid = NearestCentroid()
nearest_centroid.fit(x_train, y_train)

# Predict with the NearestCentroid classifier
y_pred_nearest_centroid = nearest_centroid.predict(x_test)

# Calculate accuracy for the NearestCentroid classifier
accuracy_nearest_centroid = accuracy_score(y_test, y_pred_nearest_centroid)
print("NearestCentroid Classifier Accuracy:", accuracy_nearest_centroid)
