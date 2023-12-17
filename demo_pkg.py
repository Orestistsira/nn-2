import time

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

import utils

# Load CIFAR-10 dataset
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

# Preprocess the data by scaling features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Create an SVM model
svm_model = SVC(kernel='linear', C=1.0)
# svm_model = SVC(kernel='poly', degree=2, coef0=1.0)
# svm_model = SVC(kernel='rbf')

# Train the SVM model
start_time = time.time()
print("Training...")
svm_model.fit(x_train, y_train)
print('Model successfully trained in %.2fs' % (time.time() - start_time))

# Make predictions on the test set
y_pred = svm_model.predict(x_test)

# Evaluate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}%')
