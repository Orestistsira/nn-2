import time

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

import utils

# Load CIFAR-10 dataset
x_train_1, y_train_1 = utils.unpickle("cifar-10/data_batch_1")
x_train_2, y_train_2 = utils.unpickle("cifar-10/data_batch_2")
x_train_3, y_train_3 = utils.unpickle("cifar-10/data_batch_3")

x = np.concatenate([x_train_1, x_train_2, x_train_3])
y = np.concatenate([y_train_1, y_train_2, y_train_3])

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)

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
