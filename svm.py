import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


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


class MySVC:
    def __init__(self, C=1.0):
        self.C = C
        self.W = None
        self.b = None

    def fit(self, x, y):
        n_samples, n_features = x.shape

        # Convert labels to {-1, 1}
        y = np.where(y == 0, -1, 1)

        # Initialize weights and bias
        self.W = np.zeros(n_features)
        self.b = 0.0

        # Gradient descent optimization
        learning_rate = 0.01
        epochs = 1000

        for epoch in range(epochs):
            for i in range(n_samples):
                if y[i] * (np.dot(x[i], self.W) - self.b) >= 1:
                    self.W -= learning_rate * (2 * self.C * self.W)
                else:
                    self.W -= learning_rate * (2 * self.C * self.W - np.dot(x[i], y[i]))
                    self.b -= learning_rate * y[i]

    def predict(self, x):
        return np.sign(np.dot(x, self.W) - self.b)


# Load CIFAR-10 dataset
x, y = unpickle("cifar-10/data_batch_1")

# Extract a subset of the dataset for simplicity
x, y = filter_samples(x, y)
x, y = x[:1000], y[:1000]

print(x.shape)

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)

# Preprocess the data by scaling features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Create an SVM model
svm_model = MySVC(C=1.0)

# Train the SVM model
svm_model.fit(x_train, y_train)

# Make predictions on the test set
y_pred = svm_model.predict(x_test)

# Evaluate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}%')
