import numpy as np
from sklearn.metrics import accuracy_score


class MySVC:
    def __init__(self, kernel='linear', C=1.0, degree=2, gamma=1.0, coef0=0.0):
        self.kernel = kernel
        self.C = C
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.W = None
        self.b = None

    def _linear_kernel(self, x):
        return np.dot(x, self.W) - self.b

    def _poly_kernel(self, x):
        return (self.gamma * np.dot(x, self.W) - self.b + self.coef0) ** self.degree

    def _rbf_kernel(self, x):
        return np.exp(-self.gamma * np.linalg.norm(x - self.W) ** 2) - self.b

    def _kernel_func(self, x):
        if self.kernel == 'linear':
            return self._linear_kernel(x)
        elif self.kernel == 'poly':
            return self._poly_kernel(x)
        elif self.kernel == 'rbf':
            return self._rbf_kernel(x)
        else:
            raise ValueError("Invalid kernel type. Supported types are 'linear', 'poly' and 'rbf'.")

    def fit(self, x, y, epochs=1000, learning_rate=0.01):
        n_samples, n_features = x.shape

        # Initialize weights and bias
        self.W = np.zeros(n_features)
        self.b = 0.0

        print('Training...')
        for epoch in range(epochs):
            # for i in range(n_samples):
            #     if y[i] * self._kernel_func(x[i]) >= 1:
            #         self.W -= learning_rate * (2 * self.C * self.W)
            #     else:
            #         self.W -= learning_rate * (2 * self.C * self.W - np.dot(x[i], y[i]))
            #         self.b -= learning_rate * y[i]

            scores = self._kernel_func(x)
            margins = y * scores

            # Update weights and bias together
            mask = margins < 1
            self.W -= learning_rate * (2 * self.C * self.W - np.dot(x[mask].T, y[mask]))
            self.b -= learning_rate * np.sum(mask * y)

        y_pred = self.predict(x)
        # Evaluate the training accuracy
        accuracy = accuracy_score(y, y_pred)
        print(f'Train Accuracy: {accuracy:.2f}')

    def predict(self, x):
        return np.sign(np.dot(x, self.W) - self.b)
