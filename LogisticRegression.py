import numpy as np


class LogisticRegression:

    def __init__(self, alpha = 0.001 , n_iterations = 1000):
        self.alpha = alpha
        n_iterations = n_iterations
        self.weights = None
        self.bias = None


    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(self.features)
        self.bias = 0

        for _ in range(self.n_iterations):
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(linear_pred)

            dw = (1/n_samples) * np.dot(X.T, (predictions - y))
            db = (1/n_samples) * np.sum(predictions - y)

            self.weights = self.weights - self.alpha * dw
            self.bias = self.bias - self.alpha * db


    @staticmethod
    def sigmoid(linear_function):
        val = 1 + np.exp(-linear_function)
        return 1/val
    

    def predict(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_pred)
        class_pred = [0 if y<=0.5 else 1 for y in y_pred]
        return class_pred




