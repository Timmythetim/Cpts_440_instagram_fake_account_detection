import numpy as np

class CustomLogisticRegression:
    def __init__(self, C=1, max_train_iterations=50) -> None:
        self.C = C
        self.weights = None
        self.max_train_iterations = max_train_iterations

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])

        # Iterate over number of training iterations
        for _ in range(self.max_train_iterations):
            # Iterate over training example
            for xt, yt in zip(X, y):
                # Predict the label
                yt_pred = np.sign(np.dot(self.weights,xt))
                if yt_pred != yt:
                    self.weights = self.weights + self.C * yt * xt

    def predict(self, X):
        decision_values = np.dot(X, self.weights)
        return np.where(decision_values > 0.5, 1, 0)