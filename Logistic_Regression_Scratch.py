import numpy as np

class CustomLogisticRegression:
    def __init__(self, C=1, learning_rate=.01, max_train_iterations=50) -> None:
        self.C = C # Regularization parameter
        self.learning_rate = learning_rate # Learning Rate
        self.weights = None # Weights
        self.max_train_iterations = max_train_iterations # Number of times to train

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        n_samples = X.shape[0]
        y = np.where(y == -1, 0, 1)

        # Iterate over number of training iterations
        for _ in range(self.max_train_iterations):
            # Iterate over training example
            for xt, yt in zip(X, y):
                # Predict the label
                yt_pred = self.sigmoid(np.dot(self.weights,xt))

                # Find gradient of log likelihood
                gradient = np.dot(xt.T, yt - yt_pred)

                # Regularization term (L2)
                regularization = (1 / n_samples) * self.weights

                # Update Weights
                self.weights = self.weights + self.learning_rate * (gradient + self.C * regularization)

    def predict(self, X):
        decision_values = self.sigmoid(np.dot(X, self.weights))
        return np.where(decision_values >= 0.5, 1, -1)