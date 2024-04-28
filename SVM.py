import numpy as np
from cvxopt import matrix, solvers

class CustomSVM:
    def __init__(self, C=1.0):
        self.C = C
        self.support_vectors = None
        self.support_vector_labels = None
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = np.dot(X[i], X[j])

        P = matrix(K)
        q = matrix(-np.ones((n_samples, 1)))
        G = matrix(np.vstack((-np.eye(n_samples), np.eye(n_samples))))
        h = matrix(np.hstack((np.zeros(n_samples), np.ones(n_samples) * self.C)))
        A = matrix(y.reshape(1, -1).astype(np.double))
        b = matrix(0.0)

        solvers.options['show_progress'] = False
        solution = solvers.qp(P, q, G, h, A, b)
        alphas = np.array(solution['x']).flatten()

        threshold = 1e-12
        support_vector_indices = np.where(alphas > threshold)[0]
        self.support_vectors = X[support_vector_indices]
        self.support_vector_labels = y[support_vector_indices]
        
        # Corrected calculation of weights
        self.weights = np.zeros(n_features)
        for i in range(len(support_vector_indices)):
            self.weights += alphas[support_vector_indices[i]] * self.support_vector_labels[i] * self.support_vectors[i]

        support_vector_index = support_vector_indices[0]
        self.bias = self.support_vector_labels[0] - np.dot(self.weights, self.support_vectors[0])

    def predict(self, X):
        decision_values = np.dot(X, self.weights) + self.bias
        return np.sign(decision_values)