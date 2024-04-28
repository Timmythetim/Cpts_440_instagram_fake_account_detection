import numpy as np
from cvxopt import matrix, solvers
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
class test:
    def __init__(self, C=0.01, gamma=0.1):
        self.C = C
        self.gamma = gamma
        self.support_vectors = None
        self.support_vector_labels = None
        self.weights = None
        self.bias = None
        self.scaler = StandardScaler()

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Normalize the input features
        X = self.scaler.fit_transform(X)
        # Compute the kernel matrix using matrix operations
        K = np.exp(-self.gamma * (np.sum(X**2, axis=1)[:, np.newaxis] + np.sum(X**2, axis=1)[np.newaxis, :] - 2 * np.dot(X, X.T)))
        K += 1e-6 * np.eye(n_samples)  # Add a small regularization term

#correct from above

        P = matrix(K)
        q = matrix(-np.ones((n_samples, 1)))
        G = matrix(np.vstack((-np.eye(n_samples), np.eye(n_samples))))
        h = matrix(np.hstack((np.zeros(n_samples), np.ones(n_samples) * self.C)))
        A = matrix(y.reshape(1, -1).astype(np.double))
        b = matrix(0.0)

        # Use the CVXOPT solver
        solution = solvers.qp(P, q, G, h, A, b)

        print("Solver status:", solution['status'])
        print("Solver iterations:", solution['iterations'])

        alphas = np.array(solution['x']).flatten()

        print(alphas.shape)
       # print("Number of support vectors:", np.sum(alphas > 1e-8))
       # print("Alphas min value:", np.min(alphas))
       # print("Alphas max value:", np.max(alphas))
        support_vector_indices = np.where(alphas)[0]
        self.support_vectors = X[support_vector_indices]
        self.support_vector_labels = y[support_vector_indices]
       # x1=np.squeeze(np.asarray(alphas[support_vector_indices] * self.support_vector_labels))
        #x2=np.squeeze(np.asarray(self.support_vectors))
        self.weights = np.dot(alphas[support_vector_indices] * self.support_vector_labels, self.support_vectors)
        #print(self.weights)
        # Weight is correct
        support_vector_index = support_vector_indices[0]
        self.bias = self.support_vector_labels[0] - np.dot(self.weights, self.support_vectors[support_vector_indices].T)
        print("Bias term:", self.bias)
        print((n_samples))

    def predict(self, X):
        # Validate input data shape
        if self.support_vectors is None or self.weights is None:
            raise ValueError("Support vectors or weights are not set.")
        
        # Kernel Calculation
        # Ensure the shape aligns: (num_samples, num_support_vectors)
        try:
            K = np.exp(-self.gamma * np.sum(np.square(X[:, np.newaxis] - self.support_vectors), axis=2))
                  
        except Exception as e:
            raise ValueError(f"Kernel calculation error: {e}")
        
        # Dot Product with Weights and Add Bias
        try:
            # Ensure that the shapes align for dot product
            print("K shape:",K.shape)
            print("X shape",X.shape)
            print("weights.shape",self.weights.shape)
            print("support vectors shape",self.support_vectors.shape)
            
            if K.shape[1] != len(self.weights):
                raise ValueError(f"Kernel matrix shape {K.shape} doesn't match with weights shape {len(self.weights)}")
            
            decision_values = np.dot(K, self.weights) + self.bias
        except Exception as e:
            raise ValueError(f"Dot product or bias addition error: {e}")

        # Sign Function for Predictions
        try:
            predictions = np.sign(decision_values)
        except Exception as e:
            raise ValueError(f"Error during predictions: {e}")
        
        return predictions