import numpy as np

class SimpleDecisionTree:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y, depth=0):
        if len(np.unique(y)) == 1 or depth == self.max_depth:
            # If all labels are the same or reached max depth, make a leaf node
            return y[0]

        # Find the best split
        best_split = self.find_best_split(X, y)
        if best_split is None:
            # If no split improves purity, make a leaf node
            return y[0]

        feature_index, threshold = best_split
        left_indices = X[:, feature_index] < threshold
        right_indices = ~left_indices

        # Recursively build left and right subtrees
        left_tree = self.fit(X[left_indices], y[left_indices], depth + 1)
        right_tree = self.fit(X[right_indices], y[right_indices], depth + 1)

        # Store the root node
        self.root = (feature_index, threshold, left_tree, right_tree)

        # Return a node representing the split
        return self.root


    def find_best_split(self, X, y):
        best_gini = 1
        best_split = None
        for feature_index in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_indices = X[:, feature_index] < threshold
                right_indices = ~left_indices

                gini = self.calculate_gini(y[left_indices], y[right_indices])
                if gini < best_gini:
                    best_gini = gini
                    best_split = (feature_index, threshold)
        return best_split

    def calculate_gini(self, left_labels, right_labels):
        n = len(left_labels) + len(right_labels)
        gini_left = 1.0 - sum((np.sum(left_labels == c) / len(left_labels))**2 for c in np.unique(left_labels))
        gini_right = 1.0 - sum((np.sum(right_labels == c) / len(right_labels))**2 for c in np.unique(right_labels))
        gini = (len(left_labels) / n) * gini_left + (len(right_labels) / n) * gini_right
        return gini

    def predict(self, X):
        return np.array([self.predict_tree(x, self.root) for x in X])

    def predict_tree(self, x, tree):
        if not isinstance(tree, tuple):
            # If tree is a leaf node, return its value
            return tree

        feature_index, threshold, left_tree, right_tree = tree
        if x[feature_index] < threshold:
            return self.predict_tree(x, left_tree)
        else:
            return self.predict_tree(x, right_tree)


class SimpleDecisionForest:
    def __init__(self, num_trees=10, max_depth=5):
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        for _ in range(self.num_trees):
            indices = np.random.choice(len(X), len(X))
            X_sample, y_sample = X[indices], y[indices]
            tree = SimpleDecisionTree(max_depth=self.max_depth)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        final_prediction = np.apply_along_axis(self.majority_vote, axis=0, arr=predictions)
        return final_prediction

    def majority_vote(self, votes):
        vote_counts = {}
        for vote in votes:
            if vote in vote_counts:
                vote_counts[vote] += 1
            else:
                vote_counts[vote] = 1
        return max(vote_counts, key=vote_counts.get)
