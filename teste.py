import numpy as np

class DiscreteNaiveBayes:
    def __init__(self):
        self.classes = None
        self.class_prior = None
        self.feature_probs = None

    def fit(self, X, y):
        # X is the training data (features), y is the target variable (labels)
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        # Initialize class prior and feature probability tables
        self.class_prior = np.zeros(n_classes, dtype=np.float64)
        self.feature_probs = [{} for _ in range(n_features)]

        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.class_prior[idx] = X_c.shape[0] / float(n_samples)

            # Calculate probabilities for each feature given the class
            for feature_idx in range(n_features):
                feature_vals, counts = np.unique(X_c[:, feature_idx], return_counts=True)
                feature_prob = {val: count / X_c.shape[0] for val, count in zip(feature_vals, counts)}
                self.feature_probs[feature_idx][c] = feature_prob

    def _predict(self, x):
        posteriors = []

        for idx, c in enumerate(self.classes):
            prior = np.log(self.class_prior[idx])
            likelihood = 0
            for feature_idx, feature_val in enumerate(x):
                # Get the feature probability given the class
                if feature_val in self.feature_probs[feature_idx][c]:
                    likelihood += np.log(self.feature_probs[feature_idx][c][feature_val])
                else:
                    # Handle unseen features (e.g., with Laplace smoothing, assume very small probability)
                    likelihood += np.log(1e-6)
            posterior = prior + likelihood
            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]

    def predict(self, X):
        # Predict class labels for samples in X
        return np.array([self._predict(x) for x in X])

# Example usage
if __name__ == "__main__":
    # Example dataset (discretized/categorical features)
    X = np.array([[1, 1], [1, 2], [2, 1], [2, 2], [2, 2], [1, 1]])
    y = np.array([0, 0, 1, 1, 1, 0])

    model = DiscreteNaiveBayes()
    model.fit(X, y)
    
    # Predictions
    predictions = model.predict(np.array([[1, 2], [2, 1]]))
    print(predictions)
