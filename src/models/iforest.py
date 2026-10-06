from sklearn.ensemble import IsolationForest

class AnomalyIsolationForest:
    def __init__(self, n_estimators=200, random_state=None):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            random_state=random_state
        )
    
    def fit(self, X):
        """
        Fit the model on normal data.
        X: numpy array of shape (n_samples, n_features)
        """
        self.model.fit(X)
        return self
        
    def predict_score(self, X):
        """
        Predict anomaly score. score = -score_samples
        Higher score means more anomalous.
        X: numpy array of shape (n_samples, n_features)
        """
        # score_samples returns opposite of anomaly score (lower is more anomalous)
        # So we negate it to have higher score = more anomalous
        scores = -self.model.score_samples(X)
        return scores
