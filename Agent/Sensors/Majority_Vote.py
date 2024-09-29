import numpy as np
from scipy.stats import mode
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier


class MajorityVoteAnomalyDetector(BaseEstimator, ClassifierMixin):
    def __init__(self, classifiers):
        self.classifiers = classifiers

    def fit(self, X, y):
        for classifier in self.classifiers:
            classifier.fit(X, y)
        return self

    def predict_proba(self, X_test):
        proba_predictions = np.array([classifier.predict_proba(X_test) for classifier in self.classifiers])
        proba_predictions_1d = np.array([proba[:, 1] if proba.ndim == 2 else proba for proba in proba_predictions])
        self.proba = np.mean(proba_predictions_1d, axis=0)
        return np.column_stack((1 - self.proba, self.proba))

    def predict(self, X_test):
        predictions = np.array([classifier.predict(X_test) for classifier in self.classifiers])
        return mode(predictions, axis=0)[0].flatten()

    def evaluate(self, X_test, y_test):
        self.y_pred = self.predict(X_test)
        self.accuracy = accuracy_score(y_test, self.y_pred)
        self.precision = precision_score(y_test, self.y_pred)
        self.recall = recall_score(y_test, self.y_pred)
        self.f1 = f1_score(y_test, self.y_pred)
        self.f2 = fbeta_score(y_test, self.y_pred, beta=2)

        print(f"Accuracy (Ensemble): {self.accuracy:.4f}")
        print(f"Precision (Ensemble): {self.precision:.4f}")
        print(f"Recall (Ensemble): {self.recall:.4f}")
        print(f"F1 score (Ensemble): {self.f1:.4f}")
        print(f"F2 score (Ensemble): {self.f2:.4f}")

        return self.accuracy, self.precision, self.recall, self.f1, self.f2
