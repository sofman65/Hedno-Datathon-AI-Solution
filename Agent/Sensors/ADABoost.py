from sklearn.ensemble import AdaBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score
import numpy as np

class AdaBoostAnomalyDetector(BaseEstimator, ClassifierMixin):
    def __init__(self, outlier_fraction=0.4, class_weight=None, sample_weights=None, threshold=None):
        self.outlier_fraction = outlier_fraction
        self.class_weight = class_weight
        self.sample_weights = sample_weights
        self.threshold = threshold
        self.model = AdaBoostClassifier()

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X_test):
        temp = self.model.predict_proba(X_test)
        self.proba = temp[:,1]
        return temp

    def predict(self, X):
        self.predict_proba(X)
        if self.threshold is None:
            threshold = np.percentile(self.proba, 100 - (self.outlier_fraction * 100))
        else:
            threshold = self.threshold
        return (self.proba > threshold).astype(int)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        self.accuracy = accuracy_score(y, y_pred, sample_weight=self.sample_weights)
        self.precision = precision_score(y, y_pred, sample_weight=self.sample_weights)
        self.recall = recall_score(y, y_pred, sample_weight=self.sample_weights)
        self.f1 = f1_score(y, y_pred, sample_weight=self.sample_weights)
        self.f2 = fbeta_score(y, y_pred, beta=2, sample_weight=self.sample_weights)

        print(f"Accuracy (AdaBoost): {self.accuracy:.4f}")
        print(f"Precision (AdaBoost): {self.precision:.4f}")
        print(f"Recall (AdaBoost): {self.recall:.4f}")
        print(f"F1 score (AdaBoost): {self.f1:.4f}")
        print(f"F2 score (AdaBoost): {self.f2:.4f}")

        return self.accuracy, self.precision, self.recall, self.f1, self.f2
        
    def get_params(self, deep=True):
        return {"outlier_fraction": self.outlier_fraction}

    def set_params(self, **params):
        for parameter, value in params.items():
            setattr(self, parameter, value)
        return self
