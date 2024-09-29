from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score
import numpy as np

class CatBoostAnomalyDetector(BaseEstimator, ClassifierMixin):
    def __init__(self, outlier_fraction=0.4, class_weight=None, sample_weights=None, threshold=None):
        self.outlier_fraction = outlier_fraction
        self.class_weight = class_weight
        self.sample_weights = sample_weights
        self.threshold = threshold
        self.model = CatBoostClassifier(verbose=0, class_weights=class_weight)

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
        
    def get_params(self, deep=True):
        return {"outlier_fraction": self.outlier_fraction}

    def set_params(self, **params):
        for parameter, value in params.items():
            setattr(self, parameter, value)
        return self