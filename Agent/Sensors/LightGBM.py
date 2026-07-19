import lightgbm as lgb
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score
import numpy as np

class LightGBMAnomalyDetector(BaseEstimator, ClassifierMixin):
    def __init__(self, outlier_fraction=0.4, class_weight=None, sample_weights=None, threshold=None):
        self.outlier_fraction = outlier_fraction
        self.class_weight = class_weight
        self.sample_weights = sample_weights
        self.threshold = threshold
        self.model = lgb.LGBMClassifier(class_weight=class_weight)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict_proba(self, X_test):
        temp = self.model.predict_proba(X_test)
        self.proba = temp[:,1]
        return temp


    def predict(self, X_test):
        self.predict_proba(X_test)
        if self.threshold is None:
            threshold = np.percentile(self.proba, 100 - (self.outlier_fraction * 100))
        else:
            threshold = self.threshold
        return (self.proba > threshold).astype(int)

    def evaluate(self, X_test, y_test):
        self.y_pred = self.predict(X_test)
        # Use stored sample weights only when they match the evaluated set
        sw = self.sample_weights if self.sample_weights is not None and len(self.sample_weights) == len(y_test) else None
        self.accuracy = accuracy_score(y_test, self.y_pred, sample_weight=sw)
        self.precision = precision_score(y_test, self.y_pred, sample_weight=sw)
        self.recall = recall_score(y_test, self.y_pred, sample_weight=sw)
        self.f1 = f1_score(y_test, self.y_pred, sample_weight=sw)
        self.f2 = fbeta_score(y_test, self.y_pred, beta=2, sample_weight=sw)

        print(f"Accuracy (LGBM): {self.accuracy:.4f}")
        print(f"Precision (LGBM): {self.precision:.4f}")
        print(f"Recall (LGBM): {self.recall:.4f}")
        print(f"F1 score (LGBM): {self.f1:.4f}")
        print(f"F2 score (LGBM): {self.f2:.4f}")

        return self.accuracy, self.precision, self.recall, self.f1, self.f2


    def get_params(self, deep=True):
        return {"outlier_fraction": self.outlier_fraction}

    def set_params(self, **params):
        for parameter, value in params.items():
            setattr(self, parameter, value)
        return self