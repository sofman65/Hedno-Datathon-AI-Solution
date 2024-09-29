import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score
from sklearn.ensemble import StackingClassifier

class StackingAnomalyDetector:
    def __init__(self, base_classifiers, meta_classifier):
        self.stacking_classifier = StackingClassifier(
            estimators=[(f"clf{i}", clf) for i, clf in enumerate(base_classifiers)],
            final_estimator=meta_classifier,
            stack_method='predict_proba',
            n_jobs=-1
        )

    def fit(self, X_train, y_train):
        self.stacking_classifier.fit(X_train, y_train)

    def predict_proba(self, X_test):
        self.proba = self.stacking_classifier.predict_proba(X_test)[:, 1]
        return self.proba

    def predict(self, X_test):
        return self.stacking_classifier.predict(X_test)

    def evaluate(self, X_test, y_test):
        self.y_pred = self.predict(X_test)
        self.accuracy = accuracy_score(y_test, self.y_pred)
        self.precision = precision_score(y_test, self.y_pred)
        self.recall = recall_score(y_test, self.y_pred)
        self.f1 = f1_score(y_test, self.y_pred)
        self.f2 = fbeta_score(y_test, self.y_pred, beta=2)

        print(f"Accuracy (Stacking): {self.accuracy:.4f}")
        print(f"Precision (Stacking): {self.precision:.4f}")
        print(f"Recall (Stacking): {self.recall:.4f}")
        print(f"F1 score (Stacking): {self.f1:.4f}")
        print(f"F2 score (Stacking): {self.f2:.4f}")

        return self.accuracy, self.precision, self.recall, self.f1, self.f2