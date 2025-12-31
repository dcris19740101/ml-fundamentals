"""
Scikit-learn specific base classes for regression and classification.

Provides common sklearn model interface (fit, predict, score) to eliminate
code duplication across concrete sklearn implementations.
"""

import numpy as np
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    confusion_matrix,
    classification_report
)
from .base import RegressionModel, ClassificationModel


class SKLearnRegressionBase(RegressionModel):
    """
    Base class for all scikit-learn regression models.
    """
    def __init__(self):
        super().__init__()
        self._model = None # Must be set by subclass

    def fit(self, X, y):
        """Fit the model to the data"""
        self._model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X):
        """Predict the target values for the input data"""
        self._check_is_fitted()
        return self._model.predict(X)

    # Evaluation metrics using sklearn
    def score(self, X, y):
        """Calculate R² score (coefficient of determination)"""
        self._check_is_fitted()
        y_pred = self.predict(X)
        return r2_score(y, y_pred)
    
    def mean_squared_error(self, X, y):
        """Calculate Mean Squared Error"""
        self._check_is_fitted()
        y_pred = self.predict(X)
        return mean_squared_error(y, y_pred)
    
    def root_mean_squared_error(self, X, y):
        """Calculate Root Mean Squared Error"""
        self._check_is_fitted()
        y_pred = self.predict(X)
        return mean_squared_error(y, y_pred, squared=False)
    
    def mean_absolute_error(self, X, y):
        """Calculate Mean Absolute Error"""
        self._check_is_fitted()
        y_pred = self.predict(X)
        return mean_absolute_error(y, y_pred)

    
        

class SKLearnClassificationBase(ClassificationModel):
    """
    Base class for all scikit-learn classification models.
    """
    def __init__(self):
        super().__init__()
        self._model = None
        self.is_fitted = False

    def fit(self, X, y):
        """Fit the model to the data"""
        self._model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X):
        """Predict class labels"""
        self._check_is_fitted()
        return self._model.predict(X)
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        self._check_is_fitted()
        return self._model.predict_proba(X)

    # Evaluation metrics using sklearn
    def score(self, X, y):
        """Calculate accuracy"""
        self._check_is_fitted()
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def confusion_matrix(self, X, y):
        """Calculate confusion matrix"""
        self._check_is_fitted()
        y_pred = self.predict(X)
        return confusion_matrix(y, y_pred)
    
    def classification_report(self, X, y, output_dict=False):
        """Generate classification report"""
        self._check_is_fitted()
        y_pred = self.predict(X)
        return classification_report(y, y_pred, output_dict=output_dict)
        
        