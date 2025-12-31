"""
Base classes for supervised learning models.

Supervised learning: Models that learn from labeled data (X, y) to predict outcomes.
Examples: predicting house prices (regression), classifying emails as spam (classification)
"""

from abc import abstractmethod
from ..base import BaseModel


class SupervisedModel(BaseModel):
    """
    Base class for all supervised learning models.
    
    Supervised models learn a mapping from input features X to output labels y.
    """
    
    @abstractmethod
    def fit(self, X, y):
        """
        Fit the model to labeled training data.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training features
        y : array-like, shape (n_samples,)
            Target values (labels)
            
        Returns
        -------
        self : object
            Fitted model
        """
        pass
    
    @abstractmethod
    def predict(self, X):
        """
        Make predictions on new data.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Features to predict on
            
        Returns
        -------
        predictions : array-like, shape (n_samples,)
            Predicted values
        """
        pass

    @abstractmethod
    def score(self, X, y):
        """
        Calculate the score of the model.
        """
        pass


class RegressionModel(SupervisedModel):
    """
    Base class for regression models.
    
    Regression: Predicting continuous numerical values.
    Examples: house prices, temperature, stock prices, sales forecasts
    
    Notes
    -----
    Concrete implementations should provide their own evaluation metrics
    appropriate to their library (sklearn, from-scratch, etc.).
    
    Common metrics:
    - R² score (coefficient of determination)
    - Mean Squared Error (MSE)
    - Root Mean Squared Error (RMSE)
    - Mean Absolute Error (MAE)
    """

    @abstractmethod
    def mean_squared_error(self, X, y):
        """
        Calculate the mean squared error of the model.
        """
        pass

    @abstractmethod
    def root_mean_squared_error(self, X, y):
        """
        Calculate the root mean squared error of the model.
        """
        pass

    @abstractmethod
    def mean_absolute_error(self, X, y):
        """
        Calculate the mean absolute error of the model.
        """
        pass


class ClassificationModel(SupervisedModel):
    """
    Base class for classification models.
    
    Classification: Predicting discrete categories/labels.
    Examples: spam detection, image recognition, disease diagnosis, sentiment analysis
    
    Notes
    -----
    Concrete implementations should provide:
    - predict_proba(X): Return class probabilities
    - Evaluation metrics: accuracy, confusion matrix, precision, recall, f1-score
    """
    
    @abstractmethod
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Features to predict on
            
        Returns
        -------
        probabilities : array-like, shape (n_samples, n_classes)
            Probability estimates for each class
        """
        pass

    @abstractmethod
    def confusion_matrix(self, X, y):
        """Calculate confusion matrix"""
        pass
    
    @abstractmethod
    def classification_report(self, X, y, output_dict=False):
        """Generate classification report"""
        pass