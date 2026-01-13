"""
Classification models from scratch.

All classes prefixed with 'Scratch' to indicate from scratch implementation.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    classification_report
)
from .base import ClassificationModel

class ScratchKNNClassifier(ClassificationModel):
    """
    K-Nearest Neighbors Classifier from scratch.
    
    Distance-based classifier that predicts class based on majority vote
    of k nearest neighbors.

    KNN is a simple, instance-based learning algorithm:
    - Non-parametric (no training phase, just stores data)
    - Predicts based on k closest neighbors
    - Distance-based (requires feature scaling!)
    
    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to use
    weights : {'uniform', 'distance'}, default='uniform'
        Weight function:
        - 'uniform': all neighbors weighted equally
        - 'distance': closer neighbors have more influence
    metric : str, default='euclidean'
        Distance metric ('euclidean', 'manhattan', 'minkowski')
    p : int, default=2
        Power parameter for the Minkowski distance
        
    Examples
    --------
    >>> from src.models.supervised import ScratchKNNClassifier
    >>> model = ScratchKNNClassifier(n_neighbors=5)
    >>> model.fit(X_train, y_train)
    >>> predictions = model.predict(X_test)
    
    Notes
    -----
    - ALWAYS scale features before using KNN (critical!)
    - Sensitive to irrelevant features (use feature selection)
    - Slow prediction for large datasets (stores all training data)
    - Good for: Small-medium datasets, multi-class problems
    - k=5 is typical default, try odd numbers to avoid ties
    """
    def __init__(self, n_neighbors=5, weights='uniform', metric='euclidean', p=2):
        super().__init__()
        self.is_fitted = False
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.p = p
        self.classes = None

    def _calculate_distance(self, x):
        """
        Calculate the distance from point x and all the training points based on the metric.

        Distance Metrics:
        - Euclidean: sqrt(sum((x_i - y_i)**2))
        - Manhattan: sum(abs(x_i - y_i))
        - Minkowski: (sum(abs(x_i - y_i)**p))^(1/p)

        Parameters
        ----------
        x : array-like, shape (n_features)
            The data point to calculate the distance from the training data.

        Returns
        -------
        distance : float
            The distance between the data point and the training data.
        """   
       
        if self.metric == 'euclidean':
            # Euclidean: L2 norm
            return np.sqrt(np.sum((x - self.X_train) ** 2, axis=1))
        elif self.metric == 'manhattan':
            # Manhattan: L1 norm
            return np.sum(np.abs(x - self.X_train), axis=1)
        elif self.metric == 'minkowski':
            # Minkowski: Generalized Lp norm
            return (np.sum(np.abs(x - self.X_train) ** self.p, axis=1)) ** (1/self.p)
        else:
            raise ValueError(f"Invalid metric: {self.metric}")

    def fit(self, X, y):
        """Fit the model to the data"""
        self.X_train = np.array(X)
        if isinstance(y, pd.Series):
            self.y_train = y.values
        else:
            self.y_train = np.array(y)
        self.classes = np.unique(y)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Predict the class of the data"""
        probabilities = self.predict_proba(X)
        class_indices = np.argmax(probabilities, axis=1)
        predictions = self.classes[class_indices]
        return np.array(predictions)

    def predict_proba(self, X):
        """Predict the class probabilities of the data"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predicting")
            
        X_test = np.array(X)
        n_test = len(X_test)
        n_classes = len(self.classes)
        probabilities = np.zeros((n_test, n_classes))

        for i in range(n_test):
            distances = self._calculate_distance(X_test[i])
            
            closest_indices = np.argsort(distances)[:self.n_neighbors]
            closest_distances = distances[closest_indices]
            closest_labels = self.y_train[closest_indices]

            # Calculate weights based on parameter
            if self.weights == 'uniform':
                # All neighbors have equal weight
                neighbor_weights = np.ones(self.n_neighbors)
            
            elif self.weights == 'distance':
                # Weight by inverse distance
                # Add small epsilon to avoid division by zero
                neighbor_weights = 1 / (closest_distances + 1e-10)
            else:
                raise ValueError(f"weights must be 'uniform' or 'distance', got {self.weights}")

            for j, cls in enumerate(self.classes):
                # Get weights for neighbors of this class
                mask = (closest_labels == cls)
                class_weight = np.sum(neighbor_weights[mask])
                
                # Normalize by total weight
                total_weight = np.sum(neighbor_weights)
                probabilities[i, j] = class_weight / total_weight

        return probabilities

    def score(self, X, y):
        """Score the model"""
        return np.mean(self.predict(X) == y)

    def confusion_matrix(self, X, y):
        """Calculate the confusion matrix"""
        return confusion_matrix(y, self.predict(X))

    def classification_report(self, X, y):
        """Generate the classification report"""
        return classification_report(y, self.predict(X))