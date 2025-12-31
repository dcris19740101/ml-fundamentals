"""
Base classes for unsupervised learning models.

Unsupervised learning: Models that discover patterns in unlabeled data (X only, no y).
Examples: customer segmentation (clustering), dimensionality reduction (PCA)
"""

from abc import abstractmethod
from ..base import BaseModel


class UnsupervisedModel(BaseModel):
    """
    Base class for all unsupervised learning models.
    
    Unsupervised models learn patterns and structure from data without labels.
    They discover hidden patterns, groupings, or representations.
    """
    
    @abstractmethod
    def fit(self, X):
        """
        Fit the model to unlabeled data.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data (no labels)
            
        Returns
        -------
        self : object
            Fitted model
        """
        pass


class ClusteringModel(UnsupervisedModel):
    """
    Base class for clustering models.
    
    Clustering: Grouping similar data points together without predefined labels.
    Examples: customer segmentation, document clustering, image segmentation
    
    Notes
    -----
    Clustering models should implement:
    - fit_predict(X): Fit model and return cluster labels
    - predict(X): Predict clusters for new data (if applicable)
    
    Not all clustering algorithms support prediction on new data.
    """
    
    @abstractmethod
    def fit_predict(self, X):
        """
        Fit model and return cluster labels.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data to cluster
            
        Returns
        -------
        labels : array, shape (n_samples,)
            Cluster label for each sample
        """
        pass
    
    def predict(self, X):
        """
        Predict cluster labels for new data.
        
        Note: Not all clustering algorithms support this operation.
        Override in subclass if the algorithm supports prediction on new data.
        
        Raises
        ------
        NotImplementedError
            If the algorithm doesn't support prediction on new data
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} doesn't support prediction on new data. "
            "Use fit_predict() instead."
        )