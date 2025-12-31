"""
Generic base class for all machine learning models.

This module contains only the most fundamental BaseModel class.
Specific base classes for supervised/unsupervised learning are in their respective submodules.
"""

from abc import ABC, abstractmethod


class BaseModel(ABC):
    """
    Abstract base class for all machine learning models.
    
    Provides common functionality for model state management.
    All concrete models (supervised/unsupervised) should inherit from this.
    
    Attributes
    ----------
    is_fitted : bool
        Whether the model has been fitted to training data
    """
    
    def __init__(self):
        self.is_fitted = False
    
    def _check_is_fitted(self):
        """
        Check if model has been fitted.
        
        Raises
        ------
        RuntimeError
            If model hasn't been fitted yet
        """
        if not self.is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} must be fitted before making predictions. "
                "Call fit() first."
            )


# ============================================================================
# Dimensionality Reduction (NEW - can be supervised or unsupervised)
# ============================================================================

class DimensionalityReductionModel(BaseModel):
    """
    Base class for dimensionality reduction models.
    
    Dimensionality reduction: Reduce number of features while preserving information.
    
    Can be:
    - Unsupervised: PCA, t-SNE, Autoencoders (don't use labels)
    - Supervised: LDA (uses labels for better class separation)
    
    Common uses:
    - Visualization (reduce to 2D or 3D)
    - Remove noise and redundant features
    - Speed up training (fewer features)
    - Avoid curse of dimensionality
    
    Notes
    -----
    Dimensionality reduction models should implement:
    - fit(X, y=None): Learn the transformation
    - transform(X): Apply the transformation
    - fit_transform(X, y=None): Fit and transform in one step
    - inverse_transform(X): Reconstruct original features (if applicable)
    
    Unlike clustering or classification, dimensionality reduction models
    don't have a predict() method - they transform data.
    """
    
    @abstractmethod
    def fit(self, X, y=None):
        """
        Learn the dimensionality reduction transformation.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,), optional
            Target values (required for supervised methods like LDA)
            
        Returns
        -------
        self : DimensionalityReductionModel
            Fitted model
        """
        pass
    
    @abstractmethod
    def transform(self, X):
        """
        Apply dimensionality reduction transformation.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data to transform
            
        Returns
        -------
        X_transformed : array, shape (n_samples, n_components)
            Transformed data with reduced dimensions
        """
        pass
    
    def fit_transform(self, X, y=None):
        """
        Fit model and transform data in one step.
        
        This is a convenience method equivalent to calling fit() then transform().
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data to fit and transform
        y : array-like, optional
            Target values (required for supervised methods like LDA)
            
        Returns
        -------
        X_transformed : array, shape (n_samples, n_components)
            Transformed data
        """
        self.fit(X, y)
        return self.transform(X)