"""
Regression models using scikit-learn.

All classes prefixed with 'SKLearn' to indicate library dependency.
Phase 1: Professional implementations using industry-standard sklearn.
Phase 2 (January 2026): Will add ScratchLinearRegression, ScratchRidgeRegression, etc.
"""

import numpy as np
from sklearn.linear_model import (
    LinearRegression as SklearnLR,
    Ridge as SklearnRidge,
    Lasso as SklearnLasso
)
from sklearn.tree import DecisionTreeRegressor as SklearnDTR
from sklearn.ensemble import RandomForestRegressor as SklearnRFR

from .sklearn_base import SKLearnRegressionBase


# ============================================================================
# LINEAR REGRESSION
# ============================================================================

class SKLearnLinearRegression(SKLearnRegressionBase):
    """
    Linear Regression using scikit-learn.
    
    Fits a linear model: y = X @ coef + intercept
    Uses Ordinary Least Squares (minimizes sum of squared residuals).
    
    Fast, interpretable, works well for linearly separable data.
    Assumes linear relationship between features and target.
    
    Attributes
    ----------
    coef_ : ndarray
        Coefficients (weights) for each feature
    intercept_ : float
        Intercept (bias) term
    
    Examples
    --------
    >>> from src.models.supervised import SKLearnLinearRegression
    >>> model = SKLearnLinearRegression()
    >>> model.fit(X_train, y_train)
    >>> y_pred = model.predict(X_test)
    >>> print(f"R² = {model.score(X_test, y_test):.4f}")
    """
    
    def __init__(self):
        super().__init__()
        self._model = SklearnLR()
        self.coef_ = None
        self.intercept_ = None
    
    def fit(self, X, y):
        """Fit linear regression model"""
        super().fit(X, y)
        self.coef_ = self._model.coef_
        self.intercept_ = self._model.intercept_
        return self


# ============================================================================
# RIDGE REGRESSION
# ============================================================================

class SKLearnRidgeRegression(SKLearnRegressionBase):
    """
    Ridge Regression (L2 regularization) using scikit-learn.
    
    Adds penalty term α * ||coef||² to prevent overfitting.
    Shrinks coefficients but doesn't set them to zero.
    
    Good when:
    - Features are correlated (multicollinearity)
    - Preventing overfitting on training data
    - You want all features to contribute
    
    Parameters
    ----------
    alpha : float, default=1.0
        Regularization strength
        - Larger values = more regularization (simpler model)
        - alpha=0 is equivalent to LinearRegression
        
    Attributes
    ----------
    coef_ : ndarray
        Regularized coefficients
    intercept_ : float
        Intercept term
        
    Examples
    --------
    >>> model = SKLearnRidgeRegression(alpha=1.0)
    >>> model.fit(X_train, y_train)
    >>> # Compare different alpha values
    >>> for alpha in [0.1, 1.0, 10.0]:
    ...     model = SKLearnRidgeRegression(alpha=alpha)
    ...     model.fit(X_train, y_train)
    ...     print(f"Alpha={alpha}: R²={model.score(X_test, y_test):.4f}")
    """
    
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        self._model = SklearnRidge(alpha=alpha)
        self.coef_ = None
        self.intercept_ = None
    
    def fit(self, X, y):
        """Fit Ridge regression"""
        super().fit(X, y)
        self.coef_ = self._model.coef_
        self.intercept_ = self._model.intercept_
        return self


# ============================================================================
# LASSO REGRESSION
# ============================================================================

class SKLearnLassoRegression(SKLearnRegressionBase):
    """
    Lasso Regression (L1 regularization) using scikit-learn.
    
    Adds penalty term α * ||coef||₁ which can drive coefficients to exactly zero.
    Performs automatic feature selection.
    
    Good when:
    - You have many features and want to select most important ones
    - Interpretability is important (sparse models)
    - You suspect many features are irrelevant
    
    Parameters
    ----------
    alpha : float, default=1.0
        Regularization strength
        - Larger values = more coefficients driven to zero
        
    Attributes
    ----------
    coef_ : ndarray
        Regularized coefficients (some may be exactly 0)
    intercept_ : float
        Intercept term
        
    Examples
    --------
    >>> model = SKLearnLassoRegression(alpha=0.1)
    >>> model.fit(X_train, y_train)
    >>> # Check which features were selected
    >>> selected_features = np.where(model.coef_ != 0)[0]
    >>> print(f"Selected {len(selected_features)} out of {len(model.coef_)} features")
    """
    
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        self._model = SklearnLasso(alpha=alpha)
        self.coef_ = None
        self.intercept_ = None
    
    def fit(self, X, y):
        """Fit Lasso regression"""
        super().fit(X, y)
        self.coef_ = self._model.coef_
        self.intercept_ = self._model.intercept_
        return self


# ============================================================================
# DECISION TREE REGRESSION
# ============================================================================

class SKLearnDecisionTreeRegressor(SKLearnRegressionBase):
    """
    Decision Tree Regressor using scikit-learn.
    
    Non-linear model that creates decision rules based on features.
    Partitions feature space into regions and predicts average value in each region.
    
    Advantages:
    - Captures non-linear relationships
    - No need for feature scaling
    - Interpretable (can visualize tree)
    - Handles mixed data types
    
    Disadvantages:
    - Prone to overfitting
    - Can be unstable (small data changes → different tree)
    - Not great for extrapolation
    
    Parameters
    ----------
    max_depth : int, optional
        Maximum depth of tree (None = unlimited, can overfit)
    min_samples_split : int, default=2
        Minimum samples required to split a node
    random_state : int, optional
        Random seed for reproducibility
        
    Examples
    --------
    >>> # Shallow tree (simple, less overfitting)
    >>> model = SKLearnDecisionTreeRegressor(max_depth=3)
    >>> model.fit(X_train, y_train)
    >>> 
    >>> # Deep tree (complex, may overfit)
    >>> model = SKLearnDecisionTreeRegressor(max_depth=None)
    >>> model.fit(X_train, y_train)
    """
    
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None, random_state=None):
        super().__init__()
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self._model = SklearnDTR(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state
        )

    def fit(self, X, y):
        """Fit Decision Tree Regressor"""
        super().fit(X, y)
        self.feature_importances_ = self._model.feature_importances_
        return self


# ============================================================================
# RANDOM FOREST REGRESSION
# ============================================================================

class SKLearnRandomForestRegressor(SKLearnRegressionBase):
    """
    Random Forest Regressor using scikit-learn.
    
    Ensemble of decision trees - combines predictions from multiple trees.
    Each tree trained on random subset of data and features.
    
    Advantages:
    - More robust than single decision tree
    - Reduces overfitting
    - Handles non-linear relationships
    - Provides feature importance
    - Often works well out-of-the-box
    
    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in forest (more = better but slower)
    max_depth : int, optional
        Maximum depth of each tree
    min_samples_split : int, default=2
        Minimum samples to split a node
    random_state : int, optional
        Random seed
        
    Examples
    --------
    >>> model = SKLearnRandomForestRegressor(n_estimators=100, max_depth=10)
    >>> model.fit(X_train, y_train)
    >>> print(f"R² = {model.score(X_test, y_test):.4f}")
    """
    
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None, random_state=None):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self._model = SklearnRFR(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state
        )

    def fit(self, X, y):
        """Fit Random Forest Regressor"""
        super().fit(X, y)
        self.feature_importances_ = self._model.feature_importances_
        return self