"""
Classification models using scikit-learn.

All classes prefixed with 'SKLearn' to indicate library dependency.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression as SklearnLogReg
from sklearn.tree import DecisionTreeClassifier as SklearnDTC
from sklearn.ensemble import RandomForestClassifier as SklearnRFC
from sklearn.neighbors import KNeighborsClassifier as SklearnKNN
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.svm import SVC as SklearnSVM

from .sklearn_base import SKLearnClassificationBase


# ============================================================================
# LOGISTIC REGRESSION
# ============================================================================

class SKLearnLogisticRegression(SKLearnClassificationBase):
    """
    Logistic Regression using scikit-learn.
    
    Despite its name, it's a classification algorithm (not regression).
    Uses logistic function to model probability of binary outcome.
    
    Good for:
    - Binary classification (spam/not spam, fraud/not fraud)
    - Multi-class classification (one-vs-rest)
    - When you need probability estimates
    - Interpretable linear decision boundaries
    
    Parameters
    ----------
    max_iter : int, default=1000
        Maximum iterations for solver convergence
    random_state : int, optional
        Random seed
        
    Examples
    --------
    >>> model = SKLearnLogisticRegression()
    >>> model.fit(X_train, y_train)
    >>> y_pred = model.predict(X_test)
    >>> print(f"Accuracy: {model.score(X_test, y_test):.4f}")
    >>> 
    >>> # Get probabilities
    >>> probabilities = model.predict_proba(X_test)
    """
    
    def __init__(self, max_iter=1000, random_state=None):
        super().__init__()
        self.max_iter = max_iter
        self.random_state = random_state
        self._model = SklearnLogReg(max_iter=max_iter, random_state=random_state)
        self.coef_ = None
        self.intercept_ = None
    
    def fit(self, X, y):
        """Fit Logistic Regression"""
        super().fit(X, y)
        self.coef_ = self._model.coef_
        self.intercept_ = self._model.intercept_
        return self

# ============================================================================
# DECISION TREE CLASSIFIER
# ============================================================================

class SKLearnDecisionTreeClassifier(SKLearnClassificationBase):
    """
    Decision Tree Classifier using scikit-learn.
    
    Creates tree of if-then-else rules based on features.
    
    Parameters
    ----------
    max_depth : int, optional
        Maximum depth of tree
    min_samples_split : int, default=2
        Minimum samples to split node
    random_state : int, optional
        Random seed
        
    Examples
    --------
    >>> model = SKLearnDecisionTreeClassifier(max_depth=5)
    >>> model.fit(X_train, y_train)
    """
    
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None, random_state=None):
        super().__init__()
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self._model = SklearnDTC(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state
        )
        self.feature_importances_ = None
    
    def fit(self, X, y):
        """Fit Decision Tree Classifier"""
        super().fit(X, y)
        self.feature_importances_ = self._model.feature_importances_
        return self


# ============================================================================
# RANDOM FOREST CLASSIFIER
# ============================================================================

class SKLearnRandomForestClassifier(SKLearnClassificationBase):
    """
    Random Forest Classifier using scikit-learn.
    
    Ensemble of decision trees for robust classification.
    
    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees
    max_depth : int, optional
        Maximum depth of each tree
    random_state : int, optional
        Random seed
        
    Examples
    --------
    >>> model = SKLearnRandomForestClassifier(n_estimators=100)
    >>> model.fit(X_train, y_train)
    """
    
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None, random_state=None):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self._model = SklearnRFC(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state
        )
        self.feature_importances_ = None
    
    def fit(self, X, y):
        """Fit Random Forest Classifier"""
        super().fit(X, y)
        self.feature_importances_ = self._model.feature_importances_
        return self


# ============================================================================
# K-NEAREST NEIGHBORS CLASSIFIER
# ============================================================================

class SKLearnKNNClassifier(SKLearnClassificationBase):
    """
    K-Nearest Neighbors Classifier using scikit-learn.
    
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
    
    Attributes
    ----------
    model : KNeighborsClassifier
        The fitted sklearn KNN model
        
    Examples
    --------
    >>> from src.models.supervised import SKLearnKNNClassifier
    >>> model = SKLearnKNNClassifier(n_neighbors=5)
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
    
    def __init__(self, n_neighbors=5, weights='uniform', metric='euclidean'):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self._model = SklearnKNN(
            n_neighbors=n_neighbors,
            weights=weights,
            metric=metric
        )


# ============================================================================
# NAIVE BAYES CLASSIFIER
# ============================================================================

class SKLearnNaiveBayes(SKLearnClassificationBase):
    """
    Naive Bayes Classifier.
    
    Naive Bayes is a probabilistic classifier based on Bayes' theorem:
    - Assumes features are independent (naive assumption)
    - Fast training and prediction
    - Works well with small datasets
    
    Parameters
    ----------
    variant : {'gaussian', 'multinomial', 'bernoulli'}, default='gaussian'
        Type of Naive Bayes:
        - 'gaussian': For continuous features (most common)
        - 'multinomial': For count/frequency data (text classification)
        - 'bernoulli': For binary features
        - var_smoothing : float, default=1e-9
            Portion of largest variance added to variances (for numerical stability)
            Only used for Gaussian variant
    
    Attributes
    ----------
    model : GaussianNB, MultinomialNB, or BernoulliNB
        The fitted sklearn Naive Bayes model
        
    Examples
    --------
    >>> from src.models.supervised import SKLearnNaiveBayes
    >>> model = SKLearnNaiveBayes(variant='gaussian')
    >>> model.fit(X_train, y_train)
    >>> predictions = model.predict(X_test)
    
    Notes
    -----
    - Fast training and prediction
    - Works well with small datasets
    - Assumes feature independence (rarely true, but often works anyway!)
    - Gaussian: Use for continuous features (Iris, etc.)
    - Multinomial: Use for text classification (word counts)
    - Performs well even when independence assumption is violated
    """
    
    def __init__(self, variant='gaussian', var_smoothing=1e-9):
        super().__init__()
        self.variant = variant
        self.var_smoothing = var_smoothing
        if variant == 'gaussian':
            self._model = GaussianNB(var_smoothing=var_smoothing)
        elif variant == 'multinomial':
            self._model = MultinomialNB()
        elif variant == 'bernoulli':
            self._model = BernoulliNB()
        else:
            raise ValueError(f"Unknown variant: {self.variant}")


# ============================================================================
# SUPPORT VECTOR MACHINE CLASSIFIER
# ============================================================================

class SKLearnSVMClassifier(SKLearnClassificationBase):
    """
    Support Vector Machine Classifier using scikit-learn.
    
    SVM finds the optimal hyperplane that maximizes the margin between classes.
    - Works well in high-dimensional spaces
    - Effective with small-medium datasets
    - Kernel trick for non-linear boundaries
    
    Parameters
    ----------
    kernel : {'linear', 'rbf', 'poly', 'sigmoid'}, default='rbf'
        Kernel type:
        - 'linear': Linear boundary (fast, interpretable)
        - 'rbf': Radial basis function (most common, flexible)
        - 'poly': Polynomial boundary
        - 'sigmoid': Sigmoid kernel
    C : float, default=1.0
        Regularization parameter (inverse of regularization strength)
        - Smaller C: wider margin, more misclassifications (regularize more)
        - Larger C: narrower margin, fewer misclassifications (fit closer)
    gamma : {'scale', 'auto'} or float, default='scale'
        Kernel coefficient for 'rbf', 'poly', 'sigmoid'
        - 'scale': 1 / (n_features × X.var())
        - 'auto': 1 / n_features
        - Higher gamma: more complex boundary (overfitting risk)
    random_state : int, default=42
        Random seed for reproducibility
    
    Attributes
    ----------
    model : SVC
        The fitted sklearn SVM model
        
    Examples
    --------
    >>> from src.models.supervised import SKLearnSVMClassifier
    >>> model = SKLearnSVMClassifier(kernel='rbf', C=1.0)
    >>> model.fit(X_train, y_train)
    >>> predictions = model.predict(X_test)
    
    Notes
    -----
    - ALWAYS scale features before using SVM (critical!)
    - RBF kernel is good default (flexible, non-linear)
    - Linear kernel for linearly separable data (faster)
    - Slow on large datasets (O(n²) to O(n³))
    - Good for: Binary classification, small-medium datasets
    - Tune C and gamma with grid search
    """
    
    def __init__(self, kernel='rbf', C=1.0, gamma='scale', random_state=42):
        super().__init__()
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.random_state = random_state
        self._model = SklearnSVM(
            kernel=kernel,
            C=C,
            gamma=gamma,
            random_state=random_state,
            probability=True  # Enable predict_proba
        )
    
    def get_support_vectors(self):
        """
        Get support vectors.
        
        Returns
        -------
        support_vectors : array, shape (n_support_vectors, n_features)
            Support vectors
        """
        if self._model is None:
            raise ValueError("Model must be fitted first")
        return self._model.support_vectors_