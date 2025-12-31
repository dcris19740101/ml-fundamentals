"""
Machine Learning Models

Organized by learning paradigm:
- supervised: Models that learn from labeled data (X, y)
- unsupervised: Models that discover patterns in unlabeled data (X only)

Each category contains:
- base.py: Abstract base classes defining interfaces
- sklearn_*.py: Implementations using scikit-learn
- keras_*.py: Implementations using TensorFlow/Keras
- scratch_*.py: From-scratch implementations (Phase 2 - January 2026)
"""

from .base import BaseModel

# Import from supervised
from .supervised import (
    # Base classes
    SupervisedModel,
    RegressionModel,
    ClassificationModel,

    # Regression models
    SKLearnLinearRegression,
    SKLearnRidgeRegression,
    SKLearnLassoRegression,
    SKLearnDecisionTreeRegressor,
    SKLearnRandomForestRegressor,

    # Classification models
    SKLearnLogisticRegression,
    SKLearnDecisionTreeClassifier,
    SKLearnRandomForestClassifier,
)

# Import from unsupervised
from .unsupervised import (
    # Base classes
    UnsupervisedModel,
    ClusteringModel,

    # Clustering models
    SKLearnKMeans,
    SKLearnDBSCAN,
)

from .neural_networks import (
    KerasANN,
    KerasCNN,
)

__all__ = [
    # Base class
    'BaseModel',
    
    # Supervised base classes
    'SupervisedModel',
    'RegressionModel',
    'ClassificationModel',

    # Supervised models
    'SKLearnLinearRegression',
    'SKLearnRidgeRegression',
    'SKLearnLassoRegression',
    'SKLearnDecisionTreeRegressor',
    'SKLearnRandomForestRegressor',
    'SKLearnLogisticRegression',
    'SKLearnDecisionTreeClassifier',
    'SKLearnRandomForestClassifier',
    
    # Unsupervised base classes
    'UnsupervisedModel',
    'ClusteringModel',
    
    # Unsupervised models
    'SKLearnKMeans',
    'SKLearnDBSCAN',

    # Neural Networks
    'KerasANN',
    'KerasCNN',
]