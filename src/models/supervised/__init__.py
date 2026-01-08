"""
Supervised learning models.

Models that learn from labeled data (X, y) to make predictions.

Organization:
- base.py: Abstract base classes (SupervisedModel, RegressionModel, ClassificationModel)
- sklearn_*.py: Implementations using scikit-learn
- keras_*.py: Implementations using TensorFlow/Keras
- scratch_*.py: From-scratch implementations (Phase 2 - January 2026)
"""

from .base import SupervisedModel, RegressionModel, ClassificationModel

from .sklearn_base import (
    SKLearnRegressionBase,
    SKLearnClassificationBase,
)

from .sklearn_regression import (
    SKLearnLinearRegression,
    SKLearnRidgeRegression,
    SKLearnLassoRegression,
    SKLearnDecisionTreeRegressor,
    SKLearnRandomForestRegressor,
)

from .sklearn_classification import (
    SKLearnLogisticRegression,
    SKLearnDecisionTreeClassifier,
    SKLearnRandomForestClassifier,
    SKLearnKNNClassifier,
    SKLearnNaiveBayes,
    SKLearnSVMClassifier,
)

from .scratch_classfication import (
    ScratchKNNClassifier,
)

from .model_selection import (
    k_fold_cross_validation,
    plot_cross_validation_results,
    grid_search_cv,
    plot_grid_search_results,
)

from .sklearn_lda import SKLearnLDA

__all__ = [
    # Base classes
    'SupervisedModel',
    'RegressionModel',
    'ClassificationModel',

    # Scikit-learn base classes
    'SKLearnRegressionBase',
    'SKLearnClassificationBase',
    
    # Regression
    'SKLearnLinearRegression',
    'SKLearnRidgeRegression',
    'SKLearnLassoRegression',
    'SKLearnDecisionTreeRegressor',
    'SKLearnRandomForestRegressor',
    
    # Classification
    'SKLearnLogisticRegression',
    'SKLearnDecisionTreeClassifier',
    'SKLearnRandomForestClassifier',
    'SKLearnKNNClassifier',
    'SKLearnNaiveBayes',
    'SKLearnSVMClassifier',

    # Scratch classification
    'ScratchKNNClassifier',

    # Model Selection
    'k_fold_cross_validation',
    'plot_cross_validation_results',
    'grid_search_cv',
    'plot_grid_search_results',
    
    # Dimensionality Reduction
    'SKLearnLDA',
]