"""
ML Fundamentals Library

Main modules:
- preprocessing: Feature transformations (DataPreprocessor)
- target_encoding: Target variable encoding
- data_utils: Split and sampling utilities
- eda: Exploratory data analysis
- data_loader: Dataset loading
- models: ML algorithms
- evaluation: Metrics and visualization
"""

__version__ = "0.1.0"

# Core preprocessing
from .preprocessing import DataPreprocessor

# Target encoding utilities
from .target_encoding import (
    encode_target_labels,
    encode_ordinal_target,
    decode_target,
)

# Data utilities
from .data_utils import (
    split_train_test,
    split_train_val_test,
)

# EDA utilities
from .eda import (
    analyze_data,
    detect_outliers_iqr,
    plot_distributions,
    plot_feature_target_relationships,
    analyze_feature_importance_for_model_selection,
    correlation_heatmap,
)

# Data loading
from . import data_loader

__all__ = [
    # Preprocessing
    'DataPreprocessor',
    
    # Target encoding
    'encode_target_labels',
    'encode_ordinal_target',
    'decode_target',
    
    # Data utilities
    'split_train_test',
    'split_train_val_test',
    
    # EDA
    'analyze_data',
    'detect_outliers_iqr',
    'plot_distributions',
    'plot_feature_target_relationships',
    'analyze_feature_importance_for_model_selection',
    'correlation_heatmap',
    
    # Modules
    'data_loader',
]