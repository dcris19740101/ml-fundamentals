"""
Unsupervised learning models.

Models that discover patterns in unlabeled data (X only, no y).

Organization:
- base.py: Abstract base classes (UnsupervisedModel, ClusteringModel)
- sklearn_*.py: Implementations using scikit-learn
- scratch_*.py: From-scratch implementations (Phase 2 - January 2026)
"""

from .base import UnsupervisedModel, ClusteringModel
from .sklearn_base import SKLearnClusteringBase
from .sklearn_clustering import (
    SKLearnKMeans,
    SKLearnDBSCAN,
    SKLearnHierarchicalClustering,
)

from .sklearn_pca import SKLearnPCA, SKLearnKernelPCA

__all__ = [
    # Base classes  
    'UnsupervisedModel',
    'ClusteringModel',
    
    # Scikit-learn base classes
    'SKLearnClusteringBase',

    # Clustering
    'SKLearnKMeans',
    'SKLearnDBSCAN',
    'SKLearnHierarchicalClustering',

    # Dimensionality Reduction
    'SKLearnPCA',
    'SKLearnKernelPCA',
]