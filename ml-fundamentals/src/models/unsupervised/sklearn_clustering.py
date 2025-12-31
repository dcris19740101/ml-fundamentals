"""
Clustering models using scikit-learn.

All classes prefixed with 'SKLearn' to indicate library dependency.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.cluster import DBSCAN as SklearnDBSCAN
from sklearn.cluster import AgglomerativeClustering as SklearnAgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from .sklearn_base import SKLearnClusteringBase


# ============================================================================
# K-MEANS CLUSTERING
# ============================================================================

class SKLearnKMeans(SKLearnClusteringBase):
    """
    K-Means clustering using scikit-learn.
    
    Partitions data into k clusters by minimizing within-cluster variance.
    
    Parameters
    ----------
    n_clusters : int, default=3
        Number of clusters to form
    random_state : int, optional
        Random seed for reproducibility
        
    Attributes
    ----------
    cluster_centers_ : ndarray, shape (n_clusters, n_features)
        Coordinates of cluster centers
    labels_ : ndarray, shape (n_samples,)
        Cluster label for each sample
        
    Examples
    --------
    >>> model = SKLearnKMeans(n_clusters=3, random_state=42)
    >>> labels = model.fit_predict(X)
    >>> print(f"Silhouette score: {model.silhouette_score(X):.4f}")
    >>> print(f"Inertia: {model.inertia():.2f}")
    
    Notes
    -----
    - Good for: Spherical clusters of similar size
    - Limitations: Must specify k beforehand, assumes spherical clusters
    - K-Means supports prediction on new data
    """
    
    def __init__(self, n_clusters=3, random_state=None):
        super().__init__()
        self.n_clusters = n_clusters
        self.random_state = random_state
        self._model = SklearnKMeans(n_clusters=n_clusters, random_state=random_state)
        self.cluster_centers_ = None
    
    def fit(self, X):
        """Fit K-Means to data."""
        self._model.fit(X)
        self.labels_ = self._model.labels_
        self.cluster_centers_ = self._model.cluster_centers_
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """
        Predict cluster for new data.
        
        K-Means supports prediction: assigns new points to nearest centroid.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data to predict
            
        Returns
        -------
        labels : array, shape (n_samples,)
            Predicted cluster labels
        """
        self._check_is_fitted()
        return self._model.predict(X)
    
    def inertia(self):
        """
        Return within-cluster sum of squares (inertia).
        
        Lower inertia = tighter clusters.
        Used in elbow method to determine optimal k.
        
        Returns
        -------
        inertia : float
            Sum of squared distances to nearest cluster center
        """
        self._check_is_fitted()
        return self._model.inertia_
    
    def get_params(self):
        """Get K-Means parameters."""
        return {
            'n_clusters': self.n_clusters,
            'random_state': self.random_state
        }


# ============================================================================
# DBSCAN CLUSTERING
# ============================================================================

class SKLearnDBSCAN(SKLearnClusteringBase):
    """DBSCAN clustering using scikit-learn."""
    
    def __init__(self, eps=0.5, min_samples=5):
        super().__init__()
        self.eps = eps
        self.min_samples = min_samples
        self._model = SklearnDBSCAN(eps=eps, min_samples=min_samples)
        self.n_clusters_ = None
    
    def fit(self, X):
        """Fit DBSCAN to data."""
        self._model.fit(X)
        self.labels_ = self._model.labels_
        self.n_clusters_ = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """
        Get cluster labels for DBSCAN.
        
        Note: DBSCAN doesn't truly "predict" on new data like K-Means.
        This method returns the labels from fitting.
        For new data, you need to refit the model.
        """
        self._check_is_fitted()
        
        # DBSCAN stores labels in labels_ after fitting
        if hasattr(self._model, 'labels_'):
            return self._model.labels_
        else:
            # If not fitted yet, fit and return labels
            return self._model.fit_predict(X)
    
    def find_optimal_k(self, X, k_range=None, method='both', show_plot=True):
        """
        DBSCAN does not support finding optimal k.
        
        DBSCAN automatically discovers the number of clusters based on
        the density parameters (eps and min_samples). It does not take
        n_clusters as a parameter.
        
        Use find_optimal_eps() instead to tune the eps parameter.
        
        Raises
        ------
        NotImplementedError
            Always (DBSCAN doesn't have n_clusters parameter)
        """
        raise NotImplementedError(
            "DBSCAN doesn't have n_clusters parameter. "
            "The number of clusters is discovered automatically based on eps and min_samples. "
            "Use find_optimal_eps() instead to tune the eps parameter."
        )
    
    def find_optimal_eps(self, X, eps_range=None, min_samples=None, show_plot=True):
        """
        Find optimal eps parameter for DBSCAN using silhouette score.
        
        Tests different eps values and evaluates clustering quality.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data to cluster (should be scaled)
        eps_range : array-like, optional
            Range of eps values to test.
            Default: np.arange(0.1, 2.0, 0.1)
        min_samples : int, optional
            min_samples parameter to use.
            If None, uses self.min_samples
        show_plot : bool, default=True
            Whether to show visualization
            
        Returns
        -------
        results : dict
            Dictionary containing:
            - 'optimal_eps': Recommended eps value
            - 'eps_values': List of eps values tested
            - 'n_clusters_list': Number of clusters found for each eps
            - 'n_noise_list': Number of noise points for each eps
            - 'silhouettes': Silhouette scores for each eps
            
        Examples
        --------
        >>> model = SKLearnDBSCAN(min_samples=5)
        >>> results = model.find_optimal_eps(X_scaled)
        >>> optimal_eps = results['optimal_eps']
        >>> 
        >>> # Train with optimal eps
        >>> final_model = SKLearnDBSCAN(eps=optimal_eps, min_samples=5)
        >>> final_model.fit(X_scaled)
        
        Notes
        -----
        - ALWAYS scale your data before using this method
        - eps controls neighborhood size (smaller = more clusters + noise)
        - min_samples controls density threshold
        - Good clustering: high silhouette, few noise points
        """
        # Default parameters
        if eps_range is None:
            eps_range = np.arange(0.1, 2.0, 0.1)
        else:
            eps_range = np.array(eps_range)
        
        if min_samples is None:
            min_samples = self.min_samples
        
        print("="*70)
        print("FINDING OPTIMAL EPS FOR DBSCAN")
        print("="*70)
        print(f"\nTesting eps values: {eps_range[:5]}...{eps_range[-5:]}")
        print(f"Using min_samples: {min_samples}\n")
        
        # Test each eps value
        eps_values = []
        n_clusters_list = []
        n_noise_list = []
        silhouettes = []
        inertias = []
        
        for eps in eps_range:
            # Create temporary model
            temp_model = SKLearnDBSCAN(eps=eps, min_samples=min_samples)
            temp_model.fit(X)
            
            n_clusters = temp_model.n_clusters_
            n_noise = np.sum(temp_model.labels_ == -1)
            
            eps_values.append(eps)
            n_clusters_list.append(n_clusters)
            n_noise_list.append(n_noise)
            
            # Calculate silhouette score if possible
            if n_clusters >= 2 and n_noise < len(X):
                try:
                    score = temp_model.silhouette_score(X, exclude_noise=True)
                    silhouettes.append(score)
                    
                    # Calculate inertia for reference
                    inertia = temp_model._calculate_inertia(X)
                    inertias.append(inertia)
                except ValueError:
                    silhouettes.append(-1)
                    inertias.append(None)
            else:
                silhouettes.append(-1)
                inertias.append(None)
            
            # Print progress
            status = f"eps={eps:.2f}: {n_clusters} clusters, {n_noise} noise"
            if silhouettes[-1] > -1:
                status += f", silhouette={silhouettes[-1]:.4f}"
            print(status)
        
        # Find optimal eps
        valid_scores = [(i, s) for i, s in enumerate(silhouettes) if s > -1]
        
        if not valid_scores:
            print("\n⚠️  No valid clustering found for any eps value!")
            print("   Try adjusting eps_range or min_samples")
            optimal_eps = eps_range[len(eps_range) // 2]
        else:
            best_idx = max(valid_scores, key=lambda x: x[1])[0]
            optimal_eps = eps_values[best_idx]
            best_silhouette = silhouettes[best_idx]
            best_n_clusters = n_clusters_list[best_idx]
            best_n_noise = n_noise_list[best_idx]
            
            print(f"\n💡 Optimal eps: {optimal_eps:.2f}")
            print(f"   Clusters: {best_n_clusters}")
            print(f"   Noise points: {best_n_noise} ({best_n_noise/len(X)*100:.1f}%)")
            print(f"   Silhouette score: {best_silhouette:.4f}")
        
        # Visualization
        if show_plot:
            self._plot_eps_results(eps_values, n_clusters_list, n_noise_list, 
                                  silhouettes, optimal_eps)
        
        return {
            'optimal_eps': optimal_eps,
            'eps_values': eps_values,
            'n_clusters_list': n_clusters_list,
            'n_noise_list': n_noise_list,
            'silhouettes': silhouettes,
            'inertias': inertias
        }
    
    def _plot_eps_results(self, eps_values, n_clusters_list, n_noise_list, 
                         silhouettes, optimal_eps):
        """Plot results of eps optimization."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Plot 1: Number of clusters vs eps
        axes[0].plot(eps_values, n_clusters_list, 'bo-', linewidth=2, markersize=6)
        axes[0].axvline(x=optimal_eps, color='red', linestyle='--', linewidth=2,
                       label=f'Optimal eps={optimal_eps:.2f}')
        axes[0].set_xlabel('eps', fontsize=12)
        axes[0].set_ylabel('Number of Clusters', fontsize=12)
        axes[0].set_title('Clusters Found vs eps', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Noise points vs eps
        axes[1].plot(eps_values, n_noise_list, 'ro-', linewidth=2, markersize=6)
        axes[1].axvline(x=optimal_eps, color='red', linestyle='--', linewidth=2,
                       label=f'Optimal eps={optimal_eps:.2f}')
        axes[1].set_xlabel('eps', fontsize=12)
        axes[1].set_ylabel('Number of Noise Points', fontsize=12)
        axes[1].set_title('Noise Points vs eps', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Silhouette score vs eps
        valid_eps = [eps for eps, s in zip(eps_values, silhouettes) if s > -1]
        valid_silhouettes = [s for s in silhouettes if s > -1]
        
        if valid_silhouettes:
            axes[2].plot(valid_eps, valid_silhouettes, 'go-', linewidth=2, markersize=6)
            axes[2].axvline(x=optimal_eps, color='red', linestyle='--', linewidth=2,
                           label=f'Optimal eps={optimal_eps:.2f}')
            axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.2)
            axes[2].set_xlabel('eps', fontsize=12)
            axes[2].set_ylabel('Silhouette Score', fontsize=12)
            axes[2].set_title('Silhouette Score vs eps', fontsize=14, fontweight='bold')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        else:
            axes[2].text(0.5, 0.5, 'No valid silhouette scores', 
                        ha='center', va='center', transform=axes[2].transAxes,
                        fontsize=14)
            axes[2].set_title('Silhouette Score vs eps', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def get_params(self):
        """Get DBSCAN parameters."""
        return {
            'eps': self.eps,
            'min_samples': self.min_samples
        }


# ============================================================================
# HIERARCHICAL CLUSTERING
# ============================================================================

class SKLearnHierarchicalClustering(SKLearnClusteringBase):
    """
    Hierarchical (Agglomerative) Clustering using scikit-learn.
    
    Builds a tree of clusters (dendrogram) from bottom-up.
    
    Parameters
    ----------
    n_clusters : int, default=3
        Number of clusters to find
    linkage : {'ward', 'complete', 'average', 'single'}, default='ward'
        Linkage criterion (ward minimizes variance)
    metric : str, default='euclidean'
        Distance metric (used with non-ward linkage)
    
    Attributes
    ----------
    labels_ : array, shape (n_samples,)
        Cluster labels for each sample
        
    Examples
    --------
    >>> model = SKLearnHierarchicalClustering(n_clusters=3, linkage='ward')
    >>> labels = model.fit_predict(X)
    >>> model.plot_dendrogram(X)
    
    Notes
    -----
    - Good for: Small-medium datasets, hierarchical structure visualization
    - Limitations: Slow on large data, cannot predict on truly new data
    - Creates dendrogram showing cluster hierarchy
    """
    
    def __init__(self, n_clusters=3, linkage='ward', metric='euclidean'):
        super().__init__()
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.metric = metric
    
    def fit(self, X):
        """Fit hierarchical clustering."""
        self._model = SklearnAgglomerativeClustering(
            n_clusters=self.n_clusters,
            linkage=self.linkage,
            metric=self.metric if self.linkage != 'ward' else 'euclidean'
        )
        self.labels_ = self._model.fit_predict(X)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """
        Return cluster labels from training.
        
        Note: Hierarchical clustering doesn't support prediction on truly new data.
        This returns the labels from fit().
        
        Parameters
        ----------
        X : array-like (ignored, but kept for API consistency)
            
        Returns
        -------
        labels : array, shape (n_samples,)
            Cluster labels from training
        """
        self._check_is_fitted()
        return self.labels_
    
    def plot_dendrogram(self, X, figsize=(12, 6), max_d=None):
        """
        Plot dendrogram showing cluster hierarchy.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Original data
        figsize : tuple, default=(12, 6)
            Figure size
        max_d : float, optional
            Maximum distance for horizontal line (shows cut point)
        """
        # Compute linkage matrix
        linkage_matrix = linkage(X, method=self.linkage)
        
        plt.figure(figsize=figsize)
        dendrogram(linkage_matrix)
        
        if max_d:
            plt.axhline(y=max_d, c='red', linestyle='--', 
                       label=f'Cut at distance {max_d:.2f}')
            plt.legend()
        
        plt.title(f'Hierarchical Clustering Dendrogram ({self.linkage} linkage)')
        plt.xlabel('Sample Index')
        plt.ylabel('Distance')
        plt.tight_layout()
        plt.show()

    def get_params(self):
        """Get hierarchical clustering parameters."""
        return {
            'n_clusters': self.n_clusters,
            'linkage': self.linkage,
            'metric': self.metric
        }