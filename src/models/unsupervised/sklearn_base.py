"""
Scikit-learn specific base class for unsupervised learning models.

Provides common sklearn clustering interface to reduce code duplication.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from .base import ClusteringModel


class SKLearnClusteringBase(ClusteringModel):
    """
    Base class for scikit-learn clustering models.
    
    Provides common functionality for sklearn clustering algorithms:
    - Fitted state tracking (is_fitted)
    - Common attributes (labels_)
    - Silhouette score calculation
    - Elbow detection for optimal k selection
    
    Attributes
    ----------
    _model : sklearn clusterer
        The underlying sklearn clustering model (must be set by subclass)
    labels_ : array, shape (n_samples,)
        Cluster labels assigned to each sample
    is_fitted : bool
        Whether the model has been fitted
        
    Notes
    -----
    Subclasses must:
    1. Call super().__init__()
    2. Set self._model to a sklearn clustering estimator
    3. Override fit() to set self.labels_ and call self.is_fitted = True
    
    Subclasses may optionally override:
    - predict() if the algorithm supports prediction on new data
    - _calculate_inertia() for elbow method support
    - Additional model-specific methods and attributes
    """
    
    def __init__(self):
        super().__init__()
        self._model = None  # Must be set by subclass
        self.labels_ = None
    
    def fit_predict(self, X):
        """
        Fit model and return cluster labels.
        
        This is a convenience method that combines fit() and returning labels.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data to cluster
            
        Returns
        -------
        labels : array, shape (n_samples,)
            Cluster labels
        """
        self.fit(X)
        return self.labels_
    
    def silhouette_score(self, X, exclude_noise=True):
        """
        Calculate silhouette score for clustering.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The data that was clustered
        exclude_noise : bool, default=True
            If True, exclude noise points (label=-1) from calculation
            Relevant for DBSCAN
            
        Returns
        -------
        float
            Silhouette score (higher is better, range [-1, 1])
        """
        from sklearn.metrics import silhouette_score as sklearn_silhouette
        
        self._check_is_fitted()
        
        # Get labels
        labels = self.predict(X)
        
        # Convert to numpy array to avoid ambiguity
        labels = np.asarray(labels).ravel()  # ← Make sure it's 1D numpy array
        
        # Check if we have at least 2 clusters
        unique_labels = np.unique(labels)
        
        # Count clusters (excluding noise if present)
        has_noise = np.any(unique_labels == -1)
        n_clusters = len(unique_labels) - (1 if has_noise else 0)
        
        if n_clusters < 2:
            print(f"⚠️  Only {n_clusters} cluster(s) found. Silhouette score requires at least 2 clusters.")
            return np.nan
        
        # Prepare data for scoring
        X_array = np.asarray(X)
        
        # Handle noise points (for DBSCAN)
        if exclude_noise and has_noise:  # ← Use the boolean we already computed
            mask = labels != -1
            
            if not np.any(mask):  # ← Use np.any() on the mask, not labels
                print("⚠️  All points are noise!")
                return np.nan
            
            X_array = X_array[mask]
            labels = labels[mask]
            
            # Recheck cluster count after removing noise
            unique_labels_no_noise = np.unique(labels)
            if len(unique_labels_no_noise) < 2:
                print(f"⚠️  Only {len(unique_labels_no_noise)} cluster(s) after removing noise.")
                return np.nan
        
        # Calculate silhouette score
        try:
            score = sklearn_silhouette(X_array, labels)
            return score
        except Exception as e:
            print(f"⚠️  Error calculating silhouette score: {e}")
            return np.nan
    
    def silhouette_samples(self, X, exclude_noise=True):
        """
        Calculate per-sample silhouette scores for clustering.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The data that was clustered
        exclude_noise : bool, default=True
            If True, exclude noise points (label=-1) from calculation
            
        Returns
        -------
        array, shape (n_samples,) or (n_samples_no_noise,)
            Silhouette coefficient for each sample
            
        Examples
        --------
        >>> model.fit(X_scaled)
        >>> scores = model.silhouette_samples(X_scaled)
        >>> print(f"Mean: {np.mean(scores):.3f}")
        """
        from sklearn.metrics import silhouette_samples as sklearn_silhouette_samples
        
        self._check_is_fitted()
        
        # Get labels
        labels = self.predict(X)
        labels = np.asarray(labels).ravel()
        
        # Check if we have at least 2 clusters
        unique_labels = np.unique(labels)
        has_noise = np.any(unique_labels == -1)
        n_clusters = len(unique_labels) - (1 if has_noise else 0)
        
        if n_clusters < 2:
            print(f"⚠️  Only {n_clusters} cluster(s) found. Silhouette requires at least 2.")
            return None
        
        # Prepare data
        X_array = np.asarray(X)
        
        # Handle noise points
        if exclude_noise and has_noise:
            mask = labels != -1
            if not np.any(mask):
                print("⚠️  All points are noise!")
                return None
            
            X_array = X_array[mask]
            labels = labels[mask]
            
            if len(np.unique(labels)) < 2:
                print(f"⚠️  Only {len(np.unique(labels))} cluster(s) after removing noise.")
                return None
        
        try:
            sample_scores = sklearn_silhouette_samples(X_array, labels)
            return sample_scores
        except Exception as e:
            print(f"⚠️  Error calculating silhouette samples: {e}")
            return None
    
    def plot_silhouette_analysis(self, X, exclude_noise=True, figsize=(10, 6)):
        """
        Plot silhouette analysis for clustering.
        
        Shows silhouette coefficient for each sample, grouped by cluster.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The data that was clustered
        exclude_noise : bool, default=True
            If True, exclude noise points from plot
        figsize : tuple, default=(10, 6)
            Figure size
            
        Examples
        --------
        >>> model.fit(X_scaled)
        >>> model.plot_silhouette_analysis(X_scaled)
        """
        import matplotlib.pyplot as plt
        
        # Get labels and scores
        labels = self.predict(X)
        labels = np.asarray(labels).ravel()
        
        # Get per-sample scores
        silhouette_vals = self.silhouette_samples(X, exclude_noise=exclude_noise)
        
        if silhouette_vals is None:
            print("⚠️  Cannot create silhouette plot")
            return
        
        # Get overall score
        overall_silhouette = self.silhouette_score(X, exclude_noise=exclude_noise)
        
        # Filter labels if noise was excluded
        unique_labels = np.unique(labels)
        has_noise = np.any(unique_labels == -1)
        
        if exclude_noise and has_noise:
            mask = labels != -1
            labels = labels[mask]
        
        # Get cluster info
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        
        print("\n" + "="*70)
        print("SILHOUETTE ANALYSIS")
        print("="*70)
        print(f"Overall Silhouette Score: {overall_silhouette:.4f}")
        print(f"Number of clusters: {n_clusters}")
        if exclude_noise and has_noise:
            print(f"ℹ️  Noise points excluded from plot")
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        y_lower = 10
        for i, cluster_label in enumerate(unique_labels):
            # Get silhouette values for this cluster
            cluster_silhouette_vals = silhouette_vals[labels == cluster_label]
            cluster_silhouette_vals.sort()
            
            size_cluster = cluster_silhouette_vals.shape[0]
            y_upper = y_lower + size_cluster
            
            # Color for this cluster
            color = plt.cm.viridis(float(i) / n_clusters)
            
            # Fill silhouette
            ax.fill_betweenx(np.arange(y_lower, y_upper),
                             0, cluster_silhouette_vals,
                             facecolor=color, edgecolor=color, alpha=0.7)
            
            # Label cluster
            ax.text(-0.05, y_lower + 0.5 * size_cluster, 
                   f'{cluster_label}', fontsize=11, fontweight='bold')
            
            y_lower = y_upper + 10
        
        # Formatting
        ax.set_xlabel('Silhouette Coefficient', fontsize=12)
        ax.set_ylabel('Cluster', fontsize=12)
        ax.set_title(f'Silhouette Analysis (n_clusters={n_clusters})', 
                    fontsize=14, fontweight='bold')
        
        # Average line
        ax.axvline(x=overall_silhouette, color="red", linestyle="--", linewidth=2,
                  label=f'Average ({overall_silhouette:.3f})')
        
        ax.set_yticks([])
        ax.set_xlim([-0.2, 1.0])
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.show()
        
        # Interpretation
        print("\n💡 How to interpret silhouette plot:")
        print("   • Width of silhouette = cluster size")
        print("   • Values > average (red line) = well-clustered samples")
        print("   • Negative values = samples likely in wrong cluster")
        print("   • Similar widths = balanced cluster sizes")
        
    def _calculate_inertia(self, X):
        """
        Calculate within-cluster sum of squares (inertia).
        
        Inertia measures how tight clusters are (lower = tighter).
        Used in elbow method to determine optimal k.
        
        This is a default implementation that works for all clustering methods.
        Subclasses can override for algorithm-specific optimizations.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data that was clustered
            
        Returns
        -------
        inertia : float
            Within-cluster sum of squared distances
            
        Notes
        -----
        - K-Means: Has built-in inertia_ attribute (more efficient)
        - DBSCAN: Uses this default implementation
        - Hierarchical: Uses this default implementation
        """
        self._check_is_fitted()
        
        # Get unique cluster labels (excluding noise if present)
        unique_labels = np.unique(self.labels_)
        unique_labels = unique_labels[unique_labels != -1]  # Exclude noise
        
        if len(unique_labels) == 0:
            raise ValueError("No valid clusters found (all points are noise)")
        
        # Calculate cluster centers (mean of points in each cluster)
        centers = []
        for label in unique_labels:
            mask = self.labels_ == label
            center = X[mask].mean(axis=0)
            centers.append(center)
        centers = np.array(centers)
        
        # Calculate inertia: sum of squared distances from points to their cluster centers
        inertia = 0.0
        for i, label in enumerate(unique_labels):
            mask = self.labels_ == label
            distances_squared = np.sum((X[mask] - centers[i])**2, axis=1)
            inertia += np.sum(distances_squared)
        
        return inertia
    
    def find_optimal_k(self, X, k_range=range(2, 11), method='both', show_plot=True):
        """
        Find optimal number of clusters using elbow method and/or silhouette analysis.
        
        NOTE: This method only works for clustering algorithms that accept
        n_clusters as a parameter (K-Means, Hierarchical Clustering).
        
        For DBSCAN, use find_optimal_eps() instead.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data to cluster (should be scaled)
        k_range : range or list, default=range(2, 11)
            Range of k values to test
        method : {'elbow', 'silhouette', 'both'}, default='both'
            Which method(s) to use
        show_plot : bool, default=True
            Whether to show visualization
            
        Returns
        -------
        results : dict
            Dictionary containing optimal k and metrics
            
        Raises
        ------
        ValueError
            If the model doesn't support n_clusters parameter
        """
        # Check if model supports n_clusters
        if not hasattr(self, 'n_clusters'):
            raise ValueError(
                f"{self.__class__.__name__} does not support find_optimal_k(). "
                f"This model doesn't have an 'n_clusters' parameter. "
                f"See model documentation for appropriate parameter tuning methods."
            )
        # Validate inputs
        if not hasattr(k_range, '__iter__'):
            raise ValueError("k_range must be iterable (e.g., range(2, 11) or [2, 3, 4, 5])")
        
        k_values = list(k_range)
        if min(k_values) < 1:
            raise ValueError("k values must be >= 1")
        
        method = method.lower()
        if method not in ['elbow', 'silhouette', 'both']:
            raise ValueError("method must be 'elbow', 'silhouette', or 'both'")
        
        # Initialize results storage
        inertias = [] if method in ['elbow', 'both'] else None
        silhouettes = [] if method in ['silhouette', 'both'] else None
        
        print("="*70)
        print("FINDING OPTIMAL NUMBER OF CLUSTERS")
        print("="*70)
        print(f"\nTesting k values: {k_values}")
        print(f"Method: {method}\n")
        
        # Test each k value
        for k in k_values:
            # Create model instance with same parameters but different k
            model_params = self.get_params()
            model_params['n_clusters'] = k
            
            # Create new instance of same class
            temp_model = self.__class__(**model_params)
            temp_model.fit(X)
            
            # Calculate metrics
            if method in ['elbow', 'both']:
                if hasattr(temp_model, 'inertia'):
                    # K-Means has built-in inertia method
                    inertia = temp_model.inertia()
                else:
                    # Use default calculation
                    inertia = temp_model._calculate_inertia(X)
                inertias.append(inertia)
            
            if method in ['silhouette', 'both'] and k >= 2:
                try:
                    score = temp_model.silhouette_score(X, exclude_noise=True)
                    silhouettes.append(score)
                except ValueError:
                    # Less than 2 clusters found (e.g., all noise in DBSCAN)
                    silhouettes.append(-1)
            elif method in ['silhouette', 'both'] and k == 1:
                silhouettes.append(None)
            
            # Print progress
            status = f"k={k:2d}:"
            if inertias:
                status += f" Inertia={inertias[-1]:8.2f}"
            if silhouettes and silhouettes[-1] is not None:
                status += f", Silhouette={silhouettes[-1]:.4f}"
            print(status)
        
        # Determine optimal k
        elbow_k = None
        silhouette_k = None
        
        if method in ['elbow', 'both']:
            elbow_k = self._find_elbow_kneedle(k_values, inertias)
            print(f"\n💡 Elbow method suggests: k={elbow_k}")
        
        if method in ['silhouette', 'both']:
            valid_silhouettes = [s for s in silhouettes if s is not None and s > -1]
            if valid_silhouettes:
                best_idx = [i for i, s in enumerate(silhouettes) if s == max(valid_silhouettes)][0]
                silhouette_k = k_values[best_idx]
                print(f"💡 Silhouette method suggests: k={silhouette_k} (score: {max(valid_silhouettes):.4f})")
        
        # Final recommendation
        if method == 'elbow':
            optimal_k = elbow_k
        elif method == 'silhouette':
            optimal_k = silhouette_k
        else:  # both
            if elbow_k == silhouette_k:
                optimal_k = elbow_k
                print(f"\n✓ Both methods agree: k={optimal_k}")
            else:
                optimal_k = silhouette_k
                print(f"\n⚠️  Methods disagree (Elbow: k={elbow_k}, Silhouette: k={silhouette_k})")
                print(f"   → Recommending k={optimal_k} (Silhouette is more reliable)")
        
        # Visualization
        if show_plot:
            self._plot_optimal_k_results(k_values, inertias, silhouettes, 
                                        elbow_k, silhouette_k, optimal_k, method)
        
        # Return results
        results = {
            'optimal_k': optimal_k,
            'elbow_k': elbow_k,
            'silhouette_k': silhouette_k,
            'k_values': k_values
        }
        if inertias:
            results['inertias'] = inertias
        if silhouettes:
            results['silhouettes'] = silhouettes
        
        return results
    
    def _find_elbow_kneedle(self, k_values, inertias):
        """
        Find elbow using Kneedle algorithm.
        
        Detects the "elbow" point by finding the point with maximum
        perpendicular distance from the line connecting first and last points.
        
        Parameters
        ----------
        k_values : list
            K values tested
        inertias : list
            Inertia values for each k
            
        Returns
        -------
        elbow_k : int
            K value at the elbow
        """
        x = np.array(k_values)
        y = np.array(inertias)
        
        # Normalize to [0, 1]
        x_norm = (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else x
        y_norm = (y - y.min()) / (y.max() - y.min()) if y.max() != y.min() else y
        
        # Line from first to last point
        p1 = np.array([x_norm[0], y_norm[0]])
        p2 = np.array([x_norm[-1], y_norm[-1]])
        
        # Calculate perpendicular distances
        distances = []
        line_vec = p2 - p1
        line_len = np.linalg.norm(line_vec)
        
        if line_len == 0:
            # All inertias are the same, return middle k
            return k_values[len(k_values) // 2]
        
        line_unitvec = line_vec / line_len
        
        for i in range(len(x_norm)):
            point = np.array([x_norm[i], y_norm[i]])
            point_vec = point - p1
            proj_length = np.dot(point_vec, line_unitvec)
            proj_point = p1 + proj_length * line_unitvec
            distance = np.linalg.norm(point - proj_point)
            distances.append(distance)
        
        # Elbow at maximum distance
        elbow_idx = np.argmax(distances)
        return k_values[elbow_idx]
    
    def _plot_optimal_k_results(self, k_values, inertias, silhouettes, 
                                elbow_k, silhouette_k, optimal_k, method):
        """
        Plot results of optimal k analysis.
        
        Parameters
        ----------
        k_values : list
            K values tested
        inertias : list or None
            Inertia values (None if not calculated)
        silhouettes : list or None
            Silhouette scores (None if not calculated)
        elbow_k : int or None
            K suggested by elbow method
        silhouette_k : int or None
            K suggested by silhouette method
        optimal_k : int
            Final recommended k
        method : str
            Method used ('elbow', 'silhouette', or 'both')
        """
        n_plots = sum([inertias is not None, silhouettes is not None])
        
        if n_plots == 0:
            return
        
        fig, axes = plt.subplots(1, n_plots, figsize=(7*n_plots, 5))
        if n_plots == 1:
            axes = [axes]
        
        plot_idx = 0
        
        # Elbow plot
        if inertias is not None:
            ax = axes[plot_idx]
            ax.plot(k_values, inertias, 'bo-', linewidth=2, markersize=8, label='Inertia')
            # Reference line
            ax.plot([k_values[0], k_values[-1]], 
                   [inertias[0], inertias[-1]], 
                   'k--', alpha=0.3, linewidth=1, label='Reference line')
            # Mark elbow
            if elbow_k:
                ax.axvline(x=elbow_k, color='red', linestyle='--', linewidth=2,
                          label=f'Elbow at k={elbow_k}')
                elbow_idx = k_values.index(elbow_k)
                ax.scatter([elbow_k], [inertias[elbow_idx]], 
                          color='red', s=300, zorder=5, marker='X', 
                          edgecolors='black', linewidths=2)
            ax.set_xlabel('Number of Clusters (k)', fontsize=12)
            ax.set_ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=12)
            ax.set_title('Elbow Method', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plot_idx += 1
        
        # Silhouette plot
        if silhouettes is not None:
            ax = axes[plot_idx]
            valid_k = [k for k, s in zip(k_values, silhouettes) if s is not None and s > -1]
            valid_scores = [s for s in silhouettes if s is not None and s > -1]
            
            if valid_scores:
                ax.plot(valid_k, valid_scores, 'go-', linewidth=2, markersize=8)
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.2)
                # Mark best
                if silhouette_k:
                    ax.axvline(x=silhouette_k, color='red', linestyle='--', linewidth=2,
                              label=f'Best k={silhouette_k}')
                    silhouette_idx = valid_k.index(silhouette_k)
                    ax.scatter([silhouette_k], [valid_scores[silhouette_idx]], 
                              color='red', s=300, zorder=5, marker='X',
                              edgecolors='black', linewidths=2)
                ax.set_xlabel('Number of Clusters (k)', fontsize=12)
                ax.set_ylabel('Silhouette Score', fontsize=12)
                ax.set_title('Silhouette Score Analysis', fontsize=14, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def get_params(self):
        """
        Get model parameters.
        
        Returns
        -------
        params : dict
            Model parameters
            
        Notes
        -----
        Subclasses should override this to return model-specific parameters.
        """
        if self._model is None:
            return {}
        return self._model.get_params()