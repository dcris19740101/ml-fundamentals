"""
PCA (Principal Component Analysis) - Unsupervised dimensionality reduction.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.decomposition import KernelPCA as SklearnKernelPCA
from ..base import DimensionalityReductionModel  # ← Import from root


# ============================================================================
# PCA (Linear)
# ============================================================================

class SKLearnPCA(DimensionalityReductionModel):
    """
    Principal Component Analysis (PCA) using scikit-learn.
    
    PCA is an UNSUPERVISED dimensionality reduction technique that:
    - Finds directions of maximum variance in the data
    - Projects data onto these directions (principal components)
    - Reduces dimensions while preserving maximum variance
    - Does NOT use labels
    
    Common uses:
    - Visualization: Reduce to 2D or 3D for plotting
    - Feature extraction: Remove redundant/correlated features
    - Noise reduction: Keep only high-variance components
    - Speed up training: Fewer features = faster algorithms
    
    Parameters
    ----------
    n_components : int, float, or None, default=None
        Number of components to keep:
        - int: Keep exactly this many components
        - float (0 < n < 1): Keep enough components to explain this much variance
        - None: Keep all components
    random_state : int, optional
        Random seed for reproducibility
        
    Attributes
    ----------
    components_ : array, shape (n_components, n_features)
        Principal components (eigenvectors)
    explained_variance_ : array, shape (n_components,)
        Variance explained by each component
    explained_variance_ratio_ : array, shape (n_components,)
        Percentage of variance explained by each component
    n_components_ : int
        Actual number of components (may differ from n_components parameter)
    
    Examples
    --------
    >>> from src.models.unsupervised import SKLearnPCA
    >>> 
    >>> # Reduce to 2D for visualization
    >>> pca = SKLearnPCA(n_components=2)
    >>> X_2d = pca.fit_transform(X_scaled)  # No y needed!
    >>> plt.scatter(X_2d[:, 0], X_2d[:, 1])
    >>> 
    >>> # Keep 95% of variance
    >>> pca = SKLearnPCA(n_components=0.95)
    >>> X_reduced = pca.fit_transform(X_scaled)
    >>> print(f"Reduced from {X.shape[1]} to {X_reduced.shape[1]} features")
    >>> 
    >>> # See variance explained
    >>> pca.plot_explained_variance()
    
    Notes
    -----
    - ALWAYS scale features before PCA (sensitive to feature scales)
    - PCA is unsupervised (doesn't use labels)
    - Components are orthogonal (uncorrelated)
    - First component captures most variance, second captures second-most, etc.
    - Cannot reduce to more components than original features
    """
    
    def __init__(self, n_components=None, random_state=None):
        super().__init__()
        self.n_components = n_components
        self.random_state = random_state
        self._model = SklearnPCA(n_components=n_components, random_state=random_state)
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.n_components_ = None
    
    def fit(self, X, y=None):
        """
        Fit PCA model.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data (SHOULD BE SCALED!)
        y : ignored
            Not used (PCA is unsupervised)
            
        Returns
        -------
        self : SKLearnPCA
            Fitted model
        """
        self._model.fit(X)
        self.components_ = self._model.components_
        self.explained_variance_ = self._model.explained_variance_
        self.explained_variance_ratio_ = self._model.explained_variance_ratio_
        self.n_components_ = self._model.n_components_
        self.is_fitted = True
        return self
    
    def transform(self, X):
        """
        Apply dimensionality reduction.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data to transform
            
        Returns
        -------
        X_transformed : array, shape (n_samples, n_components)
            Transformed data
        """
        self._check_is_fitted()
        return self._model.transform(X)
    
    def inverse_transform(self, X_transformed):
        """
        Reconstruct original features from transformed data.
        
        Parameters
        ----------
        X_transformed : array-like, shape (n_samples, n_components)
            Transformed data
            
        Returns
        -------
        X_reconstructed : array, shape (n_samples, n_features)
            Reconstructed original features (approximate)
            
        Notes
        -----
        Reconstruction is not exact if n_components < n_features
        (information was lost during dimensionality reduction)
        """
        self._check_is_fitted()
        return self._model.inverse_transform(X_transformed)
    
    def plot_explained_variance(self, figsize=(12, 5)):
        """
        Plot explained variance by component.
        
        Shows both individual and cumulative variance explained.
        Helps determine how many components to keep.
        
        Parameters
        ----------
        figsize : tuple, default=(12, 5)
            Figure size
        """
        self._check_is_fitted()
        
        n_components = len(self.explained_variance_ratio_)
        cumulative_variance = np.cumsum(self.explained_variance_ratio_)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Variance per component
        ax1.bar(range(1, n_components + 1), self.explained_variance_ratio_,
               alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Principal Component', fontsize=12)
        ax1.set_ylabel('Variance Explained', fontsize=12)
        ax1.set_title('Variance Explained by Each Component', 
                     fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Cumulative variance
        ax2.plot(range(1, n_components + 1), cumulative_variance, 
                'bo-', linewidth=2, markersize=8)
        ax2.axhline(y=0.95, color='red', linestyle='--', 
                   label='95% variance threshold')
        ax2.axhline(y=0.90, color='orange', linestyle='--', 
                   label='90% variance threshold')
        ax2.set_xlabel('Number of Components', fontsize=12)
        ax2.set_ylabel('Cumulative Variance Explained', fontsize=12)
        ax2.set_title('Cumulative Variance Explained', 
                     fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print summary
        print("\n" + "="*70)
        print("VARIANCE EXPLAINED SUMMARY")
        print("="*70)
        print(f"\nTotal components: {n_components}")
        print(f"First component: {self.explained_variance_ratio_[0]*100:.1f}%")
        
        if n_components >= 2:
            print(f"First 2 components: {cumulative_variance[1]*100:.1f}%")
        if n_components >= 3:
            print(f"First 3 components: {cumulative_variance[2]*100:.1f}%")
        
        # Find components needed for 90% and 95%
        n_for_90 = np.argmax(cumulative_variance >= 0.90) + 1
        n_for_95 = np.argmax(cumulative_variance >= 0.95) + 1
        
        print(f"\nComponents for 90% variance: {n_for_90}")
        print(f"Components for 95% variance: {n_for_95}")
    

    def plot_2d_projection(self, X, y=None, task='auto', title_suffix=""):
        """
        Plot 2D PCA projection.
        
        For classification: Colors points by class
        For regression: Colors points by continuous target value
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data to project (should be scaled)
        y : array-like, optional
            Target values (class labels or continuous values)
            If None, plots without coloring
        task : str, default='auto'
            Type of task:
            - 'classification': Discrete color map
            - 'regression': Continuous color map
            - 'auto': Detect based on y (unique values < 20 → classification)
        title_suffix : str, optional
            Additional text for title
            
        Examples
        --------
        >>> # For regression
        >>> pca = SKLearnPCA(n_components=2)
        >>> pca.fit(X_train_scaled)
        >>> pca.plot_2d_projection(X_train_scaled, y_train, task='regression')
        >>> 
        >>> # For classification
        >>> pca.plot_2d_projection(X_train_scaled, y_train, task='classification')
        >>> 
        >>> # Auto-detect
        >>> pca.plot_2d_projection(X_train_scaled, y_train)
        
        Notes
        -----
        - For regression: Shows how target values are distributed in PCA space
        - Can reveal clusters of similar values
        - Useful for identifying outliers
        - Colors indicate prediction target magnitude
        """
        self._check_is_fitted()
        
        if self.n_components_ < 2:
            raise ValueError("Need at least 2 components for 2D projection. "
                           f"Current model has {self.n_components_} component(s).")
        
        # Transform data
        X_2d = self.transform(X)
        
        # Detect task type
        if task == 'auto' and y is not None:
            n_unique = len(np.unique(y))
            task = 'classification' if n_unique < 20 else 'regression'
        
        suffix = f" ({title_suffix})" if title_suffix else ""
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if y is None:
            # No target - just plot points
            ax.scatter(X_2d[:, 0], X_2d[:, 1], 
                      alpha=0.6, edgecolors='black', s=50)
            title = f'PCA: 2D Projection{suffix}'
        
        elif task == 'classification':
            # Classification: Discrete colors
            scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y,
                               cmap='viridis', alpha=0.6, edgecolors='black', s=50)
            plt.colorbar(scatter, ax=ax, label='Class')
            title = f'PCA: 2D Projection (Classification){suffix}'
            
            print("\n" + "="*70)
            print("PCA PROJECTION - CLASSIFICATION")
            print("="*70)
            print(f"Number of classes: {len(np.unique(y))}")
            print("💡 Points of same color = same class")
            print("   Look for clusters and separation patterns")
        
        else:
            # Regression: Continuous colors
            scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y,
                               cmap='viridis', alpha=0.6, edgecolors='black', s=50)
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Target Value', fontsize=12)
            title = f'PCA: 2D Projection (Regression){suffix}'
            
            print("\n" + "="*70)
            print("PCA PROJECTION - REGRESSION")
            print("="*70)
            print(f"Target range: [{np.min(y):.2f}, {np.max(y):.2f}]")
            print("💡 Color intensity = target value magnitude")
            print("   • Darker colors = lower values")
            print("   • Lighter colors = higher values")
            print("   Look for patterns: Do similar values cluster together?")
        
        # Labels and formatting
        ax.set_xlabel(f'PC1 ({self.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
        ax.set_ylabel(f'PC2 ({self.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print variance info
        print(f"\nVariance explained:")
        print(f"  PC1: {self.explained_variance_ratio_[0]*100:.1f}%")
        print(f"  PC2: {self.explained_variance_ratio_[1]*100:.1f}%")
        print(f"  Total: {sum(self.explained_variance_ratio_[:2])*100:.1f}%")

    def plot_clusters_2d(self, X, labels, cluster_centers=None, title_suffix="", 
                     highlight_noise=True):
        """
        Plot 2D PCA projection with cluster assignments.
        
        Specifically designed for clustering visualization.
        Shows each cluster in a different color with optional centroids.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data to project (should be scaled)
        labels : array-like, shape (n_samples,)
            Cluster assignments for each point
            Use -1 for noise points (DBSCAN)
        cluster_centers : array-like, optional, shape (n_clusters, n_features)
            Cluster centers (for K-Means)
            If provided, will be projected and plotted
        title_suffix : str, optional
            Additional text for title
        highlight_noise : bool, default=True
            If True, plot noise points (-1) differently
            
        Examples
        --------
        >>> # K-Means
        >>> pca = SKLearnPCA(n_components=2)
        >>> pca.fit(X_scaled)
        >>> pca.plot_clusters_2d(X_scaled, labels, 
        ...                      cluster_centers=kmeans.cluster_centers_)
        >>> 
        >>> # DBSCAN (with noise)
        >>> pca.plot_clusters_2d(X_scaled, labels, highlight_noise=True)
        >>> 
        >>> # Hierarchical
        >>> pca.plot_clusters_2d(X_scaled, labels)
        
        Notes
        -----
        - Each cluster gets a distinct color
        - Noise points (-1) shown in black with 'x' markers
        - Cluster centers (if provided) shown as red stars
        - Automatically handles any number of clusters
        """
        self._check_is_fitted()
        
        if self.n_components_ < 2:
            raise ValueError("Need at least 2 components for 2D projection.")
        
        # Transform data
        X_2d = self.transform(X)
        
        # Get unique clusters
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = np.sum(labels == -1)
        
        suffix = f" ({title_suffix})" if title_suffix else ""
        
        print("\n" + "="*70)
        print("PCA: CLUSTER VISUALIZATION")
        print("="*70)
        print(f"Number of clusters: {n_clusters}")
        if n_noise > 0:
            print(f"Noise points: {n_noise} ({n_noise/len(labels)*100:.1f}%)")
        print(f"PCA explains {sum(self.explained_variance_ratio_[:2])*100:.1f}% of variance")
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Generate colors for each cluster
        if n_clusters > 0:
            colors = plt.cm.tab10(np.linspace(0, 1, max(10, n_clusters)))
        
        # Plot each cluster
        for i, label in enumerate(unique_labels):
            if label == -1 and highlight_noise:
                # Noise points: black X markers
                mask = labels == label
                ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                        c='black', marker='x', s=50, alpha=0.5,
                        label='Noise', linewidths=1)
            else:
                # Regular cluster
                mask = labels == label
                color_idx = label if label != -1 else 0
                
                ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                        c=[colors[color_idx % len(colors)]], 
                        marker='o', s=80, alpha=0.7,
                        edgecolors='black', linewidths=1,
                        label=f'Cluster {label}')
        
        # Plot cluster centers if provided
        if cluster_centers is not None:
            centers_2d = self.transform(cluster_centers)
            ax.scatter(centers_2d[:, 0], centers_2d[:, 1],
                    c='red', marker='*', s=500, 
                    edgecolors='black', linewidths=2,
                    label='Centroids', zorder=10)
        
        # Labels and formatting
        ax.set_xlabel(f'PC1 ({self.explained_variance_ratio_[0]*100:.1f}%)', 
                    fontsize=13, fontweight='bold')
        ax.set_ylabel(f'PC2 ({self.explained_variance_ratio_[1]*100:.1f}%)', 
                    fontsize=13, fontweight='bold')
        
        title = f'PCA: Cluster Visualization'
        if n_noise > 0:
            title += f' ({n_clusters} clusters, {n_noise} noise)'
        else:
            title += f' ({n_clusters} clusters)'
        title += suffix
        
        ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Legend
        # Put legend outside plot if many clusters
        if n_clusters > 5:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                    fontsize=10, frameon=True, shadow=True)
        else:
            ax.legend(loc='best', fontsize=11, frameon=True, shadow=True)
        
        plt.tight_layout()
        plt.show()
        
        print("\n💡 Interpretation:")
        print("   • Each color = different cluster")
        print("   • Points close together = similar in feature space")
        if cluster_centers is not None:
            print("   • Red stars = cluster centroids (K-Means)")
        if n_noise > 0:
            print("   • Black X markers = noise/outliers (not assigned to any cluster)")
    
    def plot_regression_projection_with_predictions(self, X, y_true, y_pred, title_suffix=""):
        """
        Plot 2D PCA projection showing actual vs predicted values for regression.
        
        Shows two plots side-by-side:
        - Left: Colored by actual target values
        - Right: Colored by predicted values
        
        Helps visualize:
        - Where predictions are good (similar colors)
        - Where predictions are poor (different colors)
        - Spatial patterns in errors
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data to project (should be scaled)
        y_true : array-like, shape (n_samples,)
            Actual target values
        y_pred : array-like, shape (n_samples,)
            Predicted target values
        title_suffix : str, optional
            Additional text for title
            
        Examples
        --------
        >>> # After training regression model
        >>> pca = SKLearnPCA(n_components=2)
        >>> pca.fit(X_train_scaled)
        >>> 
        >>> y_pred = model.predict(X_test_scaled)
        >>> pca.plot_regression_projection_with_predictions(
        ...     X_test_scaled, y_test, y_pred
        ... )
        """
        self._check_is_fitted()
        
        if self.n_components_ < 2:
            raise ValueError("Need at least 2 components for 2D projection.")
        
        # Transform data
        X_2d = self.transform(X)
        
        # Calculate errors
        errors = np.abs(y_true - y_pred)
        
        suffix = f" ({title_suffix})" if title_suffix else ""
        
        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Common range for colorbar
        vmin = min(np.min(y_true), np.min(y_pred))
        vmax = max(np.max(y_true), np.max(y_pred))
        
        # Plot 1: Actual values
        scatter1 = axes[0].scatter(X_2d[:, 0], X_2d[:, 1], c=y_true,
                                cmap='viridis', alpha=0.6, edgecolors='black', 
                                s=50, vmin=vmin, vmax=vmax)
        axes[0].set_xlabel(f'PC1 ({self.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
        axes[0].set_ylabel(f'PC2 ({self.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
        axes[0].set_title(f'Actual Values{suffix}', fontsize=13, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        plt.colorbar(scatter1, ax=axes[0], label='Actual')
        
        # Plot 2: Predicted values
        scatter2 = axes[1].scatter(X_2d[:, 0], X_2d[:, 1], c=y_pred,
                                cmap='viridis', alpha=0.6, edgecolors='black', 
                                s=50, vmin=vmin, vmax=vmax)
        axes[1].set_xlabel(f'PC1 ({self.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
        axes[1].set_ylabel(f'PC2 ({self.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
        axes[1].set_title(f'Predicted Values{suffix}', fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=axes[1], label='Predicted')
        
        # Plot 3: Absolute errors
        scatter3 = axes[2].scatter(X_2d[:, 0], X_2d[:, 1], c=errors,
                                cmap='Reds', alpha=0.6, edgecolors='black', s=50)
        axes[2].set_xlabel(f'PC1 ({self.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
        axes[2].set_ylabel(f'PC2 ({self.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
        axes[2].set_title(f'Absolute Errors{suffix}', fontsize=13, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        plt.colorbar(scatter3, ax=axes[2], label='|Error|')
        
        plt.tight_layout()
        plt.show()
        
        # Print statistics
        print("\n" + "="*70)
        print("REGRESSION PROJECTION ANALYSIS")
        print("="*70)
        print(f"\nTarget range: [{vmin:.2f}, {vmax:.2f}]")
        print(f"Mean absolute error: {np.mean(errors):.2f}")
        print(f"Max absolute error: {np.max(errors):.2f}")
        
        print("\n💡 Interpretation:")
        print("   • Similar colors (left vs middle) → Good predictions")
        print("   • Different colors → Prediction errors")
        print("   • Red areas (right plot) → High error regions")
        print("   • Look for spatial patterns: Are errors clustered?")

    def get_params(self):
        """Get PCA parameters."""
        return {
            'n_components': self.n_components,
            'random_state': self.random_state
        }

# ============================================================================
# Kernel PCA (Non-linear)
# ============================================================================

class SKLearnKernelPCA(DimensionalityReductionModel):
    """
    Kernel Principal Component Analysis (Kernel PCA) using scikit-learn.
    
    Kernel PCA is a NON-LINEAR extension of PCA that:
    - Uses kernel trick to capture non-linear patterns
    - Maps data to higher-dimensional space implicitly
    - Finds principal components in that space
    - Can reveal structure that linear PCA misses
    
    Differences from PCA:
    - PCA: Linear dimensionality reduction
    - Kernel PCA: Non-linear dimensionality reduction
    
    Common uses:
    - Data with circular/spiral patterns
    - Non-linear manifolds
    - When linear PCA doesn't separate classes well
    - Feature extraction for non-linear relationships
    
    Parameters
    ----------
    n_components : int, optional
        Number of components to keep
        If None, keeps all components
    kernel : str, default='rbf'
        Kernel type:
        - 'linear': Same as regular PCA
        - 'rbf': Radial basis function (Gaussian) - most common
        - 'poly': Polynomial kernel
        - 'sigmoid': Sigmoid kernel
        - 'cosine': Cosine similarity
    gamma : float, optional
        Kernel coefficient for 'rbf', 'poly', 'sigmoid'
        If None, defaults to 1/n_features
    degree : int, default=3
        Degree for 'poly' kernel
    random_state : int, optional
        Random seed for reproducibility
        
    Attributes
    ----------
    n_components_ : int
        Actual number of components
        
    Examples
    --------
    >>> from src.models.unsupervised import SKLearnKernelPCA
    >>> 
    >>> # RBF kernel (most common)
    >>> kpca = SKLearnKernelPCA(n_components=2, kernel='rbf')
    >>> X_2d = kpca.fit_transform(X_scaled)
    >>> 
    >>> # Compare with linear PCA
    >>> from src.models.unsupervised import SKLearnPCA
    >>> pca = SKLearnPCA(n_components=2)
    >>> X_pca = pca.fit_transform(X_scaled)
    >>> # Kernel PCA often shows better separation for non-linear data
    >>> 
    >>> # Try different kernels
    >>> for kernel in ['linear', 'rbf', 'poly']:
    ...     kpca = SKLearnKernelPCA(n_components=2, kernel=kernel)
    ...     X_transformed = kpca.fit_transform(X_scaled)
    
    Notes
    -----
    - ALWAYS scale features before Kernel PCA
    - RBF kernel is most commonly used
    - Cannot do inverse_transform (non-linear mapping is not invertible)
    - More computationally expensive than linear PCA
    - Does NOT have explained_variance_ratio_ (not meaningful for kernel methods)
    - Eigenvalues/eigenvectors are in kernel space (not directly interpretable)
    
    See Also
    --------
    SKLearnPCA : Linear PCA
    """
    
    def __init__(self, n_components=None, kernel='rbf', gamma=None, 
                 degree=3, random_state=None):
        super().__init__()
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.random_state = random_state
        
        self._model = SklearnKernelPCA(
            n_components=n_components,
            kernel=kernel,
            gamma=gamma,
            degree=degree,
            random_state=random_state,
            fit_inverse_transform=False
        )
        
        self.n_components_ = None  # Only keep this
    
    def fit(self, X, y=None):
        """
        Fit Kernel PCA model.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data (SHOULD BE SCALED!)
        y : ignored
            Not used (Kernel PCA is unsupervised)
            
        Returns
        -------
        self : SKLearnKernelPCA
            Fitted model
        """
        # Convert to numpy if needed
        if hasattr(X, 'values'):
            X = X.values
        
        X = np.asarray(X)
        
        # Fit the model
        self._model.fit(X)
        
        # Determine actual components
        if self.n_components is None:
            # For Kernel PCA, sklearn defaults to keeping all components
            # We get the actual number from the transformed data
            self.n_components_ = self._model.n_components
        else:
            self.n_components_ = self.n_components
        
        self.is_fitted = True
        return self
    
    def transform(self, X):
        """
        Apply Kernel PCA transformation.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data to transform
            
        Returns
        -------
        X_transformed : array, shape (n_samples, n_components)
            Transformed data
        """
        self._check_is_fitted()
        
        # Convert to numpy if needed
        if hasattr(X, 'values'):
            X = X.values
        
        return self._model.transform(X)
    
    def plot_kernel_comparison(self, X, y=None, kernels=None, figsize=(16, 10)):
        """
        Compare different kernel types on the same data.
        
        Shows how different kernels affect the 2D projection.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data to transform (should be scaled)
        y : array-like, optional
            Labels for coloring (if available)
        kernels : list of str, optional
            Kernels to compare.
            Default: ['linear', 'rbf', 'poly', 'sigmoid']
        figsize : tuple, default=(16, 10)
            Figure size
            
        Examples
        --------
        >>> kpca = SKLearnKernelPCA()
        >>> kpca.plot_kernel_comparison(X_scaled, y_train)
        """
        if kernels is None:
            kernels = ['linear', 'rbf', 'poly', 'sigmoid']
        
        # Convert to numpy if needed
        if hasattr(X, 'values'):
            X = X.values
        
        n_kernels = len(kernels)
        n_cols = 2
        n_rows = (n_kernels + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_kernels > 1 else [axes]
        
        print("\n" + "="*70)
        print("KERNEL COMPARISON")
        print("="*70)
        
        for idx, kernel in enumerate(kernels):
            # Fit Kernel PCA with this kernel
            kpca = SKLearnKernelPCA(n_components=2, kernel=kernel, 
                                   random_state=self.random_state)
            X_transformed = kpca.fit_transform(X)
            
            # Plot
            if y is not None:
                scatter = axes[idx].scatter(X_transformed[:, 0], X_transformed[:, 1],
                                           c=y, cmap='viridis', alpha=0.6, 
                                           edgecolors='black', s=50)
                plt.colorbar(scatter, ax=axes[idx], label='Class')
            else:
                axes[idx].scatter(X_transformed[:, 0], X_transformed[:, 1],
                                alpha=0.6, edgecolors='black', s=50)
            
            axes[idx].set_xlabel('Component 1', fontsize=12)
            axes[idx].set_ylabel('Component 2', fontsize=12)
            axes[idx].set_title(f'Kernel: {kernel}', fontsize=13, fontweight='bold')
            axes[idx].grid(True, alpha=0.3)
            
            print(f"\n{kernel.upper()} kernel:")
            print(f"  Transformed shape: {X_transformed.shape}")
        
        # Hide unused subplots
        for idx in range(n_kernels, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print("\n💡 Kernel Guide:")
        print("   • linear: Same as regular PCA (linear relationships)")
        print("   • rbf: Most common, good for non-linear patterns")
        print("   • poly: Polynomial relationships")
        print("   • sigmoid: S-shaped relationships")
    
    def plot_comparison_with_pca(self, X, y=None, pca_model=None, title_suffix=""):
        """
        Compare Kernel PCA with linear PCA.
        
        Shows how Kernel PCA can reveal non-linear structure.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data to transform (should be scaled)
        y : array-like, optional
            Labels for coloring
        pca_model : SKLearnPCA, optional
            Fitted PCA model. If None, creates a new one.
        title_suffix : str, optional
            Additional text for plot titles
        """
        from . import SKLearnPCA
        
        self._check_is_fitted()
        
        # Convert to numpy if needed
        if hasattr(X, 'values'):
            X = X.values
        
        # Create PCA if not provided
        if pca_model is None:
            pca_model = SKLearnPCA(n_components=2)
            pca_model.fit(X)
        
        # Get transformations
        X_pca = pca_model.transform(X)
        X_kpca = self.transform(X)
        
        suffix = f" ({title_suffix})" if title_suffix else ""
        
        # Plot comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Linear PCA
        if y is not None:
            scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y,
                                      cmap='viridis', alpha=0.6, edgecolors='black', s=50)
            plt.colorbar(scatter1, ax=axes[0], label='Class')
        else:
            axes[0].scatter(X_pca[:, 0], X_pca[:, 1],
                          alpha=0.6, edgecolors='black', s=50)
        
        axes[0].set_xlabel(f'PC1 ({pca_model.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
        axes[0].set_ylabel(f'PC2 ({pca_model.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
        axes[0].set_title(f'Linear PCA{suffix}', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Kernel PCA
        if y is not None:
            scatter2 = axes[1].scatter(X_kpca[:, 0], X_kpca[:, 1], c=y,
                                      cmap='viridis', alpha=0.6, edgecolors='black', s=50)
            plt.colorbar(scatter2, ax=axes[1], label='Class')
        else:
            axes[1].scatter(X_kpca[:, 0], X_kpca[:, 1],
                          alpha=0.6, edgecolors='black', s=50)
        
        axes[1].set_xlabel('Kernel PC1', fontsize=12)
        axes[1].set_ylabel('Kernel PC2', fontsize=12)
        axes[1].set_title(f'Kernel PCA ({self.kernel}){suffix}', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print("\n" + "="*70)
        print("PCA vs KERNEL PCA")
        print("="*70)
        print("\n💡 Key Differences:")
        print("   • Linear PCA: Finds linear combinations of features")
        print("   • Kernel PCA: Can capture non-linear relationships")
        print(f"   • Kernel used: {self.kernel}")
        print("\n💡 When to use Kernel PCA:")
        print("   • Data has non-linear patterns (circular, spiral, etc.)")
        print("   • Linear PCA doesn't separate classes well")
        print("   • Visualization of complex manifolds")
    
    def get_params(self):
        """Get Kernel PCA parameters."""
        return {
            'n_components': self.n_components,
            'kernel': self.kernel,
            'gamma': self.gamma,
            'degree': self.degree,
            'random_state': self.random_state
        }