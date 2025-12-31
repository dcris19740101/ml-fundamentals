"""
LDA (Linear Discriminant Analysis) - Supervised dimensionality reduction.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as SklearnLDA
from ..base import DimensionalityReductionModel  # ← Import from root


class SKLearnLDA(DimensionalityReductionModel):
    """
    Linear Discriminant Analysis (LDA) using scikit-learn.
    
    LDA is a SUPERVISED dimensionality reduction technique that:
    - Maximizes separation between classes
    - Minimizes variance within classes
    - Projects data to maximize class discriminability
    - REQUIRES labels
    
    Differences from PCA:
    - PCA: Unsupervised, maximizes variance (doesn't use labels)
    - LDA: Supervised, maximizes class separation (uses labels)
    
    Common uses:
    - Visualization: Reduce to 2D/3D showing class separation
    - Feature extraction: Create discriminative features
    - Preprocessing for classification
    
    Parameters
    ----------
    n_components : int, optional
        Number of components to keep.
        If None or greater than min(n_classes - 1, n_features), 
        will be automatically set to min(n_classes - 1, n_features)
        
    Attributes
    ----------
    explained_variance_ratio_ : array, shape (n_components,)
        Percentage of variance explained by each component
    classes_ : array
        Unique class labels
    n_components_ : int
        Actual number of components fitted (may differ from requested)
        
    Examples
    --------
    >>> from src.models.supervised import SKLearnLDA
    >>> 
    >>> # Request 2 components (will auto-adjust for binary classification)
    >>> lda = SKLearnLDA(n_components=2)
    >>> X_lda = lda.fit_transform(X_scaled, y)
    >>> 
    >>> # Check actual components
    >>> print(f"Requested: 2, Got: {lda.n_components_}")
    >>> # For binary: "Requested: 2, Got: 1"
    >>> # For 3+ classes: "Requested: 2, Got: 2"
    
    Notes
    -----
    - REQUIRES labels (supervised method)
    - Maximum n_components = n_classes - 1
    - For binary classification: max 1 component (auto-adjusted)
    - For 3 classes: max 2 components
    - ALWAYS scale features before LDA
    - LDA assumes Gaussian distribution and equal covariance
    """
    
    def __init__(self, n_components=None):
        super().__init__()
        self.n_components = n_components
        self._model = None  # Will be created in fit()
        self.explained_variance_ratio_ = None
        self.classes_ = None
        self.n_components_ = None
    
    def fit(self, X, y):
        """
        Fit LDA model.
        
        Automatically adjusts n_components if it exceeds the maximum allowed
        (min(n_classes - 1, n_features)).
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data (SHOULD BE SCALED!)
        y : array-like, shape (n_samples,)
            Target labels (REQUIRED for LDA - it's supervised!)
            
        Returns
        -------
        self : SKLearnLDA
            Fitted model
            
        Raises
        ------
        ValueError
            If y is None (LDA requires labels)
        """
        import numpy as np
        
        if y is None:
            raise ValueError(
                "LDA requires labels (y). It's a SUPERVISED dimensionality reduction method.\n"
                "For unsupervised dimensionality reduction, use PCA instead:\n"
                "  from src.models.unsupervised import SKLearnPCA"
            )
        
        # Convert to numpy if needed
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
        
        X = np.asarray(X)
        y = np.asarray(y).ravel()
        
        # Determine maximum possible components
        n_classes = len(np.unique(y))
        n_features = X.shape[1]
        max_components = min(n_classes - 1, n_features)
        
        # Auto-adjust n_components if needed
        if self.n_components is None:
            actual_n_components = max_components
        else:
            actual_n_components = min(self.n_components, max_components)
        
        # Warn if we had to adjust
        if self.n_components is not None and actual_n_components < self.n_components:
            print(f"\n⚠️  LDA Auto-adjustment:")
            print(f"   Requested n_components: {self.n_components}")
            print(f"   Maximum allowed (n_classes - 1): {max_components}")
            print(f"   Using: {actual_n_components} component(s)")
            
            if n_classes == 2:
                print(f"\n💡 Binary classification detected:")
                print(f"   LDA can only produce 1 component for 2 classes")
                print(f"   This finds the single best line to separate classes")
        
        # Create model with adjusted components
        self._model = SklearnLDA(n_components=actual_n_components)
        self._model.fit(X, y)
        
        self.explained_variance_ratio_ = self._model.explained_variance_ratio_
        self.classes_ = self._model.classes_
        self.n_components_ = actual_n_components
        
        self.is_fitted = True
        return self
    
    def transform(self, X):
        """
        Apply LDA transformation.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data to transform
            
        Returns
        -------
        X_transformed : array, shape (n_samples, n_components_)
            Transformed data (n_components_ may be less than requested n_components)
        """
        self._check_is_fitted()
        
        # Convert to numpy if needed
        if hasattr(X, 'values'):
            X = X.values
        
        return self._model.transform(X)
    
    def plot_explained_variance(self, figsize=(12, 5)):
        """
        Plot variance explained by each LDA component.
        
        Parameters
        ----------
        figsize : tuple, default=(12, 5)
            Figure size
        """
        self._check_is_fitted()
        
        n_components = len(self.explained_variance_ratio_)
        cumulative_variance = np.cumsum(self.explained_variance_ratio_)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Individual variance
        ax1.bar(range(1, n_components + 1), self.explained_variance_ratio_,
               alpha=0.7, edgecolor='black', color='steelblue')
        ax1.set_xlabel('Linear Discriminant', fontsize=12)
        ax1.set_ylabel('Variance Explained', fontsize=12)
        ax1.set_title('Variance Explained by Each Discriminant', 
                     fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Cumulative variance
        ax2.plot(range(1, n_components + 1), cumulative_variance,
                'go-', linewidth=2, markersize=8)
        ax2.set_xlabel('Number of Discriminants', fontsize=12)
        ax2.set_ylabel('Cumulative Variance Explained', fontsize=12)
        ax2.set_title('Cumulative Variance Explained', 
                     fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print summary
        print("\n" + "="*70)
        print("LDA VARIANCE EXPLAINED")
        print("="*70)
        print(f"\nNumber of classes: {len(self.classes_)}")
        print(f"Maximum possible components: {len(self.classes_) - 1}")
        print(f"Actual components: {n_components}")
        
        for i, var in enumerate(self.explained_variance_ratio_):
            print(f"\nLD{i+1}: {var*100:.1f}% variance")
        
        print(f"\nTotal variance explained: {cumulative_variance[-1]*100:.1f}%")
    
    def plot_comparison_with_pca(self, X, y, pca_model=None, title_suffix=""):
        """
        Compare LDA with PCA (linear or kernel) visualization.
        
        Automatically creates PCA model if not provided and shows appropriate
        visualization based on number of classes (binary vs multi-class).
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training features (already preprocessed/scaled)
        y : array-like, shape (n_samples,)
            Training labels
        pca_model : SKLearnPCA or SKLearnKernelPCA, optional
            Fitted PCA model with n_components=2.
            Can be either linear PCA or Kernel PCA.
            If None, creates and fits a new linear PCA model.
        title_suffix : str, optional
            Additional text to add to plot titles
            
        Examples
        --------
        >>> # Default: Compare with linear PCA (auto-created)
        >>> lda = SKLearnLDA(n_components=2)
        >>> lda.fit(X_train_proc, y_train)
        >>> lda.plot_comparison_with_pca(X_train_proc, y_train)
        >>> 
        >>> # Compare with linear PCA (explicit)
        >>> from src.models.unsupervised import SKLearnPCA
        >>> pca = SKLearnPCA(n_components=2)
        >>> pca.fit(X_train_proc)
        >>> lda.plot_comparison_with_pca(X_train_proc, y_train, pca_model=pca)
        >>> 
        >>> # Compare with Kernel PCA (RBF)
        >>> from src.models.unsupervised import SKLearnKernelPCA
        >>> kpca = SKLearnKernelPCA(n_components=2, kernel='rbf')
        >>> kpca.fit(X_train_proc)
        >>> lda.plot_comparison_with_pca(X_train_proc, y_train, pca_model=kpca)
        
        Notes
        -----
        - Automatically detects binary vs multi-class
        - Binary (2 classes): Shows scatter + histograms
        - Multi-class (3+ classes): Shows 2D scatter plots
        - Works with both linear PCA and Kernel PCA
        """
        from ..unsupervised import SKLearnPCA, SKLearnKernelPCA
        
        self._check_is_fitted()
        
        # Create linear PCA if not provided
        if pca_model is None:
            pca_model = SKLearnPCA(n_components=2)
            pca_model.fit(X)
        
        # Determine PCA type for titles
        is_kernel_pca = isinstance(pca_model, SKLearnKernelPCA)
        if is_kernel_pca:
            pca_title = f"Kernel PCA ({pca_model.kernel})"
            pca_label = "KPC"  # Kernel PC
        else:
            pca_title = "PCA"
            pca_label = "PC"
        
        # Get transformations
        X_pca = pca_model.transform(X)
        X_lda = self.transform(X)
        
        n_classes = len(np.unique(y))
        
        # Add suffix to titles if provided
        suffix = f" ({title_suffix})" if title_suffix else ""
        
        print("\n" + "="*70)
        print(f"{pca_title.upper()} vs LDA VISUALIZATION")
        print("="*70)
        print(f"Number of classes: {n_classes}")
        print(f"PCA type: {'Kernel PCA' if is_kernel_pca else 'Linear PCA'}")
        if is_kernel_pca:
            print(f"Kernel: {pca_model.kernel}")
        print(f"PCA components: {pca_model.n_components_}")
        print(f"LDA components: {self.n_components_}")
        
        if self.n_components_ == 1:
            # Binary classification
            self._plot_binary_comparison(X_pca, X_lda, y, pca_model, 
                                        suffix, pca_title, pca_label, is_kernel_pca)
        else:
            # Multi-class
            self._plot_multiclass_comparison(X_pca, X_lda, y, pca_model, 
                                            suffix, pca_title, pca_label, is_kernel_pca)
        
        # Print variance explained (only for linear PCA)
        print("\n" + "="*70)
        print("VARIANCE EXPLAINED")
        print("="*70)
        
        if not is_kernel_pca:
            # Linear PCA has explained_variance_ratio_
            print(f"\nPCA: {sum(pca_model.explained_variance_ratio_)*100:.1f}% total")
            for i, var in enumerate(pca_model.explained_variance_ratio_):
                print(f"  PC{i+1}: {var*100:.1f}%")
        else:
            # Kernel PCA doesn't have interpretable variance ratios
            print(f"\nKernel PCA ({pca_model.kernel} kernel):")
            print(f"  Note: Variance ratios not available for Kernel PCA")
            print(f"  Components capture non-linear patterns in feature space")
        
        print(f"\nLDA: {sum(self.explained_variance_ratio_)*100:.1f}% total")
        for i, var in enumerate(self.explained_variance_ratio_):
            print(f"  LD{i+1}: {var*100:.1f}%")

    def _plot_binary_comparison(self, X_pca, X_lda, y, pca_model, suffix, 
                                pca_title, pca_label, is_kernel_pca):
        """Plot for binary classification (1D LDA)."""
        import matplotlib.pyplot as plt
        
        print("\n💡 Binary classification: Showing scatter plots + histograms")
        
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # PCA 2D scatter
        ax1 = fig.add_subplot(gs[0, 0])
        scatter1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=y,
                            cmap='viridis', alpha=0.6, edgecolors='black', s=50)
        
        # Labels depend on PCA type
        if is_kernel_pca:
            ax1.set_xlabel(f'{pca_label}1', fontsize=12)
            ax1.set_ylabel(f'{pca_label}2', fontsize=12)
        else:
            ax1.set_xlabel(f'{pca_label}1 ({pca_model.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
            ax1.set_ylabel(f'{pca_label}2 ({pca_model.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
        
        ax1.set_title(f'{pca_title}: 2D Projection{suffix}', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter1, ax=ax1, label='Class')
        
        # LDA 1D scatter with jitter
        ax2 = fig.add_subplot(gs[0, 1])
        np.random.seed(42)
        jitter = np.random.normal(0, 0.02, len(X_lda))
        scatter2 = ax2.scatter(X_lda, jitter, c=y,
                            cmap='viridis', alpha=0.6, edgecolors='black', s=50)
        ax2.set_xlabel(f'LD1 ({self.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
        ax2.set_ylabel('Random Jitter (for visibility)', fontsize=12)
        ax2.set_title(f'LDA: 1D Projection (Binary){suffix}', fontsize=13, fontweight='bold')
        ax2.set_ylim(-0.2, 0.2)
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=ax2, label='Class')
        
        # PCA histogram
        ax3 = fig.add_subplot(gs[1, 0])
        for cls in np.unique(y):
            mask = y == cls
            ax3.hist(X_pca[mask, 0], bins=20, alpha=0.6,
                    label=f'Class {int(cls)}', edgecolor='black')
        ax3.set_xlabel(f'{pca_label}1 Value', fontsize=12)
        ax3.set_ylabel('Frequency', fontsize=12)
        ax3.set_title(f'{pca_title}: {pca_label}1 Distribution by Class', fontsize=13, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # LDA histogram
        ax4 = fig.add_subplot(gs[1, 1])
        for cls in np.unique(y):
            mask = y == cls
            ax4.hist(X_lda[mask].ravel(), bins=20, alpha=0.6,
                    label=f'Class {int(cls)}', edgecolor='black')
        ax4.set_xlabel(f'LD1 Value', fontsize=12)
        ax4.set_ylabel('Frequency', fontsize=12)
        ax4.set_title('LDA: LD1 Distribution by Class', fontsize=13, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.show()
        
        print("\n" + "="*70)
        print("KEY INSIGHTS (Binary Classification)")
        print("="*70)
        print("\n💡 Why LDA only has 1 component:")
        print("   • For 2 classes, LDA can produce at most n_classes - 1 = 1 component")
        print("   • This finds the single best line to separate the two classes")
        print("\n💡 Notice in the histograms:")
        print("   • LDA (bottom right) shows better separation")
        print(f"   • {pca_title}: {'Non-linear patterns' if is_kernel_pca else 'Linear variance maximization'}")
        print("   • LDA: Supervised - maximizes class separation using labels")

    def _plot_multiclass_comparison(self, X_pca, X_lda, y, pca_model, suffix, 
                                    pca_title, pca_label, is_kernel_pca):
        """Plot for multi-class classification (2D LDA)."""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # PCA 2D scatter
        scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y,
                                cmap='viridis', alpha=0.6, edgecolors='black', s=50)
        
        # Labels depend on PCA type
        if is_kernel_pca:
            axes[0].set_xlabel(f'{pca_label}1', fontsize=12)
            axes[0].set_ylabel(f'{pca_label}2', fontsize=12)
            axes[0].set_title(f'{pca_title} (Unsupervised)\nNon-linear Patterns{suffix}', 
                            fontsize=14, fontweight='bold')
        else:
            axes[0].set_xlabel(f'{pca_label}1 ({pca_model.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
            axes[0].set_ylabel(f'{pca_label}2 ({pca_model.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
            axes[0].set_title(f'{pca_title} (Unsupervised)\nMaximizes Variance{suffix}', 
                            fontsize=14, fontweight='bold')
        
        axes[0].grid(True, alpha=0.3)
        plt.colorbar(scatter1, ax=axes[0], label='Class')
        
        # LDA 2D scatter
        scatter2 = axes[1].scatter(X_lda[:, 0], X_lda[:, 1], c=y,
                                cmap='viridis', alpha=0.6, edgecolors='black', s=50)
        axes[1].set_xlabel(f'LD1 ({self.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
        axes[1].set_ylabel(f'LD2 ({self.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
        axes[1].set_title(f'LDA (Supervised)\nMaximizes Class Separation{suffix}', 
                        fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=axes[1], label='Class')
        
        plt.tight_layout()
        plt.show()
        
        print("\n" + "="*70)
        print("KEY INSIGHTS (Multi-class Classification)")
        print("="*70)
        print(f"\n💡 {pca_title} vs LDA:")
        if is_kernel_pca:
            print(f"   • Kernel PCA: Unsupervised - captures non-linear patterns ({pca_model.kernel} kernel)")
        else:
            print("   • PCA: Unsupervised - maximizes variance (ignores labels)")
        print("   • LDA: Supervised - maximizes class separation (uses labels)")
        print("\n💡 Notice:")
        print("   • LDA typically shows clearer class separation")
        print("   • LDA finds the directions that best distinguish between classes")

    def get_params(self):
        """Get LDA parameters."""
        return {
            'n_components': self.n_components  # Requested (not actual)
        }