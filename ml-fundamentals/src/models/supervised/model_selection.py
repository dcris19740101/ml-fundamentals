"""
Model selection and validation utilities.

This module provides tools for:
1. Model Validation: Cross-validation techniques to evaluate model performance
2. Hyperparameter Tuning: Grid search to select optimal model parameters

Both are essential components of the model selection process.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from itertools import product


def k_fold_cross_validation(model, X, y, k=5, scoring='auto', random_state=42):
    """
    Perform k-fold cross-validation.
    
    [... docstring as before ...]
    """
    # Convert to numpy arrays to avoid indexing issues
    if isinstance(X, pd.DataFrame):
        X = X.values
    elif isinstance(X, pd.Series):
        X = X.values
    
    if isinstance(y, pd.Series):
        y = y.values
    elif isinstance(y, pd.DataFrame):
        y = y.values.ravel()
    
    # Ensure X is 2D and y is 1D
    X = np.asarray(X)
    y = np.asarray(y).ravel()
    
    # Detect task type
    unique_targets = len(np.unique(y))
    is_classification = unique_targets < 50
    
    # Choose appropriate splitter
    if is_classification:
        kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
        if scoring == 'auto':
            scoring = 'accuracy'
    else:
        kfold = KFold(n_splits=k, shuffle=True, random_state=random_state)
        if scoring == 'auto':
            scoring = 'r2'
    
    # Perform cross-validation
    scores = []
    fold_details = []
    
    print("="*70)
    print(f"K-FOLD CROSS-VALIDATION (k={k})")
    print("="*70)
    print(f"Task: {'Classification' if is_classification else 'Regression'}")
    print(f"Scoring: {scoring}")
    print(f"Total samples: {len(X)}\n")
    
    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate
        if scoring == 'accuracy':
            y_pred = model.predict(X_test)
            score = accuracy_score(y_test, y_pred)
        elif scoring == 'r2':
            y_pred = model.predict(X_test)
            score = r2_score(y_test, y_pred)
        elif scoring == 'neg_mean_squared_error':
            y_pred = model.predict(X_test)
            score = -mean_squared_error(y_test, y_pred)
        else:
            # Use sklearn's cross_val_score for other metrics
            from sklearn.model_selection import cross_val_score
            score = cross_val_score(model._model, X, y, cv=kfold, scoring=scoring)[fold_idx]
        
        scores.append(score)
        fold_details.append({
            'fold': fold_idx + 1,
            'train_size': len(train_idx),
            'test_size': len(test_idx),
            'score': score
        })
        
        print(f"Fold {fold_idx+1}/{k}: {scoring}={score:.4f} "
              f"(train: {len(train_idx)}, test: {len(test_idx)})")
    
    scores = np.array(scores)
    mean_score = scores.mean()
    std_score = scores.std()
    
    print("\n" + "="*70)
    print("CROSS-VALIDATION RESULTS")
    print("="*70)
    print(f"Mean {scoring}: {mean_score:.4f} ± {std_score:.4f}")
    print(f"Min {scoring}:  {scores.min():.4f}")
    print(f"Max {scoring}:  {scores.max():.4f}")
    
    return {
        'scores': scores,
        'mean': mean_score,
        'std': std_score,
        'fold_details': fold_details,
        'scoring': scoring
    }


def plot_cross_validation_results(cv_results, figsize=(10, 6)):
    """
    Visualize cross-validation results.
    
    Parameters
    ----------
    cv_results : dict
        Results from k_fold_cross_validation()
    figsize : tuple, default=(10, 6)
        Figure size
    """
    scores = cv_results['scores']
    mean = cv_results['mean']
    std = cv_results['std']
    scoring = cv_results['scoring']
    k = len(scores)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Scores by fold
    folds = range(1, k + 1)
    ax1.plot(folds, scores, 'bo-', linewidth=2, markersize=8, label='Fold scores')
    ax1.axhline(y=mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean:.3f}')
    ax1.fill_between(folds, mean - std, mean + std, alpha=0.2, color='red', label=f'±1 std: {std:.3f}')
    ax1.set_xlabel('Fold', fontsize=12)
    ax1.set_ylabel(f'{scoring}', fontsize=12)
    ax1.set_title(f'Cross-Validation Scores (k={k})', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(folds)
    
    # Plot 2: Box plot
    ax2.boxplot([scores], labels=['CV Scores'])
    ax2.set_ylabel(f'{scoring}', fontsize=12)
    ax2.set_title('Score Distribution', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()


def grid_search_cv(model_class, param_grid, X, y, cv=5, scoring='auto', 
                   verbose=True, random_state=42):
    """
    Perform grid search with cross-validation to find best hyperparameters.
    
    [... docstring as before ...]
    """
    # Convert to numpy arrays
    if isinstance(X, pd.DataFrame):
        X = X.values
    elif isinstance(X, pd.Series):
        X = X.values
    
    if isinstance(y, pd.Series):
        y = y.values
    elif isinstance(y, pd.DataFrame):
        y = y.values.ravel()
    
    X = np.asarray(X)
    y = np.asarray(y).ravel()
    
    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(product(*param_values))
    
    n_combinations = len(param_combinations)
    
    if verbose:
        print("="*70)
        print("GRID SEARCH WITH CROSS-VALIDATION")
        print("="*70)
        print(f"Parameter grid:")
        for name, values in param_grid.items():
            print(f"  {name}: {values}")
        print(f"\nTotal combinations to test: {n_combinations}")
        print(f"Cross-validation folds: {cv}")
        print(f"Total fits: {n_combinations * cv}\n")
    
    # Detect task type
    unique_targets = len(np.unique(y))
    is_classification = unique_targets < 50
    
    if scoring == 'auto':
        scoring = 'accuracy' if is_classification else 'r2'
    
    # Test all combinations
    all_results = []
    best_score = -np.inf
    best_params = None
    
    for idx, param_values in enumerate(param_combinations):
        # Create parameter dictionary
        params = dict(zip(param_names, param_values))
        
        # Add random_state if model supports it
        if 'random_state' in model_class.__init__.__code__.co_varnames:
            params['random_state'] = random_state
        
        # Create model with these parameters
        model = model_class(**params)
        
        # Perform cross-validation
        cv_results = k_fold_cross_validation(
            model, X, y, k=cv, scoring=scoring, random_state=random_state
        )
        
        mean_score = cv_results['mean']
        std_score = cv_results['std']
        
        # Store results
        result = params.copy()
        result['mean_score'] = mean_score
        result['std_score'] = std_score
        all_results.append(result)
        
        # Update best
        if mean_score > best_score:
            best_score = mean_score
            best_params = params
        
        if verbose:
            print(f"[{idx+1}/{n_combinations}] {params}")
            print(f"  → {scoring}: {mean_score:.4f} ± {std_score:.4f}\n")
    
    # Create DataFrame of results
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('mean_score', ascending=False).reset_index(drop=True)
    
    # Train final model with best parameters
    if 'random_state' in model_class.__init__.__code__.co_varnames:
        best_params['random_state'] = random_state
    best_model = model_class(**best_params)
    best_model.fit(X, y)
    
    if verbose:
        print("="*70)
        print("GRID SEARCH RESULTS")
        print("="*70)
        print(f"\nBest parameters: {best_params}")
        print(f"Best CV {scoring}: {best_score:.4f}")
        print(f"\nTop 5 parameter combinations:")
        print(results_df.head())
    
    return {
        'best_params': best_params,
        'best_score': best_score,
        'all_results': results_df,
        'best_model': best_model,
        'scoring': scoring
    }


def plot_grid_search_results(grid_results, param1, param2=None, figsize=(12, 5)):
    """
    Visualize grid search results.
    
    [... implementation as before ...]
    """
    df = grid_results['all_results']
    scoring = grid_results['scoring']
    
    if param2 is None:
        # 1D plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for param1_val in df[param1].unique():
            subset = df[df[param1] == param1_val]
            ax.errorbar(range(len(subset)), subset['mean_score'], 
                       yerr=subset['std_score'], marker='o', label=f'{param1}={param1_val}')
        
        ax.set_xlabel('Configuration', fontsize=12)
        ax.set_ylabel(f'{scoring}', fontsize=12)
        ax.set_title(f'Grid Search: {param1}', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    else:
        # 2D heatmap
        pivot = df.pivot_table(values='mean_score', index=param2, columns=param1)
        
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(pivot, cmap='RdYlGn', aspect='auto')
        
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_yticks(range(len(pivot.index)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticklabels(pivot.index)
        
        ax.set_xlabel(param1, fontsize=12)
        ax.set_ylabel(param2, fontsize=12)
        ax.set_title(f'Grid Search: {param1} vs {param2}', fontsize=14, fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(scoring, fontsize=12)
        
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                text = ax.text(j, i, f'{pivot.iloc[i, j]:.3f}',
                             ha='center', va='center', color='black', fontsize=10)
    
    plt.tight_layout()
    plt.show()