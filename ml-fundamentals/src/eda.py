"""
Exploratory Data Analysis utilities.

Functions for understanding your data before preprocessing and model selection.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# src/eda.py

def detect_task_type(y, threshold_unique_ratio=0.05, threshold_unique_count=20):
    """
    Automatically detect if a dataset is suited for regression or classification.
    
    Uses heuristics based on target variable characteristics:
    - Few unique values + discrete → Classification
    - Many unique values + continuous → Regression
    
    Parameters
    ----------
    y : array-like
        Target variable
    threshold_unique_ratio : float, default=0.05
        If unique_values/total_samples < threshold → likely classification
    threshold_unique_count : int, default=20
        If unique_values < threshold → likely classification
        
    Returns
    -------
    task_info : dict
        Dictionary containing:
        - task_type: 'classification' or 'regression'
        - confidence: 'high', 'medium', or 'low'
        - reasoning: explanation of the decision
        - n_unique: number of unique values
        - unique_ratio: ratio of unique values to total samples
        - dtype: data type of target
        - is_integer: whether all values are integers
        
    Examples
    --------
    >>> # Classification example
    >>> y_class = np.array([0, 1, 0, 1, 2, 1, 0, 2])
    >>> info = detect_task_type(y_class)
    >>> print(info['task_type'])
    'classification'
    
    >>> # Regression example
    >>> y_reg = np.array([1.5, 2.3, 3.7, 4.2, 5.1])
    >>> info = detect_task_type(y_reg)
    >>> print(info['task_type'])
    'regression'
    """
    import pandas as pd
    import numpy as np
    
    # Convert to numpy array if pandas Series
    if isinstance(y, pd.Series):
        y_array = y.values
    else:
        y_array = np.array(y)
    
    # Get basic statistics
    n_samples = len(y_array)
    unique_values = np.unique(y_array)
    n_unique = len(unique_values)
    unique_ratio = n_unique / n_samples
    
    # Check data type
    dtype = y_array.dtype
    is_numeric = np.issubdtype(dtype, np.number)
    is_integer = np.all(y_array == y_array.astype(int)) if is_numeric else False
    
    # Initialize result
    task_info = {
        'task_type': None,
        'confidence': None,
        'reasoning': [],
        'n_unique': n_unique,
        'unique_ratio': unique_ratio,
        'dtype': str(dtype),
        'is_integer': is_integer,
        'unique_values': unique_values if n_unique <= 10 else None,
        'value_counts': None
    }
    
    # Get value distribution if categorical
    if n_unique <= 20:
        unique, counts = np.unique(y_array, return_counts=True)
        task_info['value_counts'] = dict(zip(unique, counts))
    
    # Decision logic
    reasoning = []
    classification_score = 0
    regression_score = 0
    
    # Rule 1: Check if non-numeric (strings, objects)
    if not is_numeric:
        classification_score += 10
        reasoning.append(f"Target is non-numeric (dtype: {dtype}) → Classification")
    
    # Rule 2: Very few unique values
    if n_unique <= 10:
        classification_score += 5
        reasoning.append(f"Only {n_unique} unique values → Classification")
    elif n_unique > threshold_unique_count:
        regression_score += 3
        reasoning.append(f"{n_unique} unique values → Likely regression")
    
    # Rule 3: Unique ratio
    if unique_ratio < threshold_unique_ratio:
        classification_score += 4
        reasoning.append(f"Low unique ratio ({unique_ratio:.1%}) → Classification")
    elif unique_ratio > 0.9:
        regression_score += 4
        reasoning.append(f"High unique ratio ({unique_ratio:.1%}) → Regression")
    
    # Rule 4: Integer vs float
    if is_numeric:
        if is_integer and n_unique <= threshold_unique_count:
            classification_score += 3
            reasoning.append(f"Integer values with {n_unique} classes → Classification")
        elif not is_integer:
            regression_score += 3
            reasoning.append(f"Continuous (float) values → Regression")
    
    # Rule 5: Check distribution pattern
    if is_numeric and n_unique > 10:
        # Check if values are evenly distributed (continuous) or clustered (categorical)
        if n_unique > 50:
            sorted_vals = np.sort(unique_values)
            gaps = np.diff(sorted_vals)
            avg_gap = np.mean(gaps)
            max_gap = np.max(gaps)
            
            if max_gap > 10 * avg_gap:
                classification_score += 2
                reasoning.append(f"Large gaps in values → Might be categorical")
            else:
                regression_score += 2
                reasoning.append(f"Uniformly distributed values → Continuous regression")
    
    # Rule 6: Binary target (special case)
    if n_unique == 2:
        classification_score += 5
        vals = sorted(unique_values)
        reasoning.append(f"Binary target {vals} → Binary classification")
    
    # Make decision
    if classification_score > regression_score:
        task_info['task_type'] = 'classification'
        score_diff = classification_score - regression_score
        if score_diff > 7:
            task_info['confidence'] = 'high'
        elif score_diff > 3:
            task_info['confidence'] = 'medium'
        else:
            task_info['confidence'] = 'low'
    else:
        task_info['task_type'] = 'regression'
        score_diff = regression_score - classification_score
        if score_diff > 5:
            task_info['confidence'] = 'high'
        elif score_diff > 2:
            task_info['confidence'] = 'medium'
        else:
            task_info['confidence'] = 'low'
    
    task_info['reasoning'] = reasoning
    task_info['classification_score'] = classification_score
    task_info['regression_score'] = regression_score
    
    return task_info


def print_task_detection_report(task_info):
    """
    Print a formatted report of task type detection.
    
    Parameters
    ----------
    task_info : dict
        Output from detect_task_type()
    """
    print("\n" + "="*70)
    print("AUTOMATIC TASK TYPE DETECTION")
    print("="*70)
    
    print(f"\n📊 Target Variable Analysis:")
    print(f"  - Data type: {task_info['dtype']}")
    print(f"  - Total samples: {task_info['n_unique']} unique / ? total")
    print(f"  - Unique values: {task_info['n_unique']}")
    print(f"  - Unique ratio: {task_info['unique_ratio']:.1%}")
    
    if task_info['is_integer']:
        print(f"  - Value type: Integer")
    else:
        print(f"  - Value type: Continuous (float)")
    
    if task_info['unique_values'] is not None:
        print(f"  - Unique values: {list(task_info['unique_values'])}")
    
    if task_info['value_counts'] is not None:
        print(f"\n  Value distribution:")
        for val, count in sorted(task_info['value_counts'].items()):
            print(f"    {val}: {count} samples")
    
    print(f"\n🎯 Detection Result:")
    print(f"  - Task Type: {task_info['task_type'].upper()}")
    print(f"  - Confidence: {task_info['confidence'].upper()}")
    
    confidence_emoji = {
        'high': '✅',
        'medium': '⚠️',
        'low': '❓'
    }
    emoji = confidence_emoji.get(task_info['confidence'], '❓')
    print(f"  {emoji} Confidence Level: {task_info['confidence']}")
    
    print(f"\n💡 Reasoning:")
    for reason in task_info['reasoning']:
        print(f"  • {reason}")
    
    print(f"\n📈 Scores:")
    print(f"  - Classification score: {task_info['classification_score']}")
    print(f"  - Regression score: {task_info['regression_score']}")
    
    if task_info['confidence'] == 'low':
        print(f"\n⚠️  WARNING: Low confidence detection!")
        print(f"     Please manually verify the task type.")
        print(f"     Consider the domain: What are you trying to predict?")
    
    print("="*70)

def analyze_data(X, show_stats=True):
    """
    Comprehensive data analysis showing types, missing values, and column separation.
    
    Parameters
    ----------
    X : DataFrame
        Input features to analyze
    show_stats : bool, default=True
        Whether to print basic statistics
        
    Returns
    -------
    analysis : dict
        Dictionary containing:
        - numerical_cols: list of numerical column names
        - categorical_cols: list of categorical column names
        - missing_summary: DataFrame with missing value info
        
    Examples
    --------
    >>> analysis = analyze_data(X)
    >>> print(f"Found {len(analysis['numerical_cols'])} numerical features")
    >>> print(f"Found {len(analysis['categorical_cols'])} categorical features")
    """
    print("=" * 70)
    print("DATA ANALYSIS")
    print("=" * 70)
    
    # 1. Basic info
    print(f"\nDataset shape: {X.shape[0]} rows × {X.shape[1]} columns")
    
    # 2. Data types
    print("\nData types:")
    print(X.dtypes)
    
    # 3. Missing values analysis
    missing = X.isnull().sum()
    missing_pct = (missing / len(X) * 100).round(2)
    
    missing_df = pd.DataFrame({
        'Missing_Count': missing,
        'Missing_Percent': missing_pct
    })
    missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values(
        'Missing_Percent', ascending=False
    )
    
    if len(missing_df) > 0:
        print("\nMissing values:")
        print(missing_df)
    else:
        print("\n✓ No missing values found")
    
    # 4. Separate column types
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"\nNumerical columns ({len(numerical_cols)}):")
    print(numerical_cols)
    
    print(f"\nCategorical columns ({len(categorical_cols)}):")
    print(categorical_cols)
    
    # 5. Basic statistics (optional)
    if show_stats and len(numerical_cols) > 0:
        print("\nNumerical features statistics:")
        print(X[numerical_cols].describe())
    
    print("=" * 70)
    
    return {
        'numerical_cols': numerical_cols,
        'categorical_cols': categorical_cols,
        'missing_summary': missing_df
    }


def detect_outliers_iqr(X, columns=None, threshold=1.5):
    """
    Detect outliers using IQR method.
    
    Outliers are values below Q1 - threshold*IQR or above Q3 + threshold*IQR.
    
    Purpose:
    - Identify extreme values
    - Decide: keep (legitimate), remove (errors), or transform (log scale)
    - Inform scaling choice (StandardScaler preserves outliers, MinMaxScaler squashes them)
    
    Parameters
    ----------
    X : DataFrame
        Input data
    columns : list, optional
        Columns to check (if None, check all numerical columns)
    threshold : float, default=1.5
        IQR multiplier
        - 1.5: Standard (moderate sensitivity)
        - 3.0: Conservative (only extreme outliers)
        
    Returns
    -------
    outliers : dict
        Dictionary mapping column names to boolean arrays (True = outlier)
        
    Examples
    --------
    >>> outliers = detect_outliers_iqr(X)
    >>> for col, mask in outliers.items():
    ...     print(f"{col}: {mask.sum()} outliers ({mask.sum()/len(X)*100:.1f}%)")
    >>> 
    >>> # Remove outliers if needed
    >>> X_clean = X[~outliers['MedInc']]
    """
    if columns is None:
        columns = X.select_dtypes(include=['int64', 'float64']).columns
    
    outliers = {}
    
    print("\n" + "="*70)
    print("OUTLIER DETECTION (IQR Method)")
    print("="*70)
    
    for col in columns:
        Q1 = X[col].quantile(0.25)
        Q3 = X[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        outlier_mask = (X[col] < lower_bound) | (X[col] > upper_bound)
        
        if outlier_mask.sum() > 0:
            outliers[col] = outlier_mask
            pct = outlier_mask.sum() / len(X) * 100
            print(f"{col:20s}: {outlier_mask.sum():4d} outliers ({pct:5.1f}%) "
                  f"[< {lower_bound:.2f} or > {upper_bound:.2f}]")
    
    if len(outliers) == 0:
        print("✓ No outliers detected")
    
    print("="*70)
    
    return outliers


def plot_distributions(X, numerical_cols=None, categorical_cols=None, figsize=(15, 10)):
    """
    Plot distributions of numerical and categorical features.
    
    Purpose:
    - See distribution shape (normal, skewed, bimodal)
    - Identify outliers visually (long tails)
    - Decide on transformations (log for skewed data)
    
    Note: This shows MARGINAL distributions (each feature alone).
    Use plot_feature_target_relationships() to see relationships with target.
    
    Parameters
    ----------
    X : DataFrame
        Input data
    numerical_cols : list, optional
        Numerical columns to plot
    categorical_cols : list, optional
        Categorical columns to plot
    figsize : tuple, default=(15, 10)
        Figure size
        
    Examples
    --------
    >>> plot_distributions(X)
    >>> # Look for: normal shape, skewness, outliers
    """
    if numerical_cols is None:
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    if categorical_cols is None:
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    n_numerical = len(numerical_cols)
    n_categorical = len(categorical_cols)
    
    if n_numerical > 0:
        # Plot numerical distributions
        n_cols = min(3, n_numerical)
        n_rows = (n_numerical + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_numerical > 1 else [axes]
        
        for idx, col in enumerate(numerical_cols):
            axes[idx].hist(X[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
            axes[idx].set_title(f'{col} Distribution')
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('Frequency')
            axes[idx].grid(True, alpha=0.3)
        
        # Hide extra subplots
        for idx in range(n_numerical, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Feature Distributions (Marginal)', fontsize=14, y=1.00)
        plt.tight_layout()
        plt.show()
    
    if n_categorical > 0:
        # Plot categorical distributions
        n_cols = min(3, n_categorical)
        n_rows = (n_categorical + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_categorical > 1 else [axes]
        
        for idx, col in enumerate(categorical_cols):
            value_counts = X[col].value_counts()
            axes[idx].bar(range(len(value_counts)), value_counts.values)
            axes[idx].set_title(f'{col} Distribution')
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('Count')
            axes[idx].set_xticks(range(len(value_counts)))
            axes[idx].set_xticklabels(value_counts.index, rotation=45, ha='right')
            axes[idx].grid(True, alpha=0.3, axis='y')
        
        # Hide extra subplots
        for idx in range(n_categorical, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.show()


def plot_feature_target_relationships(X, y, numerical_cols=None, figsize=(15, 10)):
    """
    Plot scatter plots of each feature vs target variable.
    
    THIS IS WHAT YOU WANT for model selection!
    
    Helps identify:
    - Linear relationships → Linear Regression will work well
    - Non-linear relationships → Need polynomial features or tree-based models
    - No relationship → Feature not useful, consider removing
    
    Parameters
    ----------
    X : DataFrame
        Input features
    y : array-like
        Target variable
    numerical_cols : list, optional
        Numerical columns to plot (if None, use all numerical)
    figsize : tuple, default=(15, 10)
        Figure size
        
    Examples
    --------
    >>> from src.eda import plot_feature_target_relationships
    >>> plot_feature_target_relationships(X, y)
    >>> # Look for linear trends, curved patterns, or no pattern
    """
    if numerical_cols is None:
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    n_features = len(numerical_cols)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for idx, col in enumerate(numerical_cols):
        ax = axes[idx]
        
        # Scatter plot
        ax.scatter(X[col], y, alpha=0.3, s=10)
        
        # Calculate correlation
        try:
            correlation, p_value = pearsonr(X[col].dropna(), y)
            
            # Add trend line
            z = np.polyfit(X[col].dropna(), y, 1)
            p = np.poly1d(z)
            x_sorted = X[col].dropna().sort_values()
            ax.plot(x_sorted, p(x_sorted), "r--", alpha=0.8, linewidth=2, label='Linear fit')
            
            # Interpret correlation
            if abs(correlation) > 0.7:
                strength = "Strong"
                color = "green"
            elif abs(correlation) > 0.4:
                strength = "Moderate"
                color = "orange"
            else:
                strength = "Weak"
                color = "red"
            
            ax.set_title(f'{col} vs Target\nCorr: {correlation:.3f} ({strength})', 
                        color=color, fontweight='bold')
        except:
            ax.set_title(f'{col} vs Target\nCorr: N/A')
        
        ax.set_xlabel(col)
        ax.set_ylabel('Target')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Hide extra subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Feature-Target Relationships (for Model Selection)', fontsize=14, y=1.00)
    plt.tight_layout()
    plt.show()


def analyze_feature_importance_for_model_selection(X, y):
    """
    Analyze features to suggest which models might work well.
    
    THIS HELPS YOU CHOOSE THE RIGHT MODEL!
    
    Returns recommendations based on:
    - Linearity of relationships (correlation with target)
    - Number of features (regularization needed?)
    - Presence of weak relationships (trees better?)
    
    Parameters
    ----------
    X : DataFrame
        Input features
    y : array-like
        Target variable
        
    Returns
    -------
    recommendations : dict
        Dictionary with model recommendations and reasoning
        
    Examples
    --------
    >>> recs = analyze_feature_importance_for_model_selection(X, y)
    >>> print("Recommended models:", recs['recommended_models'])
    >>> for reason in recs['reasoning']:
    ...     print(reason)
    """
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Calculate correlations
    correlations = {}
    for col in numerical_cols:
        try:
            corr, p_value = pearsonr(X[col].dropna(), y)
            correlations[col] = {'correlation': corr, 'p_value': p_value}
        except:
            correlations[col] = {'correlation': 0, 'p_value': 1}
    
    # Sort by absolute correlation
    sorted_features = sorted(correlations.items(), 
                            key=lambda x: abs(x[1]['correlation']), 
                            reverse=True)
    
    print("="*70)
    print("FEATURE-TARGET RELATIONSHIP ANALYSIS")
    print("="*70)
    
    print("\nFeature Correlations with Target (sorted by strength):")
    print("-"*70)
    for feature, stats in sorted_features:
        corr = stats['correlation']
        p_val = stats['p_value']
        
        if abs(corr) > 0.7:
            strength = "Strong"
        elif abs(corr) > 0.4:
            strength = "Moderate"
        else:
            strength = "Weak"
        
        significant = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        print(f"{feature:20s}: {corr:7.3f} ({strength:8s}) {significant}")
    
    # Count correlations by strength
    strong_linear = sum(1 for _, stats in correlations.items() if abs(stats['correlation']) > 0.7)
    moderate_linear = sum(1 for _, stats in correlations.items() if 0.4 < abs(stats['correlation']) <= 0.7)
    weak_linear = sum(1 for _, stats in correlations.items() if abs(stats['correlation']) <= 0.4)
    
    print("\n" + "="*70)
    print("MODEL SELECTION RECOMMENDATIONS")
    print("="*70)
    
    recommendations = {
        'strong_linear_features': strong_linear,
        'moderate_linear_features': moderate_linear,
        'weak_linear_features': weak_linear,
        'total_features': len(numerical_cols),
        'recommended_models': [],
        'reasoning': []
    }
    
    # Linear models
    if strong_linear >= 2 or (strong_linear >= 1 and moderate_linear >= 2):
        recommendations['recommended_models'].append('Linear Regression')
        recommendations['reasoning'].append(
            f"✅ Linear Regression: {strong_linear} strong + {moderate_linear} moderate linear relationships detected"
        )
        
        if len(numerical_cols) > 10:
            recommendations['recommended_models'].extend(['Ridge Regression', 'Lasso Regression'])
            recommendations['reasoning'].append(
                f"✅ Ridge/Lasso: {len(numerical_cols)} features → regularization recommended to prevent overfitting"
            )
    else:
        recommendations['reasoning'].append(
            f"⚠️  Linear Regression: Only {strong_linear} strong linear relationships → may underperform"
        )
    
    # Tree-based models
    if weak_linear >= moderate_linear or strong_linear < 3:
        recommendations['recommended_models'].extend(['Decision Tree', 'Random Forest'])
        recommendations['reasoning'].append(
            f"✅ Tree-based models: {weak_linear} features with weak linearity → trees capture non-linear patterns"
        )
    
    # Always recommend Random Forest (robust baseline)
    if 'Random Forest' not in recommendations['recommended_models']:
        recommendations['recommended_models'].append('Random Forest')
        recommendations['reasoning'].append(
            "✅ Random Forest: Robust baseline for any regression problem"
        )
    
    # Neural networks
    if len(numerical_cols) > 20 or (strong_linear < 3 and weak_linear > 5):
        recommendations['recommended_models'].append('Neural Network')
        recommendations['reasoning'].append(
            f"✅ Neural Network: {len(numerical_cols)} features with complex patterns → deep learning may help"
        )
    
    print(f"\nSummary:")
    print(f"  - Strong linear relationships: {strong_linear}")
    print(f"  - Moderate linear relationships: {moderate_linear}")
    print(f"  - Weak linear relationships: {weak_linear}")
    print(f"  - Total features: {len(numerical_cols)}")
    
    print(f"\nRecommended Models:")
    for model in recommendations['recommended_models']:
        print(f"  • {model}")
    
    print(f"\nReasoning:")
    for reason in recommendations['reasoning']:
        print(f"  {reason}")
    
    print("="*70)
    print("\n⚠️  IMPORTANT: These are suggestions based on correlations.")
    print("    Always compare multiple models and use cross-validation!")
    print("="*70)
    
    return recommendations


def correlation_heatmap(X, y=None, figsize=(12, 10)):
    """
    Plot correlation heatmap of all numerical features.
    
    Purpose:
    - See correlations between features (multicollinearity)
    - See correlations with target (if provided)
    - Identify redundant features
    
    Parameters
    ----------
    X : DataFrame
        Input features
    y : array-like, optional
        Target variable (will be added as 'Target' column)
    figsize : tuple, default=(12, 10)
        Figure size
        
    Examples
    --------
    >>> correlation_heatmap(X, y)
    >>> # Look for high correlations between features (> 0.9 = redundant)
    """
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Create correlation matrix
    if y is not None:
        # Add target to dataframe temporarily
        X_with_target = X[numerical_cols].copy()
        X_with_target['Target'] = y
        corr_matrix = X_with_target.corr()
    else:
        corr_matrix = X[numerical_cols].corr()
    
    # Plot
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Report high correlations
    if y is not None:
        print("\nTop correlations with Target:")
        target_corrs = corr_matrix['Target'].drop('Target').abs().sort_values(ascending=False)
        for feature, corr in target_corrs.head(5).items():
            print(f"  {feature:20s}: {corr:.3f}")
    
    # Find highly correlated feature pairs (multicollinearity warning)
    print("\nHighly correlated feature pairs (multicollinearity warning):")
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.9 and corr_matrix.columns[i] != 'Target' and corr_matrix.columns[j] != 'Target':
                high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
    
    if high_corr_pairs:
        for feat1, feat2, corr in high_corr_pairs:
            print(f"  {feat1} ↔ {feat2}: {corr:.3f} (consider removing one)")
    else:
        print("  ✓ No highly correlated feature pairs detected")

# ============================================================================
# CLASSIFICATION-SPECIFIC EDA FUNCTIONS
# ============================================================================

def plot_feature_distributions_by_class(X, y, numerical_cols=None, class_names=None, figsize=(15, 10)):
    """
    Plot feature distributions separated by class.
    
    For classification: Shows how features differ across classes.
    Helps identify which features are discriminative.
    
    Parameters
    ----------
    X : DataFrame
        Input features
    y : array-like
        Target classes
    numerical_cols : list, optional
        Numerical columns to plot (if None, use all numerical)
    class_names : dict, optional
        Mapping from class labels to names
        Example: {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    figsize : tuple, default=(15, 10)
        Figure size
        
    Examples
    --------
    >>> class_names = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    >>> plot_feature_distributions_by_class(X, y, class_names=class_names)
    >>> # Shows overlapping histograms colored by class
    """
    if numerical_cols is None:
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    n_features = len(numerical_cols)
    n_cols = 2
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_features > 1 else [axes]
    
    # Get unique classes
    classes = np.unique(y)
    
    # Convert y to numpy if pandas Series
    if isinstance(y, pd.Series):
        y_array = y.values
    else:
        y_array = np.array(y)
    
    for idx, col in enumerate(numerical_cols):
        ax = axes[idx]
        
        # Plot distribution for each class
        for class_val in classes:
            class_mask = y_array == class_val
            class_data = X[class_mask][col]
            label = class_names[class_val] if class_names else f'Class {class_val}'
            ax.hist(class_data, bins=20, alpha=0.5, label=label, edgecolor='black')
        
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
        ax.set_title(f'{col} by Class')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide extra subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Feature Distributions by Class', fontsize=14, y=1.00)
    plt.tight_layout()
    plt.show()


def analyze_class_separability(X, y, class_names=None):
    """
    Analyze how well features separate classes.
    
    For classification: Calculate ANOVA F-scores to identify
    most discriminative features.
    
    Parameters
    ----------
    X : DataFrame
        Input features
    y : array-like
        Target classes
    class_names : dict, optional
        Mapping from class labels to names
        Example: {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
        
    Returns
    -------
    recommendations : dict
        Model recommendations based on feature separability
        
    Examples
    --------
    >>> class_names = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    >>> recs = analyze_class_separability(X, y, class_names=class_names)
    >>> print(recs['recommended_models'])
    ['Logistic Regression', 'Decision Tree', 'Random Forest', ...]
    """
    from sklearn.feature_selection import f_classif
    
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Convert y to numpy if pandas Series
    if isinstance(y, pd.Series):
        y_array = y.values
    else:
        y_array = np.array(y)
    
    print("="*70)
    print("CLASS SEPARABILITY ANALYSIS")
    print("="*70)
    
    # Calculate F-scores for each feature (ANOVA)
    f_scores, p_values = f_classif(X[numerical_cols], y_array)
    
    # Create DataFrame
    feature_importance = pd.DataFrame({
        'Feature': numerical_cols,
        'F-Score': f_scores,
        'p-value': p_values
    }).sort_values('F-Score', ascending=False)
    
    print("\nFeature Discriminative Power (sorted by F-score):")
    print("-"*70)
    for idx, row in feature_importance.iterrows():
        f_score = row['F-Score']
        p_val = row['p-value']
        
        # Categorize strength
        if f_score > 100:
            strength = "Very Strong"
        elif f_score > 50:
            strength = "Strong"
        elif f_score > 10:
            strength = "Moderate"
        else:
            strength = "Weak"
        
        # Significance stars
        significant = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        print(f"{row['Feature']:20s}: F={f_score:8.2f} ({strength:12s}) {significant}")
    
    # Get class distribution
    unique_classes, counts = np.unique(y_array, return_counts=True)
    
    print("\n" + "="*70)
    print("CLASS DISTRIBUTION")
    print("="*70)
    for class_val, count in zip(unique_classes, counts):
        label = class_names[class_val] if class_names else f'Class {class_val}'
        pct = count / len(y_array) * 100
        print(f"{label:20s}: {count:4d} samples ({pct:5.1f}%)")
    
    # Check for class imbalance
    class_balance = max(counts) / min(counts)
    if class_balance > 3:
        print(f"\n⚠️  Class imbalance detected: {class_balance:.1f}:1 ratio")
        print("   Consider: stratified sampling, class weights, or SMOTE")
    else:
        print(f"\n✓ Classes are balanced (ratio: {class_balance:.1f}:1)")
    
    # Model recommendations
    print("\n" + "="*70)
    print("MODEL SELECTION RECOMMENDATIONS")
    print("="*70)
    
    recommendations = {
        'task_type': 'classification',
        'n_classes': len(unique_classes),
        'n_features': len(numerical_cols),
        'strong_features': len(feature_importance[feature_importance['F-Score'] > 50]),
        'class_balance': class_balance,
        'recommended_models': [],
        'reasoning': []
    }
    
    n_classes = len(unique_classes)
    
    # Logistic Regression
    if recommendations['strong_features'] >= 2 or n_classes == 2:
        recommendations['recommended_models'].append('Logistic Regression')
        if n_classes == 2:
            recommendations['reasoning'].append(
                "✅ Logistic Regression: Binary classification with linear decision boundary"
            )
        else:
            recommendations['reasoning'].append(
                f"✅ Logistic Regression: Multi-class ({n_classes} classes) with strong linear separability"
            )
    
    # Tree-based models (always good)
    recommendations['recommended_models'].extend(['Decision Tree', 'Random Forest'])
    recommendations['reasoning'].append(
        "✅ Decision Trees/Random Forests: Handle non-linear boundaries, good baseline"
    )
    
    # KNN if features are well-separated
    if recommendations['strong_features'] >= len(numerical_cols) * 0.5:
        recommendations['recommended_models'].append('K-Nearest Neighbors')
        recommendations['reasoning'].append(
            "✅ KNN: Features show good class separation"
        )
    
    # SVM for binary or small multi-class
    if n_classes <= 3:
        recommendations['recommended_models'].append('SVM')
        recommendations['reasoning'].append(
            f"✅ SVM: Works well for {n_classes}-class problems"
        )
    
    # Neural Networks for complex problems
    if len(numerical_cols) > 10 or n_classes > 3:
        recommendations['recommended_models'].append('Neural Network')
        recommendations['reasoning'].append(
            f"✅ Neural Network: {len(numerical_cols)} features, {n_classes} classes → deep learning may help"
        )
    
    print(f"\nRecommended Models:")
    for model in recommendations['recommended_models']:
        print(f"  • {model}")
    
    print(f"\nReasoning:")
    for reason in recommendations['reasoning']:
        print(f"  {reason}")
    
    print("="*70)
    print("\n⚠️  IMPORTANT: Always compare multiple models with cross-validation!")
    print("="*70)
    
    return recommendations

# ============================================================================
# CLUSTERING-SPECIFIC EDA FUNCTIONS
# ============================================================================

def analyze_clustering_suitability(X):
    """
    Analyze if dataset is suitable for clustering.
    
    Checks:
    - Feature variance (low variance features won't help clustering)
    - Feature correlations (highly correlated features are redundant)
    - Dataset size (need enough samples)
    
    Parameters
    ----------
    X : DataFrame
        Input features
        
    Returns
    -------
    recommendations : dict
        Clustering method recommendations
        
    Examples
    --------
    >>> X = data_loader.load_customer_segmentation()
    >>> recs = analyze_clustering_suitability(X)
    >>> print(recs['recommended_models'])
    ['K-Means', 'DBSCAN', 'Hierarchical Clustering']
    """
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print("="*70)
    print("CLUSTERING SUITABILITY ANALYSIS")
    print("="*70)
    
    # Calculate feature statistics
    print("\nFeature Variance Analysis:")
    print("-"*70)
    
    variances = X[numerical_cols].var()
    low_variance_features = variances[variances < 0.01].index.tolist()
    
    for col in numerical_cols:
        var = variances[col]
        if var < 0.01:
            status = "⚠️  Very Low (may not help clustering)"
        elif var < 0.1:
            status = "Low"
        elif var < 1.0:
            status = "Moderate"
        else:
            status = "High"
        print(f"{col:20s}: {var:10.4f} ({status})")
    
    if low_variance_features:
        print(f"\n⚠️  Warning: {len(low_variance_features)} features have very low variance")
        print(f"   Consider removing: {low_variance_features}")
    
    # Feature correlations
    print("\n" + "="*70)
    print("FEATURE CORRELATION ANALYSIS")
    print("="*70)
    
    corr_matrix = X[numerical_cols].corr()
    high_corr_pairs = []
    
    for i in range(len(numerical_cols)):
        for j in range(i+1, len(numerical_cols)):
            corr = abs(corr_matrix.iloc[i, j])
            if corr > 0.9:
                high_corr_pairs.append((numerical_cols[i], numerical_cols[j], corr))
    
    if high_corr_pairs:
        print(f"\n⚠️  {len(high_corr_pairs)} highly correlated feature pairs found:")
        for feat1, feat2, corr in high_corr_pairs[:5]:
            print(f"   {feat1} <-> {feat2}: {corr:.3f}")
        print(f"   Consider removing redundant features")
    else:
        print("\n✓ No highly correlated features (> 0.9)")
    
    # Dataset size check
    print("\n" + "="*70)
    print("DATASET SIZE ANALYSIS")
    print("="*70)
    
    n_samples = len(X)
    n_features = len(numerical_cols)
    
    print(f"\nSamples: {n_samples}")
    print(f"Features: {n_features}")
    print(f"Samples per feature: {n_samples / n_features:.1f}")
    
    if n_samples < 50:
        print("⚠️  Small dataset (< 50 samples) - clustering may be unreliable")
    elif n_samples < 200:
        print("⚠️  Medium dataset - results should be interpreted carefully")
    else:
        print("✓ Sufficient samples for clustering")
    
    # Model recommendations
    print("\n" + "="*70)
    print("CLUSTERING METHOD RECOMMENDATIONS")
    print("="*70)
    
    recommendations = {
        'task_type': 'clustering',
        'n_samples': n_samples,
        'n_features': n_features,
        'low_variance_features': len(low_variance_features),
        'high_corr_pairs': len(high_corr_pairs),
        'recommended_models': [],
        'reasoning': []
    }
    
    # K-Means (always a good baseline)
    recommendations['recommended_models'].append('K-Means')
    recommendations['reasoning'].append(
        "✅ K-Means: Fast, works well for spherical clusters"
    )
    
    # DBSCAN for non-spherical clusters
    recommendations['recommended_models'].append('DBSCAN')
    recommendations['reasoning'].append(
        "✅ DBSCAN: Finds arbitrary-shaped clusters, handles outliers"
    )
    
    # Hierarchical for small datasets
    if n_samples < 1000:
        recommendations['recommended_models'].append('Hierarchical Clustering')
        recommendations['reasoning'].append(
            "✅ Hierarchical: Creates dendrogram, no need to specify k"
        )
    
    # Gaussian Mixture for overlapping clusters
    if n_features <= 10:
        recommendations['recommended_models'].append('Gaussian Mixture Models')
        recommendations['reasoning'].append(
            "✅ GMM: Soft clustering, handles overlapping clusters"
        )
    
    print(f"\nRecommended Methods:")
    for model in recommendations['recommended_models']:
        print(f"  • {model}")
    
    print(f"\nReasoning:")
    for reason in recommendations['reasoning']:
        print(f"  {reason}")
    
    print("\n" + "="*70)
    print("IMPORTANT NOTES:")
    print("="*70)
    print("• Always scale features before clustering (StandardScaler)")
    print("• Try multiple values of k (K-Means) or eps/min_samples (DBSCAN)")
    print("• Use elbow method or silhouette score to choose optimal k")
    print("• Visualize results with PCA or t-SNE for high-dimensional data")
    print("="*70)
    
    return recommendations


def plot_feature_pairwise(X, sample_size=None, figsize=(15, 10)):
    """
    Plot pairwise scatter plots for clustering visualization.
    
    Useful for clustering to see natural groupings.
    
    Parameters
    ----------
    X : DataFrame
        Input features
    sample_size : int, optional
        Sample this many points if dataset is large
    figsize : tuple
        Figure size
    """
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Limit to first 4 features if too many
    if len(numerical_cols) > 4:
        print(f"⚠️  Too many features ({len(numerical_cols)}), showing first 4")
        numerical_cols = numerical_cols[:4]
    
    # Sample if dataset is large
    if sample_size and len(X) > sample_size:
        X_plot = X.sample(n=sample_size, random_state=42)
        print(f"⚠️  Large dataset, showing sample of {sample_size} points")
    else:
        X_plot = X
    
    n_features = len(numerical_cols)
    
    fig, axes = plt.subplots(n_features, n_features, figsize=figsize)
    
    for i in range(n_features):
        for j in range(n_features):
            ax = axes[i, j] if n_features > 1 else axes
            
            if i == j:
                # Diagonal: histogram
                ax.hist(X_plot[numerical_cols[i]], bins=20, edgecolor='black', alpha=0.7)
                ax.set_ylabel('Frequency')
            else:
                # Off-diagonal: scatter plot
                ax.scatter(X_plot[numerical_cols[j]], X_plot[numerical_cols[i]], 
                          alpha=0.5, s=20)
                ax.set_xlabel(numerical_cols[j] if i == n_features-1 else '')
                ax.set_ylabel(numerical_cols[i] if j == 0 else '')
            
            ax.grid(True, alpha=0.3)
    
    plt.suptitle('Pairwise Feature Relationships (Look for natural clusters)', 
                 fontsize=14, y=1.00)
    plt.tight_layout()
    plt.show()