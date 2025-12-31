"""
Data manipulation utilities.

Functions for splitting, sampling, and basic data operations.
"""

from sklearn.model_selection import train_test_split


def split_train_test(X, y, test_size=0.2, random_state=42, stratify=None):
    """
    Split data into training and test sets.
    
    Wrapper around sklearn's train_test_split with sensible defaults.
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Features
    y : array-like, shape (n_samples,)
        Target variable
    test_size : float, default=0.2
        Proportion of data for test set (0.0 to 1.0)
    random_state : int, default=42
        Random seed for reproducibility
    stratify : array-like, optional
        If not None, split in stratified fashion using this as class labels
        Use stratify=y for classification to preserve class distribution
        
    Returns
    -------
    X_train, X_test, y_train, y_test : arrays
        Split datasets
        
    Examples
    --------
    >>> # Regression (no stratification)
    >>> X_train, X_test, y_train, y_test = split_train_test(X, y)
    
    >>> # Classification (stratified by class)
    >>> X_train, X_test, y_train, y_test = split_train_test(X, y, stratify=y)
    """
    return train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=stratify
    )


def split_train_val_test(X, y, val_size=0.15, test_size=0.15, random_state=42):
    """
    Split data into training, validation, and test sets.
    
    Parameters
    ----------
    X : array-like
        Features
    y : array-like
        Target
    val_size : float, default=0.15
        Proportion for validation set
    test_size : float, default=0.15
        Proportion for test set
    random_state : int, default=42
        Random seed
        
    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test : arrays
        
    Examples
    --------
    >>> X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(X, y)
    >>> print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    """
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Second split: separate train and validation
    val_ratio = val_size / (1 - test_size)  # Adjust for remaining data
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test