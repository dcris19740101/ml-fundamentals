"""
Target variable (y) encoding utilities.

Use these when your target variable is categorical and needs encoding.
"""

import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder


def encode_target_labels(y):
    """
    Encode categorical target variable to integers (0, 1, 2, ...).
    
    Use for classification when target is categorical.
    
    Parameters
    ----------
    y : array-like
        Target variable with categorical labels
        Example: ['cat', 'dog', 'cat', 'bird']
        
    Returns
    -------
    y_encoded : ndarray
        Integer-encoded target
        Example: [0, 1, 0, 2]
    encoder : LabelEncoder
        Fitted encoder (save for inverse_transform later)
        
    Examples
    --------
    >>> y = ['spam', 'ham', 'spam', 'ham']
    >>> y_encoded, encoder = encode_target_labels(y)
    >>> y_encoded
    array([1, 0, 1, 0])
    >>> encoder.classes_
    array(['ham', 'spam'])
    >>> encoder.inverse_transform(y_encoded)
    array(['spam', 'ham', 'spam', 'ham'])
    """
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    return y_encoded, encoder


def encode_ordinal_target(y, categories):
    """
    Encode ordinal target variable preserving order.
    
    Use when target has natural ordering (e.g., 'low' < 'medium' < 'high').
    
    Parameters
    ----------
    y : array-like
        Target variable with ordinal categories
    categories : list
        Ordered list of categories (low to high)
        Example: ['low', 'medium', 'high']
        
    Returns
    -------
    y_encoded : ndarray
        Integer-encoded target preserving order
        Example: [0, 1, 2] for ['low', 'medium', 'high']
    encoder : OrdinalEncoder
        Fitted encoder
        
    Examples
    --------
    >>> y = ['medium', 'high', 'low', 'medium']
    >>> categories = ['low', 'medium', 'high']
    >>> y_encoded, encoder = encode_ordinal_target(y, categories)
    >>> y_encoded
    array([1, 2, 0, 1])
    """
    encoder = OrdinalEncoder(categories=[categories])
    y_encoded = encoder.fit_transform(np.array(y).reshape(-1, 1)).ravel()
    return y_encoded, encoder


def decode_target(y_encoded, encoder):
    """
    Convert encoded target back to original labels.
    
    Parameters
    ----------
    y_encoded : array-like
        Integer-encoded predictions
    encoder : LabelEncoder or OrdinalEncoder
        The encoder used for encoding
        
    Returns
    -------
    y_original : ndarray
        Original categorical labels
        
    Examples
    --------
    >>> y_encoded = [0, 1, 0, 1]
    >>> y_original = decode_target(y_encoded, encoder)
    >>> y_original
    array(['ham', 'spam', 'ham', 'spam'])
    """
    return encoder.inverse_transform(y_encoded)