# src/data_loader.py

"""
Dataset loading utilities.

Provides easy access to common ML datasets organized by task type:
- Regression: Predict continuous values
- Classification: Predict categories
- Clustering: Discover groups (unsupervised)
"""

import pandas as pd
import numpy as np
from sklearn.datasets import (
    fetch_california_housing,
    fetch_openml,
    load_iris as sklearn_load_iris,
    load_breast_cancer as sklearn_load_breast_cancer,
    make_blobs
)


# ============================================================================
# REGRESSION DATASETS (Predict Continuous Values)
# ============================================================================

def load_california_housing():
    """
    Load California housing dataset.
    
    Regression task: Predict median house value.
    
    Returns
    -------
    X : DataFrame, shape (20640, 8)
        Features: MedInc, HouseAge, AveRooms, AveBedrms, Population,
                  AveOccup, Latitude, Longitude
    y : Series, shape (20640,)
        Target: Median house value (in $100,000s)
        
    Examples
    --------
    >>> X, y = load_california_housing()
    >>> print(f"Target range: ${y.min():.1f}k - ${y.max():.1f}k")
    """
    housing = fetch_california_housing(as_frame=True)
    return housing.data, housing.target


def load_insurance():
    """
    Load medical insurance cost dataset.
    
    Regression task: Predict insurance charges.
    Has categorical features (good for testing preprocessing).
    
    Returns
    -------
    X : DataFrame, shape (1338, 6)
        Features: age, sex, bmi, children, smoker, region
    y : Series, shape (1338,)
        Target: Medical charges (in dollars)
        
    Examples
    --------
    >>> X, y = load_insurance()
    >>> print(X.dtypes)  # Has categorical columns!
    age           int64
    sex          object  # Categorical
    bmi         float64
    children      int64
    smoker       object  # Categorical
    region       object  # Categorical
    """
    url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
    df = pd.read_csv(url)
    X = df.drop('charges', axis=1)
    y = df['charges']
    return X, y


def load_ames_housing():
    """
    Load Ames housing dataset.
    
    Regression task: Predict house sale price.
    Complex dataset with many features and missing values.
    
    Returns
    -------
    X : DataFrame, shape (1460, 79)
        Features: Many categorical and numerical housing features
    y : Series, shape (1460,)
        Target: Sale price (in dollars)
        
    Examples
    --------
    >>> X, y = load_ames_housing()
    >>> print(f"Features: {X.shape[1]}")
    >>> print(f"Missing values: {X.isnull().sum().sum()}")
    """
    ames = fetch_openml(name="house_prices", as_frame=True, parser='auto')
    X = ames.data
    y = ames.target
    return X, y


# ============================================================================
# CLASSIFICATION DATASETS (Predict Categories)
# ============================================================================

def load_iris():
    """
    Load Iris flower dataset.
    
    Multi-class classification: Predict flower species (3 classes).
    Classic ML dataset, small and clean.
    
    Returns
    -------
    X : DataFrame, shape (150, 4)
        Features: sepal length, sepal width, petal length, petal width
    y : Series, shape (150,)
        Target: species (0=setosa, 1=versicolor, 2=virginica)
        
    Examples
    --------
    >>> X, y = load_iris()
    >>> print(y.value_counts())
    0    50  # setosa
    1    50  # versicolor
    2    50  # virginica
    """
    iris = sklearn_load_iris(as_frame=True)
    return iris.data, iris.target


def load_breast_cancer():
    """
    Load Breast Cancer Wisconsin dataset.
    
    Binary classification: Predict tumor malignancy.
    
    Returns
    -------
    X : DataFrame, shape (569, 30)
        Features: tumor characteristics
    y : Series, shape (569,)
        Target: 0=malignant, 1=benign
        
    Examples
    --------
    >>> X, y = load_breast_cancer()
    >>> print(f"Class distribution:")
    >>> print(y.value_counts())
    1    357  # benign
    0    212  # malignant
    """
    cancer = sklearn_load_breast_cancer(as_frame=True)
    return cancer.data, cancer.target


def load_titanic():
    """
    Load Titanic survival dataset.
    
    Binary classification with categorical features and missing values.
    Good for testing complete preprocessing pipeline.
    
    Returns
    -------
    X : DataFrame, shape (891, 7)
        Features: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
    y : Series, shape (891,)
        Target: Survived (0=No, 1=Yes)
        
    Examples
    --------
    >>> X, y = load_titanic()
    >>> print("Missing values:")
    >>> print(X.isnull().sum())
    Age         177
    Embarked      2
    >>> print("Categorical features:", X.select_dtypes('object').columns.tolist())
    """
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)
    
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    X = df[features]
    y = df['Survived']
    
    return X, y


# ============================================================================
# CLUSTERING DATASETS (Unsupervised - No Target)
# ============================================================================

def load_clustering_blobs(n_samples=300, centers=3, random_state=42):
    """
    Generate synthetic clustering dataset.
    
    Creates well-separated clusters for testing clustering algorithms.
    
    Parameters
    ----------
    n_samples : int, default=300
        Number of samples
    centers : int, default=3
        Number of clusters
    random_state : int, default=42
        Random seed
        
    Returns
    -------
    X : ndarray, shape (n_samples, 2)
        2D data points
    y_true : ndarray, shape (n_samples,)
        True cluster labels (for evaluation only, NOT for training!)
        
    Examples
    --------
    >>> X, y_true = load_clustering_blobs(centers=4)
    >>> # For clustering, we DON'T use y_true during training
    >>> from src.models.unsupervised import SKLearnKMeans
    >>> model = SKLearnKMeans(n_clusters=4)
    >>> labels = model.fit_predict(X)  # No y_true here!
    >>> # Use y_true only to evaluate: how well did clustering match true groups?
    """
    X, y_true = make_blobs(
        n_samples=n_samples,
        centers=centers,
        cluster_std=0.6,
        random_state=random_state
    )
    
    # Convert to DataFrame for consistency
    X = pd.DataFrame(X, columns=['Feature_1', 'Feature_2'])
    
    return X, y_true


def load_customer_segmentation():
    """
    Load mall customer segmentation dataset.
    
    Real-world clustering: segment customers by spending behavior.
    No target variable - this is unsupervised!
    
    Returns
    -------
    X : DataFrame, shape (200, 3)
        Features: Age, Annual Income (k$), Spending Score (1-100)
        
    Examples
    --------
    >>> X = load_customer_segmentation()  # No y!
    >>> from src.models.unsupervised import SKLearnKMeans
    >>> model = SKLearnKMeans(n_clusters=5)
    >>> customer_segments = model.fit_predict(X)
    >>> # Analyze segments
    >>> for segment in range(5):
    ...     print(f"Segment {segment}:")
    ...     print(X[customer_segments == segment].describe())
    """
    url = "https://raw.githubusercontent.com/SteffiPeTaffy/machineLearningAZ/master/Machine%20Learning%20A-Z%20Template%20Folder/Part%204%20-%20Clustering/Section%2025%20-%20Hierarchical%20Clustering/Mall_Customers.csv"
    df = pd.read_csv(url)
    
    X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
    
    return X  # No y! Unsupervised