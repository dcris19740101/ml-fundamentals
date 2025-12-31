"""
Feature preprocessing transformations.

Handles scaling, encoding, and missing values for input features (X).
Does NOT handle target variable (y) encoding.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer



class DataPreprocessor:
    """
    Smart preprocessing of input features (X) for machine learning models.

    Handles:
    - Missing value imputation
    - Numerical feature scaling
    - Categorical feature encoding (nominal and ordinal)
    
    Does NOT handle:
    - Target variable (y) encoding → See target_encoding.py
    - Train/test splitting → See data_utils.py
    - Exploratory analysis → See eda.py
    
    Parameters
    ----------
    scale : bool, default=True
        Whether to scale numerical features
    encode_categorical : bool, default=True
        Whether to encode categorical features
    ordinal_features : dict, optional
        Dictionary mapping ordinal feature names to their ordered categories.
        Example: {'Education': ['High School', 'Bachelor', 'Master', 'PhD']}
    auto_strategy : bool, default=True
        If True, automatically determine best imputation strategy per column
    polynomial_degree : int, optional
            If provided, create polynomial features of this degree
            Applied AFTER encoding, BEFORE scaling
    """
    
    def __init__(self, scale=True, encode_categorical=True, ordinal_features=None, auto_strategy=True, polynomial_degree=None):
        self.scale = scale
        self.encode_categorical = encode_categorical
        self.ordinal_features = ordinal_features or {}
        self.auto_strategy = auto_strategy
        self.polynomial_degree = polynomial_degree
        
        # Fitted components
        self.numerical_imputer = None
        self.nominal_imputer = None
        self.ordinal_imputer = None
        self.nominal_encoder = None
        self.ordinal_encoder = None
        self.polynomial_features = None
        self.scaler = None
        
        # Column tracking
        self.numerical_cols = None
        self.nominal_cols = None  # Categorical without order
        self.ordinal_cols = None  # Categorical with order
        self.columns_to_drop = None
        self.cols_to_scale = None
        self.feature_names_out = None
    
    def fit_transform(self, X):
        """Fit on training data and transform"""
        X = pd.DataFrame(X).copy()
        
        print("\n" + "="*60)
        print("DataPreprocessor: fit_transform")
        print("="*60)
        
        # Identify column types
        self.numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        all_categorical = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Separate ordinal from nominal categorical
        self.ordinal_cols = [col for col in all_categorical if col in self.ordinal_features]
        self.nominal_cols = [col for col in all_categorical if col not in self.ordinal_features]
        
        print(f"\n1. Column types:")
        print(f"   Numerical: {len(self.numerical_cols)} columns")
        print(f"   Nominal categorical: {len(self.nominal_cols)} columns")
        print(f"   Ordinal categorical: {len(self.ordinal_cols)} columns")
        
        if self.ordinal_cols:
            print(f"\n   Ordinal features with defined order:")
            for col in self.ordinal_cols:
                print(f"      {col}: {self.ordinal_features[col]}")
        
        # Handle numerical columns
        print(f"\n2. Processing numerical columns...")
        if self.numerical_cols:
            self.numerical_imputer = SimpleImputer(strategy='median')
            X[self.numerical_cols] = self.numerical_imputer.fit_transform(X[self.numerical_cols])
            print(f"   ✓ Applied median imputation")
        
        # Handle nominal categorical columns
        print(f"\n3. Processing nominal categorical columns...")
        if self.nominal_cols:
            self.nominal_imputer = SimpleImputer(strategy='most_frequent')
            X[self.nominal_cols] = self.nominal_imputer.fit_transform(X[self.nominal_cols])
            print(f"   ✓ Applied most_frequent imputation")
            
            if self.encode_categorical:
                print(f"   ✓ One-hot encoding {len(self.nominal_cols)} nominal columns...")
                self.nominal_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                encoded = self.nominal_encoder.fit_transform(X[self.nominal_cols])
                encoded_names = self.nominal_encoder.get_feature_names_out(self.nominal_cols)
                
                encoded_df = pd.DataFrame(encoded, columns=encoded_names, index=X.index)
                X = X.drop(columns=self.nominal_cols)
                X = pd.concat([X, encoded_df], axis=1)
                print(f"   → Created {len(encoded_names)} one-hot features")
        
        # Handle ordinal categorical columns
        print(f"\n4. Processing ordinal categorical columns...")
        if self.ordinal_cols:
            self.ordinal_imputer = SimpleImputer(strategy='most_frequent')
            X[self.ordinal_cols] = self.ordinal_imputer.fit_transform(X[self.ordinal_cols])
            print(f"   ✓ Applied most_frequent imputation")
            
            if self.encode_categorical:
                print(f"   ✓ Ordinal encoding {len(self.ordinal_cols)} columns...")
                # Create ordered categories list for OrdinalEncoder
                categories_list = [self.ordinal_features[col] for col in self.ordinal_cols]
                
                self.ordinal_encoder = OrdinalEncoder(
                    categories=categories_list,
                    handle_unknown='use_encoded_value',
                    unknown_value=-1  # Assign -1 to unknown categories
                )
                X[self.ordinal_cols] = self.ordinal_encoder.fit_transform(X[self.ordinal_cols])
                print(f"   → Preserved ordinal relationships")
        
        # Create polynomial features (ONLY on numerical features!)
        if self.polynomial_degree and self.numerical_cols:
            print(f"\n5. Creating polynomial features (degree={self.polynomial_degree})...")
            
            # Separate numerical and non-numerical
            numerical_cols = [col for col in X.columns if col in self.numerical_cols or 
                            any(orig in col for orig in self.numerical_cols)]
            
            X_numerical = X[self.numerical_cols]
            X_other = X.drop(columns=self.numerical_cols)
            
            # Create polynomial on numerical only
            self.polynomial_features = PolynomialFeatures(
                degree=self.polynomial_degree,
                include_bias=False
            )
            X_numerical_poly = self.polynomial_features.fit_transform(X_numerical)
            poly_names = self.polynomial_features.get_feature_names_out(self.numerical_cols)
            X_numerical_poly = pd.DataFrame(X_numerical_poly, columns=poly_names, index=X.index)
            
            # Recombine
            X = pd.concat([X_numerical_poly, X_other], axis=1)

            # Track which columns to scale (polynomial features)
            self.cols_to_scale = list(poly_names)
            
            print(f"   → Created {len(poly_names)} polynomial features")
        else:
            # If no polynomial, scale original numerical columns
            self.cols_to_scale = self.numerical_cols.copy()

        # Scale ONLY the numerical-derived features
        if self.scale and self.cols_to_scale:
            print(f"\n6. Scaling features...")
            self.scaler = StandardScaler()
            X[self.cols_to_scale] = self.scaler.fit_transform(X[self.cols_to_scale])
            print(f"   ✓ Scaled {len(self.cols_to_scale)} features")
        
        # Store feature names
        self.feature_names_out = X.columns.tolist()
    
        print(f"\n{'='*60}")
        print(f"Final shape: {X.shape[0]} rows × {X.shape[1]} features")
        print(f"{'='*60}\n")
        
        return X.values
    
    def transform(self, X):
        """Transform new data using fitted parameters"""
        X = pd.DataFrame(X).copy()
        
        # Numerical imputation
        if self.numerical_cols and self.numerical_imputer:
            X[self.numerical_cols] = self.numerical_imputer.transform(X[self.numerical_cols])
        
        # Nominal categorical
        if self.nominal_cols and self.nominal_imputer:
            X[self.nominal_cols] = self.nominal_imputer.transform(X[self.nominal_cols])
            
            if self.encode_categorical and self.nominal_encoder:
                encoded = self.nominal_encoder.transform(X[self.nominal_cols])
                encoded_names = self.nominal_encoder.get_feature_names_out(self.nominal_cols)
                # Reset index for encoded features
                encoded_df = pd.DataFrame(encoded, columns=encoded_names).reset_index(drop=True)
                X = X.drop(columns=self.nominal_cols).reset_index(drop=True)
                X = pd.concat([X, encoded_df], axis=1)
        
        # Ordinal categorical
        if self.ordinal_cols and self.ordinal_imputer:
            X[self.ordinal_cols] = self.ordinal_imputer.transform(X[self.ordinal_cols])
            
            if self.encode_categorical and self.ordinal_encoder:
                X[self.ordinal_cols] = self.ordinal_encoder.transform(X[self.ordinal_cols])
        
        # Polynomial features
        if self.polynomial_degree and self.polynomial_features:
            X_numerical = X[self.numerical_cols].copy()
            X_other = X.drop(columns=self.numerical_cols).copy()
            
            X_numerical_poly = self.polynomial_features.transform(X_numerical)
            poly_names = self.polynomial_features.get_feature_names_out(self.numerical_cols)
            
            # Reset indices before concat
            X_numerical_poly = pd.DataFrame(X_numerical_poly, columns=poly_names).reset_index(drop=True)
            X_other = X_other.reset_index(drop=True)
            
            X = pd.concat([X_numerical_poly, X_other], axis=1)
        
        # Reorder columns to match training
        if self.feature_names_out:
            X = X[self.feature_names_out]
        
        # Scaling
        if self.scale and self.scaler and self.cols_to_scale:
            X[self.cols_to_scale] = self.scaler.transform(X[self.cols_to_scale])
    
        return X.values

    
    def get_feature_names(self):
        """Return feature names after transformation"""
        return self.feature_names_out

