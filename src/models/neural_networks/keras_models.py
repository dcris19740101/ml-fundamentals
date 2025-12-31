"""
Neural Network models using TensorFlow/Keras.

Standalone implementations - neural networks have fundamentally
different APIs than traditional ML models.
"""

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, r2_score


class KerasANN:
    """
    Artificial Neural Network (fully-connected) using Keras/TensorFlow.
    
    Can be used for both classification and regression by specifying
    the task type and appropriate output configuration.
    
    Parameters
    ----------
    input_dim : int
        Number of input features
    hidden_layers : list of int, default=[64, 32]
        Number of neurons in each hidden layer
    task : str, default='classification'
        Type of task: 'classification' or 'regression'
    output_dim : int, default=1
        For classification: number of classes (1 for binary, n for multi-class)
        For regression: number of targets (1 for single, n for multi-output)
    activation : str, default='relu'
        Activation function for hidden layers
        
    Examples
    --------
    >>> # Binary classification
    >>> model = KerasANN(input_dim=10, task='classification', output_dim=1)
    >>> model.fit(X_train, y_train, epochs=50)
    >>> 
    >>> # Multi-class classification
    >>> model = KerasANN(input_dim=10, task='classification', output_dim=3)
    >>> model.fit(X_train, y_train_onehot, epochs=50)
    >>> 
    >>> # Regression
    >>> model = KerasANN(input_dim=10, task='regression', output_dim=1)
    >>> model.fit(X_train, y_train, epochs=50)
    """
    
    def __init__(self, input_dim, hidden_layers=[64, 32], task='classification',
                 output_dim=1, activation='relu'):
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.task = task
        self.output_dim = output_dim
        self.activation = activation
        self._model = None
        self.is_fitted = False
        self.history = None
        self._build_model()
    
    def _build_model(self):
        """Build neural network architecture"""
        model = tf.keras.models.Sequential()
        
        # Input + hidden layers
        model.add(tf.keras.layers.Dense(
            self.hidden_layers[0],
            activation=self.activation,
            input_dim=self.input_dim
        ))
        
        for neurons in self.hidden_layers[1:]:
            model.add(tf.keras.layers.Dense(neurons, activation=self.activation))
        
        # Output layer - depends on task
        if self.task == 'classification':
            if self.output_dim == 1:
                # Binary classification
                model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
                model.compile(
                    optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
            else:
                # Multi-class classification
                model.add(tf.keras.layers.Dense(self.output_dim, activation='softmax'))
                model.compile(
                    optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
        else:  # regression
            model.add(tf.keras.layers.Dense(self.output_dim, activation='linear'))
            model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae']
            )
        
        self._model = model
    
    def fit(self, X, y, epochs=50, batch_size=32, validation_split=0.2, 
            verbose=1, callbacks=None):
        """
        Train the neural network.
        
        Parameters
        ----------
        X : array-like
            Training features
        y : array-like
            Training labels/targets
        epochs : int, default=50
            Number of training epochs
        batch_size : int, default=32
            Batch size for training
        validation_split : float, default=0.2
            Fraction of training data to use for validation
        verbose : int, default=1
            Verbosity (0=silent, 1=progress bar, 2=one line per epoch)
        callbacks : list, optional
            Keras callbacks (e.g., EarlyStopping, ModelCheckpoint)
            
        Returns
        -------
        self
            Fitted model
        """
        self.history = self._model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose,
            callbacks=callbacks
        )
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """
        Predict class labels (classification) or values (regression).
        
        Parameters
        ----------
        X : array-like
            Input features
            
        Returns
        -------
        array
            Predicted labels or values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predicting")
        
        predictions = self._model.predict(X, verbose=0)
        
        if self.task == 'classification':
            if self.output_dim == 1:
                return (predictions > 0.5).astype(int).flatten()
            else:
                return np.argmax(predictions, axis=1)
        else:  # regression
            return predictions.flatten() if self.output_dim == 1 else predictions
    
    def predict_proba(self, X):
        """
        Predict class probabilities (classification only).
        
        Parameters
        ----------
        X : array-like
            Input features
            
        Returns
        -------
        array
            Predicted probabilities
        """
        if self.task != 'classification':
            raise ValueError("predict_proba only available for classification tasks")
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predicting")
        
        return self._model.predict(X, verbose=0)
    
    def score(self, X, y):
        """
        Calculate accuracy (classification) or R² score (regression).
        
        Parameters
        ----------
        X : array-like
            Test features
        y : array-like
            True labels/values
            
        Returns
        -------
        float
            Accuracy or R² score
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before scoring")
        
        y_pred = self.predict(X)
        
        if self.task == 'classification':
            # Handle one-hot encoded y
            if len(y.shape) > 1 and y.shape[1] > 1:
                y = np.argmax(y, axis=1)
            return accuracy_score(y, y_pred)
        else:  # regression
            return r2_score(y, y_pred)
    
    def get_config(self):
        """Get model configuration"""
        return {
            'input_dim': self.input_dim,
            'hidden_layers': self.hidden_layers,
            'task': self.task,
            'output_dim': self.output_dim,
            'activation': self.activation
        }
    
    def summary(self):
        """Print model architecture summary"""
        if self._model is not None:
            self._model.summary()
        else:
            print("Model not built yet")


class KerasCNN:
    """
    Convolutional Neural Network using Keras/TensorFlow.
    
    Specialized for image data. Uses convolutional layers to learn
    spatial features hierarchically.
    
    Parameters
    ----------
    input_shape : tuple
        Shape of input images (height, width, channels)
        Example: (28, 28, 1) for grayscale MNIST
                 (32, 32, 3) for RGB images
    num_classes : int
        Number of output classes
    filters : list of int, default=[32, 64, 64]
        Number of filters in each convolutional layer
    dense_units : int, default=64
        Number of units in dense layer before output
        
    Examples
    --------
    >>> # MNIST (28x28 grayscale)
    >>> model = KerasCNN(input_shape=(28, 28, 1), num_classes=10)
    >>> model.fit(X_train, y_train, epochs=10)
    >>> 
    >>> # Custom architecture
    >>> model = KerasCNN(
    ...     input_shape=(32, 32, 3),
    ...     num_classes=10,
    ...     filters=[64, 128, 256],
    ...     dense_units=128
    ... )
    """
    
    def __init__(self, input_shape, num_classes, filters=[32, 64, 64], 
                 dense_units=64):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.filters = filters
        self.dense_units = dense_units
        self._model = None
        self.is_fitted = False
        self.history = None
        self._build_model()
    
    def _build_model(self):
        """Build CNN architecture"""
        layers = []
        
        # First convolutional block
        layers.append(tf.keras.layers.Conv2D(
            self.filters[0], (3, 3), 
            activation='relu', 
            input_shape=self.input_shape
        ))
        layers.append(tf.keras.layers.MaxPooling2D((2, 2)))
        
        # Additional convolutional blocks
        for filters in self.filters[1:]:
            layers.append(tf.keras.layers.Conv2D(filters, (3, 3), activation='relu'))
            layers.append(tf.keras.layers.MaxPooling2D((2, 2)))
        
        # Flatten and dense layers
        layers.append(tf.keras.layers.Flatten())
        layers.append(tf.keras.layers.Dense(self.dense_units, activation='relu'))
        layers.append(tf.keras.layers.Dense(self.num_classes, activation='softmax'))
        
        model = tf.keras.models.Sequential(layers)
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self._model = model
    
    def fit(self, X, y, epochs=10, batch_size=32, validation_split=0.2,
            verbose=1, callbacks=None):
        """Train the CNN"""
        self.history = self._model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose,
            callbacks=callbacks
        )
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Predict class labels"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predicting")
        
        predictions = self._model.predict(X, verbose=0)
        return np.argmax(predictions, axis=1)
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predicting")
        
        return self._model.predict(X, verbose=0)
    
    def score(self, X, y):
        """Calculate accuracy"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before scoring")
        
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)
    
    def get_config(self):
        """Get model configuration"""
        return {
            'input_shape': self.input_shape,
            'num_classes': self.num_classes,
            'filters': self.filters,
            'dense_units': self.dense_units
        }
    
    def summary(self):
        """Print model architecture summary"""
        if self._model is not None:
            self._model.summary()
        else:
            print("Model not built yet")