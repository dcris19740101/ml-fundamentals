"""
Helper module for optimized TensorFlow imports.

This module provides lazy loading and optimized configuration for TensorFlow
to reduce kernel restart times.
"""
import os

# Optimize TensorFlow startup - set before importing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations (can speed up import)

def get_tf():
    """
    Lazy import TensorFlow with optimized settings.
    
    Returns:
        tensorflow module
        
    Usage:
        tf = get_tf()
        model = tf.keras.models.Sequential()
    """
    import tensorflow as tf
    return tf

def get_keras():
    """
    Lazy import Keras from TensorFlow.
    
    Returns:
        tensorflow.keras module
        
    Usage:
        keras = get_keras()
        model = keras.models.Sequential()
    """
    tf = get_tf()
    return tf.keras

# Optional: Configure GPU settings to speed up initialization
def configure_tf_for_cpu_only():
    """
    Configure TensorFlow to use CPU only (faster initialization if GPU not needed).
    Call this BEFORE importing TensorFlow.
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

