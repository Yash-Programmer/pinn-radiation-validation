"""
KAVACH BL4S - Baseline Neural Network
Standard feedforward neural network (no physics constraints).
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class BaselineNN:
    """
    Standard feedforward neural network.
    No physics knowledge - pure data-driven learning.
    """
    
    def __init__(self, input_dim=5, hidden_layers=[64, 64, 64, 32], 
                 activation='relu', learning_rate=0.001):
        """
        Args:
            input_dim: Number of input features
            hidden_layers: List of hidden layer sizes
            activation: Activation function
            learning_rate: Adam optimizer learning rate
        """
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.learning_rate = learning_rate
        self.model = None
        self.history = None
        
    def build_model(self):
        """Construct the neural network architecture."""
        
        model = keras.Sequential(name='baseline_nn')
        
        # Input layer
        model.add(layers.Input(shape=(self.input_dim,), name='input'))
        
        # Hidden layers
        for i, units in enumerate(self.hidden_layers):
            model.add(layers.Dense(
                units,
                activation=self.activation,
                kernel_initializer='glorot_uniform',
                name=f'hidden_{i+1}'
            ))
        
        # Output layer
        model.add(layers.Dense(1, activation='linear', name='output'))
        
        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=200, 
              batch_size=32, verbose=1):
        """
        Train the model.
        
        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=30,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=15,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return self.history
    
    def predict(self, X):
        """Make predictions."""
        return self.model.predict(X, verbose=0)
    
    def evaluate(self, X_test, y_test):
        """Evaluate on test set."""
        return self.model.evaluate(X_test, y_test, verbose=0)
    
    def save(self, filepath):
        """Save model."""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def get_summary(self):
        """Print model architecture."""
        return self.model.summary()


if __name__ == '__main__':
    # Quick test
    model = BaselineNN(input_dim=3)
    model.build_model()
    model.get_summary()
    print("\n✓ Baseline NN test passed!")
