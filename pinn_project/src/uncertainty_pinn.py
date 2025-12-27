"""
Uncertainty Quantification for Physics-Informed Neural Networks
KAVACH BL4S - Validation Framework

Implements:
1. Monte Carlo Dropout for prediction uncertainty
2. Statistical uncertainty from MC simulations
3. Confidence intervals and calibration metrics
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Tuple, Optional
import warnings


class UncertaintyPINN:
    """
    Physics-Informed Neural Network with Uncertainty Quantification.
    
    Uses Monte Carlo Dropout to estimate epistemic uncertainty
    in predictions, enabling confidence intervals.
    """
    
    def __init__(
        self,
        input_dim: int = 5,
        hidden_layers: list = [64, 64, 64, 32],
        dropout_rate: float = 0.1,
        lambda_smooth: float = 0.1,
        lambda_positive: float = 0.01,
        lambda_gradient: float = 0.05
    ):
        """
        Initialize PINN with dropout for uncertainty estimation.
        
        Args:
            input_dim: Number of input features
            hidden_layers: Neurons per hidden layer
            dropout_rate: Dropout probability for MC dropout
            lambda_*: Physics constraint weights
        """
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.lambda_smooth = lambda_smooth
        self.lambda_positive = lambda_positive
        self.lambda_gradient = lambda_gradient
        
        self.model = self._build_model()
        
    def _build_model(self) -> keras.Model:
        """Build neural network with dropout layers."""
        inputs = keras.Input(shape=(self.input_dim,))
        x = inputs
        
        for units in self.hidden_layers:
            x = keras.layers.Dense(units, activation='tanh')(x)
            # Dropout layer - active during training AND inference for MC dropout
            x = keras.layers.Dropout(self.dropout_rate)(x, training=True)
        
        outputs = keras.layers.Dense(1)(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model
    
    def predict_with_uncertainty(
        self, 
        X: np.ndarray, 
        n_samples: int = 100,
        confidence_level: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty estimates using MC Dropout.
        
        Args:
            X: Input features (n_samples, n_features)
            n_samples: Number of forward passes for MC estimation
            confidence_level: Confidence level for intervals (e.g., 0.95)
            
        Returns:
            mean: Mean prediction
            std: Standard deviation (uncertainty)
            lower: Lower confidence bound
            upper: Upper confidence bound
        """
        predictions = []
        
        for _ in range(n_samples):
            # Forward pass with dropout active (training=True)
            pred = self.model(X, training=True)
            predictions.append(pred.numpy())
        
        predictions = np.array(predictions).squeeze()
        
        # Compute statistics
        mean = np.mean(predictions, axis=0)
        std = np.std(predictions, axis=0)
        
        # Confidence intervals using percentiles
        alpha = 1 - confidence_level
        lower = np.percentile(predictions, 100 * alpha / 2, axis=0)
        upper = np.percentile(predictions, 100 * (1 - alpha / 2), axis=0)
        
        return mean, std, lower, upper
    
    def predict_deterministic(self, X: np.ndarray) -> np.ndarray:
        """Single forward pass without dropout (deterministic)."""
        return self.model(X, training=False).numpy()
    
    @tf.function
    def _compute_gradients(self, X: tf.Tensor) -> tf.Tensor:
        """Compute gradients for physics constraints."""
        with tf.GradientTape() as tape:
            tape.watch(X)
            y = self.model(X, training=True)
        return tape.gradient(y, X)
    
    def physics_loss(self, X: tf.Tensor, y_true: tf.Tensor) -> dict:
        """Compute all loss components."""
        y_pred = self.model(X, training=True)
        
        # Data loss
        data_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        
        # Smoothness: penalize large gradients
        with tf.GradientTape() as tape:
            tape.watch(X)
            y = self.model(X, training=True)
        grads = tape.gradient(y, X)
        depth_grad = grads[:, 1:2]  # depth is feature index 1
        smooth_loss = tf.reduce_mean(tf.square(depth_grad))
        
        # Positivity: penalize negative predictions
        pos_loss = tf.reduce_mean(tf.square(tf.maximum(0.0, -y_pred)))
        
        # Total loss
        total_loss = (
            data_loss + 
            self.lambda_smooth * smooth_loss +
            self.lambda_positive * pos_loss
        )
        
        return {
            'total_loss': total_loss,
            'data_loss': data_loss,
            'smooth_loss': smooth_loss,
            'pos_loss': pos_loss
        }
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 200,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        patience: int = 30,
        verbose: int = 1
    ) -> dict:
        """
        Train the PINN with physics constraints.
        
        Returns:
            history: Training history dictionary
        """
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Convert to tensors
        X_train_tf = tf.constant(X_train, dtype=tf.float32)
        y_train_tf = tf.constant(y_train, dtype=tf.float32)
        X_val_tf = tf.constant(X_val, dtype=tf.float32)
        y_val_tf = tf.constant(y_val, dtype=tf.float32)
        
        history = {
            'loss': [], 'val_loss': [],
            'data_loss': [], 'smooth_loss': [], 'pos_loss': []
        }
        
        best_val_loss = float('inf')
        wait = 0
        best_weights = None
        
        n_batches = len(X_train) // batch_size
        
        for epoch in range(epochs):
            # Shuffle training data
            indices = np.random.permutation(len(X_train))
            X_shuffled = X_train_tf.numpy()[indices]
            y_shuffled = y_train_tf.numpy()[indices]
            
            epoch_losses = []
            
            for i in range(n_batches):
                start = i * batch_size
                end = start + batch_size
                X_batch = tf.constant(X_shuffled[start:end], dtype=tf.float32)
                y_batch = tf.constant(y_shuffled[start:end], dtype=tf.float32)
                
                with tf.GradientTape() as tape:
                    losses = self.physics_loss(X_batch, y_batch)
                    total_loss = losses['total_loss']
                
                gradients = tape.gradient(total_loss, self.model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                epoch_losses.append(total_loss.numpy())
            
            # Validation
            val_losses = self.physics_loss(X_val_tf, y_val_tf)
            val_loss = val_losses['total_loss'].numpy()
            
            # Record history
            avg_loss = np.mean(epoch_losses)
            history['loss'].append(avg_loss)
            history['val_loss'].append(val_loss)
            history['data_loss'].append(losses['data_loss'].numpy())
            history['smooth_loss'].append(losses['smooth_loss'].numpy())
            history['pos_loss'].append(losses['pos_loss'].numpy())
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = self.model.get_weights()
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    if verbose:
                        print(f"\nEarly stopping at epoch {epoch}")
                    break
            
            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}: loss={avg_loss:.6f}, val_loss={val_loss:.6f}")
        
        # Restore best weights
        if best_weights:
            self.model.set_weights(best_weights)
        
        return history
    
    def save(self, filepath: str):
        """Save model weights."""
        self.model.save(filepath)
    
    def load(self, filepath: str):
        """Load model weights."""
        self.model = keras.models.load_model(filepath)


def calculate_mc_uncertainty(
    energy_deposit: np.ndarray, 
    n_histories: int
) -> np.ndarray:
    """
    Calculate statistical uncertainty from Monte Carlo simulation.
    
    Based on Poisson statistics: σ = √N, relative uncertainty = 1/√N
    
    Args:
        energy_deposit: Energy deposited per bin (MeV)
        n_histories: Number of histories simulated
        
    Returns:
        Absolute uncertainty per bin (MeV)
    """
    # Relative uncertainty from counting statistics
    relative_uncertainty = 1.0 / np.sqrt(n_histories)
    
    # Absolute uncertainty
    absolute_uncertainty = energy_deposit * relative_uncertainty
    
    return absolute_uncertainty


def calibration_error(
    y_true: np.ndarray,
    y_pred_mean: np.ndarray,
    y_pred_std: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Compute Expected Calibration Error (ECE) for uncertainty estimates.
    
    A well-calibrated model should have:
    - 68% of true values within ± 1 std
    - 95% of true values within ± 2 std
    
    Args:
        y_true: True values
        y_pred_mean: Predicted mean
        y_pred_std: Predicted standard deviation
        n_bins: Number of bins for calibration
        
    Returns:
        ECE score (lower is better, 0 = perfectly calibrated)
    """
    # Compute z-scores
    z_scores = np.abs(y_true - y_pred_mean) / (y_pred_std + 1e-8)
    
    # Expected coverage at various confidence levels
    confidence_levels = np.linspace(0.1, 0.99, n_bins)
    
    ece = 0.0
    for conf in confidence_levels:
        from scipy.stats import norm
        z_threshold = norm.ppf((1 + conf) / 2)
        actual_coverage = np.mean(z_scores <= z_threshold)
        ece += np.abs(conf - actual_coverage)
    
    return ece / n_bins


def prediction_interval_coverage(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray
) -> float:
    """
    Compute fraction of true values within prediction interval.
    
    For 95% confidence intervals, this should be ~0.95.
    """
    in_interval = (y_true >= lower) & (y_true <= upper)
    return np.mean(in_interval)


if __name__ == "__main__":
    # Example usage
    print("Uncertainty PINN Module")
    print("=" * 50)
    
    # Create model
    pinn = UncertaintyPINN(
        input_dim=5,
        dropout_rate=0.1,
        lambda_smooth=0.1,
        lambda_positive=0.01
    )
    
    # Example prediction with uncertainty
    X_test = np.random.randn(10, 5).astype(np.float32)
    mean, std, lower, upper = pinn.predict_with_uncertainty(X_test, n_samples=50)
    
    print(f"Predictions shape: {mean.shape}")
    print(f"Mean prediction: {mean[:3]}")
    print(f"Uncertainty (std): {std[:3]}")
    print(f"95% CI: [{lower[:3]}, {upper[:3]}]")
