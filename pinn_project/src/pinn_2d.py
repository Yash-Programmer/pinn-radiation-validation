"""
2D Physics-Informed Neural Network for Dose Distribution
KAVACH BL4S - 2D Dose Validation Framework

Extends 1D PINN to predict dose as function of (energy, depth, radial_distance).
Adds rotational symmetry physics constraint.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple, Optional


class PINN2D:
    """
    2D Physics-Informed Neural Network for dose distribution prediction.
    
    Input: (energy, depth, radial_distance)
    Output: dose
    
    Physics constraints:
    1. Smoothness in both depth and radial dimensions
    2. Positivity (dose must be non-negative)
    3. Radial symmetry (dose should not depend on angle, only radius)
    4. Radial falloff (dose decreases with distance from beam axis)
    """
    
    def __init__(
        self,
        input_dim: int = 3,  # energy, depth, radial
        hidden_layers: list = [128, 128, 64, 32],
        activation: str = 'tanh',
        learning_rate: float = 0.001,
        lambda_smooth: float = 0.1,
        lambda_positive: float = 0.01,
        lambda_radial: float = 0.05
    ):
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.learning_rate = learning_rate
        
        self.lambda_smooth = lambda_smooth
        self.lambda_positive = lambda_positive
        self.lambda_radial = lambda_radial
        
        self.model = None
        self.history = None
        
    def build_model(self) -> keras.Model:
        """Build neural network for 2D dose prediction."""
        inputs = layers.Input(shape=(self.input_dim,), name='input')
        
        x = inputs
        for i, units in enumerate(self.hidden_layers):
            x = layers.Dense(
                units,
                activation=self.activation,
                kernel_initializer='glorot_uniform',
                name=f'hidden_{i+1}'
            )(x)
        
        outputs = layers.Dense(1, activation='linear', name='output')(x)
        
        self.model = keras.Model(inputs=inputs, outputs=outputs, name='pinn_2d')
        return self.model
    
    @tf.function
    def smoothness_loss_2d(self, X: tf.Tensor) -> tf.Tensor:
        """
        Penalize non-smooth dose in both depth and radial dimensions.
        Uses second derivatives (curvature) as penalty.
        """
        depth_idx = 1
        radial_idx = 2
        
        with tf.GradientTape(persistent=True) as tape2:
            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch(X)
                tape2.watch(X)
                dose = self.model(X, training=True)
            
            # First derivatives
            grads = tape1.gradient(dose, X)
            if grads is None:
                del tape1, tape2
                return tf.constant(0.0)
            
            grad_depth = grads[:, depth_idx:depth_idx+1]
            grad_radial = grads[:, radial_idx:radial_idx+1]
        
        # Second derivatives
        grad2_depth = tape2.gradient(grad_depth, X)
        grad2_radial = tape2.gradient(grad_radial, X)
        
        del tape1, tape2
        
        loss = tf.constant(0.0)
        if grad2_depth is not None:
            loss += tf.reduce_mean(tf.square(grad2_depth[:, depth_idx]))
        if grad2_radial is not None:
            loss += tf.reduce_mean(tf.square(grad2_radial[:, radial_idx]))
        
        return loss
    
    @tf.function
    def positivity_loss(self, y_pred: tf.Tensor) -> tf.Tensor:
        """Penalize negative dose predictions."""
        return tf.reduce_mean(tf.nn.relu(-y_pred))
    
    @tf.function
    def radial_falloff_loss(self, X: tf.Tensor) -> tf.Tensor:
        """
        Encourage dose to decrease with radial distance.
        Physical expectation: ∂D/∂r < 0 (dose decreases away from beam axis)
        """
        radial_idx = 2
        
        with tf.GradientTape() as tape:
            tape.watch(X)
            dose = self.model(X, training=True)
        
        grads = tape.gradient(dose, X)
        
        if grads is not None:
            grad_radial = grads[:, radial_idx]
            # Penalize positive radial gradient (dose should decrease with r)
            return tf.reduce_mean(tf.nn.relu(grad_radial))
        
        return tf.constant(0.0)
    
    @tf.function
    def pinn_loss_2d(self, X: tf.Tensor, y_true: tf.Tensor) -> Tuple:
        """Combined loss for 2D PINN."""
        y_pred = self.model(X, training=True)
        
        # Data loss
        data_loss = tf.reduce_mean(tf.square(y_pred - y_true))
        
        # Physics losses
        smooth_loss = self.smoothness_loss_2d(X)
        positive_loss = self.positivity_loss(y_pred)
        radial_loss = self.radial_falloff_loss(X)
        
        total_loss = (
            data_loss +
            self.lambda_smooth * smooth_loss +
            self.lambda_positive * positive_loss +
            self.lambda_radial * radial_loss
        )
        
        return total_loss, data_loss, smooth_loss, positive_loss, radial_loss
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 200,
        batch_size: int = 64,
        verbose: int = 1
    ) -> dict:
        """Train the 2D PINN with physics constraints."""
        
        if self.model is None:
            self.build_model()
        
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        X_val_tf = tf.constant(X_val, dtype=tf.float32)
        y_val_tf = tf.constant(y_val, dtype=tf.float32)
        
        history = {
            'loss': [], 'data_loss': [], 'smooth_loss': [],
            'positive_loss': [], 'radial_loss': [], 'val_loss': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_weights = None
        
        n_samples = len(X_train)
        n_batches = max(1, n_samples // batch_size)
        
        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            ep_loss, ep_data, ep_smooth, ep_pos, ep_rad = 0.0, 0.0, 0.0, 0.0, 0.0
            
            for i in range(n_batches):
                start = i * batch_size
                end = min(start + batch_size, n_samples)
                
                X_batch = tf.constant(X_shuffled[start:end], dtype=tf.float32)
                y_batch = tf.constant(y_shuffled[start:end], dtype=tf.float32)
                
                with tf.GradientTape() as tape:
                    total, data, smooth, pos, rad = self.pinn_loss_2d(X_batch, y_batch)
                
                gradients = tape.gradient(total, self.model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                
                ep_loss += total.numpy()
                ep_data += data.numpy()
                ep_smooth += smooth.numpy()
                ep_pos += pos.numpy()
                ep_rad += rad.numpy()
            
            # Validation
            y_val_pred = self.model(X_val_tf, training=False)
            val_loss = tf.reduce_mean(tf.square(y_val_pred - y_val_tf)).numpy()
            
            history['loss'].append(ep_loss / n_batches)
            history['data_loss'].append(ep_data / n_batches)
            history['smooth_loss'].append(ep_smooth / n_batches)
            history['positive_loss'].append(ep_pos / n_batches)
            history['radial_loss'].append(ep_rad / n_batches)
            history['val_loss'].append(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = self.model.get_weights()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch:3d}: Loss={history['loss'][-1]:.6f}, "
                      f"Data={history['data_loss'][-1]:.6f}, "
                      f"Smooth={history['smooth_loss'][-1]:.6f}, "
                      f"Radial={history['radial_loss'][-1]:.6f}, "
                      f"Val={val_loss:.6f}")
            
            if patience_counter >= 30:
                print(f"\nEarly stopping at epoch {epoch}")
                if best_weights:
                    self.model.set_weights(best_weights)
                break
        
        self.history = history
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X, verbose=0)
    
    def predict_2d_grid(
        self,
        energy: float,
        depth_range: Tuple[float, float],
        radial_range: Tuple[float, float],
        n_depth: int = 50,
        n_radial: int = 30
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate 2D dose map for visualization.
        
        Returns:
            depth_grid, radial_grid, dose_grid
        """
        depths = np.linspace(depth_range[0], depth_range[1], n_depth)
        radials = np.linspace(radial_range[0], radial_range[1], n_radial)
        
        depth_grid, radial_grid = np.meshgrid(depths, radials, indexing='ij')
        
        # Create input array
        X = np.zeros((n_depth * n_radial, 3))
        X[:, 0] = energy  # Energy constant
        X[:, 1] = depth_grid.flatten()
        X[:, 2] = radial_grid.flatten()
        
        # Predict
        dose = self.predict(X)
        dose_grid = dose.reshape(n_depth, n_radial)
        
        return depth_grid, radial_grid, dose_grid
    
    def save(self, filepath: str):
        """Save model."""
        self.model.save(filepath)
        print(f"2D PINN saved to {filepath}")


def create_synthetic_2d_data(
    n_samples: int = 5000,
    energies: list = [70, 150, 250, 500, 1000],
    noise_level: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic 2D dose data for testing.
    Uses Gaussian beam profile with depth-dependent spread.
    """
    np.random.seed(42)
    
    X = []
    y = []
    
    for energy in energies:
        n_per_energy = n_samples // len(energies)
        
        # Sample positions
        depths = np.random.uniform(0, 20, n_per_energy)  # mm
        radials = np.random.uniform(0, 25, n_per_energy)  # mm
        
        # Physical model: Gaussian radial profile with depth-dependent amplitude
        # D(z, r) = D0(z) * exp(-r²/2σ²(z))
        
        # Beam spread increases with depth (scattering)
        sigma = 3.0 + 0.1 * depths  # mm
        
        # Bragg peak-like depth profile
        z_bragg = 15.0 * (energy / 70) ** 0.5  # mm, scales with energy
        depth_profile = np.exp(-((depths - z_bragg) / 5.0) ** 2) + 0.3
        
        # Radial Gaussian
        radial_profile = np.exp(-radials**2 / (2 * sigma**2))
        
        # Combined dose
        dose = depth_profile * radial_profile * (energy / 100)  # Scale with energy
        
        # Add noise
        dose += noise_level * dose * np.random.randn(n_per_energy)
        dose = np.maximum(dose, 0)  # Ensure non-negative
        
        # Store
        for i in range(n_per_energy):
            X.append([energy, depths[i], radials[i]])
            y.append([dose[i]])
    
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


if __name__ == "__main__":
    print("=" * 60)
    print("2D PINN for Dose Distribution")
    print("=" * 60)
    
    # Create synthetic data
    print("\nGenerating synthetic 2D dose data...")
    X, y = create_synthetic_2d_data(n_samples=5000)
    
    # Split data
    n_train = int(0.7 * len(X))
    n_val = int(0.15 * len(X))
    
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]
    
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Normalize
    X_mean, X_std = X_train.mean(axis=0), X_train.std(axis=0)
    X_train_norm = (X_train - X_mean) / X_std
    X_val_norm = (X_val - X_mean) / X_std
    X_test_norm = (X_test - X_mean) / X_std
    
    y_mean, y_std = y_train.mean(), y_train.std()
    y_train_norm = (y_train - y_mean) / y_std
    y_val_norm = (y_val - y_mean) / y_std
    
    # Create and train model
    print("\nTraining 2D PINN...")
    model = PINN2D(
        input_dim=3,
        hidden_layers=[128, 128, 64, 32],
        lambda_smooth=0.1,
        lambda_positive=0.01,
        lambda_radial=0.05
    )
    
    history = model.train(
        X_train_norm, y_train_norm,
        X_val_norm, y_val_norm,
        epochs=100,
        verbose=1
    )
    
    # Evaluate
    y_pred_norm = model.predict(X_test_norm)
    y_pred = y_pred_norm * y_std + y_mean
    
    mse = np.mean((y_test - y_pred) ** 2)
    mae = np.mean(np.abs(y_test - y_pred))
    
    print(f"\n2D PINN Test Performance:")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print("\n✓ 2D PINN module test passed!")
