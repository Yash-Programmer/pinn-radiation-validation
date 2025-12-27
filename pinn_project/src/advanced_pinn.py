"""
KAVACH BL4S - Advanced Physics-Informed Neural Network
Multiple physics constraints for maximum accuracy.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class AdvancedPINN:
    """
    PINN with multiple physics constraints:
    1. Smoothness: dose varies smoothly
    2. Non-negativity: dose must be positive
    3. Gradient correlation: dE/dx matches theory
    """
    
    def __init__(self, input_dim=5, hidden_layers=[64, 64, 64, 32],
                 activation='tanh', learning_rate=0.001,
                 lambda_smooth=0.1, lambda_positive=0.01, lambda_gradient=0.05):
        """
        Args:
            lambda_smooth: Weight for smoothness constraint
            lambda_positive: Weight for non-negativity constraint
            lambda_gradient: Weight for gradient correlation
        """
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.learning_rate = learning_rate
        
        self.lambda_smooth = lambda_smooth
        self.lambda_positive = lambda_positive
        self.lambda_gradient = lambda_gradient
        
        self.model = None
        self.history = None
        
    def build_model(self):
        """Build neural network."""
        
        inputs = layers.Input(shape=(self.input_dim,), name='input')
        
        x = inputs
        for i, units in enumerate(self.hidden_layers):
            x = layers.Dense(
                units,
                activation=self.activation,
                kernel_initializer='glorot_uniform',
                name=f'hidden_{i+1}'
            )(x)
            x = layers.Dropout(0.1)(x)  # MC Dropout
        
        outputs = layers.Dense(1, activation='linear', name='output')(x)
        
        self.model = keras.Model(inputs=inputs, outputs=outputs, name='advanced_pinn')
        return self.model

    def predict(self, X):
        """Make predictions."""
        return self.model.predict(X, verbose=0)

    def predict_with_uncertainty(self, X, n_iterations=100):
        """MC Dropout for uncertainty quantification."""
        predictions = []
        X_tf = tf.constant(X, dtype=tf.float32)
        
        for _ in range(n_iterations):
            # Keep dropout active during inference
            y_pred = self.model(X_tf, training=True)
            predictions.append(y_pred.numpy())
        
        predictions = np.array(predictions)
        mean_prediction = predictions.mean(axis=0)
        std_prediction = predictions.std(axis=0)
        
        return mean_prediction, std_prediction
        """Penalize spiky dose profiles."""
        depth_idx = 1
        
        with tf.GradientTape(persistent=True) as tape2:
            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch(X)
                tape2.watch(X)
                dose = self.model(X, training=True)
            
            ddose = tape1.gradient(dose, X)
            if ddose is not None:
                ddose_ddepth = ddose[:, depth_idx:depth_idx+1]
            else:
                del tape1, tape2
                return tf.constant(0.0)
        
        if ddose is not None:
            d2dose = tape2.gradient(ddose_ddepth, X)
            if d2dose is not None:
                loss = tf.reduce_mean(tf.square(d2dose[:, depth_idx]))
            else:
                loss = tf.constant(0.0)
        else:
            loss = tf.constant(0.0)
        
        del tape1, tape2
        return loss
    
    @tf.function
    def positivity_loss(self, y_pred):
        """Penalize negative dose predictions."""
        return tf.reduce_mean(tf.nn.relu(-y_pred))
    
    @tf.function
    def gradient_loss(self, X, physics_ref):
        """
        Penalize if dose gradient doesn't correlate with physics reference.
        physics_ref: theoretical dE/dx from NIST/Bethe-Bloch
        """
        depth_idx = 1
        
        with tf.GradientTape() as tape:
            tape.watch(X)
            dose = self.model(X, training=True)
        
        ddose = tape.gradient(dose, X)
        
        if ddose is not None and physics_ref is not None:
            ddose_ddepth = ddose[:, depth_idx]
            
            # Normalize both for comparison
            ddose_norm = tf.abs(ddose_ddepth) / (tf.reduce_max(tf.abs(ddose_ddepth)) + 1e-8)
            ref_norm = tf.abs(physics_ref[:, 0]) / (tf.reduce_max(tf.abs(physics_ref[:, 0])) + 1e-8)
            
            return tf.reduce_mean(tf.square(ddose_norm - ref_norm))
        
        return tf.constant(0.0)
    
    @tf.function
    def pinn_loss(self, X, y_true, physics_ref=None):
        """Combined loss: Data + All Physics Constraints."""
        
        y_pred = self.model(X, training=True)
        
        # Data loss
        data_loss = tf.reduce_mean(tf.square(y_pred - y_true))
        
        # Physics losses
        smooth_loss = self.smoothness_loss(X)
        positive_loss = self.positivity_loss(y_pred)
        
        if physics_ref is not None:
            grad_loss = self.gradient_loss(X, physics_ref)
        else:
            grad_loss = tf.constant(0.0)
        
        total_loss = (data_loss +
                     self.lambda_smooth * smooth_loss +
                     self.lambda_positive * positive_loss +
                     self.lambda_gradient * grad_loss)
        
        return total_loss, data_loss, smooth_loss, positive_loss, grad_loss
    
    def train(self, X_train, y_train, X_val, y_val, physics_train=None,
              epochs=200, batch_size=32, verbose=1):
        """Custom training loop with all physics constraints."""
        
        if self.model is None:
            self.build_model()
        
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        X_val_tf = tf.constant(X_val, dtype=tf.float32)
        y_val_tf = tf.constant(y_val, dtype=tf.float32)
        
        history = {
            'loss': [], 'data_loss': [], 'smooth_loss': [],
            'positive_loss': [], 'gradient_loss': [], 'val_loss': []
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
            
            if physics_train is not None:
                physics_shuffled = physics_train[indices]
            
            ep_loss, ep_data, ep_smooth, ep_pos, ep_grad = 0.0, 0.0, 0.0, 0.0, 0.0
            
            for i in range(n_batches):
                start = i * batch_size
                end = min(start + batch_size, n_samples)
                
                X_batch = tf.constant(X_shuffled[start:end], dtype=tf.float32)
                y_batch = tf.constant(y_shuffled[start:end], dtype=tf.float32)
                
                if physics_train is not None:
                    physics_batch = tf.constant(physics_shuffled[start:end], dtype=tf.float32)
                else:
                    physics_batch = None
                
                with tf.GradientTape() as tape:
                    total, data, smooth, pos, grad = self.pinn_loss(
                        X_batch, y_batch, physics_batch
                    )
                
                gradients = tape.gradient(total, self.model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                
                ep_loss += total.numpy()
                ep_data += data.numpy()
                ep_smooth += smooth.numpy()
                ep_pos += pos.numpy()
                ep_grad += grad.numpy()
            
            # Validation
            y_val_pred = self.model(X_val_tf, training=False)
            val_loss = tf.reduce_mean(tf.square(y_val_pred - y_val_tf)).numpy()
            
            history['loss'].append(ep_loss / n_batches)
            history['data_loss'].append(ep_data / n_batches)
            history['smooth_loss'].append(ep_smooth / n_batches)
            history['positive_loss'].append(ep_pos / n_batches)
            history['gradient_loss'].append(ep_grad / n_batches)
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
                      f"Val={val_loss:.6f}")
            
            if patience_counter >= 30:
                print(f"\nEarly stopping at epoch {epoch}")
                if best_weights:
                    self.model.set_weights(best_weights)
                break
        
        self.history = history
        return history
    
    def predict(self, X):
        """Make predictions."""
        return self.model.predict(X, verbose=0)
    
    def evaluate(self, X_test, y_test):
        """Evaluate on test set."""
        y_pred = self.model(tf.constant(X_test, dtype=tf.float32), training=False)
        mse = tf.reduce_mean(tf.square(y_pred - y_test)).numpy()
        mae = tf.reduce_mean(tf.abs(y_pred - y_test)).numpy()
        return mse, mae
    
    def save(self, filepath):
        """Save model."""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def get_summary(self):
        """Print architecture."""
        return self.model.summary()


if __name__ == '__main__':
    model = AdvancedPINN(input_dim=3)
    model.build_model()
    model.get_summary()
    print("\n✓ Advanced PINN test passed!")
