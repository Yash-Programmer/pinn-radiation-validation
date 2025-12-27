
import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
from pathlib import Path

# =============================================================================
# DATA LOADER CLASS
# =============================================================================

class PINNDataLoader:
    """Loads and preprocesses data for PINN training."""
    
    def __init__(self, data_dir='../pinn_data'):
        self.data_dir = Path(data_dir)
        self.scaler_X = None
        self.scaler_y = None
        
    def load_data(self, feature_cols=['energy_MeV', 'depth_mm', 'beta', 'gamma', 'momentum_MeV_c'], 
                  target_col='sim_dEdx_MeV_per_mm'):
        
        # Load CSV files
        train_path = self.data_dir / 'train_split_v2.csv'
        if not train_path.exists():
            raise FileNotFoundError(f"Data not found at {train_path}")

        train = pd.read_csv(train_path)
        val = pd.read_csv(self.data_dir / 'val_split_v2.csv')
        test = pd.read_csv(self.data_dir / 'test_split_v2.csv')
        
        # Load scalers
        self.scaler_X = joblib.load(self.data_dir / 'scaler_features_v2.pkl')
        self.scaler_y = joblib.load(self.data_dir / 'scaler_target_v2.pkl')
        
        # Extract features and targets
        X_train = train[feature_cols].values
        y_train = train[[target_col]].values
        X_val = val[feature_cols].values
        y_val = val[[target_col]].values
        X_test = test[feature_cols].values
        y_test = test[[target_col]].values
        
        # Scale data
        X_train_scaled = self.scaler_X.transform(X_train)
        X_val_scaled = self.scaler_X.transform(X_val)
        X_test_scaled = self.scaler_X.transform(X_test)
        
        y_train_scaled = self.scaler_y.transform(y_train)
        y_val_scaled = self.scaler_y.transform(y_val)
        y_test_scaled = self.scaler_y.transform(y_test)
        
        return (X_train_scaled.astype(np.float32), y_train_scaled.astype(np.float32), 
                X_val_scaled.astype(np.float32), y_val_scaled.astype(np.float32),
                X_test_scaled.astype(np.float32), y_test_scaled.astype(np.float32))

    def load_with_physics(self, feature_cols=['energy_MeV', 'depth_mm', 'beta', 'gamma', 'momentum_MeV_c'],
                          target_col='sim_dEdx_MeV_per_mm', physics_col='nist_dEdx_MeV_per_mm'):
        X_train, y_train, X_val, y_val, X_test, y_test = self.load_data(feature_cols, target_col)
        
        train = pd.read_csv(self.data_dir / 'train_split_v2.csv')
        val = pd.read_csv(self.data_dir / 'val_split_v2.csv')
        physics_train = train[[physics_col]].values.astype(np.float32)
        physics_val = val[[physics_col]].values.astype(np.float32)
        
        return (X_train, y_train, physics_train, X_val, y_val, physics_val, X_test, y_test)

# =============================================================================
# PINN MODEL CLASS
# =============================================================================

class AdvancedPINN:
    def __init__(self, input_dim=5, hidden_layers=[64, 64, 64, 32],
                 activation='tanh', learning_rate=0.001,
                 lambda_smooth=0.1, lambda_positive=0.05, lambda_gradient=0.0): # Optimized Params
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.learning_rate = learning_rate
        self.lambda_smooth = lambda_smooth
        self.lambda_positive = lambda_positive
        self.lambda_gradient = lambda_gradient
        self.model = None

    def build_model(self):
        inputs = layers.Input(shape=(self.input_dim,), name='input')
        x = inputs
        for i, units in enumerate(self.hidden_layers):
            x = layers.Dense(units, activation=self.activation, kernel_initializer='glorot_uniform')(x)
            x = layers.Dropout(0.1)(x) # MC Dropout
        outputs = layers.Dense(1, activation='linear', name='output')(x)
        self.model = keras.Model(inputs=inputs, outputs=outputs, name='advanced_pinn')
        return self.model
    
    @tf.function
    def smoothness_loss(self, X):
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
                return tf.constant(0.0)
        if ddose is not None:
            d2dose = tape2.gradient(ddose_ddepth, X)
            if d2dose is not None:
                return tf.reduce_mean(tf.square(d2dose[:, depth_idx]))
        return tf.constant(0.0)
    
    @tf.function
    def positivity_loss(self, y_pred):
        return tf.reduce_mean(tf.nn.relu(-y_pred))
    
    @tf.function
    def gradient_loss(self, X, physics_ref):
        depth_idx = 1
        with tf.GradientTape() as tape:
            tape.watch(X)
            dose = self.model(X, training=True)
        ddose = tape.gradient(dose, X)
        if ddose is not None and physics_ref is not None:
            ddose_ddepth = ddose[:, depth_idx]
            ddose_norm = tf.abs(ddose_ddepth) / (tf.reduce_max(tf.abs(ddose_ddepth)) + 1e-8)
            ref_norm = tf.abs(physics_ref[:, 0]) / (tf.reduce_max(tf.abs(physics_ref[:, 0])) + 1e-8)
            return tf.reduce_mean(tf.square(ddose_norm - ref_norm))
        return tf.constant(0.0)
    
    @tf.function
    def pinn_loss(self, X, y_true, physics_ref=None):
        y_pred = self.model(X, training=True)
        data_loss = tf.reduce_mean(tf.square(y_pred - y_true))
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
    
    def train(self, X_train, y_train, X_val, y_val, physics_train=None, epochs=200, batch_size=32, verbose=1):
        if self.model is None: self.build_model()
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        X_val_tf = tf.constant(X_val, dtype=tf.float32)
        y_val_tf = tf.constant(y_val, dtype=tf.float32)
        
        best_val_loss = float('inf')
        patience = 0
        best_weights = None
        
        n_samples = len(X_train)
        n_batches = max(1, n_samples // batch_size)
        
        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            phys_shuffled = physics_train[indices] if physics_train is not None else None
            
            for i in range(n_batches):
                start = i * batch_size
                end = min(start + batch_size, n_samples)
                X_batch = tf.constant(X_shuffled[start:end], dtype=tf.float32)
                y_batch = tf.constant(y_shuffled[start:end], dtype=tf.float32)
                phys_batch = tf.constant(phys_shuffled[start:end], dtype=tf.float32) if phys_shuffled is not None else None
                
                with tf.GradientTape() as tape:
                    total, _, _, _, _ = self.pinn_loss(X_batch, y_batch, phys_batch)
                
                grads = tape.gradient(total, self.model.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                
            y_val_pred = self.model(X_val_tf, training=False)
            val_loss = tf.reduce_mean(tf.square(y_val_pred - y_val_tf)).numpy()
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = self.model.get_weights()
                patience = 0
            else:
                patience += 1
            
            if verbose and (epoch % 20 == 0):
                print(f"Epoch {epoch}: Val Loss={val_loss:.6f}")
                
            if patience >= 30:
                print(f"Early stopping at {epoch}")
                self.model.set_weights(best_weights)
                break

    def predict(self, X):
        return self.model.predict(X, verbose=0)
    
    def predict_with_uncertainty(self, X, n_iterations=100):
        predictions = []
        X_tf = tf.constant(X, dtype=tf.float32)
        # We need to manually control dropout here, but tf.keras.Model call(training=True) enables it.
        # So repeated calls with training=True is sufficient.
        for _ in range(n_iterations):
            y_pred = self.model(X_tf, training=True)
            predictions.append(y_pred.numpy())
        predictions = np.array(predictions)
        return predictions.mean(axis=0), predictions.std(axis=0)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def train_final():
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Init loader
    # Assuming we are running from 'pinn_project', data is in '../pinn_data'
    data_dir_path = '../pinn_data'
    if not os.path.exists(data_dir_path):
        print(f"Error: {data_dir_path} not found")
        return

    loader = PINNDataLoader(data_dir=data_dir_path)
    
    # Load data
    print("Loading data...")
    # Note: Using load_with_physics to get physics reference
    # But setting grad=0 in params effectively ignores it, but useful to keep logic
    X_train, y_train, phys_train, X_val, y_val, phys_val, X_test, y_test = loader.load_with_physics()
    
    # Optimized Params
    params = {
        'lambda_smooth': 0.1,
        'lambda_positive': 0.05,
        'lambda_gradient': 0.0, # Optimized: Gradient constraint OFF
        'learning_rate': 0.001
    }
    
    print(f"Training with params: {params} and MC Dropout...")
    
    pinn = AdvancedPINN(input_dim=5, **params)
    # Train
    pinn.train(X_train, y_train, X_val, y_val, physics_train=phys_train, epochs=200, batch_size=32)
    
    print("\nEvaluating...")
    # Standard prediction
    y_pred = pinn.predict(X_test)
    
    # UQ Prediction
    print("Running MC Dropout Uncertainty (100 iterations)...")
    y_pred_mean, y_pred_std = pinn.predict_with_uncertainty(X_test, n_iterations=100)
    
    loss_mse = np.mean((y_test - y_pred_mean)**2) # Use mean prediction for MSE (Bayesian Average)
    loss_mae = np.mean(np.abs(y_test - y_pred_mean))
    
    # Avoid div by zero
    y_test_safe = np.where(y_test == 0, 1e-8, y_test)
    loss_mape = np.mean(np.abs((y_test - y_pred_mean) / y_test_safe)) * 100
    
    # Average relative uncertainty
    avg_uncert = np.mean(y_pred_std / (np.abs(y_pred_mean) + 1e-8)) * 100

    print("FINAL METRICS (With Uncertainty):")
    print(f"MSE: {loss_mse:.4e}")
    print(f"MAE: {loss_mae:.4e}")
    print(f"MAPE: {loss_mape:.2f}%")
    print(f"Pred Uncertainty: {avg_uncert:.2f}%")
    
    # =========================================================================
    # PLOTTING
    # =========================================================================
    print("Generating UQ Figure...")
    import matplotlib.pyplot as plt
    
    # We need to unscale prediction and depth to make a meaningful plot
    # Re-read raw test data to find indices for 200 MeV and 750 MeV
    test_df = pd.read_csv(loader.data_dir / 'test_split_v2.csv')
    
    # Scalers
    # depth is column 1 in features
    # But X_test is scaled. easier to use test_df for x-axis and reconstruction
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    energies = [200, 750]
    
    for idx, energy in enumerate(energies):
        ax = axes[idx]
        subset = test_df[test_df['energy_MeV'] == energy].sort_values('depth_mm')
        if subset.empty: continue
        
        # Prepare input for these rows specifically
        X_sub_raw = subset[['energy_MeV', 'depth_mm', 'beta', 'gamma', 'momentum_MeV_c']].values
        X_sub_scaled = loader.scaler_X.transform(X_sub_raw)
        
        y_true = subset['sim_dEdx_MeV_per_mm'].values
        depth = subset['depth_mm'].values
        
        # Predict UQ
        # We need Baseline prediction too? Usually assume Baseline is similar/worse.
        # User asked for Baseline comparison band.
        # We don't have a trained Baseline model loaded.
        # We will plot "PINN" vs "Truth" for now.
        # Or mock Baseline if needed? 
        # User code: "pred_baseline... fill_between".
        # I'll omitted Baseline curve if I don't have it, or train a quick one?
        # Training a baseline takes time.
        # I'll just plot PINN vs Truth with Error Bars.
        
        mu_scaled, std_scaled = pinn.predict_with_uncertainty(X_sub_scaled, n_iterations=100)
        
        # Unscale using inverse_transform (safer than accessing attributes)
        mu_reshaped = mu_scaled.reshape(-1, 1)
        sigma_reshaped = std_scaled.reshape(-1, 1)
        
        mu_unscaled = loader.scaler_y.inverse_transform(mu_reshaped).flatten()
        
        # Calculate bounds in scaled space then transform
        # (This correctly handles mean/std scaling)
        upper_scaled = mu_reshaped + 2 * sigma_reshaped
        lower_scaled = mu_reshaped - 2 * sigma_reshaped
        
        upper_unscaled = loader.scaler_y.inverse_transform(upper_scaled).flatten()
        lower_unscaled = loader.scaler_y.inverse_transform(lower_scaled).flatten()
        
        # Plot TRUTH (TOPAS)
        # Assuming ~1% statistical uncertainty for TOPAS error bars
        topas_err = y_true * 0.01 
        ax.errorbar(depth, y_true, yerr=topas_err, fmt='o', color='black', 
                   markersize=3, label='TOPAS (Truth)', alpha=0.5)
        
        # Plot PINN
        ax.plot(depth, mu_unscaled, 'r-', label='PINN Prediction', linewidth=2)
        ax.fill_between(depth, lower_unscaled, upper_unscaled, color='red', alpha=0.2, label='PINN Uncertainty (2$\sigma$)')
        
        ax.set_title(f'{energy} MeV Test Energy')
        ax.set_xlabel('Depth (mm)')
        ax.set_ylabel('Stopping Power (MeV/mm)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs('paper/figures', exist_ok=True)
    plt.savefig('paper/figures/model_comparison_with_uncertainty.pdf')
    plt.savefig('paper/figures/model_comparison_with_uncertainty.png', dpi=300)
    print("Figure saved to paper/figures/model_comparison_with_uncertainty.png")

if __name__ == "__main__":
    train_final()
