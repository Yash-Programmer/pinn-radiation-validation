"""
KAVACH BL4S - PINN Data Loader
Loads and preprocesses data for Physics-Informed Neural Network training.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path


class PINNDataLoader:
    """Loads and preprocesses data for PINN training."""
    
    def __init__(self, data_dir='../pinn_data'):
        self.data_dir = Path(data_dir)
        self.scaler_X = None
        self.scaler_y = None
        self.feature_cols = None
        self.target_col = None
        
    def load_data(self, feature_cols=['energy_MeV', 'depth_mm', 'beta', 'gamma', 'momentum_MeV_c'], 
                  target_col='sim_dEdx_MeV_per_mm'):
        """
        Load train/val/test splits with scaling.
        
        Args:
            feature_cols: Input feature columns
            target_col: Target column name
            
        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        self.feature_cols = feature_cols
        self.target_col = target_col
        
        # Load CSV files
        train = pd.read_csv(self.data_dir / 'train_split_v2.csv')
        val = pd.read_csv(self.data_dir / 'val_split_v2.csv')
        test = pd.read_csv(self.data_dir / 'test_split_v2.csv')
        
        print(f"Loaded data:")
        print(f"  Train: {len(train)} samples ({train['energy_MeV'].nunique()} energies)")
        print(f"  Val: {len(val)} samples ({val['energy_MeV'].nunique()} energies)")
        print(f"  Test: {len(test)} samples ({test['energy_MeV'].nunique()} energies)")
        
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
        
        print(f"\nFeature shape: {X_train_scaled.shape}")
        print(f"Target shape: {y_train_scaled.shape}")
        print(f"Feature range: [{X_train_scaled.min():.2f}, {X_train_scaled.max():.2f}]")
        print(f"Target range: [{y_train_scaled.min():.4f}, {y_train_scaled.max():.4f}]")
        
        return (X_train_scaled.astype(np.float32), y_train_scaled.astype(np.float32), 
                X_val_scaled.astype(np.float32), y_val_scaled.astype(np.float32),
                X_test_scaled.astype(np.float32), y_test_scaled.astype(np.float32))
    
    def load_with_physics(self, feature_cols=['energy_MeV', 'depth_mm', 'beta', 'gamma', 'momentum_MeV_c'],
                          target_col='sim_dEdx_MeV_per_mm', physics_col='nist_dEdx_MeV_per_mm'):
        """
        Load data including physics reference column for advanced PINN.
        
        Returns:
            Tuple of (X_train, y_train, physics_train, X_val, y_val, physics_val, X_test, y_test)
        """
        X_train, y_train, X_val, y_val, X_test, y_test = self.load_data(
            feature_cols, target_col
        )
        
        # Load physics reference
        train = pd.read_csv(self.data_dir / 'train_split_v2.csv')
        val = pd.read_csv(self.data_dir / 'val_split_v2.csv')
        
        physics_train = train[[physics_col]].values.astype(np.float32)
        physics_val = val[[physics_col]].values.astype(np.float32)
        
        return (X_train, y_train, physics_train,
                X_val, y_val, physics_val,
                X_test, y_test)
    
    def inverse_transform_target(self, y_scaled):
        """Convert scaled predictions back to original units."""
        return self.scaler_y.inverse_transform(y_scaled)
    
    def inverse_transform_features(self, X_scaled):
        """Convert scaled features back to original units."""
        return self.scaler_X.inverse_transform(X_scaled)


if __name__ == '__main__':
    # Test data loading
    loader = PINNDataLoader()
    X_train, y_train, X_val, y_val, X_test, y_test = loader.load_data()
    print("\n✓ Data loading test passed!")
