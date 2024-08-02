import time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def apply_pca(X, explained_variance=0.95):
    """Apply PCA to reduce dimensionality while retaining the specified variance."""
    start_time = time.time()
    pca = PCA(n_components=explained_variance)
    X_pca = pca.fit_transform(X)
    end_time = time.time()
    pca_time = end_time - start_time
    
    print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum()}")
    print(f"PCA Training Time: {pca_time:.2f} seconds")
    
    return X_pca, pca


def apply_rfe(X, y, n_features=50):
    """Apply RFE to select the top n features."""
    rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
    rfe = RFE(estimator=rf_model, n_features_to_select=n_features)
    
    start_time = time.time()
    rfe.fit(X, y)
    end_time = time.time()
    rfe_time = end_time - start_time
    
    X_rfe = rfe.transform(X)
    print(f"Training Time for RFE: {rfe_time:.2f} seconds")
    
    return X_rfe, rfe


def evaluate_model(model, X_train, X_test, y_train, y_test, saleprice_scaler):
    """Train the model, make predictions, and evaluate performance."""
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    training_time = end_time - start_time
    
    y_pred = model.predict(X_test)
    y_pred_original = saleprice_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    y_test_original = saleprice_scaler.inverse_transform(y_test.values.reshape(-1, 1)).flatten()
    
    mae = mean_absolute_error(y_test_original, y_pred_original)
    mse = mean_squared_error(y_test_original, y_pred_original)
    
    print(f"Model Performance:")
    print(f"Mean Absolute Error (MAE): ${mae:.2f}")
    print(f"Mean Squared Error (MSE): ${mse:.2f}")
    print(f"Training Time: {training_time:.2f} seconds")
    
    return mae, mse, training_time, y_pred_original, y_test_original
