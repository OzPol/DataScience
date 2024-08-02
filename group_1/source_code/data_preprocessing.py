import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    """Preprocesses the cleaned Ames Housing dataset."""
    # Encoding Categorical Variables
    df = pd.get_dummies(df, drop_first=True)
    print("Categorical variables encoded.")

    # Standardize all numerical features except 'SalePrice'
    scaler = StandardScaler()
    num_features = df.select_dtypes(include=[np.number]).columns
    num_features = num_features.drop('SalePrice')
    df[num_features] = scaler.fit_transform(df[num_features])
    print("Numerical features standardized.")

    # Scale 'SalePrice' separately
    saleprice_scaler = StandardScaler()
    df['SalePrice'] = saleprice_scaler.fit_transform(df['SalePrice'].values.reshape(-1, 1))
    print("'SalePrice' standardized.")

    return df, saleprice_scaler
