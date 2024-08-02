import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

def load_and_clean_data(filepath):
    """Loads and cleans the Ames Housing dataset."""
    # Load the Ames dataset
    ames_df = pd.read_csv(filepath)

    # Specifically replace blanks in 'MasVnrArea' with 0
    ames_df.replace({'MasVnrArea': 'nan'}, 0, inplace=True)

    # Identify columns with missing values
    missing_values = ames_df.isnull().sum()
    print("\nColumns with missing values and their counts before preprocessing:")
    print(missing_values[missing_values > 0])

    # Handle missing values for numerical columns
    num_imputer = SimpleImputer(strategy='median')
    num_cols_with_missing = ames_df.select_dtypes(include=[np.number]).columns[
        ames_df.select_dtypes(include=[np.number]).isnull().any()
    ].tolist()
    ames_df[num_cols_with_missing] = num_imputer.fit_transform(ames_df[num_cols_with_missing])

    # Handle missing values for categorical columns
    cat_imputer = SimpleImputer(strategy='most_frequent')
    cat_cols_with_missing = ames_df.select_dtypes(exclude=[np.number]).columns[
        ames_df.select_dtypes(exclude=[np.number]).isnull().any()
    ].tolist()
    ames_df[cat_cols_with_missing] = cat_imputer.fit_transform(ames_df[cat_cols_with_missing])

    # Ensure no missing values remain
    print("\nColumns with missing values after imputation and their counts:")
    print(ames_df.isnull().sum()[ames_df.isnull().sum() > 0])

    # Convert boolean columns to integer
    bool_cols = ames_df.select_dtypes(include=[bool]).columns.tolist()
    ames_df[bool_cols] = ames_df[bool_cols].astype(int)

    # Check the number of rows and columns in the dataset before outlier removal
    num_rows, num_columns = ames_df.shape
    print(f"\nThe dataset contains {num_rows} rows and {num_columns} columns before outlier removal.")

    # Remove outliers based on 'GrLivArea'
    initial_row_count = ames_df.shape[0]
    ames_df = ames_df.drop(ames_df[ames_df['GrLivArea'] > 4000].index)
    final_row_count = ames_df.shape[0]
    print(f"Number of rows removed based on 'GrLivArea' > 4000: {initial_row_count - final_row_count}")

    # Check the number of rows and columns in the dataset after outlier removal
    num_rows, num_columns = ames_df.shape
    print(f"\nThe dataset contains {num_rows} rows and {num_columns} columns after outlier removal.")

    return ames_df
