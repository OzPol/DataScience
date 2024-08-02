# House Price Prediction using Machine Learning Models

## Overview

The goal of this project is to predict house prices using a combination of advanced regression techniques and machine learning models. Accurate prediction of house prices is complex due to the multitude of influencing factors such as location, structural details, and neighborhood characteristics. We utilize advanced machine learning techniques like Support Vector Regression (SVR), Artificial Neural Networks (ANN), and XGBoost to achieve this goal.

## Dataset

The primary dataset used in this project is the Ames Housing dataset, which contains 79 features (explanatory variables) describing various aspects of residential homes in Ames, Iowa. Additionally, we utilize the train and test datasets from the House Prices - Advanced Regression Techniques competition on Kaggle.

- **Original Dataset:** [Ames Housing Dataset](https://www.kaggle.com/datasets/marcopale/housing/data)
- **Competition Dataset:** [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

### Data Directory Structure

data/
│── AmesData.csv
│── data_description.txt


## Repository Structure

group_1/
├── report.pdf
├── source_code/
│   ├── data_cleaning.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── nb1_exploring_and_visualization.ipynb
│   ├── nb2_baseline_simple_linear_regression.ipynb
│   ├── nb3_multivariable_linear_regression.ipynb
│   ├── nb4_lasso_regression.ipynb
│   ├── nb5_ridge_regression.ipynb
│   ├── nb6_elasticnet_regression.ipynb
│   ├── nb7_random_forest.ipynb
│   ├── nb8_gradient_boosting.ipynb
│   ├── nb9_xgboost.ipynb
│   ├── nb10_svr.ipynb
│   ├── nb11_ann.ipynb
│   ├── nb12_ann.ipynb
│   ├── nb13_ensemble_votingregressor.ipynb
│   ├── nb14_ensemble_xgboost_svr_gradientbooster.ipynb
│   ├── readme.md
├── .devcontainer/
│   ├── devcontainer.json
├── data/
│   ├── AmesData.csv
│   ├── data_description.txt
├── requirements.txt
├── setup.sh
└── readme.md


## Prerequisites

- Python 3.11
- Jupyter Notebook or JupyterLab

# Data Science Project

## Setup Instructions

1. **Navigate to the project directory**:

    ```sh
    cd /path/to/your/DataScience
    ```

2. **Ensure you have Python 3.11 installed**:

    Check your Python version by running:

    ```sh
    python --version
    ```

    If Python 3.11 is not your default version, find the path to the Python 3.11 executable. For example, it might be something like `C:\Users\yourusername\AppData\Local\Programs\Python\Python311\python.exe`.

3. **Install the required packages**:

    Replace `/path/to/python311` with your actual Python 3.11 path.

    ```sh
    /path/to/python311/python.exe -m pip install --upgrade pip
    /path/to/python311/python.exe -m pip install -r requirements.txt
    ```

4. **Run Jupyter Notebook** (if required):

    ```sh
    /path/to/python311/python.exe -m jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
    ```

5. **Run the data cleaning script**:

    ```sh
    /path/to/python311/python.exe group_1/source_code/data_cleaning.py
    ```

### Example Paths

For instance, if your Python 3.11 executable is located at `C:\Users\yourusername\AppData\Local\Programs\Python\Python311\python.exe`, you would replace `/path/to/python311/python.exe` with `C:/Users/yourusername/AppData/Local/Programs/Python/Python311/python.exe`.

## Important Notes

- Make sure you replace `/path/to/python311` with the actual path to your Python 3.11 executable.
- The above commands should be run in your terminal or command prompt.


## Setting Up the Environment

### Using GitHub Codespaces

1. **Open Codespaces:**
   - Navigate to your repository on GitHub.
   - Click on the `Code` button and select `Open with Codespaces`.

2. **Configuration:**
   - The `.devcontainer/devcontainer.json` file is set up to automatically install necessary packages from `requirements.txt` when the Codespace starts.

### Local Setup

1. Clone the Repository:

    ```sh
    git clone https://github.com/OzPol/DataScience.git
    cd DataScience
    ```

2. Install Dependencies:

    ```sh
    pip install -r requirements.txt
    ```

3. Start Jupyter Notebook:

    ```sh
    jupyter notebook
    ```

### Alternative Setup without `requirements.txt`

If you prefer to install the libraries manually, you can do so using the following commands:

    ```sh
    pip install pandas numpy matplotlib seaborn plotly scikit-learn catboost xgboost lightgbm tensorflow keras torch scipy statsmodels feature-engine dask openpyxl pyarrow jupyterlab mlxtend tqdm joblib imblearn
    ```


## Data Processing and Analysis

### Load and Clean Data
The initial data loading and cleaning are performed in the `data_cleaning.py` script. This script is imported and used in all the notebooks.

#### data_cleaning.py
**Methods:** 
- `load_data(filepath)`: Loads the dataset from the given filepath.
- `remove_outliers(df, column, threshold)`: Removes outliers based on a specified threshold.
- `fill_missing_values(df)`: Fills missing values in the dataset.
- `clean_data(df)`: Applies all cleaning steps to the dataset.

### Data Preprocessing
The data preprocessing steps, including encoding and scaling, are performed in the `data_preprocessing.py` script. This script is imported into the model notebooks.

#### data_preprocessing.py
**Methods:**
- `preprocess_data(df)`: Preprocesses the dataset by handling categorical and numerical features.

### Exploratory Data Analysis (EDA)
The initial data analysis and processing are performed in the `nb1_exploring_and_visualization.ipynb` notebook.

#### nb1_exploring_and_visualization.ipynb
**Contents:**
- Initial data exploration and visualization.
- Summary statistics.
- Distribution plots.


## Machine Learning Models

Each model has its own notebook, where the cleaned and preprocessed data is used. Evaluation metrics include Mean Squared Error (MSE) and Mean Absolute Error (MAE), as well as runtime metrics for computational efficiency and model training time.

### nb1_edas.ipynb
**Contents:**
- Exploratory Data Analysis (EDA) and Visualization.

### nb1_exploring_and_visualization.ipynb
**Contents:**
- Exploratory Data Analysis (EDA).
- Data visualization techniques.

### nb1_visuals_afterprocessing.ipynb
**Contents:**
- Visualization after data preprocessing.
- Additional EDA after preprocessing steps.

### nb2_baseline_simple_linear_regression.ipynb
**Contents:**
- Baseline simple linear regression model.

### nb3_feature_engineering.ipynb
**Contents:**
- Feature engineering techniques.
- Creation of new features.

### nb4_multivariable_linear_regression.ipynb
**Contents:**
- Multivariable linear regression model.

### nb5_ridge_regression.ipynb
**Contents:**
- Ridge regression model.

### nb6_lasso_regression.ipynb
**Contents:**
- Lasso regression model.

### nb7_elasticnet_regression.ipynb
**Contents:**
- ElasticNet regression model.

### nb8_gradient_boosting.ipynb
**Contents:**
- Gradient boosting model.

### nb9_random_forest.ipynb
**Contents:**
- Random forest model.

### nb10_xgboost.ipynb
**Contents:**
- XGBoost model.

### nb11_svr.ipynb
**Contents:**
- Support Vector Regression (SVR) model.

### nb12_ann.ipynb
**Contents:**
- Artificial Neural Networks (ANN) model.

### nb13_ensemble_votingregressor.ipynb
**Contents:**
- Ensemble model using Voting Regressor.

### nb14_ensemble_xgboost_svr_gradientbooster.ipynb
**Contents:**
- Ensemble model using XGBoost, SVR, and Gradient Boosting.

## Model Tuning

Hyperparameters are tuned using GridSearchCV or RandomizedSearchCV across various models.

## Evaluation Metrics

- **Accuracy:** Mean Squared Error (MSE) and Mean Absolute Error (MAE) on the test set.
- **Runtime:** Computational efficiency and model training time.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Roadmap

### Objective:

- Understand the dataset and clean it up if required.
- Build regression models to predict the sales with respect to single and multiple features.
- Evaluate the models and compare their respective scores like MAE and MSE.

### Strategic Plan of Action:

1. Data Exploration
2. Exploratory Data Analysis (EDA)
3. Data Pre-processing
4. Data Manipulation
5. Feature Selection/Extraction
6. Predictive Modeling

### Tools and Methods:

#### Programming Languages and Environment:

- Python
- Jupyter Notebooks

#### Machine Learning Libraries:

- TensorFlow
- Scikit-learn
- XGBoost

#### Data Collection and Preprocessing:

- Pandas
- NumPy

#### Data Visualization:

- Matplotlib
- Seaborn

#### Model Evaluation:

- SciPy

## References

- https://jse.amstat.org/v19n3/decock.pdf
- https://www.kaggle.com/datasets/marcopale/housing/data
- https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques
- https://www.mdpi.com/2220-9964/12/5/200
- https://seaborn.pydata.org/tutorial.html
- https://matplotlib.org/stable/gallery/index
- https://pandas.pydata.org/pandas-docs/stable/reference/plotting.html
- https://www.tensorflow.org/tutorials/keras/text_classification
- https://www.tensorflow.org/tutorials/keras/regression
- https://www.geeksforgeeks.org/house-price-prediction-using-machine-learning-in-python/
