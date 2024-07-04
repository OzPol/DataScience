
# House Price Prediction using Machine Learning Models

## Overview

The goal of this project is to predict house prices using a combination of advanced regression techniques & machine learning models. Accurate prediction of house prices is complex due to the multitude of influencing factors such as location, structural details, and neighborhood characteristics. We utilize advanced machine learning techniques like Support Vector Regression (SVR), Artificial Neural Networks (ANN), and XGBoost to achieve this goal.

## Dataset

The primary dataset used in this project is the Ames Housing dataset, which contains 79 features (explanatory variables) describing various aspects of residential homes in Ames, Iowa. Additionally, we utilize the train and test datasets from the House Prices - Advanced Regression Techniques competition on Kaggle.

Original Dataset:
Courtesy of Prof. Dean De Cock - Thank you! 

https://www.kaggle.com/datasets/marcopale/housing/data
https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques


### Original Dataset

- **Target:** 734 rows in `target.csv`
- **Test:** 734 rows in `test.csv`
- **Train:** 2198 rows in `train.csv`

### Competition Dataset

- **AmesHousing:** 2931 rows in `AmesHousing.csv`
- **Sample Submission:** 1460 rows in `sample_submission.csv`
- **Test:** 1460 rows in `test.csv`
- **Train:** 1461 rows in `train.csv`

The competition dataset is divided differently than the original dataset, allowing us to test our models on both and compare their performance. The dataset is manageable, with the following characteristics:

- **Volume:** Moderate size with over 80 features.
- **Variety:** Diverse attributes including numerical, categorical, and ordinal data.
- **Velocity:** Static dataset, no real-time data.
- **Veracity:** High quality with well-documented attributes.
- **Value:** Significant value in accurately predicting prices for buyers, sellers, and real estate professionals.


### Data Directory Structure

      data/
      ├── original/
      │ ├── AmesHousing.csv
      │ ├── target.csv
      │ ├── test.csv
      │ ├── train.csv
      ├── competition/
      │ ├── AmesHousing.csv
      │ ├── data_description.txt
      │ ├── sample_submission.csv
      │ ├── test.csv
      │ ├── train.csv


### Repository Structure


      ├── .devcontainer/
      │ └── devcontainer.json
      ├── data/
      │ ├── original/
      │ │ ├── AmesHousing.csv
      │ │ ├── target.csv
      │ │ ├── test.csv
      │ │ ├── train.csv
      │ ├── competition/
      │ │ ├── AmesHousing.csv
      │ │ ├── data_description.txt
      │ │ ├── sample_submission.csv
      │ │ ├── test.csv
      │ │ ├── train.csv
      ├── notebooks/
      │ └── initial_analysis.ipynb
      ├── requirements.txt
      ├── setup.sh
      └── README.md



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

## Data Processing and Analysis

### Load and Analyze Data:

The initial data analysis and processing are performed in the `notebooks/initial_analysis.ipynb` notebook.

### Handling Categorical Columns:

One-hot encoding is used for categorical features.

### Handling Missing Values:

Missing values are handled using appropriate imputation techniques.

### Handling Outliers:

Outliers are identified and treated to improve model performance.

### Feature Engineering:

New features are created from existing data to enhance model accuracy.

## Machine Learning Models

### Baseline Models:

- Linear Regression
- Decision Trees

### Advanced Models:

- Support Vector Regression (SVR)
- Artificial Neural Networks (ANN)
- XGBoost
- Random Forest
- Gradient Boosting

### Model Tuning:

Hyperparameters are tuned using GridSearchCV.

## Evaluation Metrics

- **Accuracy:** Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE) on the test set.
- **Runtime:** Computational efficiency and model training time.

## Contributions

### Branching:

Each feature or bug fix should be worked on in its own branch. Use meaningful branch names to easily identify the purpose of the branch (e.g., feature-data-preprocessing, bugfix-missing-values).

### Pull Requests:

Ensure your branch is up to date with the main branch before creating a pull request. Provide a clear description of the changes and the problem it solves.

## License

This project is licensed under the MIT License. See the LICENSE file for details.


## Roadmap

### Objective:
- Understand the dataset and clean it up if required.
- Build regression models to predict the sales with respect to single and multiple features.
- Evaluate the models and compare their respective scores like R2, RMSE, etc.

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


### References:

https://jse.amstat.org/v19n3/decock.pdf
https://www.kaggle.com/datasets/marcopale/housing/data
https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques
https://www.mdpi.com/2220-9964/12/5/200
https://seaborn.pydata.org/tutorial.html
https://matplotlib.org/stable/gallery/index
https://pandas.pydata.org/pandas-docs/stable/reference/plotting.html
https://www.tensorflow.org/tutorials/keras/text_classification
https://www.tensorflow.org/tutorials/keras/regression
https://www.geeksforgeeks.org/house-price-prediction-using-machine-learning-in-python/



A Survey of Methods and Input Data Types for House Price Prediction
https://www.mdpi.com/2220-9964/12/5/200

House Price Prediction using a Machine Learning Model: A Survey of Literature:
https://www.irjet.net/archives/V9/i5/IRJET-V9I5455.pdf

https://www.researchgate.net/profile/Ubaidullah-Nor-Hasbiah/publication/347584803_House_Price_Prediction_using_a_Machine_Learning_Model_A_Survey_of_Literature/links/600617a7a6fdccdcb8642a88/House-Price-Prediction-using-a-Machine-Learning-Model-A-Survey-of-Literature.pdf


