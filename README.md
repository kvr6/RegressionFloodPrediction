# Flood Probability Prediction

This repository contains the code for predicting the probability of flooding in a region based on various features. The primary objective is to build a regression model that predicts the `FloodProbability` using machine learning techniques. This problem is from a [Kaggle competition](https://www.kaggle.com/competitions/playground-series-s4e5/overview)

## Overview

The project uses polynomial features, scaling, and regularization techniques to improve the model's performance. The best model is selected based on the R2 score after hyperparameter tuning with cross-validation.

## Dataset

The dataset consists of the following files:
- `train.csv`: The training dataset with features and the target variable `FloodProbability`.
- `test.csv`: The test dataset for which predictions need to be made.
- `sample_submission.csv`: A sample submission file with the correct format for the predictions.

## Approach

1. **Feature Engineering**: Polynomial features are generated to capture non-linear relationships.
2. **Scaling**: Features are scaled using `StandardScaler`.
3. **Modeling**: Regularized linear models (Ridge and Lasso regression) are used.
4. **Hyperparameter Tuning**: Grid search with cross-validation is performed to find the best parameters.
5. **Prediction**: The best model is used to make predictions on the test set.

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Conda (Miniconda or Anaconda)

### Installation

1. **Clone the repository**:
   ```sh
   git clone https://github.com/yourusername/flood-probability-prediction.git
   cd flood-probability-prediction

2. **Create and activate a virtual environment**:
   ```sh
   conda create -n ml-env python=3.8
   conda activate ml-env

3. **Install the required packages**:
   ```sh
   pip install numpy pandas scikit-learn xgboost

### Running the Code

1. Place the dataset files (train.csv and test.csv) in the project directory.
2. Run the script:
   ```sh
   python main.py

### Code Explanation

1. Loading the data:

   ```sh
   train_data = pd.read_csv('path/to/train.csv')
   test_data = pd.read_csv('path/to/test.csv')

2. Separating features and target variable:

   ```sh
   X = train_data.drop(columns=['id', 'FloodProbability'])
   y = train_data['FloodProbability']

3. Generating polynomial features:

   ```sh
   poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
   X_poly = poly.fit_transform(X)

4. Scaling features

   ```sh
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X_poly)

5. Train-test split:

   ```sh
   X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

6. Hyperparameter tuning with GridSearchCV:

   ```sh
   param_grid_ridge = {'alpha': [0.1, 1.0, 10.0, 100.0]}
   grid_ridge = GridSearchCV(Ridge(), param_grid_ridge, scoring='r2', cv=5, n_jobs=-1)
   grid_ridge.fit(X_train, y_train)

7. Training the best model

   ```sh
   best_model = grid_ridge if grid_ridge.best_score_ > grid_lasso.best_score_ else grid_lasso
   best_model.fit(X_train, y_train)

8. Making predictions

   ```sh
   y_test_pred = best_model.predict(X_test_scaled)

9. Creating the submission file:

   ```sh
   submission = pd.DataFrame({'id': test_data['id'], 'FloodProbability': y_test_pred})
   submission.to_csv('submission.csv', index=False)

### Results

  The best model achieved an R2 score of approximately ~0.85 on the validation set. Further improvements could be explored through additional feature engineering, model tuning, 
  and ensemble methods.




 

