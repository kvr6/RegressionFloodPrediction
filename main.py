import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import r2_score, make_scorer
import time

print("Starting execution...")

# Load the data
print("Loading data...")
train_data = pd.read_csv('path/to/train.csv')  # Update the path to your train.csv
test_data = pd.read_csv('path/to/test.csv')    # Update the path to your test.csv

# Separate features and target variable
print("Separating features and target variable...")
X = train_data.drop(columns=['id', 'FloodProbability'])
y = train_data['FloodProbability']

# Feature engineering: Polynomial features
print("Generating polynomial features...")
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
X_poly = poly.fit_transform(X)

# Scaling features
print("Scaling features using StandardScaler...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_poly)

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define R2 scorer
r2_scorer = make_scorer(r2_score)

# Define parameter grids for Ridge and Lasso
param_grid_ridge = {
    'alpha': [0.1, 1.0, 10.0, 100.0]
}

param_grid_lasso = {
    'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0]
}

# GridSearchCV for Ridge
print("Performing Grid Search for Ridge Regression...")
grid_ridge = GridSearchCV(Ridge(), param_grid_ridge, scoring=r2_scorer, cv=5, n_jobs=-1)
grid_ridge.fit(X_train, y_train)
print(f"Best parameters for Ridge: {grid_ridge.best_params_}, Best R2 Score: {grid_ridge.best_score_:.4f}")

# GridSearchCV for Lasso
print("Performing Grid Search for Lasso Regression...")
grid_lasso = GridSearchCV(Lasso(), param_grid_lasso, scoring=r2_scorer, cv=5, n_jobs=-1)
grid_lasso.fit(X_train, y_train)
print(f"Best parameters for Lasso: {grid_lasso.best_params_}, Best R2 Score: {grid_lasso.best_score_:.4f}")

# Choose the best model based on R2 score
best_model = grid_ridge if grid_ridge.best_score_ > grid_lasso.best_score_ else grid_lasso
print(f"Best Model: {'Ridge' if best_model == grid_ridge else 'Lasso'} Regressor with R2 Score: {best_model.best_score_:.4f}")

# Train the best model on full training data
print("Training the best model on full training data...")
best_model.fit(X_train, y_train)

# Predict on the validation set and test set
print("Predicting on the validation and test set...")
y_val_pred = best_model.predict(X_val)
X_test = test_data.drop(columns=['id'])
X_test_poly = poly.transform(X_test)
X_test_scaled = scaler.transform(X_test_poly)
y_test_pred = best_model.predict(X_test_scaled)

# Evaluate on validation set
val_r2 = r2_score(y_val, y_val_pred)
print(f"Validation R2 Score: {val_r2:.4f}")

# Prepare the submission file
print("Preparing submission file...")
submission = pd.DataFrame({'id': test_data['id'], 'FloodProbability': y_test_pred})
submission.to_csv('submission.csv', index=False)
print("Submission file created: 'submission.csv'")

print("Execution completed.")
