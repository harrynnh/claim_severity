import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

# Load the training and test datasets
train_df = pd.read_csv("../data/allstate-claims-severity/train.csv")
test_df = pd.read_csv("../data/allstate-claims-severity/test.csv")

# Preprocessing data
# Encoding categorical features to numeric using LabelEncoder
cat_columns_train = train_df.select_dtypes(include="object").columns
cat_columns_test = test_df.select_dtypes(include="object").columns
label_encoder = LabelEncoder()
train_df[cat_columns_train] = train_df[cat_columns_train].apply(
    label_encoder.fit_transform
)
test_df[cat_columns_test] = test_df[cat_columns_test].apply(
    label_encoder.fit_transform
)

# Extract features (X) and target variable (y) from the training dataset
y = train_df["loss"].values
X = train_df.drop("loss", axis=1).values

# Extract features from the test dataset
X_test = test_df.values

# Check for missing values
print(np.any(np.isnan(np.sum(y))))
print(np.any(np.isnan(np.sum(X))))

# Compute correlation matrix
corr_mat = np.corrcoef(X, rowvar=False)
corr_threshold = 0.8

# Identify redundant features
mask = np.abs(corr_mat) >= corr_threshold
np.fill_diagonal(mask, False)
redundant_feature_indices = np.where(mask.any(axis=0))[0]
print(redundant_feature_indices.shape)

# Remove redundant features from X
X_filtered = np.delete(X, redundant_feature_indices, axis=1)

# Feature selection on non-redundant features using SelectKBest
selector = SelectKBest(score_func=f_regression, k=102)
X_selected = selector.fit_transform(X_filtered, y)
selected_indices = selector.get_support(indices=True)
print(f"Selected features: {selected_indices}")

# Select corresponding features in the test dataset
X_test_filtered = np.delete(X_test, redundant_feature_indices, axis=1)
X_test_selected = selector.transform(X_test_filtered)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_selected, y, test_size=0.2, random_state=42
)

# Model training and evaluation
# Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
rmse = np.sqrt(mse)
print(f"Linear Regression MSE: {mse}")
print(f"Linear Regression RMSE: {rmse}")

# XGBoost
model_xgb = XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=3)
model_xgb.fit(X_train, y_train)
y_pred_xgb = model_xgb.predict(X_val)
mse_xgb = mean_squared_error(y_val, y_pred_xgb)
rmse_xgb = np.sqrt(mse_xgb)
print(f"XGBoost MSE: {mse_xgb}")
print(f"XGBoost RMSE: {rmse_xgb}")
