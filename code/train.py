import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

train_df = pd.read_csv("../data/allstate-claims-severity/train.csv")
test_df = pd.read_csv("../data/allstate-claims-severity/test.csv")
# Preprocessing data
# Encoding categorical features to numeric
cat_columns_train = train_df.select_dtypes(include="object").columns
cat_columns_test = test_df.select_dtypes(include="object").columns
label_encoder = LabelEncoder()
train_df[cat_columns_train] = train_df[cat_columns_train].apply(
    label_encoder.fit_transform
)
test_df[cat_columns_test] = test_df[cat_columns_test].apply(
    label_encoder.fit_transform
)


y = train_df["loss"].values
X = train_df.drop("loss", axis=1).values


X_test = test_df.values
# check na for features and target
print(np.any(np.isnan(np.sum(y))))
print(np.any(np.isnan(np.sum(X))))
corr_mat = np.corrcoef(X, rowvar=False)
corr_threshold = 0.8
# identify redundant features
mask = np.abs(corr_mat) >= corr_threshold
np.fill_diagonal(mask, False)
# get indices of redundant features
redundant_feature_indices = np.where(mask.any(axis=0))[0]
print(redundant_feature_indices.shape)
X_filtered = np.delete(X, redundant_feature_indices, axis=1)
# feature selection on non redundant feature
selector = SelectKBest(score_func=f_regression, k=102)
X_selected = selector.fit_transform(X_filtered, y)
selected_indices = selector.get_support(indices=True)
print(f"selected features: {selected_indices}")

# select features in test dataset
X_test_filtered = np.delete(X_test, redundant_feature_indices, axis=1)
X_test_selected = selector.transform(X_test_filtered)


X_train, X_val, y_train, y_val = train_test_split(
    X_selected, y, test_size=0.2, random_state=42
)
# simple model using OLS
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
print(mean_squared_error(y_val, y_pred))
print(mean_squared_error(y_val, y_pred, squared=False))

# Extreme Gradient Boosting

model_xgb = XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=3)
model_xgb.fit(X_train, y_train)
y_pred_xgb = model_xgb.predict(
    X_val,
)
print(mean_squared_error(y_pred_xgb, y_val))
print(mean_squared_error(y_pred_xgb, y_val, squared=False))
