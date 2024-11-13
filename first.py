# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
import joblib

# Step 1: Load Data
df = pd.read_csv(r"data\london_merged.csv")  # Replace with your file path

# Ensure 'timestamp' is in datetime format
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Step 2: Extract time-based features from 'timestamp'
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['month'] = df['timestamp'].dt.month

# Drop 'timestamp' column as we’ve extracted useful features from it
df.drop(columns=['timestamp'], inplace=True)

# Step 3: Data Preprocessing
df.fillna(df.mean(), inplace=True)
df = pd.get_dummies(df, columns=['season', 'weather_code'], drop_first=True)

# Scale Numerical Features (optional for some models)
scaler = StandardScaler()
df[['t1', 't2', 'hum', 'wind_speed']] = scaler.fit_transform(df[['t1', 't2', 'hum', 'wind_speed']])

# Step 4: Original Data Visualizations

# Distribution of bike shares
plt.figure(figsize=(8, 5))
sns.histplot(df['cnt'], bins=30, kde=True)
plt.title("Distribution of Bike Shares (cnt)")
plt.xlabel("Bike Shares (cnt)")
plt.ylabel("Frequency")
plt.show()

# Scatter plot for Temperature vs. Bike Shares
plt.figure(figsize=(8, 5))
sns.scatterplot(x='t1', y='cnt', data=df)
plt.title("Temperature vs. Bike Shares")
plt.xlabel("Temperature (t1)")
plt.ylabel("Bike Shares (cnt)")
plt.show()

# Step 5: Prepare Features and Target Variable
X = df.drop(columns=['cnt'])
y = df['cnt']

# Step 6: Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Train Initial Model (Random Forest)
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)

# Make predictions and evaluate the initial model
y_pred = rf.predict(X_test)
print("Random Forest MAE:", mean_absolute_error(y_test, y_pred))
print("Random Forest RMSE:", mean_squared_error(y_test, y_pred, squared=False))
print("Random Forest R²:", r2_score(y_test, y_pred))

# Step 8: Model Optimization with Grid Search on XGBoost
xgb = XGBRegressor(random_state=42)
param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.7, 0.9],
}
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# Predict and evaluate with the optimized model
y_pred_optimized = best_model.predict(X_test)
print("Optimized XGBoost MAE:", mean_absolute_error(y_test, y_pred_optimized))
print("Optimized XGBoost RMSE:", mean_squared_error(y_test, y_pred_optimized, squared=False))
print("Optimized XGBoost R²:", r2_score(y_test, y_pred_optimized))

# Step 9: Model Result Visualizations

# Plot of actual vs. predicted values
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred_optimized, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color="red", linestyle="--")  # Reference line
plt.xlabel("Actual Bike Shares (cnt)")
plt.ylabel("Predicted Bike Shares (cnt)")
plt.title("Actual vs. Predicted Bike Shares")
plt.show()

# Residual Plot
residuals = y_test - y_pred_optimized
plt.figure(figsize=(8, 5))
sns.histplot(residuals, bins=30, kde=True)
plt.title("Residuals of Predicted Bike Shares")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.show()

# Feature Importance
importances = best_model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance_df)
plt.title("Feature Importance for Bike Share Prediction")
plt.show()

# Step 10: Save and Load the Model
# Save the best model
joblib.dump(best_model, "bike_share_model.pkl")

# Load the saved model (for later use)
loaded_model = joblib.load("bike_share_model.pkl")
