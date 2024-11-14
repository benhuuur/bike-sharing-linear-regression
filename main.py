
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib


df = pd.read_csv(r"data\london_merged.csv")


df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['month'] = df['timestamp'].dt.month
df.drop(columns=['timestamp'], inplace=True)


label_encoder = LabelEncoder()
df['season'] = label_encoder.fit_transform(df['season'])
df['weather_code'] = label_encoder.fit_transform(df['weather_code'])


scaler = StandardScaler()
df[['t1', 't2', 'hum', 'wind_speed']] = scaler.fit_transform(
    df[['t1', 't2', 'hum', 'wind_speed']])


X = df.drop(columns=['cnt'])
y = df['cnt']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


svr = SVR(kernel='rbf')
svr.fit(X_train, y_train)


y_pred_svr = svr.predict(X_test)
print("SVR MAE:", mean_absolute_error(y_test, y_pred_svr))
print("SVR RMSE:", root_mean_squared_error(y_test, y_pred_svr))
print("SVR R²:", r2_score(y_test, y_pred_svr))


param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto'],
    'epsilon': [0.1, 0.2, 0.3],
}
grid_search = GridSearchCV(estimator=svr, param_grid=param_grid,
                           cv=5, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)


best_model_svr = grid_search.best_estimator_


y_pred_svr_optimized = best_model_svr.predict(X_test)
print("Optimized SVR MAE:", mean_absolute_error(y_test, y_pred_svr_optimized))
print("Optimized SVR RMSE:", root_mean_squared_error(
    y_test, y_pred_svr_optimized))
print("Optimized SVR R²:", r2_score(y_test, y_pred_svr_optimized))


def model_stats(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"Model Performance Statistics:")
    print(f"  Mean Absolute Error (MAE): {mae:.4f}")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"  R²: {r2:.4f}")


print("\n--- SVR Model Stats ---")
model_stats(y_test, y_pred_svr)

print("\n--- Optimized SVR Model Stats ---")
model_stats(y_test, y_pred_svr_optimized)


correlation_matrix = df.corr()
plt.figure(figsize=(20, 25))
sns.heatmap(correlation_matrix, annot=True,
            cmap='coolwarm', fmt='.2f', cbar=True)
plt.yticks(rotation=45)
plt.xticks(rotation=45)
plt.title("Correlation Heatmap of Features")
plt.show()


feature_target_corr = correlation_matrix['cnt'].sort_values(ascending=False)
plt.figure(figsize=(20, 25))
sns.barplot(x=feature_target_corr.index,
            y=feature_target_corr.values, palette='viridis')
plt.title("Correlation of Features with Bike Shares (cnt)")
plt.xticks(rotation=45)
plt.ylabel("Correlation Coefficient")
plt.yticks(rotation=45)
plt.show()


plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_svr_optimized, alpha=0.6, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(),
         y_test.max()], color="red", linestyle="--")
plt.xlabel("Actual Bike Shares (cnt)")
plt.ylabel("Predicted Bike Shares (cnt)")
plt.title("Actual vs. Predicted Bike Shares (Optimized SVR)")
plt.show()


joblib.dump(best_model_svr, "bike_share_svr_model.pkl")
loaded_model_svr = joblib.load("bike_share_svr_model.pkl")
