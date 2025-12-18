import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =====================
# LOAD DATA 
# =====================
X_train = np.load("X_train.npy")
X_test  = np.load("X_test.npy")
y_train = np.load("y_train.npy")
y_test  = np.load("y_test.npy")

# =====================
# EXPERIMENT
# =====================
mlflow.set_experiment("HousePrices_CI")

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

# =====================
# LOGGING
# =====================
mlflow.log_param("model", "RandomForest")
mlflow.log_param("n_estimators", 200)
mlflow.log_param("max_depth", 15)

mlflow.log_metric("MAE", mae)
mlflow.log_metric("RMSE", rmse)
mlflow.log_metric("R2", r2)

mlflow.sklearn.log_model(model, "model")

print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)
