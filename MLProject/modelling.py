import numpy as np
import mlflow
import mlflow.sklearn
import json

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.sparse import issparse

# =====================
# LOAD DATA (CI-SAFE PATH)
# =====================
X_train = np.load("X_train.npy", allow_pickle=True)
X_test  = np.load("X_test.npy", allow_pickle=True)
y_train = np.load("y_train.npy", allow_pickle=True)
y_test  = np.load("y_test.npy", allow_pickle=True)

# =====================
# FIX SPARSE / OBJECT
# =====================
if isinstance(X_train, np.ndarray) and X_train.dtype == object:
    X_train = X_train.item()
if isinstance(X_test, np.ndarray) and X_test.dtype == object:
    X_test = X_test.item()

if issparse(X_train):
    X_train = X_train.toarray()
if issparse(X_test):
    X_test = X_test.toarray()

# =====================
# EXPERIMENT
# =====================
mlflow.set_experiment("HousePrices_CI")

with mlflow.start_run(run_name="RandomForest_CI"):

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
    # LOG PARAMS & METRICS
    # =====================
    mlflow.log_param("model", "RandomForest")
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("max_depth", 15)

    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R2", r2)

    # =====================
    # ARTEFAK (ADVANCE POINT)
    # =====================
    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    mlflow.log_artifact("metrics.json")

    # =====================
    # LOG MODEL
    # =====================
    mlflow.sklearn.log_model(model, "model")

    print("MAE :", mae)
    print("RMSE:", rmse)
    print("R2  :", r2)
