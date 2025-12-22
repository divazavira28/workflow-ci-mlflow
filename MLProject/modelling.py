import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.sparse import issparse

mlflow.set_experiment("HousePrices_Experiment")

# =====================
# LOAD DATA
# =====================
X_train = np.load("../preprocessing/houseprices_preprocessing/X_train.npy", allow_pickle=True)
X_test  = np.load("../preprocessing/houseprices_preprocessing/X_test.npy", allow_pickle=True)
y_train = np.load("../preprocessing/houseprices_preprocessing/y_train.npy", allow_pickle=True)
y_test  = np.load("../preprocessing/houseprices_preprocessing/y_test.npy", allow_pickle=True)

# FIX sparse
if isinstance(X_train, np.ndarray) and X_train.dtype == object:
    X_train = X_train.item()
if isinstance(X_test, np.ndarray) and X_test.dtype == object:
    X_test = X_test.item()

if issparse(X_train):
    X_train = X_train.toarray()
if issparse(X_test):
    X_test = X_test.toarray()

# =====================
# AUTOLOG ONLY
# =====================
mlflow.sklearn.autolog()

with mlflow.start_run(run_name="RandomForest_Baseline"):
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

    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R2:", r2)
