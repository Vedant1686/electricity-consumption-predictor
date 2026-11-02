# ==========================================
# SMART ELECTRICITY CONSUMPTION PREDICTION
# MODEL TRAINING
# ==========================================

import os
import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "energydata_complete.csv"

MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)


print("\n===================================")
print("SMART ELECTRICITY PREDICTION SYSTEM")
print("===================================")


# ==========================================
# 2. LOAD DATASET
# ==========================================

print("\nLoading dataset...")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        "energydata_complete.csv not found. "
        "Place it in the project folder."
    )

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")

print("Dataset shape:", df.shape)


# ==========================================
# 3. DATA PREPROCESSING
# ==========================================

print("\nPreprocessing data...")

# Convert date using exact dataset format
df["date"] = pd.to_datetime(
    df["date"],
    format="%Y-%m-%d %H:%M:%S",
    errors="raise"
)

# Sort chronologically
df = df.sort_values("date").reset_index(drop=True)

# Remove duplicate records
df = df.drop_duplicates(subset=["date"])

# Extract time features
df["hour"] = df["date"].dt.hour

df["day_of_week"] = df["date"].dt.dayofweek

# Circular time features
df["hour_sin"] = np.sin(
    2 * np.pi * df["hour"] / 24
)

df["hour_cos"] = np.cos(
    2 * np.pi * df["hour"] / 24
)

print("Preprocessing completed!")


# ==========================================
# 4. SELECT INPUT FEATURES
# ==========================================

features = [
    "lights",
    "T1",
    "RH_1",
    "T2",
    "RH_2",
    "T_out",
    "RH_out",
    "Windspeed",
    "hour_sin",
    "hour_cos",
    "day_of_week"
]

target = "Appliances"

X = df[features].copy()

y = df[target].copy()

# Check missing values
if X.isnull().any().any() or y.isnull().any():
    raise ValueError("Dataset contains missing values.")

print("\nSelected features:")
print(features)

print("\nTarget variable:", target)


# ==========================================
# 5. TRAIN-TEST SPLIT
# ==========================================

print("\nSplitting dataset...")

split = int(len(df) * 0.80)

X_train = X.iloc[:split]

X_test = X.iloc[split:]

y_train = y.iloc[:split]

y_test = y.iloc[split:]

print("Training records:", len(X_train))

print("Testing records:", len(X_test))


# ==========================================
# 6. DEFINE ML ALGORITHMS
# ==========================================

models = {

    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        max_depth=18,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": HistGradientBoostingRegressor(
        max_iter=100,
        learning_rate=0.08,
        random_state=42
    )

}


# ==========================================
# 7. MODEL TRAINING AND EVALUATION
# ==========================================

results = {}

best_model = None

best_model_name = None

best_mae = float("inf")

best_predictions = None

print("\nStarting model training...")


for name, model in models.items():

    print("\nTraining:", name)

    # Train model
    model.fit(X_train, y_train)

    # Predict test data
    predictions = model.predict(X_test)

    # Calculate evaluation metrics
    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    # Store results
    results[name] = {

        "MAE": round(float(mae), 3),

        "RMSE": round(float(rmse), 3),

        "R2 Score": round(float(r2), 4)

    }

    print("MAE:", round(mae, 3))

    print("RMSE:", round(rmse, 3))

    print("R2 Score:", round(r2, 4))

    # Select best model based on MAE
    if mae < best_mae:

        best_mae = mae

        best_model = model

        best_model_name = name

        best_predictions = predictions


# ==========================================
# 8. SAVE BEST MODEL
# ==========================================

print("\nSaving best model...")

joblib.dump(
    best_model,
    MODEL_DIR / "best_model.pkl"
)


# ==========================================
# 9. SAVE EVALUATION RESULTS
# ==========================================

# Save first 100 test predictions
comparison = pd.DataFrame({

    "Actual": y_test.iloc[:100].tolist(),

    "Predicted": best_predictions[:100].tolist()

})

output = {

    "best_model": best_model_name,

    "results": results,

    "features": features,

    "defaults": {
        key: float(value)
        for key, value in X_train.median().items()
    },

    "test_actual": comparison["Actual"].tolist(),

    "test_predicted": comparison["Predicted"].tolist(),

    "dataset_records": int(len(df)),

    "training_records": int(len(X_train)),

    "testing_records": int(len(X_test))

}


with open(MODEL_DIR / "metrics.json", "w") as file:

    json.dump(output, file, indent=4)


# ==========================================
# 10. FINAL OUTPUT
# ==========================================

print("\n===================================")

print("MODEL TRAINING COMPLETED!")

print("===================================")

print("\nBest Model:", best_model_name)

print("Best MAE:", round(best_mae, 3))

print("\nAll Model Results:")

for name, scores in results.items():

    print("\n", name)

    print(scores)


print("\nFiles Generated:")

print("models/best_model.pkl")

print("models/metrics.json")

print("\nYou can now run the Streamlit application!")

print("===================================")