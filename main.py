import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==================================================
# LOAD DATASET
# ==================================================

df = pd.read_csv("data/crime_dataset_india.csv")

# ==================================================
# CLEANING
# ==================================================

df["Weapon Used"] = df["Weapon Used"].fillna("Unknown")

df.drop(columns=["Date Case Closed"], inplace=True)

# ==================================================
# CREATE SEVERITY SCORE
# ==================================================

severity_map = {
    "PUBLIC INTOXICATION": 1,
    "TRAFFIC VIOLATION": 2,
    "SHOPLIFTING": 2,
    "VANDALISM": 3,
    "IDENTITY THEFT": 4,
    "FRAUD": 4,
    "COUNTERFEITING": 4,
    "VEHICLE - STOLEN": 5,
    "CYBERCRIME": 5,
    "ILLEGAL POSSESSION": 5,
    "BURGLARY": 6,
    "DRUG OFFENSE": 6,
    "EXTORTION": 7,
    "ROBBERY": 8,
    "ASSAULT": 8,
    "DOMESTIC VIOLENCE": 8,
    "FIREARM OFFENSE": 8,
    "KIDNAPPING": 9,
    "SEXUAL ASSAULT": 9,
    "ARSON": 9,
    "HOMICIDE": 10
}

df["Severity Score"] = df["Crime Description"].map(severity_map)

# ==================================================
# FEATURE SELECTION
# ==================================================

selected_columns = [
    "Crime Description",
    "City",
    "Victim Age",
    "Victim Gender",
    "Weapon Used",
    "Crime Domain",
    "Police Deployed",
    "Case Closed",
    "Severity Score"
]

df = df[selected_columns]

# ==================================================
# LABEL ENCODING
# ==================================================

label_encoder = LabelEncoder()

categorical_columns = [
    "Crime Description",
    "City",
    "Victim Gender",
    "Weapon Used",
    "Crime Domain",
    "Case Closed"
]

for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])

# ==================================================
# FEATURES & TARGET
# ==================================================

X = df.drop("Severity Score", axis=1)
y = df["Severity Score"]

# ==================================================
# TRAIN TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ==================================================
# LINEAR REGRESSION
# ==================================================

lr_model = LinearRegression()

lr_model.fit(X_train, y_train)

lr_predictions = lr_model.predict(X_test)

# ==================================================
# KNN REGRESSOR
# ==================================================

knn_model = KNeighborsRegressor(n_neighbors=5)

knn_model.fit(X_train, y_train)

knn_predictions = knn_model.predict(X_test)

# ==================================================
# EVALUATION FUNCTION
# ==================================================

def evaluate_model(name, y_true, y_pred):

    mae = mean_absolute_error(y_true, y_pred)

    rmse = mean_squared_error(
        y_true,
        y_pred
    ) ** 0.5

    r2 = r2_score(
        y_true,
        y_pred
    )

    print("\n==============================")
    print(name)
    print("==============================")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")

# ==================================================
# RESULTS
# ==================================================

evaluate_model(
    "LINEAR REGRESSION",
    y_test,
    lr_predictions
)

evaluate_model(
    "KNN REGRESSOR",
    y_test,
    knn_predictions
)
# ==================================================
# VISUALIZATIONS
# ==================================================

import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("results", exist_ok=True)

# ==================================================
# 1. CRIME DISTRIBUTION
# ==================================================

plt.figure(figsize=(12,6))

df["Crime Description"].value_counts().plot(kind="bar")

plt.title("Crime Type Distribution")
plt.xlabel("Crime Type")
plt.ylabel("Count")

plt.tight_layout()
plt.savefig("results/crime_distribution.png")
plt.close()

# ==================================================
# 2. CORRELATION HEATMAP
# ==================================================

numeric_df = df[[
    "Victim Age",
    "Police Deployed",
    "Severity Score"
]]

plt.figure(figsize=(6,4))

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.tight_layout()
plt.savefig("results/heatmap.png")
plt.close()

# ==================================================
# 3. ACTUAL VS PREDICTED (LR)
# ==================================================

plt.figure(figsize=(7,5))

plt.scatter(
    y_test,
    lr_predictions,
    alpha=0.5
)

plt.xlabel("Actual Severity")
plt.ylabel("Predicted Severity")

plt.title("Linear Regression: Actual vs Predicted")

plt.tight_layout()
plt.savefig("results/actual_vs_pred_lr.png")
plt.close()

# ==================================================
# 4. ACTUAL VS PREDICTED (KNN)
# ==================================================

plt.figure(figsize=(7,5))

plt.scatter(
    y_test,
    knn_predictions,
    alpha=0.5
)

plt.xlabel("Actual Severity")
plt.ylabel("Predicted Severity")

plt.title("KNN: Actual vs Predicted")

plt.tight_layout()
plt.savefig("results/actual_vs_pred_knn.png")
plt.close()

# ==================================================
# 5. R² COMPARISON
# ==================================================

lr_r2 = r2_score(y_test, lr_predictions)
knn_r2 = r2_score(y_test, knn_predictions)

plt.figure(figsize=(6,5))

plt.bar(
    ["Linear Regression", "KNN"],
    [lr_r2, knn_r2]
)

plt.ylabel("R² Score")
plt.title("R² Comparison")

plt.tight_layout()
plt.savefig("results/r2_comparison.png")
plt.close()

print("\nGraphs saved in results folder.")