import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("Details.csv")
df.columns = df.columns.str.strip()  # remove extra spaces

# -----------------------------
# Features and target
# -----------------------------
X = df[["Quantity", "Amount", "Category", "Sub-Category", "PaymentMode"]]
y = df["Profit"]

# Fill missing values
for col in ["Quantity", "Amount"]:
    X[col] = X[col].fillna(0)
for col in ["Category", "Sub-Category", "PaymentMode"]:
    X[col] = X[col].fillna("Unknown")

# -----------------------------
# Preprocessing pipeline
# -----------------------------
categorical_cols = ["Category", "Sub-Category", "PaymentMode"]
numeric_cols = ["Quantity", "Amount"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ],
    remainder="passthrough"  # numeric features stay as-is
)

# Pipeline with Linear Regression
model_v2 = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# -----------------------------
# Train model
# -----------------------------
model_v2.fit(X, y)

# -----------------------------
# Save model
# -----------------------------
joblib.dump(model_v2, "revenue_model_v2.pkl")
print("Improved model v2 saved using Quantity, Amount, Category, Sub-Category, PaymentMode")
