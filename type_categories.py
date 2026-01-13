import pandas as pd

# Load dataset
df = pd.read_csv("Details.csv")

# Select categorical columns (non-numeric)
categorical_cols = df.select_dtypes(include='object').columns

print("Categorical columns and their unique values:\n")
for col in categorical_cols:
    unique_values = df[col].unique()
    print(f"{col} ({len(unique_values)} types): {unique_values}\n")
