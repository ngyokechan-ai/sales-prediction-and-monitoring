import pandas as pd

# Load your dataset
df = pd.read_csv("Details.csv")  # Change to your file name if different

# Check unique categories in the 'Category' column
unique_categories = df['Category'].unique()
print("Unique categories:", unique_categories)

# Count how many categories there are
num_categories = df['Category'].nunique()
print("Number of unique categories:", num_categories)

# Count how many rows are in each category
category_counts = df['Category'].value_counts()
print("\nCategory counts:\n", category_counts)
