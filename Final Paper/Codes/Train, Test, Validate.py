# pip install kagglehub[pandas-datasets]
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Set the path to the file inside the dataset you'd like to load
file_path = "student_lifestyle_100k.csv"

# Load the latest version using the recommended dataset_load
df = kagglehub.dataset_load(
  KaggleDatasetAdapter.PANDAS,
  "aldinwhyudii/student-depression-and-lifestyle-100k-data",
  file_path
)

print(df.head())

import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "Depression"

X = df.drop(columns=[TARGET])
y = df[TARGET]

# Set random seed
RANDOM_SEED = 42

# Step 1: Training (70%) and remaining (30%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    random_state=RANDOM_SEED,
    stratify=y
)

# Step 2: Testing (20%) and Validation (10%)
X_test, X_val, y_test, y_val = train_test_split(
    X_temp, y_temp,
    test_size=1/3,
    random_state=RANDOM_SEED,
    stratify=y_temp
)

# Combine features and target
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)
val_data = pd.concat([X_val, y_val], axis=1)

# Print datasets separately
print("\n========== TRAINING DATASET (70%) ==========")
print(train_data.head())
print("Shape:", train_data.shape)
print("Class distribution:\n", train_data[TARGET].value_counts())

print("\n========== TESTING DATASET (20%) ==========")
print(test_data.head())
print("Shape:", test_data.shape)
print("Class distribution:\n", test_data[TARGET].value_counts())

print("\n========== VALIDATION DATASET (10%) ==========")
print(val_data.head())
print("Shape:", val_data.shape)
print("Class distribution:\n", val_data[TARGET].value_counts())

# Split summary
total = len(df)

print("\n========== DATASET SPLIT SUMMARY ==========")
print(f"Training:   {len(train_data)} ({len(train_data)/total*100:.2f}%)")
print(f"Testing:    {len(test_data)} ({len(test_data)/total*100:.2f}%)")
print(f"Validation: {len(val_data)} ({len(val_data)/total*100:.2f}%)")

# Save datasets
train_data.to_csv("data/train.csv", index=False)
test_data.to_csv("data/test.csv", index=False)
val_data.to_csv("data/validation_unseen.csv", index=False)

from google.colab import files

# Download Training Dataset
files.download("data/train.csv")

# Download Testing Dataset
files.download("data/test.csv")

# Download Validation Dataset
files.download("data/validation_unseen.csv")
