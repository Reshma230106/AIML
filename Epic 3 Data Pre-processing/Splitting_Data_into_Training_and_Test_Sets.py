# ==========================================
# Project: Smart Lender
# Epic 3: Data Pre-processing
# Story: Splitting Data into Training and Test Sets
# ==========================================

import pandas as pd
from sklearn.model_selection import train_test_split

# Load scaled dataset
df = pd.read_csv(
    r"C:\Users\shaik\OneDrive\Desktop\AIML Project\Epic 3 Data Pre-processing\scaled_loan_data.csv"
)

print("========== DATASET ==========")
print(df.head())

# Features and Target
X = df.drop(["Loan_ID", "Loan_Status"], axis=1)
y = df["Loan_Status"]

# Convert categorical columns into numeric values
X = pd.get_dummies(X, drop_first=True)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Features Shape:", X_train.shape)
print("Testing Features Shape :", X_test.shape)
print("Training Labels Shape  :", y_train.shape)
print("Testing Labels Shape   :", y_test.shape)

# Save files
X_train.to_csv(
    r"C:\Users\shaik\OneDrive\Desktop\AIML Project\Epic 3 Data Pre-processing\X_train.csv",
    index=False
)
X_test.to_csv(
    r"C:\Users\shaik\OneDrive\Desktop\AIML Project\Epic 3 Data Pre-processing\X_test.csv",
    index=False
)
y_train.to_csv(
    r"C:\Users\shaik\OneDrive\Desktop\AIML Project\Epic 3 Data Pre-processing\y_train.csv",
    index=False
)
y_test.to_csv(
    r"C:\Users\shaik\OneDrive\Desktop\AIML Project\Epic 3 Data Pre-processing\y_test.csv",
    index=False
)

print("\n========== DATA SPLITTING COMPLETED ==========")
print("Files created:")
print("X_train.csv")
print("X_test.csv")
print("y_train.csv")
print("y_test.csv")