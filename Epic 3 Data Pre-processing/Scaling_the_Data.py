# ==========================================
# Project: Smart Lender
# Epic 3: Data Pre-processing
# Story: Scaling the Data
# ==========================================

import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load balanced dataset
df = pd.read_csv("balanced_loan_data.csv")

print("========== DATASET ==========")
print(df.head())

# Numerical columns
numerical_columns = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term"
]

# Fill missing values
df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].mean())

# Scale data
scaler = StandardScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

print("\n========== SCALED DATA ==========")
print(df[numerical_columns].head())

# Save scaled dataset
df.to_csv("scaled_loan_data.csv", index=False)

print("\nScaled dataset saved as scaled_loan_data.csv")