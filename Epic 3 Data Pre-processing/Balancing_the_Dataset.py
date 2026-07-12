# ==========================================
# Project: Smart Lender
# Epic 3: Data Pre-processing
# Story: Balancing the Dataset
# ==========================================

import pandas as pd
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv(r"C:\Users\shaik\OneDrive\Desktop\AIML Project\Epic 2 Visualizing and Analysing the Data\train.csv")

print("========== ORIGINAL DATASET ==========")
print(df.shape)

print("\nOriginal Loan Status Count:")
print(df["Loan_Status"].value_counts())

# Separate majority and minority classes
majority = df[df["Loan_Status"] == "Y"]
minority = df[df["Loan_Status"] == "N"]

# Upsample minority class
minority_upsampled = resample(
    minority,
    replace=True,
    n_samples=len(majority),
    random_state=42
)

# Combine both classes
balanced_df = pd.concat([majority, minority_upsampled])

# Shuffle the dataset
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

print("\nBalanced Loan Status Count:")
print(balanced_df["Loan_Status"].value_counts())

print("\nBalanced Dataset Shape:")
print(balanced_df.shape)

# Plot
plt.figure(figsize=(6,4))
sns.countplot(data=balanced_df, x="Loan_Status")
plt.title("Balanced Loan Status")
plt.show()

# Save balanced dataset
balanced_df.to_csv(
    r"C:\Users\shaik\OneDrive\Desktop\AIML Project\Epic 3 Data Pre-processing\balanced_loan_data.csv",
    index=False
)

print("\nBalanced dataset saved as balanced_loan_data.csv")