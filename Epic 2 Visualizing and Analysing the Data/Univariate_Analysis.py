# ==========================================
# Project: Smart Lender - Applicant Credibility Prediction
# Epic 2: Visualizing and Analysing the Data
# Story: Univariate Analysis
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv(r"C:\Users\shaik\OneDrive\Desktop\AIML Project\Epic 2 Visualizing and Analysing the Data\train.csv")

print("========== DATASET ==========")
print(df.head())

print("\nDataset Shape:", df.shape)

sns.set(style="whitegrid")

# -----------------------------
# Loan Status Distribution
# -----------------------------
plt.figure(figsize=(6,4))
sns.countplot(data=df, x="Loan_Status")
plt.title("Loan Status Distribution")
plt.xlabel("Loan Status")
plt.ylabel("Count")
plt.show()

# -----------------------------
# Gender Distribution
# -----------------------------
plt.figure(figsize=(6,4))
sns.countplot(data=df, x="Gender")
plt.title("Gender Distribution")
plt.show()

# -----------------------------
# Education Distribution
# -----------------------------
plt.figure(figsize=(6,4))
sns.countplot(data=df, x="Education")
plt.title("Education Distribution")
plt.show()

# -----------------------------
# Property Area Distribution
# -----------------------------
plt.figure(figsize=(6,4))
sns.countplot(data=df, x="Property_Area")
plt.title("Property Area Distribution")
plt.show()

# -----------------------------
# Applicant Income
# -----------------------------
plt.figure(figsize=(7,4))
sns.histplot(df["ApplicantIncome"], bins=20, kde=True)
plt.title("Applicant Income Distribution")
plt.show()

# -----------------------------
# Loan Amount
# -----------------------------
plt.figure(figsize=(7,4))
sns.histplot(df["LoanAmount"], bins=20, kde=True)
plt.title("Loan Amount Distribution")
plt.show()

print("\n========== UNIVARIATE ANALYSIS COMPLETED ==========")