# ==========================================
# Project: Smart Lender - Applicant Credibility Prediction
# Epic 2: Visualizing and Analysing the Data
# Story: Bivariate Analysis
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

# Gender vs Loan Status
plt.figure(figsize=(6,4))
sns.countplot(data=df, x="Gender", hue="Loan_Status")
plt.title("Gender vs Loan Status")
plt.show()

# Married vs Loan Status
plt.figure(figsize=(6,4))
sns.countplot(data=df, x="Married", hue="Loan_Status")
plt.title("Married vs Loan Status")
plt.show()

# Education vs Loan Status
plt.figure(figsize=(6,4))
sns.countplot(data=df, x="Education", hue="Loan_Status")
plt.title("Education vs Loan Status")
plt.show()

# Property Area vs Loan Status
plt.figure(figsize=(6,4))
sns.countplot(data=df, x="Property_Area", hue="Loan_Status")
plt.title("Property Area vs Loan Status")
plt.show()

# Credit History vs Loan Status
plt.figure(figsize=(6,4))
sns.countplot(data=df, x="Credit_History", hue="Loan_Status")
plt.title("Credit History vs Loan Status")
plt.show()

# Applicant Income vs Loan Status
plt.figure(figsize=(8,5))
sns.boxplot(data=df, x="Loan_Status", y="ApplicantIncome")
plt.title("Applicant Income vs Loan Status")
plt.show()

# Loan Amount vs Loan Status
plt.figure(figsize=(8,5))
sns.boxplot(data=df, x="Loan_Status", y="LoanAmount")
plt.title("Loan Amount vs Loan Status")
plt.show()

print("\n========== BIVARIATE ANALYSIS COMPLETED ==========")