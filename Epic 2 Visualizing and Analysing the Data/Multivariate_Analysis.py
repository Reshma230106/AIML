# ==========================================
# Project: Smart Lender - Applicant Credibility Prediction
# Epic 2: Visualizing and Analysing the Data
# Story: Multivariate Analysis
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv(r"C:\Users\shaik\OneDrive\Desktop\AIML Project\Epic 2 Visualizing and Analysing the Data\train.csv")

print("========== DATASET ==========")
print(df.head())

print("\nDataset Shape:", df.shape)

# Select only numerical columns
numeric_df = df.select_dtypes(include=['int64', 'float64'])

# Correlation Matrix
print("\n========== CORRELATION MATRIX ==========")
print(numeric_df.corr())

# Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()

print("\n========== MULTIVARIATE ANALYSIS COMPLETED ==========")