import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv(r"C:\Users\shaik\OneDrive\Desktop\AIML Project\train.csv")
print("Dataset Shape:", df.shape)

# -----------------------------
# 1. Check Missing Values
# -----------------------------
print("\nMissing Values:")
print(df.isnull().sum())

# Fill numerical missing values with median
numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns

for col in numeric_columns:
    df[col] = df[col].fillna(df[col].median())


# Fill categorical missing values with mode
categorical_columns = df.select_dtypes(include=['object']).columns

for col in categorical_columns:
    if not df[col].mode().empty:
        df[col] = df[col].fillna(df[col].mode()[0])
# -----------------------------
# 2. Remove Duplicate Records
# -----------------------------
duplicates = df.duplicated().sum()
print("\nDuplicate Records:", duplicates)

df = df.drop_duplicates()

# -----------------------------
# 3. Handle Invalid Values
# -----------------------------

# Remove records with negative income
if "ApplicantIncome" in df.columns:
    df = df[df["ApplicantIncome"] >= 0]

# Remove records with negative loan amount
if "LoanAmount" in df.columns:
    df = df[df["LoanAmount"] >= 0]

# Validate Age (if available)
if "Age" in df.columns:
    df = df[(df["Age"] >= 18) & (df["Age"] <= 100)]

# -----------------------------
# 4. Standardize Categorical Values
# -----------------------------
if "Gender" in df.columns:
    df["Gender"] = (
        df["Gender"]
        .replace({
            "male": "Male",
            "MALE": "Male",
            "M": "Male",
            "female": "Female",
            "F": "Female",
            "FEMALE": "Female"
        })
    )

# -----------------------------
# 5. Convert Data Types
# -----------------------------
if "LoanAmount" in df.columns:
    df["LoanAmount"] = pd.to_numeric(df["LoanAmount"], errors="coerce")

if "ApplicantIncome" in df.columns:
    df["ApplicantIncome"] = pd.to_numeric(df["ApplicantIncome"], errors="coerce")

# -----------------------------
# 6. Detect Outliers (IQR Method)
# -----------------------------
# def remove_outliers(dataframe, column):
#     Q1 = dataframe[column].quantile(0.25)
#     Q3 = dataframe[column].quantile(0.75)
#     IQR = Q3 - Q1
#
#     lower = Q1 - 1.5 * IQR
#     upper = Q3 + 1.5 * IQR
#
#     return dataframe[
#         (dataframe[column] >= lower) &
#         (dataframe[column] <= upper)
#     ]
#
# for col in numeric_columns:
#     if col in df.columns:
#         df = remove_outliers(df, col)

# -----------------------------
# 7. Final Validation
# -----------------------------
print("\nFinal Missing Values:")
print(df.isnull().sum())

print("\nFinal Dataset Shape:", df.shape)

# -----------------------------
# 8. Save Clean Dataset
# -----------------------------
df.to_csv(r"C:\Users\shaik\OneDrive\Desktop\AIML Project\loan_data_cleaned.csv", index=False)
print("\nData preprocessing completed successfully.")
print("Clean dataset saved as loan_data_cleaned.csv")