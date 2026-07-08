import pandas as pd

df = pd.read_csv(r"C:\Users\shaik\OneDrive\Desktop\AIML Project\Epic 2 Visualizing and Analysing the Data\train.csv")

print(df.head())
print("\nDataset Shape:", df.shape)
print("\nDataset Information:")
df.info()
print("\nSummary Statistics:")
print(df.describe())