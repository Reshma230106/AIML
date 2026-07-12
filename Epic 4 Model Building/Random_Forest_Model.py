# ==========================================
# Project: Smart Lender
# Epic 4: Model Building
# Story: Random Forest Model
# ==========================================

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load training and testing datasets
X_train = pd.read_csv(
    r"C:\Users\shaik\OneDrive\Desktop\AIML Project\Epic 3 Data Pre-processing\X_train.csv"
)

X_test = pd.read_csv(
    r"C:\Users\shaik\OneDrive\Desktop\AIML Project\Epic 3 Data Pre-processing\X_test.csv"
)

y_train = pd.read_csv(
    r"C:\Users\shaik\OneDrive\Desktop\AIML Project\Epic 3 Data Pre-processing\y_train.csv"
).squeeze()

y_test = pd.read_csv(
    r"C:\Users\shaik\OneDrive\Desktop\AIML Project\Epic 3 Data Pre-processing\y_test.csv"
).squeeze()

print("========== TRAINING RANDOM FOREST MODEL ==========")

# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))