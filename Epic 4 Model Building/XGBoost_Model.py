# ==========================================
# Project: Smart Lender
# Epic 4: Model Building
# Story: XGBoost Model
# ==========================================

import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

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

# Fill missing values
X_train = X_train.fillna(X_train.mean(numeric_only=True))
X_test = X_test.fillna(X_test.mean(numeric_only=True))

# Convert labels (Y/N) to 1/0
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)

print("========== TRAINING XGBOOST MODEL ==========")

# Create model
model = XGBClassifier(
    random_state=42,
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    eval_metric="logloss"
)

# Train model
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