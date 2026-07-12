import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
df = pd.read_csv("balanced_loan_data.csv")

# Select only the 5 features used in the HTML form
X = df[[
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History"
]]

# Fill missing values
X = X.fillna(X.mean(numeric_only=True))

# Target column
y = df["Loan_Status"]

# Convert target to numeric
y = y.map({"N": 0, "Y": 1})

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

print("========== TRAINING RANDOM FOREST MODEL ==========")
print()
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print()
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print()
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "Random_Forest_Model.pkl")
print("\nModel saved successfully as Random_Forest_Model.pkl")
