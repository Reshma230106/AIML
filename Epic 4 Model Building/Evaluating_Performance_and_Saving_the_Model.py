# ==========================================
# Project: Smart Lender
# Epic 4: Model Building
# Story: Evaluating Performance and Saving the Model
# ==========================================

import pandas as pd
import joblib

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# ==========================================
# Load Dataset
# ==========================================

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

# Encode labels
encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train)
y_test_encoded = encoder.transform(y_test)

# ==========================================
# Train Models
# ==========================================

decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, y_train)

random_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
random_forest.fit(X_train, y_train)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

xgboost = XGBClassifier(
    random_state=42,
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    eval_metric="logloss"
)
xgboost.fit(X_train, y_train_encoded)

# ==========================================
# Evaluate Models
# ==========================================

dt_acc = accuracy_score(y_test, decision_tree.predict(X_test))

rf_acc = accuracy_score(y_test, random_forest.predict(X_test))

knn_acc = accuracy_score(y_test, knn.predict(X_test))

xgb_acc = accuracy_score(
    y_test_encoded,
    xgboost.predict(X_test)
)

print("\n========== MODEL ACCURACY ==========\n")

print(f"Decision Tree : {dt_acc*100:.2f}%")
print(f"Random Forest : {rf_acc*100:.2f}%")
print(f"KNN           : {knn_acc*100:.2f}%")
print(f"XGBoost       : {xgb_acc*100:.2f}%")

# ==========================================
# Save Best Model
# ==========================================

best_model = random_forest

joblib.dump(
    best_model,
    r"C:\Users\shaik\OneDrive\Desktop\AIML Project\Epic 4 Model Building\Random_Forest_Model.pkl"
)

print("\nBest Model: Random Forest")
print("Model saved successfully as Random_Forest_Model.pkl")