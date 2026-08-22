import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)

# 1. Load data
df = pd.read_csv("Employee-Attrition.csv")

# 2. Keep only the selected features + target
selected_features = [
    "Age", "MonthlyIncome", "DistanceFromHome", "YearsAtCompany",
    "JobSatisfaction", "EnvironmentSatisfaction", "WorkLifeBalance",
    "NumCompaniesWorked", "PercentSalaryHike",
]

df = df[selected_features + ["Attrition"]]

# 3. Encode target (IBM dataset also uses "Yes"/"No")
df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})

X = df[selected_features]
y = df["Attrition"]

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Train
model = DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42)
model.fit(X_train_scaled, y_train)

# 7. Evaluate
y_pred = model.predict(X_test_scaled)

print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:   ", recall_score(y_test, y_pred))
print("F1:       ", f1_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n", classification_report(y_test, y_pred, target_names=["No", "Yes"]))

# 8. Retrain on full data for the deployed model
final_scaler = StandardScaler()
X_all_scaled = final_scaler.fit_transform(X)
final_model = DecisionTreeClassifier(max_depth=2, class_weight="balanced", random_state=42)
final_model.fit(X_all_scaled, y)

joblib.dump(final_model, "model.pkl")
joblib.dump(final_scaler, "scaler.pkl")
print("\nSaved model.pkl and scaler.pkl")