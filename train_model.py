import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("employee_attrition.csv")

df["Attrition"] = df["Attrition"].map({"Yes":1, "No":0})

X = df.drop("Attrition", axis=1)
y = df["Attrition"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_scaled, y)

joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Files created successfully")