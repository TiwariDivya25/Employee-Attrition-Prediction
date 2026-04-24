from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

FEATURE_COLUMNS = [
    "Age",
    "MonthlyIncome",
    "DistanceFromHome",
    "YearsAtCompany",
    "JobSatisfaction",
    "EnvironmentSatisfaction",
    "WorkLifeBalance",
    "NumCompaniesWorked",
    "PercentSalaryHike"
]


@app.route("/")
def home():
    return render_template("index.html")


# Single prediction
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    values = pd.DataFrame([{
        "Age": float(data["age"]),
        "MonthlyIncome": float(data["income"]),
        "DistanceFromHome": float(data["distance"]),
        "YearsAtCompany": float(data["years"]),
        "JobSatisfaction": float(data["job"]),
        "EnvironmentSatisfaction": float(data["env"]),
        "WorkLifeBalance": float(data["balance"]),
        "NumCompaniesWorked": float(data["companies"]),
        "PercentSalaryHike": float(data["hike"])
    }])

    scaled = scaler.transform(values)

    pred = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0][1]

    result = "Likely to Leave" if pred == 1 else "Likely to Stay"

    return jsonify({
        "prediction": result,
        "confidence": round(prob * 100, 2)
    })


# Bulk CSV prediction
@app.route("/bulk_predict", methods=["POST"])
def bulk_predict():
    if "file" not in request.files:
        return "No file uploaded"

    file = request.files["file"]

    if file.filename == "":
        return "No selected file"

    df = pd.read_csv(file)

    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        return f"Missing columns: {', '.join(missing)}"

    X = df[FEATURE_COLUMNS]

    scaled = scaler.transform(X)

    preds = model.predict(scaled)
    probs = model.predict_proba(scaled)[:, 1]

    df["Prediction"] = [
        "Likely to Leave" if p == 1 else "Likely to Stay"
        for p in preds
    ]

    df["RiskScore(%)"] = np.round(probs * 100, 2)

    output_file = "bulk_results.csv"
    df.to_csv(output_file, index=False)

    return send_file(output_file, as_attachment=True)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)