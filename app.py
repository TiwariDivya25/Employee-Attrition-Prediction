from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    values = np.array([[
        float(data['age']),
        float(data['income']),
        float(data['distance']),
        float(data['years']),
        float(data['job']),
        float(data['env']),
        float(data['balance']),
        float(data['companies']),
        float(data['hike'])
    ]])

    scaled = scaler.transform(values)

    pred = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0][1]

    if pred == 1:
        result = "Likely to Leave"
    else:
        result = "Likely to Stay"

    return jsonify({
        "prediction": result,
        "confidence": round(prob*100,2)
    })

if __name__ == "__main__":
    app.run(debug=True)