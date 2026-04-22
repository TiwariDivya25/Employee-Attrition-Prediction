# 🧠 Employee Attrition Predictor

A machine learning web application that predicts whether an employee is likely to leave or stay at a company based on key HR metrics. Built with Flask, scikit-learn, and a modern glassmorphism UI.

---

## Interface 
> **Home / Prediction Form**
>
> <img width="1365" height="632" alt="image" src="https://github.com/user-attachments/assets/06900d81-6e2e-46b3-b800-37235a61fa7e" />


---

> **Prediction Result**
>
> <img width="1365" height="632" alt="image" src="https://github.com/user-attachments/assets/3736ccaf-a613-4352-b7d6-d9beccfc29e4" />


---

## 🚀 Features

- Predicts employee attrition risk (Leave / Stay) in real time
- Displays confidence percentage alongside the prediction
- Clean, responsive glassmorphism UI with animated gradients
- Decision Tree classifier trained on synthetic HR data
- REST API endpoint (`/predict`) for easy integration
- Lightweight — no database required

---

## 🛠️ Tech Stack

| Layer        | Technology                        |
|--------------|-----------------------------------|
| Backend      | Python, Flask                     |
| ML Model     | scikit-learn (Decision Tree)      |
| Preprocessing| StandardScaler (joblib)           |
| Frontend     | HTML, CSS (Glassmorphism), JS     |
| Data         | Custom HR attrition CSV dataset   |

---

## 📁 Project Structure

```
employee-attrition-predictor/
│
├── app.py                  # Flask app & prediction API
├── train_model.py          # Model training script
├── employee_attrition.csv  # Training dataset
├── model.pkl               # Saved trained model
├── scaler.pkl              # Saved StandardScaler
├── requirements.txt        # Python dependencies
│
├── templates/
│   └── index.html          # Frontend HTML
│
└── static/
    └── style.css           # Glassmorphism styles
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/employee-attrition-predictor.git
cd employee-attrition-predictor
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model

```bash
python train_model.py
```

This generates `model.pkl` and `scaler.pkl` in the project root.

### 4. Run the Flask app

```bash
python app.py
```

Open your browser and navigate to `http://127.0.0.1:5000`

---

## 🧪 Input Features

| Field                        | Description                          | Range      |
|------------------------------|--------------------------------------|------------|
| Age                          | Employee age                         | 18–60      |
| Monthly Income               | Monthly salary in currency units     | Any        |
| Distance From Home           | Commute distance (km/miles)          | 1–30       |
| Years At Company             | Tenure at current company            | 0–40       |
| Job Satisfaction             | Self-rated job satisfaction          | 1–4        |
| Environment Satisfaction     | Workplace environment rating         | 1–4        |
| Work-Life Balance            | Work-life balance rating             | 1–4        |
| Number of Companies Worked   | Total companies worked at before     | 0–9        |
| Percent Salary Hike          | Last salary hike percentage          | 10–25      |

---

## 📡 API Reference

**Endpoint:** `POST /predict`

**Request Body (JSON):**
```json
{
  "age": 30,
  "income": 75000,
  "distance": 10,
  "years": 5,
  "job": 3,
  "env": 2,
  "balance": 3,
  "companies": 2,
  "hike": 15
}
```

**Response (JSON):**
```json
{
  "prediction": "Likely to Stay",
  "confidence": 82.35
}
```

---

## 🤖 Model Details

- **Algorithm:** Decision Tree Classifier
- **Max Depth:** 5
- **Preprocessing:** StandardScaler (zero mean, unit variance)
- **Target:** Binary — `1` (Attrition: Yes) / `0` (Attrition: No)
- **Training Data:** 4,000+ employee records across 10 features

---

## 📦 Requirements

```
flask
pandas
numpy
scikit-learn
joblib
```

## 🙌 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.
