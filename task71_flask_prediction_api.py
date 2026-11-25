"""
Task 2: Simple Flask REST API for Mobile Price predictions.
Test POST /predict with JSON containing a "features" object.
"""
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Train once when the server starts.
data = pd.read_csv("train.csv")
if "price_range" not in data.columns:
    raise ValueError("train.csv must contain the 'price_range' column.")
data = data.drop(columns=["id"], errors="ignore")
X = data.drop(columns=["price_range"]).select_dtypes(include=np.number)
y = data["price_range"]
if X.empty:
    raise ValueError("No numeric feature columns found.")

model = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
model.fit(X, y)
FEATURES = list(X.columns)

@app.get("/")
def home():
    return jsonify({
        "message": "Mobile Price Prediction API",
        "endpoint": "POST /predict",
        "required_features": FEATURES
    })

@app.post("/predict")
def predict():
    body = request.get_json(silent=True)
    if not isinstance(body, dict) or not isinstance(body.get("features"), dict):
        return jsonify({"error": "Send JSON with a 'features' object."}), 400

    supplied = body["features"]
    missing = [name for name in FEATURES if name not in supplied]
    extra = [name for name in supplied if name not in FEATURES]
    if missing:
        return jsonify({"error": "Missing features", "missing": missing}), 400
    if extra:
        return jsonify({"error": "Unknown features", "unknown": extra}), 400

    try:
        row = pd.DataFrame([[float(supplied[name]) for name in FEATURES]], columns=FEATURES)
    except (TypeError, ValueError):
        return jsonify({"error": "Every feature value must be numeric."}), 400
    if not np.isfinite(row.to_numpy()).all():
        return jsonify({"error": "Feature values must be finite numbers."}), 400

    prediction = int(model.predict(row)[0])
    probabilities = model.predict_proba(row)[0]
    return jsonify({
        "predicted_price_range": prediction,
        "class_probabilities": {
            str(label): round(float(prob), 4)
            for label, prob in zip(model.classes_, probabilities)
        }
    })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
