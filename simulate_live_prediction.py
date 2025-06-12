# simulate_live_prediction.py

import pandas as pd
import time
import joblib

# Load necessary files
model = joblib.load("traffic_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Load original dataset
df = pd.read_csv("messaging_traffic_data.csv")

# Drop target column and store it separately
X_live = df.drop(columns=["Traffic state"])
y_true = df["Traffic state"]

# Encode categorical columns
for col in label_encoders:
    if col != "Traffic state":
        X_live[col] = label_encoders[col].transform(X_live[col])

# Scale features
X_scaled = scaler.transform(X_live)

# Predict in loop
print("🚀 Starting live traffic prediction...\n")
for i, row in enumerate(X_scaled):
    sample = pd.DataFrame([row], columns=X_live.columns)
    pred_class = model.predict(sample)[0]
    label = label_encoders["Traffic state"].inverse_transform([pred_class])[0]

    print(f"🕒 Sample {i + 1}: Predicted Traffic → {label}")
    time.sleep(3)  # wait for 5 seconds before next prediction
