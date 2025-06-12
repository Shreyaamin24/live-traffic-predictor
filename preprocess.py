import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

# Load dataset
df = pd.read_csv("messaging_traffic_data.csv")

# Optional: Drop rows with missing values
df.dropna(inplace=True)

# Separate features and target
X = df.drop("Traffic state", axis=1)
y = df["Traffic state"]

# Encode categorical columns if any
label_encoders = {}
for column in X.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])
    label_encoders[column] = le

# Encode target
target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)
label_encoders["Traffic state"] = target_encoder

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save processed files
pd.DataFrame(X_scaled, columns=X.columns).to_csv("X_scaled.csv", index=False)
pd.DataFrame(y_encoded, columns=["Traffic state"]).to_csv("y.csv", index=False)

# Save the scaler and encoders
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

print("✅ Preprocessing complete.")
