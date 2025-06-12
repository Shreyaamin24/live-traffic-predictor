# train_model.py

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load preprocessed data
X = pd.read_csv("X_scaled.csv")
y = pd.read_csv("y.csv").values.ravel()

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Predict on training data (or use train_test_split for real evaluation)
y_pred = model.predict(X)

# Evaluation
print("✅ Model Accuracy:", accuracy_score(y, y_pred)*100,"%")
print("📊 Classification Report:\n", classification_report(y, y_pred))

# Save model
joblib.dump(model, "traffic_model.pkl")
print("✅ Model saved as traffic_model.pkl")
