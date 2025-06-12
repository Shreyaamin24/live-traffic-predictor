import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

# Load preprocessed data
X = pd.read_csv("X_scaled.csv")
y = pd.read_csv("y.csv").values.ravel()

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluation
print("✅ Model Accuracy:", accuracy_score(y_test, y_pred) * 100, "%")
print("📊 Classification Report:\n", classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "traffic_model.pkl")
print("✅ Model saved as traffic_model.pkl")
