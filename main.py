from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
import joblib
import numpy as np
import pandas as pd
import random

app = FastAPI()

# Mount static and template folders
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load model & tools
model = joblib.load("traffic_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")
label_decoder = label_encoders['Traffic state']

features = [
    "Active users", "New users", "Message rate", "Media sharing", "Spam ratio",
    "User sentiment", "Server load", "Time of day", "Latency", "Bandwidth usage"
]

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/predict")
def predict():
    row = [
        random.randint(1000, 5000),  # Active users
        random.randint(10, 500),     # New users
        random.uniform(50, 500),     # Message rate
        round(random.uniform(0.1, 0.9), 2),  # Media sharing
        round(random.uniform(0.01, 0.5), 2), # Spam ratio
        random.randint(0, 2),        # User sentiment
        random.randint(10, 100),     # Server load
        random.randint(0, 3),        # Time of day
        random.randint(20, 300),     # Latency
        random.randint(50, 500)      # Bandwidth usage
    ]

    row_df = pd.DataFrame([row], columns=features)
    scaled = scaler.transform(row_df)
    pred = model.predict(scaled)[0]
    traffic_label = label_decoder.inverse_transform([pred])[0]

    return {
        "input": dict(zip(features, row)),
        "prediction": traffic_label
    }
