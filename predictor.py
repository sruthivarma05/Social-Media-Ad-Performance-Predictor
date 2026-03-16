import joblib
import pandas as pd

model = joblib.load("engagement_model.pkl")

def predict_engagement(features: dict) -> float:
    df = pd.DataFrame([features])
    prediction = model.predict(df)[0]
    return round(prediction * 100, 2)  # return %
