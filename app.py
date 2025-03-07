
from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# Cargar el modelo previamente entrenado
model = joblib.load("modelo.pkl")

@app.get("/")
def home():
    return {"message": "API de predicción desplegada con FastAPI"}

@app.post("/predict/")
def predict(data: dict):
    try:
        df = pd.DataFrame([data])  
        prediction = model.predict(df)  
        return {"prediction": prediction.tolist()}
    except Exception as e:
        return {"error": str(e)}
