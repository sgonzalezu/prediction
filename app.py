

import os
import gdown

# URL del modelo en Google Drive (reemplaza con tu enlace)
url = "https://drive.google.com/file/d/11TZMYmLSTh42bGkBRRYvnmLQJxlRdmQV/view?usp=share_link"

# Descargar el modelo si no existe
if not os.path.exists("modelo.pkl"):
    print("Descargando modelo desde Google Drive...")
    gdown.download(url, "modelo.pkl", quiet=False)


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
