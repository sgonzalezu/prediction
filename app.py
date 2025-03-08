import os
import gdown

# URL del modelo en Google Drive (reemplaza con tu enlace)
url = "https://drive.google.com/uc?id=11TZMYmLSTh42bGkBRRYvnmLQJxlRdmQV"

# Descargar el modelo si no existe
if not os.path.exists("modelo.pkl"):
    print("Descargando modelo desde Google Drive...")
    gdown.download(url, "modelo.pkl", quiet=False)

import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

app = FastAPI()

# Cargar el modelo entrenado
model_path = "modelo.pkl"
if not os.path.exists(model_path):
    raise RuntimeError("El archivo modelo.pkl no se encuentra en el servidor.")
    
model = joblib.load(model_path)

# Cargar la estructura original de las columnas del modelo
column_structure_path = "column_structure.pkl"
if not os.path.exists(column_structure_path):
    raise RuntimeError("El archivo column_structure.pkl no se encuentra en el servidor.")
    
original_columns = joblib.load(column_structure_path)  # Lista de columnas con las que el modelo fue entrenado

@app.get("/")
def home():
    return {"message": "API de predicción desplegada con FastAPI"}

@app.post("/predict/")
def predict(data: dict):
    try:
        # Validar si todas las columnas requeridas están en los datos recibidos
        expected_columns = [
            "HomePlanet", "CryoSleep", "Age", "VIP",
            "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"
        ]
        
        for col in expected_columns:
            if col not in data:
                raise HTTPException(status_code=400, detail=f"Falta la columna: {col}")
                
        # Convertir el diccionario en un DataFrame
        df = pd.DataFrame([data])
        
        # Aplicar One-Hot Encoding a las variables categóricas
        df = pd.get_dummies(df)
        
        # Asegurar que las columnas coincidan con las del modelo
        for col in original_columns:
            if col not in df.columns:
                df[col] = 0  # Agregar columnas faltantes con valor 0
                
        df = df[original_columns]  # Ordenar columnas en el mismo orden que en el entrenamiento
        
        # Realizar la predicción
        prediction = model.predict(df)
        return {"prediction": bool(prediction[0])}  # Convertir a booleano si es binaria
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        