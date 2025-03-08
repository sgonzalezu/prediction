import os
import gdown

# URL del modelo en Google Drive (reemplaza con tu enlace)
url = "https://drive.google.com/uc?id=11TZMYmLSTh42bGkBRRYvnmLQJxlRdmQV"

# Descargar el modelo si no existe
if not os.path.exists("modelo.pkl"):
    print("Descargando modelo desde Google Drive...")
    gdown.download(url, "modelo.pkl", quiet=False)

import os
from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd

app = FastAPI()

# Cargar el modelo previamente entrenado
model_path = "modelo.pkl"
if not os.path.exists(model_path):
    raise RuntimeError("El archivo modelo.pkl no se encuentra en el servidor.")
    
model = joblib.load(model_path)

# Definir las categorías esperadas para codificación
categorical_mappings = {
    "HomePlanet": {"Earth": 0, "Europa": 1, "Mars": 2},
    "CryoSleep": {False: 0, True: 1},
    "VIP": {False: 0, True: 1}
}

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
        
        # Convertir variables categóricas a sus valores numéricos
        for col, mapping in categorical_mappings.items():
            if col in df:
                df[col] = df[col].map(mapping)
                
        # Revisar si hay valores NaN después de la conversión
        if df.isnull().values.any():
            raise HTTPException(status_code=400, detail="Error en la conversión de datos. Revisa los valores enviados.")
            
        # Realizar la predicción
        prediction = model.predict(df)
        return {"prediction": bool(prediction[0])}  # Convertir a booleano si es binaria
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
        