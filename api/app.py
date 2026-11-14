"""API de inferencia: carga el modelo y responde con la calidad del vino"""
import csv
import os
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from prometheus_client import REGISTRY, Counter
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

LOGS = Path(os.getenv("LOGS_DIR", "logs"))

modelo = joblib.load(Path(os.getenv("MODEL_DIR", "models")) / "model.pkl")
app = FastAPI(title="API de predicción de calidad de vino")
Instrumentator().instrument(app).expose(app)
try:
    PREDICCIONES = Counter("predicciones_total", "Predicciones por puntaje", ["puntaje"])
except ValueError:  # ya existe: pasa cuando las pruebas vuelven a cargar la API
    PREDICCIONES = REGISTRY._names_to_collectors["predicciones_total"]


class Vino(BaseModel):
    """Columnas del dataset."""

    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: int
    total_sulfur_dioxide: int
    density: float
    pH: float
    sulphates: float
    alcohol: float

def guardar_entrada(datos):
    """Agrega una fila a logs/entradas.csv con lo que recibió la API."""
    LOGS.mkdir(exist_ok=True)
    archivo = LOGS / "entradas.csv"
    nuevo = not archivo.exists()
    with open(archivo, "a", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=list(datos))
        if nuevo:
            escritor.writeheader()
        escritor.writerow(datos)

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(vino: Vino):
    guardar_entrada(vino.model_dump())
    # Mismo one-hot que en src/prepare.py; reindex descarta las categorías de referencia
    # (drop_first) y rellena con 0 las columnas que el cliente no activa.
    X = pd.get_dummies(pd.DataFrame([vino.model_dump()]), dtype=int)
    X = X.reindex(columns=modelo.feature_names_in_, fill_value=0)
    prediction = round(modelo.predict(X)[0])
    PREDICCIONES.labels(puntaje=prediction).inc()
    return {"prediccion": prediction}

