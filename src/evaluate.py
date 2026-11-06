"""Etapa evaluate: mide el modelo en el conjunto de prueba y escribe metrics.json."""
import json
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.metrics import mean_squared_error, root_mean_squared_error

target = yaml.safe_load(open("params.yaml", encoding="utf-8"))["prepare"]["target"]

test = pd.read_csv("data/prepared/test.csv")
X, y = test.drop(columns=[target]), test[target]
pred = joblib.load("models/model.pkl").predict(X)

metrics = {
    "mean_error": round(mean_squared_error(y, pred), 4),
    "root_sq_error": round(root_mean_squared_error(y, pred), 4),
}
Path("metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
