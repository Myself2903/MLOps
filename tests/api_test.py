import importlib
import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TEST_CASE = {
    "fixed_acidity": 11.2,
    "volatile_acidity": 0.28,
    "citric_acid": 0.56,
    "residual_sugar": 1.9,
    "chlorides": 0.075,
    "free_sulfur_dioxide": 17,
    "total_sulfur_dioxide": 60,
    "density": 0.998,
    "pH": 3.16,
    "sulphates": 0.58,
    "alcohol": 9.8
} 

TEST_LIST = [
    {
        "fixed_acidity": 7.4,
        "volatile_acidity": 0.7,
        "citric_acid": 0,
        "residual_sugar": 1.9,
        "chlorides": 0.76,
        "free_sulfur_dioxide": 11,
        "total_sulfur_dioxide": 34,
        "density": 0.9978,
        "pH": 3.51,
        "sulphates": 0.56,
        "alcohol": 9.4
    },
    {
        "fixed_acidity": 7.8,
        "volatile_acidity": 0.88,
        "citric_acid": 0,
        "residual_sugar": 2.6,
        "chlorides": 0.98,
        "free_sulfur_dioxide": 25,
        "total_sulfur_dioxide": 67,
        "density": 0.9968,
        "pH": 3.2,
        "sulphates": 0.68,
        "alcohol": 9.8
    }
]

@pytest.fixture()
def wine(tmp_path, monkeypatch):
    X = pd.DataFrame(TEST_LIST)
    y = [5,5]

    joblib.dump(RandomForestRegressor().fit(X, y), tmp_path / "model.pkl")
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("LOGS_DIR", str(tmp_path / "logs"))
    import api.app as modulo
    importlib.reload(modulo)
    return TestClient(modulo.app)


def test_health(wine):
    assert wine.get("/health").json() == {"status": "ok"}


def test_predict(wine):
    r = wine.post("/predict", json=TEST_CASE)
    assert r.status_code == 200
    assert 0 <= r.json()["prediccion"] <= 10 


def test_dato_invalido(wine):
    test_case = TEST_CASE
    test_case["pH"] = "Hello world!"
    assert wine.post("/predict", json=test_case).status_code == 422