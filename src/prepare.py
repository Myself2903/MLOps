"""Etapa prepare: limpia el dataset y lo divide en entrenamiento y prueba."""
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

params = yaml.safe_load(open("params.yaml", encoding="utf-8"))["prepare"]
target = params["target"]
df = pd.read_csv("data/raw/winequality-red.csv", delimiter=";")
df = df.drop(columns=params.get("drop_columns") or [], errors="ignore")

# Columnas de texto que en realidad son números (por ejemplo, TotalCharges en Telco)
for col in df.columns.drop(target):
    if not pd.api.types.is_numeric_dtype(df[col]):
        as_number = pd.to_numeric(df[col], errors="coerce")
        if as_number.notna().mean() > 0.95:
            df[col] = as_number

df = df.dropna()
y = df[target]
X = pd.get_dummies(df.drop(columns=[target]), drop_first=True, dtype=int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=params["test_size"], random_state=params["random_state"]
)

out = Path("data/prepared")
out.mkdir(parents=True, exist_ok=True)
X_train.assign(**{target: y_train}).to_csv(out / "train.csv", index=False)
X_test.assign(**{target: y_test}).to_csv(out / "test.csv", index=False)
