# Laboratorio 4 · MLflow en el pipeline del equipo

Sábado del fin de semana 1 · 40 min · por equipos · hito E1

## Objetivo
Que cada entrenamiento quede registrado en MLflow con sus parámetros, métricas y modelo, y que el mejor modelo tenga el alias `champion`.

## Pasos

### 0. Rama de trabajo
```bash
git switch entrega/e1
git pull
git switch -c feat/registro-mlflow
```

### 1. Nueva versión de src/train.py
Reemplacen todo el contenido de `src/train.py` por:
```python
"""Etapa train: entrena, registra el run en MLflow y guarda models/model.pkl."""
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

params = yaml.safe_load(open("params.yaml", encoding="utf-8"))
target = params["prepare"]["target"]
p = params["train"]
cfg = params["mlflow"]

train = pd.read_csv("data/prepared/train.csv")
test = pd.read_csv("data/prepared/test.csv")
X_train, y_train = train.drop(columns=[target]), train[target]
X_test, y_test = test.drop(columns=[target]), test[target]

# Versión del dataset según DVC: queda como etiqueta del run
dvc_file = yaml.safe_load(open("data/raw/dataset.csv.dvc", encoding="utf-8"))
data_md5 = dvc_file["outs"][0]["md5"]

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment(cfg["experiment"])

with mlflow.start_run():
    mlflow.set_tag("data_md5", data_md5)
    mlflow.log_params(p)

    model = RandomForestClassifier(
        n_estimators=p["n_estimators"],
        max_depth=p["max_depth"],
        random_state=params["prepare"]["random_state"],
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    mlflow.log_metrics({
        "accuracy": accuracy_score(y_test, pred),
        "f1_macro": f1_score(y_test, pred, average="macro"),
    })
    mlflow.sklearn.log_model(
        model,
        name="model",
        registered_model_name=cfg["model_name"],
        input_example=X_train.head(3),
        serialization_format="cloudpickle",
    )

Path("models").mkdir(exist_ok=True)
joblib.dump(model, "models/model.pkl")
```
Revisen que `experiment` y `model_name` en `params.yaml` tengan el nombre del equipo.

### 2. Al menos tres experimentos
Cambien `max_depth` o `n_estimators` en `params.yaml` y corran el pipeline cada vez:
```bash
dvc repro
```
Cada vez que se entrena se crea un run nuevo y una versión nueva del modelo registrado. Usen valores que no hayan probado antes: si repiten uno, DVC reutiliza el resultado guardado y no entrena (ver Errores frecuentes).

### 3. Comparar en la interfaz
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Abrir http://localhost:5000, entrar al experimento del equipo, seleccionar los runs y pulsar **Compare**.

### 4. Alias champion
En la interfaz: **Models › modelo-equipo-X › la mejor versión › Aliases › Add** y escribir `champion`.

O desde Python:
```python
from mlflow import MlflowClient
client = MlflowClient(tracking_uri="sqlite:///mlflow.db")
client.set_registered_model_alias("modelo-equipo-X", "champion", 3)  # 3 = número de versión
```

### 5. Guardar y Pull Request hacia entrega/e1
```bash
git add src/train.py params.yaml dvc.lock metrics.json
git commit -m "Registra experimentos en MLflow"
dvc push
git push -u origin feat/registro-mlflow
```
`mlflow.db` y `mlruns/` no se suben: ya están en `.gitignore`.

En GitHub: **Pull requests › New pull request**, con `base: entrega/e1` ← `compare: feat/registro-mlflow`. Borren la plantilla de entrega y escriban qué cambia. Otro integrante aprueba, hace el merge y borra la rama. Después, todos: `git switch entrega/e1` y `git pull`.

### 6. Abrir el Pull Request de la entrega E1
1. Tomen dos capturas de la MLflow UI: la comparación de runs y el modelo con alias `champion` en el Model Registry.
2. En GitHub: **Pull requests › New pull request**, con `base: master` ← `compare: entrega/e1`.
3. Título: `Entrega E1 · <nombre del equipo>`. Esta vez dejen la plantilla: marquen el checklist y arrastren las dos capturas al cuadro de texto.
4. **Create pull request.** Nadie del equipo lo aprueba ni hace merge: lo aprueba y lo une el docente en la verificación.

Detalle en [docs/entregas.md](../../docs/entregas.md).

## Listo cuando
- [ ] Hay al menos 3 runs en el experimento del equipo.
- [ ] Un modelo registrado tiene el alias `champion`.
- [ ] `mlflow.db` y `mlruns/` están en `.gitignore`.
- [ ] El PR de `feat/registro-mlflow` está aprobado y unido a `entrega/e1`.
- [ ] Hay un PR abierto de `entrega/e1` hacia `master` con el checklist y las capturas.

## Errores frecuentes
- **`The saved sklearn model references untrusted types`**: falta `serialization_format="cloudpickle"` en `log_model`. Las versiones recientes de MLflow guardan con otro formato por defecto.
- **Aviso `Inferred schema contains integer column(s)`**: es solo un aviso, se puede ignorar.
- **`mlflow ui` no muestra nada**: se abrió sin `--backend-store-uri sqlite:///mlflow.db`, o desde otra carpeta. Ejecutarlo en la raíz del repositorio.
- **El puerto 5000 está ocupado** (pasa en macOS): `mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001`.
- **`dvc repro` no crea un run nuevo** (dice `Stage 'train' is cached`): no cambió ningún parámetro, o ese valor ya lo habían probado antes y DVC reutilizó el resultado guardado (run-cache) en vez de volver a entrenar. Prueben un valor nuevo o fuercen el entrenamiento con `dvc repro --force`.
- **El modelo de cada integrante es distinto**: cada uno tiene su propio `mlflow.db`. Para el E1 basta con que una máquina tenga los 3 runs; en el FS2 veremos cómo compartir el modelo.
