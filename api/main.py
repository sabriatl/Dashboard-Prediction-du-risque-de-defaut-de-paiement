from fastapi import FastAPI
from fastapi.responses import FileResponse
import pickle
from pydantic import BaseModel
import numpy as np
import pandas as pd

app = FastAPI()

# Charger le modèle  au démarrage
with open("models/model.pkl", "rb") as file:
    model = pickle.load(file)
print("Modèle chargédans")


@app.get("/favicon.ico")
def favicon():
    return FileResponse("favicon.ico")

@app.get("/")
def read_root():
    return {"message": "API prête avec modèle déjà en mémoire"}

# Schéma d'entrée
class PredictRequest(BaseModel):
    data: list[list] 
    columns: list


@app.post("/predict")
def predict(request: PredictRequest):

    # Recréer un DataFrame 
    X_input = pd.DataFrame(request.data, columns=request.columns)

     # Prédictions
    y_pred = model.predict(X_input)
    y_proba = model.predict_proba(X_input)[:, 1]  # proba d'appartenir à la classe 1

    return {
        "predictions": [int(y) for y in y_pred],
        "probas_class_1": [float(p) for p in y_proba]
    }



