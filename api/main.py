from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
import pickle
from pydantic import BaseModel
import pandas as pd
import os

app = FastAPI()

# Calcul du chemin absolu du modèle à partir du fichier actuel
chemin_fichier = os.path.dirname(__file__)  # dossier où se trouve le script
chemin_modele = os.path.join(chemin_fichier, "models", "model.pkl")
# chemin_favicon = os.path.join(chemin_fichier, "favicon.ico")

# Chargement du modèle
with open(chemin_modele, "rb") as file:
    model = pickle.load(file)

print("Modèle chargé")


@app.get("/favicon.ico")
def favicon():
    return FileResponse("../assets/favicon.ico")


@app.get("/")
def read_root():
    return {"message": "API prête avec modèle déjà en mémoire"}


# Schéma d'entrée
class PredictRequest(BaseModel):
    data: list[list]
    columns: list


@app.post("/predict")
def predict(request: PredictRequest):
    try:
        # Recréer un DataFrame
        X_input = pd.DataFrame(request.data, columns=request.columns)

        # Prédictions
        y_pred = model.predict(X_input)
        y_proba = model.predict_proba(X_input)[:, 1]  # proba d'appartenir à la classe 1

        return {
            "predictions": [int(y) for y in y_pred],
            "probas_class_1": [float(p) for p in y_proba],
        }
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Erreur lors de la prédiction : {e}"
        )
