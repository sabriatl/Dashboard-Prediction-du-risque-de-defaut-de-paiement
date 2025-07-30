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
chemin_favicon = os.path.abspath(os.path.join(chemin_fichier, "..", "assets", "favicon.ico"))

# Chargement du modèle
with open(chemin_modele, "rb") as file:
    model = pickle.load(file)

print("Modèle chargé")


@app.get("/favicon.ico")
def favicon():
    return FileResponse(chemin_favicon)

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

        # Prédictions probabilistes
        y_proba = model.predict_proba(X_input)[:, 1]  # proba d'appartenir à la classe 1

        # Application du seuil métier 
        seuil_optimal = 0.47
        y_pred = (y_proba >= seuil_optimal).astype(int)
     
        return {
            "predictions": [int(y) for y in y_pred],
            "probas_class_1": [float(p) for p in y_proba]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur lors de la prédiction : {e}")


