from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
import pickle
from pydantic import BaseModel
import pandas as pd
import os
import shap
import numpy as np

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

# Récupération du modèle et du preprocessor
preprocessor = model.named_steps["preprocessor"]
model_lgb = model.named_steps["model"]

# Colonnes du preprocessor en entrée
colonnes_entree_modele = preprocessor.feature_names_in_.tolist()

# Colonnes scalées (on récupère dynamiquement les colonnes du scaler)
scaler = preprocessor.named_transformers_["scaler_continuous_features"]
colonnes_scalées = scaler.get_feature_names_out()

# Colonnes passthrough = toutes celles en entrée - celles qui ont été scalées
colonnes_passthrough = [
    col for col in colonnes_entree_modele if col not in scaler.feature_names_in_
]

# Noms finaux des features transformées
feature_names = list(colonnes_scalées) + list(colonnes_passthrough)

# Explainer SHAP
explainer = shap.TreeExplainer(model_lgb)

# --- Schéma Pydantic ---
class ShapGlobalRequest(BaseModel):
    data: list[list]
    columns: list

@app.post("/shap_global")
def shap_global_endpoint(request: ShapGlobalRequest):
    try:
        # 1. Recréer DataFrame depuis la requête
        df = pd.DataFrame(request.data, columns=request.columns)
        print("df.shape =", df.shape)

        # 2. Transformer avec le préprocesseur
        df_transformed = preprocessor.transform(df)
        print("df_transformed.shape =", df_transformed.shape)

        # 3. SHAP explainer
        shap_output = explainer.shap_values(df_transformed)
        
        if isinstance(shap_output, list):
            shap_values = shap_output[1]  # classe 1
        else:
            shap_values = shap_output

         # Nettoyage
        shap_values_clean = np.nan_to_num(shap_values)
        df_clean = np.nan_to_num(df_transformed)

        return {
            "shap_values": shap_values_clean.tolist(),
            "feature_names": feature_names,
            "features_transformed": df_clean.tolist()
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur SHAP global : {e}")

# --- Schéma Pydantic ---
class ShapLocalRequest(BaseModel):
    data: list[list]
    columns: list

@app.post("/shap_local")
def shap_local_endpoint(request: ShapLocalRequest):
    try:
        # 1. Recréation DataFrame à partir de la requête
        df = pd.DataFrame(request.data, columns=request.columns)  
        print("Client reçu pour SHAP local :", df.shape)

        # 2. Prétraitement
        df_transformed = preprocessor.transform(df)

        # 3. SHAP local avec explainer 
        shap_explanation = explainer(df_transformed)

        # 4. Récupération des valeurs SHAP pour ce client
        shap_values = shap_explanation.values[0]
        base_value = shap_explanation.base_values[0]

        # Nettoyage des valeurs avant JSON
        shap_values_clean = np.nan_to_num(shap_values)
        features_clean = np.nan_to_num(df_transformed[0])

        return {
            "shap_values": shap_values_clean.tolist(),
            "feature_names": feature_names,
            "features_transformed": features_clean.tolist(),
            "base_value": float(base_value)
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur SHAP local : {e}")