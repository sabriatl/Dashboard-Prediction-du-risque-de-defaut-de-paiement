import streamlit as st
import pandas as pd
import requests
import numpy as np

st.title("Prédiction du risque de defaut banquaire")

# Uploader CSV
uploaded_file = st.file_uploader("Uploader un fichier CSV avec les clients à prédire", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Aperçu des données :", data.head())

    # Nettoyer les données pour JSON (convertir NaN, inf)
    data_json = [
        [None if (isinstance(x, float) and (np.isnan(x) or np.isinf(x))) else x
         for x in row]
        for row in data.values.tolist()
    ]
    columns_json = data.columns.tolist()


    # Bouton pour lancer la prédiction
    if st.button("Prédire"):
        try:
            response = requests.post(
                "https://projet7-credit-default-risk.onrender.com/predict",
                json={"data": data_json, "columns": columns_json}
            )
            if response.status_code == 200:
                predictions = response.json()
                st.success("Prédictions reçues !")
                st.write(predictions)
            else:
                st.error(f"Erreur API : {response.status_code} - {response.text}")
        except Exception as erreur:
            st.error(f"Erreur lors de l'appel API : {erreur}")

            