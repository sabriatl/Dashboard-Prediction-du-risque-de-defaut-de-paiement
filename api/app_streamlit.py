import streamlit as st
import pandas as pd
import requests
import numpy as np


st.image("../assets/banner.png", use_container_width=True)
st.title("Dashboard - Scoring client")

# Uploader CSV
uploaded_file = st.file_uploader(
    "Uploader un fichier CSV avec les clients à prédire", type="csv"
)

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Aperçu des données :")
    st.dataframe(data)

    # Selection du client
    id_clients = data["SK_ID_CURR"].unique().tolist()

    id_selectionne = st.selectbox(
        "Selectionner un client via son SK_ID_CURR :", id_clients
    )

    # Afficher les infos du client sélectionné
    client_selectionne = data[data["SK_ID_CURR"] == id_selectionne]
    st.write("Client selectionne :", client_selectionne)

    # Bouton pour lancer la prédiction
    if st.button("Prédire"):
        # Préparer les données du client sélectionné pour l'API (convertir NaN, inf)
        data_client = client_selectionne.drop(columns=["SK_ID_CURR"])

        data_json = [
            [
                None if (isinstance(x, float) and (np.isnan(x) or np.isinf(x))) else x
                for x in row
            ]
            for row in data_client.values.tolist()
        ]
        columns_json = data_client.columns.tolist()

        try:
            response = requests.post(
                #"https://projet7-credit-default-risk.azurewebsites.net/predict",
                "http://127.0.0.1:8000/predict",
                json={"data": data_json, "columns": columns_json},
            )
            if response.status_code == 200:
                predictions = response.json()

                st.dataframe(
                    pd.DataFrame(
                        {
                            "Prediction": [predictions["predictions"][0]],
                            "Score Client": [predictions["probas_class_1"][0] * 100],
                        }
                    )
                )

                # Jauge
                st.progress(
                    predictions["probas_class_1"][0],
                    text=f"Probabilité: {predictions['probas_class_1'][0]:.0%}",
                )

                if predictions["predictions"][0] == 1:
                    st.warning("**Attention : risque de défaut élevé**")
                else:
                    st.success("**Risque faible**")

            else:
                st.error(f"Erreur API : {response.status_code} - {response.text}")
        except Exception as erreur:
            st.error(f"Erreur lors de l'appel API : {erreur}")