import streamlit as st
import pandas as pd
import requests
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import plotly.graph_objects as go
import os

plt.style.use('fivethirtyeight')

st.set_page_config(
    page_title="Dashboard Scoring Client - API",
    page_icon="📊",
    layout="wide",  
    initial_sidebar_state="auto"
)

chemin_fichier = os.path.dirname(__file__)  # dossier du script actuel
chemin_banner = os.path.abspath(os.path.join(chemin_fichier, "..", "assets", "banner.png"))
st.image(chemin_banner, use_container_width=True)

st.markdown("# <div style='text-align: center;'>Dashboard - Scoring client</div>", unsafe_allow_html=True)


# SIDEBAR : Upload CSV 
with st.sidebar:
    uploaded_file = st.file_uploader("Uploader un fichier CSV avec les clients à prédire", type="csv")

if uploaded_file is not None:
    # Création des onglets
    tab1, tab2, tab3 = st.tabs(["Info Client", "Valeurs SHAP", "Comparaison des données"])

    with tab1:
        data = pd.read_csv(uploaded_file)
        st.write("### Aperçu des données :")
        st.dataframe(data, height=200)

        id_clients = data["SK_ID_CURR"].unique().tolist()
        id_selectionne = st.selectbox("Selectionner un client via son SK_ID_CURR :", id_clients)

        st.write("### Information du client sélectionné :")
        # Initialisation ou changement de client
        if "client_selectionne" not in st.session_state or st.session_state["client_id"] != id_selectionne:
            st.session_state["client_selectionne"] = data[data["SK_ID_CURR"] == id_selectionne]
            st.session_state["client_id"] = id_selectionne

        # Reset
        if st.button("Reset"):
            st.session_state["client_selectionne"] = data[data["SK_ID_CURR"] == id_selectionne]

        st.session_state["client_selectionne"] = st.data_editor(
            st.session_state["client_selectionne"],
            num_rows="fixed",
            use_container_width=True,
            hide_index=True
        )

        client_selectionne = st.session_state["client_selectionne"]

        col1, col2, col3 = st.columns(3)
        col1.metric("ID", client_selectionne["SK_ID_CURR"])
        if client_selectionne["CODE_GENDER"].iloc[0] == 0:
            col2.metric("Sexe", "Homme")
        else:
            col2.metric("Sexe", "Femme")
        col3.metric("Âge", f"{int(abs(client_selectionne['DAYS_BIRTH'].iloc[0]) / 365)} ans")
        

   
        if uploaded_file is not None:
            # Bouton Prédire
            if st.button("Prédire"):
                data_client = client_selectionne.drop(columns=["SK_ID_CURR"])
                data_json = [
                    [None if (isinstance(x, float) and (np.isnan(x) or np.isinf(x))) else x for x in row]
                    for row in data_client.values.tolist()
                ]
                columns_json = data_client.columns.tolist()

                try:
                    response = requests.post(
                        "https://projet7-credit-default-risk.azurewebsites.net/predict",
                        #"http://127.0.0.1:8000/predict",
                        json={"data": data_json, "columns": columns_json},
                    )
                    if response.status_code == 200:
                        pred_json = response.json()
                        st.session_state["predictions"] = pd.DataFrame({
                            "Prediction": [pred_json["predictions"][0]],
                            "Score Client (%)": [pred_json["probas_class_1"][0] * 100],
                        })
                    else:
                        st.error(f"Erreur API : {response.status_code} - {response.text}")
                except Exception as e:
                    st.error(f"Erreur lors de l'appel API : {e}")

            # Affichage prédiction 
            if "predictions" in st.session_state:
                st.write("### Résultat de la prédiction")
                #st.dataframe(st.session_state["predictions"])
                score = st.session_state["predictions"]["Score Client (%)"].values[0]               
                seuil = 47

                # Jauge Plotly
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=score,
                    number={'prefix': " Score client : "},
                    delta={'reference': seuil, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "black"},
                        'steps': [
                            {'range': [0, seuil-10], 'color': "green"},
                            {'range': [seuil-10, seuil], 'color': "orange"},
                            {'range': [seuil, 100], 'color': "red"},
                        ]}
                    ))
                fig.update_layout(width=450, height=450)
                st.plotly_chart(fig)

                # Message d'information
                if score < seuil - 10:
                    st.success(f"Score client : {score:.0f} / 100 — Risque faible (seuil = {seuil})")
                elif seuil - 10 <= score < seuil:
                    st.warning(f"Score client : {score:.0f} / 100 — À surveiller (seuil = {seuil})")
                else:
                    st.error(f"Score client : {score:.0f} / 100 — Risque élevé (seuil = {seuil})")


    with tab2:
        col1, col2 = st.columns(2)
        if uploaded_file is not None:
            with col1:
                # Bouton SHAP Local
                if st.button("Calcul SHAP Local (Waterfall)"):
                    data_client = client_selectionne.drop(columns=["SK_ID_CURR"])
                    data_json = [
                        [None if (isinstance(x, float) and (np.isnan(x) or np.isinf(x))) else x for x in row]
                        for row in data_client.values.tolist()
                    ]
                    columns_json = data_client.columns.tolist()

                    try:
                        response = requests.post(
                            "https://projet7-credit-default-risk.azurewebsites.net/shap_local",
                            #"http://127.0.0.1:8000/shap_local",
                            json={"data": data_json, "columns": columns_json},
                        )
                        if response.status_code == 200:
                            res_json = response.json()

                            # Nettoyage et extraction
                            st.session_state["shap_local_values"] = np.array(res_json["shap_values"])
                            st.session_state["shap_local_features"] = np.array(res_json["features_transformed"])
                            st.session_state["shap_local_names"]  = res_json["feature_names"]
                            st.session_state["shap_local_base_value"] = res_json["base_value"] 

                        else:
                            st.error(f"Erreur API : {response.status_code} - {response.text}")
                            
                    except Exception as e:
                        st.error(f"Erreur lors de l'appel API : {e}")

                # Affichage Waterfall
                if "shap_local_values" in st.session_state:
                    st.write("### Interprétation locale (SHAP - Waterfall)")
                    exp = shap.Explanation(
                        values=st.session_state["shap_local_values"],
                        data=st.session_state["shap_local_features"],
                        feature_names=st.session_state["shap_local_names"],
                        base_values=float(st.session_state["shap_local_base_value"])
                    )

                    # Appel du waterfall sans création préalable de fig/ax
                    shap.plots.waterfall(exp, show=False)

                    # Récupérer la figure créée automatiquement par SHAP
                    fig = plt.gcf()

                    # Optionnel : ajuster la taille si besoin
                    fig.set_size_inches(8, 10)

                    st.pyplot(fig)

                    

            with col2:
                # Bouton SHAP global
                if st.button("Calcul SHAP Global (Beeswarm)"):
                    data_pour_api = data.drop(columns=["SK_ID_CURR"], errors="ignore")
                    data_json = [
                        [None if (isinstance(x, float) and (np.isnan(x) or np.isinf(x))) else x for x in row]
                        for row in data_pour_api.values.tolist()
                    ]
                    columns_json = data_pour_api.columns.tolist()

                    try:
                        response = requests.post(
                            "https://projet7-credit-default-risk.azurewebsites.net/shap_global",
                            #"http://127.0.0.1:8000/shap_global",
                            json={"data": data_json, "columns": columns_json},
                        )
                        if response.status_code == 200:
                            res_json = response.json()
                            st.session_state["shap_values"] = np.array(res_json["shap_values"])
                            st.session_state["feature_names"] = res_json["feature_names"]
                            st.session_state["features_transformed"] = np.array(res_json["features_transformed"])
                        else:
                            st.error(f"Erreur API : {response.status_code} - {response.text}")
                    except Exception as e:
                        st.error(f"Erreur lors de l'appel API : {e}")

                # Affichage SHAP Global si existant
                if "shap_values" in st.session_state:
                    st.write("### Feature Importance Globale (SHAP)")
                    fig, ax = plt.subplots(figsize=(8, 10))
                    shap.summary_plot(
                        st.session_state["shap_values"],
                        features=st.session_state["features_transformed"],
                        feature_names=st.session_state["feature_names"],
                        show=False
                    )
                    st.pyplot(fig)

    with tab3:
        # filtre 
        filtre = st.radio(
            "Filtrer les clients similaires :",
            ["Vue globale", "Même sexe", "Même tranche d'âge", "Même sexe et tranche d'âge"]
        )
        if filtre == "Vue globale":
            data_filtre = data
        elif filtre == "Même sexe":
            data_filtre = data[data["CODE_GENDER"] == client_selectionne["CODE_GENDER"].iloc[0]]
        elif filtre == "Même tranche d'âge":
            data_filtre = data[(data["DAYS_BIRTH"] >= client_selectionne['DAYS_BIRTH'].iloc[0] - 1825) & (data["DAYS_BIRTH"] <= client_selectionne['DAYS_BIRTH'].iloc[0] + 1825)]
        elif filtre == "Même sexe et tranche d'âge":
            data_filtre = data[(data["DAYS_BIRTH"] >= client_selectionne['DAYS_BIRTH'].iloc[0] - 1825) & (data["DAYS_BIRTH"] <= client_selectionne['DAYS_BIRTH'].iloc[0] + 1825) & (data["CODE_GENDER"] == client_selectionne["CODE_GENDER"].iloc[0])]

        col1, col2 = st.columns(2)
        if uploaded_file is not None:
            with col1:
                st.write("Comparaison univariée :")
                 # 1. Liste des colonnes possibles à visualiser (hors ID)
                features_disponibles = [col for col in data.columns if col != "SK_ID_CURR"]

                # 2. Choix de la variable à visualiser
                variable_choisie = st.selectbox("Choisissez une variable à comparer :", features_disponibles)

                # 3. Création du graphique
                fig, ax = plt.subplots(figsize=(8, 4))
                
                # Histogramme ou KDE de tous les clients
                sns.histplot(data_filtre[variable_choisie], kde=True, color="green", bins=30, ax=ax)

                # Valeur du client
                valeur_client = client_selectionne[variable_choisie].values[0]
                ax.axvline(valeur_client, color="red", linestyle="--", linewidth=3, label="Client sélectionné")
                
                ax.legend(loc='upper left')
                st.pyplot(fig)

        with col2:
            st.write("Comparaison bivariée :")

            # Choix des 2 variables
            variable_x = st.selectbox("Variable X :", features_disponibles, key="x_bivariee")
            variable_y = st.selectbox("Variable Y :", features_disponibles, key="y_bivariee")

            # Création du plot
            fig, ax = plt.subplots(figsize=(6, 6))

            # 1. Tous les clients 
            sns.scatterplot(
                data=data_filtre,
                x=variable_x,
                y=variable_y,
                color="green",
                ax=ax,
                label="Population"
            )

            # 2. Ajout du client sélectionné
            sns.scatterplot(
                x=client_selectionne[variable_x],
                y=client_selectionne[variable_y],
                color="red",
                s=100,  
                ax=ax,
                label="Client sélectionné"
            )

            # Mise en forme
            ax.set_xlabel(variable_x)
            ax.set_ylabel(variable_y)
            ax.legend(loc='upper left')

            st.pyplot(fig)
                
