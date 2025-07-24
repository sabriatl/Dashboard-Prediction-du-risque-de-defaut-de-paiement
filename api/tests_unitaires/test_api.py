from fastapi.testclient import TestClient
from api.main import app
import pandas as pd
import numpy as np
import os

client = TestClient(app)


def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_predict_single_client():
    # Charge le CSV exemple
    chemin_fichier = os.path.dirname(__file__)
    chemin_csv = os.path.join(chemin_fichier, "../../data", "sample_clients.csv")
    data = pd.read_csv(chemin_csv)

    # Choix d'un SK_ID_CURR existant
    id_selectionne = data["SK_ID_CURR"].iloc[0]

    # Filtrer la ligne du client sélectionné
    client_selectionne = data[data["SK_ID_CURR"] == id_selectionne]

    # Supprimer la colonne ID
    data_client = client_selectionne.drop(columns=["SK_ID_CURR"])

    # Préparer les données au format JSON
    data_json = [
        [
            None if (isinstance(x, float) and (np.isnan(x) or np.isinf(x))) else x
            for x in row
        ]
        for row in data_client.values.tolist()
    ]
    columns_json = data_client.columns.tolist()

    # Appel API
    response = client.post(
        "/predict", json={"data": data_json, "columns": columns_json}
    )

    # Vérifications
    assert response.status_code == 200
    json_response = response.json()
    assert "predictions" in json_response
    assert "probas_class_1" in json_response
    assert json_response["predictions"][0] in [0, 1]
    assert 0 <= json_response["probas_class_1"][0] <= 1
