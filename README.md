# Implémentation d’une API REST pour un modèle de scoring crédit

Projet réalisé dans le cadre du parcours diplômant de Data Scientist d'OpenClassrooms

## Projet n°7 : Implémentation d’un modèle de scoring crédit

---

### Contexte

La société financière **Prêt à dépenser** propose des crédits à la consommation pour des personnes ayant peu ou pas du tout d’historique de prêt.

L’objectif est de développer un outil de **scoring crédit** permettant de calculer automatiquement la probabilité qu’un client rembourse son crédit, puis de classer la demande en crédit accordé ou refusé.

Un souci important est la **transparence** des décisions, demandée par les clients et valorisée par l’entreprise. Pour cela, un dashboard interactif est également développé afin de permettre aux chargés de relation client d’expliquer les décisions et d’explorer facilement les données clients.

---

### Données

Les données utilisées proviennent du concours Kaggle **Home Credit Default Risk** :  
[https://www.kaggle.com/c/home-credit-default-risk/data](https://www.kaggle.com/c/home-credit-default-risk/data)

---

### Objectifs du projet

- Construire un modèle supervisé de scoring capable de prédire la probabilité de défaut d’un client.
- Implémenter un pipeline complet allant du prétraitement des données à la prédiction.
- Intégrer le suivi des expérimentations via MLflow.
- Déployer le modèle sous forme d’une API REST.
- Développer un dashboard Streamlit pour tester l’API et visualiser les résultats.
- Réaliser un tableau d’analyse de data drift avec Evidently.
- Utiliser Git pour le versioning et un pipeline CI/CD pour déployer automatiquement l’API.

---

### Déploiement de l’API

L’API de prédiction est déployée sur Azure et accessible ici :  
[https://projet7-credit-default-risk.azurewebsites.net/predict](https://projet7-credit-default-risk.azurewebsites.net/predict)

Cet endpoint attend des requêtes POST avec des données JSON au format attendu et retourne la prédiction du score de défaut.

---

### Compétences évaluées

- Modélisation supervisée sur données déséquilibrées (gestion SMOTE, sample weights)
- Mise en place et suivi d’expérimentations avec MLflow
- Déploiement d’un modèle via une API REST
- Création d’un dashboard interactif Streamlit
- Analyse de data drift avec Evidently
- Versioning Git et CI/CD

---

### Lancer le projet localement

1. Cloner le repository  
2. Créer un environnement virtuel Python et installer les dépendances :

    ```bash
    pip install -r requirements.txt
    ```

3. Lancer l’API FastAPI :

    ```bash
    uvicorn api.main:app --reload
    ```

4. Lancer l’application Streamlit :

    ```bash
    streamlit run api/app_streamlit.py
    ```
