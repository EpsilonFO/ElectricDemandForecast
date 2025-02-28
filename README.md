# Deep Learning Challenge
Auteurs : Lylian CHALLIER & Félix OLLIVIER

Ce projet de Deep Learning consiste à prédire des consommations électriques par région et métropoles de l'année 2022, à partir de données météos et des consommations électriques de 2017 à 2021 (et 2022 pour la météo).

Il est constitué de 5 fichiers :
- `prepare_data.py`
- `traitement_meteo.py`
- `model.py`
- `train.py`
- `predict.py`.

## Comment lancer le projet
Tout d'abord, téléchargez les jeux de données via ce lien (trop volumineux pour être ajoutés au repo GitHub) : 
[Data](https://drive.google.com/drive/folders/19CdmxhwE5sEEytkxwUzmQj2EoLTNOL8o?usp=sharing)

Ensuite, ajoutez ce dossier au dossier du repo, et renommez le `Data`. Une fois ceci fait, lancez le script `prepare_data.py`.
Le projet est prêt à être éxécuté.

### `train.py`
Ce fichier contient le code nécessaire à l'entraînement du modèle MLP. Il charge les fichiers de données 'csv', ajuste les colonnes du jeu de données d'entraînement et de prédiction de la manière la plus optimisée selon notre modèle, puis entraine ce dernier avec la fonction `training_loop`. Enfin, il sauvegarde l'entraînement du modèle dans un fichier `pred_model.pth`, ce dernier dans le dossier `Model`.

### `predict.py`
Ce fichier charge le modèle entraîné par le fichier `train.py` puis fait une prédiction des consommations électriques sur toutes les colonnes entrées en argument de `y_train` dans le fichier `train.py`. Il enregistre enfin la prediction dans le fichier `pred.csv`. Il suffit ensuite de créer une archive `zip` avec ce fichier et envoyer cette archive sur [Codabench](https://www.codabench.org/competitions/5206/#/participate-tab).

### `model.py`
Ce fichier contient simplement notre architecture de modèle `DRAGMLP`.

### `traitement_meteo.py`
Ce fichier montre comment nous traitons le fichier `meteo.parquet` : On regroupe par région en faisant la moyenne, puis on utilise la fonction `pivot` pour créer les colonnes par région et par variable, on sélectionne les régions (ou si besoin la base la plus proche),