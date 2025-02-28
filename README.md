# Deep Learning Challenge
Auteurs : Lylian CHALLIER & Félix OLLIVIER

Ce projet de Deep Learning consiste à prédire des consommations électriques par région et métropoles de l'année 2022, à partir de données météos et des consommations électriques de 2017 à 2021 (et 2022 pour la météo).

## Structure du projet
```
ElectricDemandForecast/
├── README.md        <-- Vous êtes ici
├── dragon_model.py
├── model.py
├── predict.py
├── prepare_data.py
├── train.py
├── traitement_meteo.py
├── Data/            # Après l'archive Data téléchargé et prepare_data.py lancé
|   ├── jours_feries_metropole.csv
|   ├── population_region.csv
|   ├── X_2022_final.csv
|   ├── X_train_final.csv
|   ├── y_train.csv
├── Model/           # Une fois le fichier train.py lancé
|   ├── model_params.pt
|   ├── pred_model.pth
|   ├── scaler_x.pt
|   ├── scaler_y.pt
|   ├── total_columns.pt
|   └── X_2022_prepared.csv
├── save/            # Après l'archive Data téléchargé et prepare_data.py lancé
|   └── test_mutant/
|       └── best_model/
|           ├── x.pkl
|           └── best_model.pth
└── Solutions/
    └── pred.csv
```

## Comment lancer le projet
Tout d'abord, téléchargez les jeux de données via ce lien (trop volumineux pour être ajoutés au repo GitHub) : 
[Data](https://drive.google.com/drive/folders/19CdmxhwE5sEEytkxwUzmQj2EoLTNOL8o?usp=sharing)

Ensuite, ajoutez ce dossier au dossier du repo, et renommez le `Data`. Une fois ceci fait, lancez le script `prepare_data.py`.
Le projet est prêt à être éxécuté, lancez d'abord `train.py` pour entraîner le modèle et enregistrer les informations dans le dossier `Model` puis lancez `predict.py` pour mettre à jour le fichier `pred.csv` dans `Solutions`.

## Descriptions des fichiers
### `train.py`
Ce fichier contient le code nécessaire à l'entraînement du modèle MLP. Il charge les fichiers de données 'csv', ajuste les colonnes du jeu de données d'entraînement et de prédiction de la manière la plus optimisée selon notre modèle, puis entraine ce dernier avec la fonction `training_loop`. Enfin, il sauvegarde l'entraînement du modèle dans un fichier `pred_model.pth`, ce dernier dans le dossier `Model`.

### `predict.py`
Ce fichier charge le modèle entraîné par le fichier `train.py` puis fait une prédiction des consommations électriques sur toutes les colonnes entrées en argument de `y_train` dans le fichier `train.py`. Il enregistre enfin la prediction dans le fichier `pred.csv`. Il suffit ensuite de créer une archive `zip` avec ce fichier et envoyer cette archive sur [Codabench](https://www.codabench.org/competitions/5206/#/participate-tab).

### `model.py`
Ce fichier contient simplement notre architecture de modèle `MLP`.

### `traitement_meteo.py`
Ce fichier montre comment nous traitons le fichier `meteo.parquet` : On regroupe par région en faisant la moyenne, puis on utilise la fonction `pivot` pour créer les colonnes par région et par variable, on sélectionne les régions (ou si besoin la base la plus proche), on met à jour les noms des colonnes, puis on utilise les fonctions `resample` et `interpolate` pour créer des lignes de relevé toutes les 30 minutes, on ajoute la colonne des jours feriés, on crée les colonnes pour la France globale, on enlève les dates problématiques liées au changement d'heure, on complète les dates manquantes pour `X_2022` puis on enregistre les jeux de données.

### `dragon_model.py`
Ce fichier utilise la bibliothèque `DRAGON` pour créer un modèle efficace et puissant. 
(WARNING : L'éxécution du fichier est très longue. Pour simplifier le processus, nous avons donné dans le dossier Data le meilleur modèle, avec les fichiers `best_model.pth` et `x.pkl`, présents dans le dossier save/test_mutant/best_model).

### `prepare_data.py`
Ce fichier sert à initialiser correctement l'archive, en créant les dossiers nécessaires pour stocker les modèles et prédictions à venir.
