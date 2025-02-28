# Deep Learning Challenge
Auteurs : Lylian CHALLIER & Félix OLLIVIER

Ce projet de Deep Learning consiste à prédire des consommations électriques par région et métropoles de l'année 2022, à partir de données météos et des consommations électriques de 2017 à 2021 (et 2022 pour la météo).

Il est constitué de 4 fichiers :
`prepare_data.py`
`traitement_meteo.py`
`train.py`
`predict.py`.

## Comment lancer le projet
Tout d'abord, téléchargez les jeux de données via ce lien (trop volumineux pour être ajoutés au repo GitHub) : 
[Data](https://drive.google.com/drive/folders/19CdmxhwE5sEEytkxwUzmQj2EoLTNOL8o?usp=sharing)

Ensuite, ajoutez ce dossier au dossier du repo, et renommez le `Data`. Une fois ceci fait, lancez le script `prepare_data.py`.
Le projet est prêt à être éxécuté.

### `train.py`
Ce fichier contient le code nécessaire à l'entraînement du modèle MLP. Il charge les fichiers de données 'csv',
