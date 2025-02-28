import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from model import MLP

def configurer_dispositif():
    """Configure et retourne le dispositif de calcul (CPU ou GPU)"""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Utilisation du dispositif: {device}")
    return device

def charger_modele(device):
    """
    Charge le modèle entraîné depuis le fichier sauvegardé
    
    Args:
        device: Dispositif PyTorch (CPU/GPU)
        
    Returns:
        Modèle chargé
    """
    # Chargement des paramètres du modèle
    model_params = torch.load("Model/model_params.pt")
    input_size = model_params['input_size']
    output_size = model_params['output_size']
    
    # Création du modèle avec les dimensions sauvegardées
    model = MLP(input_size, output_size).to(device)
    
    # Chargement des poids du modèle
    model.load_state_dict(torch.load("Model/pred_model.pth"))
    
    # Passage en mode évaluation
    model.eval()
    
    print(f"Modèle chargé avec succès (entrée: {input_size}, sortie: {output_size})")
    return model

def charger_donnees():
    """Charge les données préparées pour la prédiction"""
    # Chargement des données préparées
    X_2022 = pd.read_csv("Model/X_2022_prepared.csv")
    
    # Chargement des colonnes pour les prédictions
    total = torch.load("Model/total_columns.pt")
    
    # Chargement des scalers
    scaler_x = torch.load("Model/scaler_x.pt")
    scaler_y = torch.load("Model/scaler_y.pt")
    
    return X_2022, total, scaler_x, scaler_y

def preparer_donnees_prediction(X_2022, scaler_x):
    """
    Prépare les données pour la prédiction
    
    Args:
        X_2022: DataFrame contenant les caractéristiques pour la prédiction
        scaler_x: Scaler pour normaliser les données d'entrée
        
    Returns:
        Tenseur normalisé pour la prédiction
    """
    # Normalisation des données
    X_2022_scaled = scaler_x.transform(X_2022)
    
    # Conversion en tenseur
    X_2022_tensor = torch.tensor(X_2022_scaled, dtype=torch.float32)
    
    print(f"Forme des données à prédire: {X_2022_tensor.shape}")
    return X_2022_tensor

def faire_predictions(model, X_2022_tensor, scaler_y, total, device):
    """
    Réalise des prédictions avec le modèle chargé
    
    Args:
        model: Modèle PyTorch chargé
        X_2022_tensor: Données d'entrée pour la prédiction
        scaler_y: Scaler pour dénormaliser les sorties
        total: Liste des colonnes de sortie
        device: Dispositif PyTorch (CPU/GPU)
        
    Returns:
        DataFrame contenant les prédictions
    """
    print("Calcul des prédictions...")
    with torch.no_grad():
        X_2022_tensor = X_2022_tensor.to(device)
        pred_2022 = model(X_2022_tensor)
    
    # Dénormalisation des prédictions
    pred_2022 = scaler_y.inverse_transform(pred_2022.cpu())
    
    # Conversion en DataFrame
    pred_df = pd.DataFrame(pred_2022, columns=total)
    
    print(f"Prédictions terminées pour {len(pred_df)} instances")
    return pred_df

def sauvegarder_resultats(pred_df, total):
    """
    Sauvegarde les résultats de prédiction
    
    Args:
        pred_df: DataFrame contenant les prédictions
        total: Liste des colonnes de sortie
    """
    # Chargement du fichier de prédiction existant
    writing_pred = pd.read_csv("Solutions/pred.csv")
    
    # Ajout des prédictions
    for t in total:
        writing_pred["pred_"+t] = pred_df[t]
    
    # Sauvegarde des prédictions
    writing_pred.to_csv("Solutions/pred.csv", index=False)
    print("Prédictions sauvegardées dans Solutions/pred.csv")

def main():
    """Fonction principale exécutant le pipeline de prédiction"""
    print("Démarrage du processus de prédiction...")
    
    # Configuration du dispositif
    device = configurer_dispositif()
    
    # Chargement du modèle entraîné
    model = charger_modele(device)
    
    # Chargement des données et des scalers
    X_2022, total, scaler_x, scaler_y = charger_donnees()
    
    # Préparation des données pour la prédiction
    X_2022_tensor = preparer_donnees_prediction(X_2022, scaler_x)
    
    # Réalisation des prédictions
    pred_df = faire_predictions(model, X_2022_tensor, scaler_y, total, device)
    
    # Sauvegarde des résultats
    sauvegarder_resultats(pred_df, total)
    
    print("Processus de prédiction terminé avec succès !")

if __name__ == "__main__":
    main()