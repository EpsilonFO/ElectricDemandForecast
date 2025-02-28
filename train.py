import pandas as pd
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from model import MLP

def configurer_dispositif():
    """Configure et retourne le dispositif de calcul (CPU ou GPU)"""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    return device

def training_loop(model, data_loader, num_epochs, optimizer, scaler_y, device):
    """
    Fonction d'entraînement du modèle
    
    Args:
        model: Modèle PyTorch à entraîner
        data_loader: DataLoader contenant les données d'entraînement
        num_epochs: Nombre d'époques d'entraînement
        optimizer: Optimiseur PyTorch
        scaler_y: Normaliseur pour transformer inversement les prédictions
        device: Dispositif PyTorch (CPU/GPU)
        
    Returns:
        Modèle entraîné
    """
    model.train()
    loss_memory = []
    i = 0
    
    for epoch in range(num_epochs):
        ep_loss = 0

        for X, y in data_loader:
            # Transfert des données vers le dispositif
            inputs = X.to(device)
            target = y.to(device)
            
            # Réinitialisation des gradients
            optimizer.zero_grad()
            
            # Passe avant
            output = model(inputs)

            # Gestion des valeurs NaN dans les cibles
            nan_mask = ~torch.isnan(target)
            target = torch.where(nan_mask, target, torch.tensor(0.0, device=device))
            output = torch.where(nan_mask, output, torch.tensor(0.0, device=device))
            
            # Calcul de la perte RMSE pour chaque région et somme
            loss = torch.nansum(torch.sqrt(torch.sum((output - target)**2, dim=0) / nan_mask.sum(axis=0)))
            
            # Passe arrière et optimisation
            loss.backward()
            optimizer.step()

            # Dénormalisation des sorties et cibles pour le rapport
            output_descaled = scaler_y.inverse_transform(output.detach().cpu())
            target_descaled = scaler_y.inverse_transform(target.cpu())
            
            # Calcul du RMSE sur les valeurs dénormalisées
            nan_mask = ~np.isnan(target_descaled)
            rmse = np.nansum(np.sqrt(np.where(nan_mask, (output_descaled - target_descaled) ** 2, 0).sum(axis=0) / nan_mask.sum(axis=0)))

            ep_loss += rmse
            i += 1
            
        # Suivi et rapport de progression
        loss_memory.append(ep_loss/len(data_loader))
        if (epoch+1) % 10 == 0:
            print(f'Époque [{epoch+1}/{num_epochs}], Loss: {ep_loss/len(data_loader):.0f}')
            
    print(f"Min epoch loss : {np.min(loss_memory)}, à la {np.argmin(loss_memory)+1}ème époque")
    return model

def charger_donnees():
    """Charge les données d'entraînement et de test depuis les fichiers CSV"""
    X_train = pd.read_csv("Data/X_train_final.csv")
    y_train = pd.read_csv("Data/y_train.csv")
    X_2022 = pd.read_csv("Data/X_2022_final.csv")
    return X_train, y_train, X_2022

def definir_colonnes():
    """Définit les colonnes à agréger et les régions/métropoles à prédire"""
    columns_to_aggregate = ['t']
    region = ['France', 'Auvergne-Rhône-Alpes', 'Bourgogne-Franche-Comté', 'Bretagne',
           'Centre-Val de Loire', 'Grand Est', 'Hauts-de-France', 'Normandie',
           'Nouvelle-Aquitaine', 'Occitanie', 'Pays de la Loire',
           "Provence-Alpes-Côte d'Azur", 'Île-de-France']
    metro = ['Montpellier Méditerranée Métropole', 'Métropole Européenne de Lille',
           'Métropole Grenoble-Alpes-Métropole', "Métropole Nice Côte d'Azur",
           'Métropole Rennes Métropole', 'Métropole Rouen Normandie',
           "Métropole d'Aix-Marseille-Provence", 'Métropole de Lyon',
           'Métropole du Grand Nancy', 'Métropole du Grand Paris',
           'Nantes Métropole', 'Toulouse Métropole']
    temporal = ["is_holiday", 'date']
    total = region + metro
    return columns_to_aggregate, region, metro, temporal, total

def preparer_donnees(X_train, y_train, X_2022, columns_to_aggregate, temporal, total):
    """Prépare les données pour l'entraînement et la prédiction"""
    # Sélection des colonnes
    X_col = temporal.copy()
    for agg in columns_to_aggregate:
        X_col += [agg+"_"+r for r in total]
    X_train, y_train, X_2022 = X_train[X_col], y_train[total], X_2022[X_col]
    
    # Conversion des dates
    X_train = X_train.copy()
    X_2022 = X_2022.copy()
    X_train['date'] = pd.to_datetime(X_train['date'], utc=True, errors="coerce")
    X_2022['date'] = pd.to_datetime(X_2022['date'], utc=True, errors="coerce")

    # Extraction des caractéristiques temporelles
    X_train['minute'] = X_train['date'].dt.minute
    X_train['month'] = X_train['date'].dt.month
    X_train['day'] = X_train['date'].dt.day
    X_train['hour'] = X_train['date'].dt.hour
    X_train['weekday'] = X_train['date'].dt.weekday

    X_2022['minute'] = X_2022['date'].dt.minute
    X_2022['month'] = X_2022['date'].dt.month
    X_2022['day'] = X_2022['date'].dt.day
    X_2022['hour'] = X_2022['date'].dt.hour
    X_2022['weekday'] = X_2022['date'].dt.weekday

    # Encodage one-hot des caractéristiques temporelles
    X_train = pd.get_dummies(X_train, columns=['hour', "month",'weekday'], dtype=int)
    X_train = X_train.drop(columns="date")

    X_2022 = pd.get_dummies(X_2022, columns=['hour', "month",'weekday'], dtype=int)
    X_2022 = X_2022.drop(columns="date")
    
    return X_train, y_train, X_2022

def normaliser_donnees(X_train, y_train):
    """Normalise les données d'entrée et de sortie pour l'entraînement"""
    scaler_x, scaler_y = StandardScaler(), StandardScaler()
    X_train_scaled, y_train_scaled = scaler_x.fit_transform(X_train), scaler_y.fit_transform(y_train)
    
    # Conversion en tenseurs
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
    
    print(f"Forme des entrées: {X_train_tensor.shape}")
    print(f"Forme des cibles: {y_train_tensor.shape}")
    
    # Sauvegarde des scalers pour la prédiction
    torch.save(scaler_x, "Model/scaler_x.pt")
    torch.save(scaler_y, "Model/scaler_y.pt")
    
    return X_train_tensor, y_train_tensor, scaler_y

def creer_et_entrainer_modele(X_train_tensor, y_train_tensor, scaler_y, device):
    """Crée et entraîne le modèle de prédiction"""
    # Initialisation du modèle et de l'optimiseur
    input_size = X_train_tensor.shape[1]
    output_size = y_train_tensor.shape[1]
    model = MLP(input_size, output_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Paramètres d'entraînement
    batch_size = 1024
    num_epochs = 50
    
    # Création du DataLoader
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=False)
    
    # Entraînement du modèle
    start = time.time()
    model = training_loop(model, train_loader, num_epochs, optimizer, scaler_y, device)
    training_time = (time.time()-start)/60
    print(f"Temps d'entraînement: {training_time} minutes")
    
    return model

def sauvegarder_modele(model, input_size, output_size):
    """Sauvegarde le modèle entraîné et ses paramètres"""
    # Sauvegarde du modèle
    torch.save(model.state_dict(), "Model/pred_model.pth")
    
    # Sauvegarde des dimensions pour recréer le modèle lors de la prédiction
    torch.save({
        'input_size': input_size,
        'output_size': output_size
    }, "Model/model_params.pt")
    
    print("Modèle et paramètres sauvegardés avec succès !")

def main():
    """Fonction principale exécutant le pipeline d'entraînement"""
    # Configuration du dispositif
    device = configurer_dispositif()
    
    # Chargement des données
    X_train, y_train, X_2022 = charger_donnees()
    
    # Définition des colonnes
    columns_to_aggregate, region, metro, temporal, total = definir_colonnes()
    
    # Préparation des données
    X_train, y_train, X_2022 = preparer_donnees(X_train, y_train, X_2022, columns_to_aggregate, temporal, total)
    
    # Sauvegarde de X_2022 pour la prédiction
    X_2022.to_csv("Model/X_2022_prepared.csv", index=False)
    
    # Sauvegarde des colonnes pour la prédiction
    torch.save(total, "Model/total_columns.pt")
    
    # Normalisation des données d'entraînement
    X_train_tensor, y_train_tensor, scaler_y = normaliser_donnees(X_train, y_train)
    
    # Création et entraînement du modèle
    model = creer_et_entrainer_modele(X_train_tensor, y_train_tensor, scaler_y, device)
    
    # Sauvegarde du modèle et de ses paramètres
    sauvegarder_modele(model, X_train_tensor.shape[1], y_train_tensor.shape[1])
    
    print("Entraînement terminé avec succès !")

if __name__ == "__main__":
    main()