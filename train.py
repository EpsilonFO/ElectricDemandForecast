import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def configurer_dispositif():
    """Configure et retourne le dispositif de calcul (CPU ou GPU)"""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    return device

class DRAGMLP(nn.Module):
    """
    Architecture de réseau neuronal pour prédire la consommation d'énergie régionale
    """
    def __init__(self, input_size, output_size, hidden_size=339):
        super(DRAGMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(input_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h = nn.ELU()(self.fc1(x))
        x = nn.GELU()(h)
        x = nn.Tanh()(x)
        x = self.fc3(x)
        return x

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
            print(f'Époque [{epoch+1}/{num_epochs}], Perte: {ep_loss/len(data_loader):.0f}')
            
    print(f"Perte d'époque minimale: {np.min(loss_memory)}, à la {np.argmin(loss_memory)+1}ème époque")
    return model

def test_loop(model, X_test, y_test, scaler_y, device):
    """
    Évaluer le modèle sur les données de test
    
    Args:
        model: Modèle PyTorch entraîné
        X_test: Données d'entrée de test
        y_test: Données cibles de test
        scaler_y: Normaliseur pour transformer inversement les prédictions
        device: Dispositif PyTorch (CPU/GPU)
        
    Returns:
        Tuple de (prédictions dénormalisées, cibles dénormalisées)
    """
    model.eval()
    with torch.no_grad():
        inputs = X_test.to(device)
        target = y_test.to(device)
        output = model(X_test)
        
        # Dénormalisation des prédictions et cibles
        output_descaled = scaler_y.inverse_transform(output.cpu())
        target_descaled = scaler_y.inverse_transform(target.cpu())
        
        # Calcul du RMSE
        nan_mask = ~np.isnan(target_descaled)
        rmse = np.nansum(np.sqrt(np.where(nan_mask, (output_descaled - target_descaled) ** 2, 0).sum(axis=0) / nan_mask.sum(axis=0)))
        print(f'Erreur quadratique moyenne sur l\'ensemble de test: {rmse.item():.0f}')
        
    return output_descaled, target_descaled

def preprocess_data():
    """
    Charger et prétraiter les données d'entraînement
    
    Returns:
        Tuple de (tenseur X_train, tenseur y_train, tenseur X_test, tenseur y_test, scaler_x, scaler_y, caractéristiques sélectionnées)
    """
    # Chargement des données
    X_train = pd.read_csv("Data/X_train_final.csv")
    y_train = pd.read_csv("Data/y_train.csv")

    # Définition des groupes de caractéristiques
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
    
    # Construction de la liste de caractéristiques
    X_col = temporal.copy()
    for agg in columns_to_aggregate:
       X_col += [agg+"_"+r for r in region]
       X_col += [agg+"_"+m for m in metro]
       
    # Sélection des caractéristiques et cibles  
    X_train, y_train = X_train[X_col], y_train[region]
    X_train = X_train.copy()
    X_train['date'] = pd.to_datetime(X_train['date'], utc=True, errors="coerce")

    # Ajout de caractéristiques temporelles
    X_train.loc[:,'minute'] = X_train['date'].dt.minute
    X_train.loc[:,'month'] = X_train['date'].dt.month
    X_train.loc[:,'day'] = X_train['date'].dt.day
    X_train.loc[:,'hour'] = X_train['date'].dt.hour
    X_train.loc[:,'weekday'] = X_train['date'].dt.weekday  # jours de la semaine (lundi=0, dimanche=6)

    # Conversion des caractéristiques catégorielles en one-hot
    X_train = pd.get_dummies(X_train, columns=['hour', "month", 'weekday'], dtype=int)
    X_train = X_train.drop(columns="date")  # Suppression de la colonne datetime

    # Stockage des caractéristiques sélectionnées pour une utilisation ultérieure
    selected_features = X_train.columns
    
    # Mise à l'échelle des caractéristiques et cibles
    scaler_x, scaler_y = StandardScaler(), StandardScaler()
    X_train_scaled, y_train_scaled = scaler_x.fit_transform(X_train), scaler_y.fit_transform(y_train)

    # Division en ensembles d'entraînement/test (division basée sur le temps)
    X_train_final, X_test, y_train_final, y_test = train_test_split(
        X_train_scaled, y_train_scaled, test_size=0.205, shuffle=False
    )

    # Conversion en tenseurs PyTorch
    X_train_tensor = torch.tensor(X_train_final, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_final, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    
    print(f"Forme des entrées: {X_train_tensor.shape}")
    print(f"Forme des cibles: {y_train_tensor.shape}")
    
    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, scaler_x, scaler_y, selected_features, region

def plot_results(target, output, region_name):
    """
    Tracer les valeurs réelles vs prédites
    
    Args:
        target: Valeurs réelles
        output: Valeurs prédites
        region_name: Nom de la région pour le titre du graphique
    """
    plt.figure(figsize=(14, 7))
    plt.plot(target[:,0], label="Réelle", color="skyblue")
    plt.plot(output[:,0], label='Prédictions', color='darkgreen', linestyle='--')
    plt.title(f"Prédiction de l'ensemble test sur {region_name}")
    plt.xlabel('Index (temps)')
    plt.ylabel('Consommation')
    plt.legend()
    plt.show()
    
def sauvegarder_meilleur_modele(model, output, target, region, training_time, selected_features, num_epochs, batch_size, scaler_x, optimizer):
    """
    Évalue le modèle et le sauvegarde s'il est meilleur que le précédent.
    Args:
        model: Le modèle à évaluer
        output: Les prédictions du modèle
        target: Les valeurs cibles réelles
        region: La région concernée
        training_time: Le temps d'entraînement en minutes
        selected_features: Les caractéristiques utilisées
        num_epochs: Le nombre d'époques d'entraînement
        batch_size: La taille du batch
        scaler_x: Le scaler utilisé
        optimizer: L'optimiseur utilisé
    Returns:
        float: Le RMSE actuel
    """
    # Calcul du RMSE actuel
    nan_mask = ~np.isnan(target)
    rmse = np.nansum(np.sqrt(np.where(nan_mask, (output - target) ** 2, 0).sum(axis=0) / nan_mask.sum(axis=0)))
    print(f"RMSE actuel: {rmse}")

    # Créer le dossier Model s'il n'existe pas
    os.makedirs("Model", exist_ok=True)

    # Chemin du fichier qui stocke le meilleur RMSE
    rmse_file_path = "Model/best_rmse.txt"

    # Récupérer le meilleur RMSE précédent s'il existe
    if os.path.exists(rmse_file_path):
        with open(rmse_file_path, 'r') as file:
            past_rmse = float(file.read().strip())
    else:
        # Valeur par défaut si le fichier n'existe pas
        past_rmse = 10000

    print(f"Meilleur RMSE précédent: {past_rmse}")

    # Comparer avec le RMSE précédent
    if rmse <= past_rmse:
        print(f"Amélioration: {past_rmse} à {rmse}. Bien joué!")
        
        # Sauvegarder le nouveau meilleur RMSE
        with open(rmse_file_path, 'w') as file:
            file.write(str(rmse))
            
        # Sauvegarder l'état du modèle
        torch.save(model.state_dict(), "Model/all_reg_best_model.pth")
        
        # Écrire les détails du modèle
        with open("Model/all_reg_best_model.txt", 'w') as file:
            file.write("Architecture du modèle:\n")
            file.write(str(model) + "\n\n")
            file.write(f"Performance:\n") 
            file.write(f"RMSE sur {region} : {rmse} \n")
            file.write(f"Temps d'entraînement : {training_time} minutes \n\n")
            file.write(f"Caractéristiques sélectionnées:\n")
            file.write(f"{selected_features}\n\n")
            file.write("Hyperparamètres:\n")
            file.write(f"Nombre d'époques : {num_epochs}\n")
            file.write(f"Taille de batch : {batch_size}\n") 
            file.write(f"Scaler : {scaler_x}\n")
            file.write(f"Critère : RMSE pondéré\n")
            file.write(f"Optimiseur : {optimizer}\n")
            
        print("Détails du modèle sauvegardés!")
    else:
        print("Pas d'amélioration! :-(")
        
    return rmse

def main():
    """
    Fonction principale pour exécuter l'ensemble du processus d'entraînement
    """

    # Configuration du dispositif
    device = configurer_dispositif()
    
    # Prétraitement des données
    X_train, y_train, X_test, y_test, scaler_x, scaler_y, selected_features, region = preprocess_data()
    
    # Initialisation du modèle et des paramètres d'entraînement
    model = DRAGMLP(input_size=X_train.shape[1], output_size=y_train.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    batch_size = 1024
    num_epochs = 50
    
    # Création du chargeur de données
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=False)
    
    # Entraînement du modèle
    start = time.time()
    model = training_loop(model, train_loader, num_epochs, optimizer, scaler_y, device)
    training_time = (time.time()-start)/60  # en minutes
    print(f"Temps d'entraînement: {training_time}min")
    
    # Test du modèle
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    output, target = test_loop(model, X_test, y_test, scaler_y, device)
    
    # Tracé des résultats
    plot_results(target, output, region[0])
    
    # Sauvegarder le modèle si c'est le meilleur
    sauvegarder_meilleur_modele(model, output, target, region, training_time, selected_features, 
                              num_epochs, batch_size, scaler_x, optimizer)
    
if __name__ == "__main__":
    main()
