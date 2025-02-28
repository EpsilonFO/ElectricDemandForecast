import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import graphviz
from dragon.search_space.bricks_variables import mlp_var, identity_var, operations_var, mlp_const_var, dag_var, node_var, dropout
from dragon.search_space.base_variables import ArrayVar
from dragon.search_operators.base_neighborhoods import ArrayInterval
from dragon.search_algorithm.mutant_ucb import Mutant_UCB
from dragon.utils.plot_functions import draw_cell, load_archi, str_operations
from model import MetaArchi

# Variables globales
batch_size = 128
num_epochs = 5
X_test = None
y_test = None
train_loader = None
input_shape = None
labels = None
scaler_y = None
device = None


# Configuration de l'appareil de calcul
def configurer_appareil():
    """Détecte et configure l'appareil de calcul (GPU ou CPU)"""
    global device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    return device


# Boucle d'entraînement
def training_loop(model, data_loader, num_epochs, optimizer, scaler_y, device, verbose=False):
    """Fonction pour entraîner le modèle"""
    global X_test, y_test
    
    model.train()
    loss_memory = []
    for epoch in range(num_epochs):
        ep_loss = 0

        for X, y in data_loader:
            inputs = X.to(device)
            target = y.to(device)
            optimizer.zero_grad()
            output = model(inputs)

            # Calcul de la perte RMSE
            nan_mask = ~torch.isnan(target)
            target = torch.where(nan_mask, target, torch.tensor(0.0, device=device))
            output = torch.where(nan_mask, output, torch.tensor(0.0, device=device))
            loss = torch.nansum(torch.sqrt(torch.sum((output - target)**2, dim=0) / nan_mask.sum(axis=0)))
            loss.backward()
            optimizer.step()

            # Calcul du RMSE pour l'époque
            output_descaled = scaler_y.inverse_transform(output.detach().cpu())
            target_descaled = scaler_y.inverse_transform(target.cpu())
            nan_mask = ~np.isnan(target_descaled)
            rmse = np.nansum(np.sqrt(np.where(nan_mask, (output_descaled - target_descaled) ** 2, 0).sum(axis=0) / nan_mask.sum(axis=0)))

            ep_loss += rmse

        loss_memory.append(ep_loss/len(data_loader))
        if verbose == True:
            if (epoch+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {ep_loss/len(data_loader):.0f}')
                output, target, rmsetest = test_loop(model, X_test, y_test, scaler_y, device, True)
    if verbose == True:
        print(f"Min epoch loss : {np.min(loss_memory)}, on the {np.argmin(loss_memory)+1}th epoch ")
    return model


# Évaluation du modèle
def test_loop(model, X_test, y_test, scaler_y, device, verbose=False):
    """Fonction pour évaluer le modèle sur les données de test"""
    model.eval()
    with torch.no_grad():
        inputs = X_test.to(device)
        target = y_test.to(device)
        output = model(X_test)
        # Déscaler les prédictions
        output_descaled = scaler_y.inverse_transform(output.cpu())
        target_descaled = scaler_y.inverse_transform(target.cpu())
        # Calcul du RMSE
        nan_mask = ~np.isnan(target_descaled)
        rmse = np.nansum(np.sqrt(np.where(nan_mask, (output_descaled - target_descaled) ** 2, 0).sum(axis=0) / nan_mask.sum(axis=0)))
        if verbose==True:
            print(f'Root Mean Squared Error on Test Set: {rmse.item():.0f}')
            return output_descaled, target_descaled, rmse
    return rmse


# Fonction de perte pour l'évaluation des modèles
def loss_function(args, idx):
    """Calcule la perte pour une configuration d'architecture donnée"""
    global labels, input_shape, train_loader, num_epochs, scaler_y, device, X_test, y_test
    
    args = dict(zip(labels, args))
    model = MetaArchi(args, input_shape).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model = training_loop(model, train_loader, num_epochs, optimizer, scaler_y, device)
    rmse = test_loop(model, X_test, y_test, scaler_y, device)
    print(f"Idx = {idx}, RMSE : {rmse:.0f}\n")
    return rmse, model


# Visualisation du graphe du modèle
def draw_graph(n_dag, m_dag, output_file, act="Identity()", name="Input"):
    """Génère une visualisation du graphe du modèle"""
    G = graphviz.Digraph(output_file, format='pdf',
                         node_attr={'nodesep': '0.02', 'shape': 'box', 'rankstep': '0.02', 'fontsize': '20', "fontname": "sans-serif"})

    G, g_nodes = draw_cell(G, n_dag, m_dag, "#ffa600", [], name_input=name, color_input="#ef5675")
    G.node(','.join(["MLP", "25", act]), style="rounded,filled", color="black", fillcolor="#ef5675", fontcolor="#ECECEC")
    G.edge(g_nodes[-1], ','.join(["MLP", "25", act]))
    return G


# Chargement et préparation des données
def charger_donnees():
    """Charge et prépare les données pour l'entraînement"""
    # Import des données
    X_train = pd.read_csv("Data/X_train_final.csv")
    y_train = pd.read_csv("Data/y_train.csv")
    X_2022 = pd.read_csv("Data/X_2022_final.csv")
    
    # Définition des colonnes à utiliser
    region = ["France", 'Auvergne-Rhône-Alpes', 'Bourgogne-Franche-Comté', 'Bretagne',
           'Centre-Val de Loire', 'Grand Est', 'Hauts-de-France', 'Normandie',
           'Nouvelle-Aquitaine', 'Occitanie', 'Pays de la Loire',
           "Provence-Alpes-Côte d'Azur", "Île-de-France"]
    metro = ['Montpellier Méditerranée Métropole', 'Métropole Européenne de Lille',
           'Métropole Grenoble-Alpes-Métropole', "Métropole Nice Côte d'Azur",
           'Métropole Rennes Métropole', 'Métropole Rouen Normandie',
           "Métropole d'Aix-Marseille-Provence", 'Métropole de Lyon',
           'Métropole du Grand Nancy', 'Métropole du Grand Paris',
           'Nantes Métropole', 'Toulouse Métropole']
    temporal = ['is_holiday', 'date']
    X_col = temporal.copy()
    X_col += ["t_"+r for r in region]
    X_col += ["t_"+m for m in metro]
    
    # Filtrage des colonnes
    X_train, y_train, X_2022 = X_train[X_col], y_train[region+metro], X_2022[X_col]
    X_train, X_2022 = X_train.copy(), X_2022.copy()
    
    # Conversion des dates
    X_train['date'] = pd.to_datetime(X_train['date'], utc=True, errors="coerce")
    X_2022['date'] = pd.to_datetime(X_2022['date'], utc=True, errors="coerce")
    
    # Ajout de caractéristiques temporelles
    X_train.loc[:,'minute'] = X_train['date'].dt.minute
    X_train.loc[:,'month'] = X_train['date'].dt.month
    X_train.loc[:,'day'] = X_train['date'].dt.day
    X_train.loc[:,'hour'] = X_train['date'].dt.hour
    X_train.loc[:,'weekday'] = X_train['date'].dt.weekday
    
    X_2022.loc[:,'minute'] = X_2022['date'].dt.minute
    X_2022.loc[:,'month'] = X_2022['date'].dt.month
    X_2022.loc[:,'day'] = X_2022['date'].dt.day
    X_2022.loc[:,'hour'] = X_2022['date'].dt.hour
    X_2022.loc[:,'weekday'] = X_2022['date'].dt.weekday
    
    # One-hot encoding des caractéristiques temporelles
    X_train = pd.get_dummies(X_train, columns=["month", "hour"], dtype=int)
    X_2022 = pd.get_dummies(X_2022, columns=["month", "hour"], dtype=int)
    
    # Suppression de la colonne date
    X_train = X_train.drop(columns="date")
    X_2022 = X_2022.drop(columns="date")
    selected_features = X_train.columns
    
    return X_train, y_train, X_2022, region, metro, selected_features


# Préparation des données pour l'entraînement
def preparer_donnees(X_train, y_train, X_2022):
    """Prépare les données pour l'entraînement (scaling, conversion en tenseurs)"""
    global X_test, y_test, scaler_y
    
    # Division train/test
    X_train, X_test_df, y_train, y_test_df = train_test_split(X_train, y_train, test_size=0.205, shuffle=False)
    
    # Scaling des données
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_train_scaled = scaler_x.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train)
    X_test_scaled = scaler_x.transform(X_test_df)
    y_test_scaled = scaler_y.transform(y_test_df)
    X_2022_scaled = scaler_x.transform(X_2022)
    
    # Conversion en tenseurs
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
    X_2022_tensor = torch.tensor(X_2022_scaled, dtype=torch.float32)
    X_test = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test = torch.tensor(y_test_scaled, dtype=torch.float32)
    
    print(f"The train shape: {X_train_tensor.shape}")
    print(f"The targets shape: {y_train_tensor.shape}")
    
    return X_train_tensor, y_train_tensor, X_2022_tensor


# Configuration de l'espace de recherche
def configurer_espace_recherche(num_classes):
    """Configure l'espace de recherche pour l'algorithme d'optimisation"""
    # Définition des opérations candidates
    candidate_operations = operations_var("Candidate operations", size=5, candidates=[mlp_var("MLP"), identity_var("Identity"), dropout("Dropout")])
    dag = dag_var("Dag", candidate_operations)
    print(f'An example of a generated DAG: {dag.random()}')
    
    # Configuration de la couche de sortie
    activation_function = nn.Identity()
    out = node_var("Out", operation=mlp_const_var('Operation', num_classes), activation_function=activation_function)
    search_space = ArrayVar(dag, out, label="Search Space", neighbor=ArrayInterval())
    
    return search_space


# Exécution de la recherche d'architecture
def executer_recherche(search_space):
    """Exécute l'algorithme de recherche d'architecture"""
    search_algorithm = Mutant_UCB(search_space, save_dir=f"save/test_mutant", T=20, N=10, K=10, E=0.01, evaluation=loss_function)
    search_algorithm.run()
    return search_algorithm


# Entraînement du meilleur modèle
def entrainer_meilleur_modele(search_space):
    """Charge et entraîne le meilleur modèle trouvé"""
    global X_test, y_test, scaler_y, device, input_shape, labels
    
    # Chargement du meilleur modèle
    best_model_params = load_archi(f"save/test_mutant/best_model/x.pkl")
    best_model_params = dict(zip(labels, best_model_params))
    
    # Visualisation de l'architecture
    m_dag = best_model_params['Dag'].matrix
    n_dag = str_operations(best_model_params["Dag"].operations)
    graph = draw_graph(n_dag, m_dag, "dragon_model/best_archi")
    graph
    
    # Création et chargement du modèle
    bestmodel = MetaArchi(best_model_params, input_shape)
    bestmodel.load_state_dict(torch.load(f"save/test_mutant/best_model/best_model.pth"))
    
    # Évaluation du modèle
    output, target, rmse = test_loop(bestmodel, X_test, y_test, scaler_y, device, True)
    
    # Entraînement supplémentaire
    batch_size = 128
    train_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
    num_epochs = 5
    optimizer = torch.optim.Adam(bestmodel.parameters(), lr=0.001)
    bestmodel = training_loop(bestmodel, train_loader, num_epochs, optimizer, scaler_y, device, True)
    
    return bestmodel


# Génération des prédictions
def generer_predictions(bestmodel, X_2022, region, metro):
    """Génère et sauvegarde les prédictions pour les données de 2022"""
    global scaler_y
    
    bestmodel.eval()
    with torch.no_grad():
        pred_2022 = bestmodel(X_2022)
    pred_2022 = scaler_y.inverse_transform(pred_2022.cpu())
    
    # Sauvegarde du modèle et des prédictions
    torch.save(bestmodel.state_dict(), "Model/predD_model.pth")
    writing_pred = pd.read_csv("Solutions/MLP/pred.csv")
    pred_2022 = pd.DataFrame(pred_2022, columns=region+metro)
    writing_pred[["pred_"+r for r in region]] = pred_2022[region]
    writing_pred[["pred_"+m for m in metro]] = pred_2022[metro]
    writing_pred.to_csv("Solutions/MLP/pred.csv", index=False)
    print(f"Model and predictions saved!")


# Fonction principale
def main():
    """Fonction principale orchestrant l'ensemble du processus"""
    global batch_size, train_loader, num_epochs, input_shape, labels, device
    
    # Configuration de l'appareil de calcul
    device = configurer_appareil()
    
    # Chargement et préparation des données
    X_train_df, y_train_df, X_2022_df, region, metro, selected_features = charger_donnees()
    X_train, y_train, X_2022 = preparer_donnees(X_train_df, y_train_df, X_2022_df)
    
    # Configuration de l'entraînement
    batch_size = 128
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=False)
    num_epochs = 5
    input_shape = (X_train.shape[1],)
    
    # Configuration de l'espace de recherche
    num_classes = 25
    search_space = configurer_espace_recherche(num_classes)
    labels = [e.label for e in search_space]
    
    # Test des configurations aléatoires
    p1, p2 = search_space.random(2)
    print("P1 ==> loss: ", loss_function(p1, 1))
    print("P2 ==> loss: ", loss_function(p2, 2))
    
    # Exécution de la recherche d'architecture
    search_algorithm = executer_recherche(search_space)
    
    # Entraînement du meilleur modèle
    bestmodel = entrainer_meilleur_modele(search_space)
    
    # Génération des prédictions
    generer_predictions(bestmodel, X_2022, region, metro)


if __name__ == "__main__":
    main()