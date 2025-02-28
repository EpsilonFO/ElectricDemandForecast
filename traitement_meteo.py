import pandas as pd

def charger_donnees():
    """
    Charge les données météorologiques depuis un fichier parquet.
    
    Returns:
        DataFrame: Les données météorologiques brutes
    """
    return pd.read_parquet('Data/meteo.parquet')

def convertir_colonnes_numeriques(df, colonnes):
    """
    Convertit les colonnes spécifiées en format numérique.
    
    Args:
        df (DataFrame): DataFrame à traiter
        colonnes (list): Liste des colonnes à convertir
        
    Returns:
        DataFrame: DataFrame avec les colonnes converties
    """
    for col in colonnes:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def agreger_par_region(df, colonnes_a_agreger):
    """
    Agrège les données par date et région.
    
    Args:
        df (DataFrame): DataFrame à traiter
        colonnes_a_agreger (list): Colonnes à agréger
        
    Returns:
        DataFrame: DataFrame pivoté avec les moyennes par région
    """
    # Regrouper par date et région, calculer les moyennes
    df_grouped = df.groupby(['date', 'nom_reg'], as_index=False)[colonnes_a_agreger].mean()
    
    # Pivoter pour avoir les colonnes par région
    df_pivot = df_grouped.pivot(index='date', columns='nom_reg', values=colonnes_a_agreger).reset_index()
    df_pivot.columns.name = None
    
    # Formater les dates et noms de colonnes
    df_pivot['date'] = pd.to_datetime(df_pivot['date'])
    df_pivot.columns = [f'{col[0]}_{col[1]}' if col[1] else col[0] for col in df_pivot.columns]
    df_pivot.reset_index(inplace=True)
    
    return df_pivot

def filtrer_et_fusionner_metropoles(df, df_pivot, colonnes_a_agreger, noms_metropoles, noms_supp):
    """
    Filtre les données pour les métropoles et villes supplémentaires et les fusionne avec le DataFrame principal.
    
    Args:
        df (DataFrame): DataFrame original
        df_pivot (DataFrame): DataFrame pivoté par région
        colonnes_a_agreger (list): Colonnes à agréger
        noms_metropoles (list): Liste des noms de métropoles
        noms_supp (list): Liste des noms supplémentaires
        
    Returns:
        DataFrame: DataFrame fusionné avec les données des métropoles et villes
    """
    # Filtrer les données pour les métropoles et villes spécifiques
    df_filtered = df[df['libgeo'].isin(noms_supp) | df['nom_epci'].isin(noms_metropoles)]
    
    # Pour chaque colonne à agréger, créer des pivots par ville et par EPCI
    for col in colonnes_a_agreger:
        # Pivot par ville
        df_pivot_additional = df_filtered.pivot(index='date', columns='libgeo', values=col).reset_index()
        df_pivot_additional.columns.name = None
        df_pivot_additional.columns = [f'{col}_{libgeo}' if libgeo != 'date' else 'date' for libgeo in df_pivot_additional.columns]

        # Pivot par EPCI (métropole)
        df_pivot_additional_epci = df_filtered.pivot(index='date', columns='nom_epci', values=col).reset_index()
        df_pivot_additional_epci.columns.name = None
        df_pivot_additional_epci.columns = [f'{col}_{epci}' if epci != 'date' else 'date' for epci in df_pivot_additional_epci.columns]

        # Fusionner avec le DataFrame principal
        df_pivot = df_pivot.merge(df_pivot_additional, on='date', how='left')
        df_pivot = df_pivot.merge(df_pivot_additional_epci, on='date', how='left')
    
    return df_pivot

def reorganiser_colonnes(df_pivot, colonnes_a_agreger, noms_regions, noms_metropoles, noms_supp):
    """
    Réorganise les colonnes du DataFrame.
    
    Args:
        df_pivot (DataFrame): DataFrame à réorganiser
        colonnes_a_agreger (list): Colonnes agrégées
        noms_regions (list): Noms des régions
        noms_metropoles (list): Noms des métropoles
        noms_supp (list): Noms supplémentaires
        
    Returns:
        DataFrame: DataFrame avec colonnes réorganisées
    """
    df_pivot.reset_index(inplace=True)
    
    # Générer la liste ordonnée des noms de colonnes
    noms_colonnes = []
    for agg in colonnes_a_agreger:
        noms_colonnes += [agg+"_"+nom for nom in noms_regions+noms_metropoles+noms_supp]
    noms_colonnes.append('date')
    
    # Filtrer pour ne garder que ces colonnes
    df_pivot = df_pivot[noms_colonnes]
    df_pivot.set_index('date', inplace=True)
    
    return df_pivot

def resample_et_renommer(df_pivot, colonnes_a_agreger):
    """
    Effectue un rééchantillonnage à 30 minutes et renomme certaines colonnes.
    
    Args:
        df_pivot (DataFrame): DataFrame à traiter
        colonnes_a_agreger (list): Colonnes agrégées
        
    Returns:
        DataFrame: DataFrame rééchantillonné avec colonnes renommées
    """
    # Rééchantillonner à 30 minutes avec interpolation linéaire
    df_resampled = df_pivot.resample('30min').interpolate(method='linear')
    df_resampled.reset_index(inplace=True)
    
    # Renommer les colonnes
    for agg in colonnes_a_agreger:
        df_resampled[agg+'_Métropole de Lyon'] = df_resampled[agg+'_Colombier-Saugnieu']
        df_resampled = df_resampled.rename(columns={agg+"_Mauguio":agg+"_Montpellier Méditerranée Métropole"})
        df_resampled = df_resampled.rename(columns={agg+"_Colombier-Saugnieu":agg+"_Métropole Grenoble-Alpes-Métropole"})
        df_resampled = df_resampled.rename(columns={agg+"_Rennes Métropole":agg+"_Métropole Rennes Métropole"})
        df_resampled = df_resampled.rename(columns={agg+"_Thuilley-aux-Groseilles":agg+"_Métropole du Grand Nancy"})
    
    # Remplir les valeurs manquantes
    df_resampled = df_resampled.bfill()
    
    return df_resampled

def ajouter_jours_feries(df_resampled):
    """
    Ajoute une colonne indiquant si la date est un jour férié.
    
    Args:
        df_resampled (DataFrame): DataFrame à traiter
        
    Returns:
        DataFrame: DataFrame avec colonne de jours fériés
    """
    # Charger les jours fériés
    feries = pd.read_csv("Data/jours_feries_metropole.csv")
    feries = feries[(feries['date']>="2017-01-01") & (feries['date']<"2023-01-01")]
    feries = feries[['date']]
    feries['date'] = pd.to_datetime(feries['date'])
    
    # Préparer les dates pour la comparaison
    df_resampled['date'] = pd.to_datetime(df_resampled['date'])
    df_resampled['date_only'] = df_resampled['date'].dt.date
    feries['date_only'] = feries['date'].dt.date
    
    # Ajouter la colonne des jours fériés (1 si férié, 0 sinon)
    df_resampled['is_holiday'] = df_resampled['date_only'].isin(feries['date_only']).astype(int)
    df_resampled = df_resampled.drop(columns=['date_only'])
    
    return df_resampled

def ajouter_agregat_france(df_resampled, colonnes_a_agreger):
    """
    Ajoute des colonnes agrégées pour la France entière, pondérées par population.
    
    Args:
        df_resampled (DataFrame): DataFrame à traiter
        colonnes_a_agreger (list): Colonnes à agréger
        
    Returns:
        DataFrame: DataFrame avec colonnes agrégées pour la France
    """
    # Charger les données de population
    population_region = pd.read_csv("Data/population_region.csv", sep=";")
    
    # Calculer les poids proportionnels à la population
    poids = population_region["p21_pop"]/population_region["p21_pop"].sum()
    
    # Calculer les moyennes pondérées pour la France
    noms_regions = population_region["nom_reg"].tolist()
    for agg in colonnes_a_agreger:
        df_resampled[agg + "_France"] = df_resampled[[agg + "_" + nomreg for nomreg in noms_regions]].mul(poids.values, axis=1).sum(axis=1)
    
    return df_resampled

def supprimer_dates_problematiques(df_resampled):
    """
    Supprime les dates problématiques (changements d'heure).
    
    Args:
        df_resampled (DataFrame): DataFrame à traiter
        
    Returns:
        DataFrame: DataFrame sans les dates problématiques
    """
    # Liste des dates à supprimer (changements d'heure)
    dates_a_supprimer = [
        "2017-10-29 02:00:00+02:00",
        "2017-10-29 02:30:00+02:00",
        "2018-10-28 02:00:00+02:00",
        "2018-10-28 02:30:00+02:00",
        "2019-10-27 02:00:00+02:00",
        "2019-10-27 02:30:00+02:00",
        "2020-10-25 02:00:00+02:00",
        "2020-10-25 02:30:00+02:00",
        "2021-10-31 02:00:00+02:00",
        "2021-10-31 02:30:00+02:00",
        "2022-10-30 02:00:00+02:00",
        "2022-10-30 02:30:00+02:00"
    ]
    dates_a_supprimer = pd.to_datetime(dates_a_supprimer, utc=True)
    
    # Filtrer le DataFrame pour exclure ces dates
    return df_resampled[~df_resampled['date'].isin(dates_a_supprimer)]

def separer_train_test(df_resampled):
    """
    Sépare les données en ensembles d'entraînement et de test.
    
    Args:
        df_resampled (DataFrame): DataFrame complet
        
    Returns:
        tuple: (DataFrame d'entraînement, DataFrame de test)
    """
    # Ensemble d'entraînement : 2017-2021
    df_train = df_resampled[(df_resampled['date']>="2017-02-13 00:30:00+00:00") & 
                            (df_resampled['date']<"2021-12-31 23:00:00+00:00")]
    
    # Ensemble de test : 2022
    df_2022 = df_resampled[(df_resampled['date']>="2022-01-01")]
    
    return df_train, df_2022

def completer_donnees_2022(df_2022):
    """
    Complète les données 2022 avec des dates manquantes.
    
    Args:
        df_2022 (DataFrame): DataFrame des données 2022
        
    Returns:
        DataFrame: DataFrame 2022 complété
    """
    # Charger les données X_2022 existantes
    X_2022 = pd.read_csv("Data/X_2022_final.csv")
    france2022_train = X_2022[['date']]
    
    # Ajouter les premières heures de l'année 2022
    date_range = pd.date_range(start='2022-01-01 00:00:00+01:00', 
                             end='2022-01-01 00:30:00+01:00', 
                             freq='30min')
    new_rows = pd.DataFrame()
    for date in date_range:
        new_row = france2022_train.iloc[0].copy()
        new_row['date'] = date
        new_rows = pd.concat([new_rows, new_row.to_frame().T], ignore_index=True)
    france2022_train = pd.concat([france2022_train, new_rows], ignore_index=True)
    
    # Ajouter des heures manquantes du 27 mars 2022
    date_range = pd.date_range(start='2022-03-27 04:00:00+02:00', 
                             end='2022-03-27 04:30:00+02:00', 
                             freq='30min')
    new_rows = pd.DataFrame()
    for date in date_range:
        new_row = france2022_train.iloc[0].copy()
        new_row['date'] = date
        new_rows = pd.concat([new_rows, new_row.to_frame().T], ignore_index=True)
    france2022_train = pd.concat([france2022_train, new_rows], ignore_index=True)
    
    # Fusionner avec df_2022
    df_2022['date'] = pd.to_datetime(df_2022['date'])
    france2022_train['date'] = pd.to_datetime(france2022_train['date'], utc=True)
    df_2022 = df_2022.merge(france2022_train, on='date')
    
    # Ajouter les dernières heures de l'année 2022
    date_range = pd.date_range(start='2022-12-31 22:00:00+01:00', 
                             end='2022-12-31 23:30:00+01:00', 
                             freq='30min')
    new_rows = pd.DataFrame()
    for date in date_range:
        new_row = df_2022.iloc[-1].copy()
        new_row['date'] = date
        new_rows = pd.concat([new_rows, new_row.to_frame().T], ignore_index=True)
    df_2022 = pd.concat([df_2022, new_rows], ignore_index=True)
    
    return df_2022

def enregistrer_donnees(df_train, df_2022):
    """
    Enregistre les données d'entraînement et de test dans des fichiers CSV.
    
    Args:
        df_train (DataFrame): DataFrame d'entraînement
        df_2022 (DataFrame): DataFrame de test
    """
    df_train.to_csv("Data/X_train_final.csv", index=False)
    df_2022.to_csv("Data/X_2022_final.csv", index=False)

def main():
    """
    Fonction principale qui exécute l'ensemble du processus de traitement des données.
    """
    # Définir les colonnes à agréger
    colonnes_a_agreger = ['dd','ff','per','pmer','pres','rafper','t','td','tend','u','vv',
                        'rr1','rr3','rr6','nnuage1','hnuage1']
    
    # Définir les noms de métropoles et villes
    noms_metropoles = ['Métropole Européenne de Lille', 'Métropole Nice Côte d\'Azur',
                      'Rennes Métropole', 'Métropole Rouen Normandie',
                      'Métropole d\'Aix-Marseille-Provence', 'Métropole du Grand Paris',
                      'Nantes Métropole', 'Toulouse Métropole']
    noms_supp = ['Mauguio', 'Colombier-Saugnieu', 'Thuilley-aux-Groseilles']
    noms_regions = ['Auvergne-Rhône-Alpes', 'Bourgogne-Franche-Comté',
                   'Bretagne', 'Centre-Val de Loire', 'Grand Est',
                   'Hauts-de-France', 'Normandie', 'Nouvelle-Aquitaine',
                   'Occitanie', 'Pays de la Loire', 'Provence-Alpes-Côte d\'Azur',
                   'Île-de-France']
    
    # Étape 1: Charger les données
    df = charger_donnees()
    
    # Étape 2: Convertir les colonnes en format numérique
    df = convertir_colonnes_numeriques(df, colonnes_a_agreger)
    
    # Étape 3: Agréger par région
    df_pivot = agreger_par_region(df, colonnes_a_agreger)
    
    # Étape 4: Filtrer et fusionner les données des métropoles
    df_pivot = filtrer_et_fusionner_metropoles(df, df_pivot, colonnes_a_agreger, 
                                             noms_metropoles, noms_supp)
    
    # Étape 5: Réorganiser les colonnes
    df_pivot = reorganiser_colonnes(df_pivot, colonnes_a_agreger, noms_regions, 
                                  noms_metropoles, noms_supp)
    
    # Étape 6: Rééchantillonner et renommer
    df_resampled = resample_et_renommer(df_pivot, colonnes_a_agreger)
    
    # Étape 7: Ajouter les jours fériés
    df_resampled = ajouter_jours_feries(df_resampled)
    
    # Étape 8: Ajouter les agrégats France
    df_resampled = ajouter_agregat_france(df_resampled, colonnes_a_agreger)
    
    # Étape 9: Supprimer les dates problématiques
    df_resampled = supprimer_dates_problematiques(df_resampled)
    
    # Étape 10: Séparer en train/test
    df_train, df_2022 = separer_train_test(df_resampled)
    
    # Étape 11: Compléter les données 2022
    df_2022 = completer_donnees_2022(df_2022)
    
    # Étape 12: Enregistrer les données
    enregistrer_donnees(df_train, df_2022)
    
    print("Traitement des données météorologiques terminé avec succès!")

# Exécuter le programme si le script est appelé directement
if __name__ == "__main__":
    main()
