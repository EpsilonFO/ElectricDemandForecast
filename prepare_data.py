import os
import shutil

# Définition des dossiers
data_dir = "Data"
model_dir = "Model"
solutions_dir = "Solutions"
pred_file = os.path.join(data_dir, "pred.csv")
dest_file = os.path.join(solutions_dir, "pred.csv")

# Création des dossiers si non existants
os.makedirs(model_dir, exist_ok=True)
os.makedirs(solutions_dir, exist_ok=True)

# Déplacement du fichier pred.csv si présent
if os.path.exists(pred_file):
    shutil.move(pred_file, dest_file)
    print(f"Fichier déplacé : {pred_file} → {dest_file}")
else:
    print(f"Attention : '{pred_file}' n'existe pas ! Assurez-vous d'avoir téléchargé les données.")

print("Préparation terminée.")
