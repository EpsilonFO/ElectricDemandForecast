import os
import shutil
import subprocess

try:
    # Exécuter la commande git lfs pull
    result = subprocess.run(['git', 'lfs', 'pull'], check=True, text=True, capture_output=True)
except subprocess.CalledProcessError as e:
    print(f"Une erreur s'est produite : {e}")
    print(f"Sortie standard : {e.stdout}")
    print(f"Erreur standard : {e.stderr}")

# Définition des dossiers
data_dir = "Data"
model_dir = "Model"
solutions_dir = "Solutions"
save_dir = "save/test_mutant/best_model"
pred_file = os.path.join(data_dir, "pred.csv")
dest_file = os.path.join(solutions_dir, "pred.csv")
dragon_x_file = os.path.join(data_dir, "x.pkl")
dragon_bm_file = os.path.join(data_dir, "best_model.pth")
dragon_x_dest = os.path.join(save_dir, "x.pkl")
dragon_bm_dest = os.path.join(save_dir, "best_model.pth")

# Création des dossiers si non existants
os.makedirs(model_dir, exist_ok=True)
os.makedirs(solutions_dir, exist_ok=True)
os.makedirs(save_dir, exist_ok=True)

# Déplacement du fichier pred.csv si présent
if os.path.exists(pred_file):
    shutil.move(pred_file, dest_file)
    print(f"Fichier déplacé : {pred_file} → {dest_file}")
else:
    print(f"Attention : '{pred_file}' n'existe pas ! Assurez-vous d'avoir téléchargé les données.")

print("Préparation terminée.")

if os.path.exists(dragon_x_file):
    shutil.move(dragon_x_file, os.path.join(save_dir, "x.pkl"))
    print(f"Fichier déplacé : {dragon_x_file} → {dragon_x_dest}")
else:
    print(f"Attention : '{dragon_x_file}' n'existe pas ! Assurez-vous d'avoir téléchargé les données.")

if os.path.exists(dragon_bm_file):
    shutil.move(dragon_bm_file, os.path.join(save_dir, "best_model.pth"))
    print(f"Fichier déplacé : {dragon_bm_file} → {dragon_bm_dest}")
else:
    print(f"Attention : '{dragon_bm_file}' n'existe pas ! Assurez-vous d'avoir téléchargé les données.")
