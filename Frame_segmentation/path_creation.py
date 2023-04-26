import os
import requests
import json

def define_paths():
    """define_paths
    This function defines the paths for the Frame_segmentation folders.

    Returns:
        str: returns the paths of the folders
    """
    # Chemin du dossier Data
    data_dir = os.getcwd() + "/Data"

    # Chemin où se trouve le JSON
    json_path = os.path.join(data_dir,"Raw","Raw_json","raw.json")
    
    # Chemin où seront enregistrés les photos
    pictures_path = os.path.join(data_dir,"Raw", "Raw_pictures")
    
    # Chemin où sont téléchergées les images de frames bar 
    frames_bar_path = os.path.join(data_dir, "Raw","Frame_bar_pictures")

        # Chemin où seront enregistrés les masques
    frame_masks_path = os.path.join(data_dir, "Raw","Frame_masks_pictures")
    
    return data_dir, json_path, pictures_path, frames_bar_path, frame_masks_path

def load_json(json_path):
    """load_json
    
    Loads data from a JSON file.
    
    Args:
        json_path (str): the path to the JSON file

    Returns:
        array :  a list of items contained in the JSON file
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    items = []
    for raw in data:
        objects = raw
        items.append(raw)

    return items
#appeler cette fonction avec le chemin vers votre fichier JSON
#items = load_json(json_path)

def create_folder(path):
    """create folder
    
    This function checks if a folder already exists at a given path,
    and if it doesn't, it creates it.

    Args:
        path (str): path to the folder where the images are saved
    """
    # Vérifier si le dossier existe déjà sinon le créer
    if not os.path.exists(path):
        os.makedirs(path)
        print("Le dossier a été créé avec succès.")
    else:
        print("Le dossier existe déjà.")

def rename_files(frame_masks_path):
    """rename_files
    
    the function renames all image files 
    in a folder to give them a name without extension.
    
    Args:
        masks_path (str): path for the masks
    """
    frame_masks_path = "./structure/masks/"
    print(frame_masks_path)
    # Boucle sur tous les fichiers dans le dossier
    for file in os.listdir(frame_masks_path):
        # Vérifie que le fichier est un fichier et non un dossier
        if os.path.isfile(os.path.join(frame_masks_path, file)):
            # Vérifie que le fichier a l'extension ".jpg"
            if ".jpg" in file:
                # Construit le nouveau nom de fichier sans l'extension ".jpg"
                new_name = file.replace(".jpg", "")
                # Renomme le fichier en utilisant le nouveau nom
                os.rename(os.path.join(frame_masks_path, file), os.path.join(frame_masks_path, new_name))
