import os
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
    json_path = os.path.join(data_dir,"Raw","Raw_json")
    
    # Chemin où seront enregistrés les photos
    pictures_path = os.path.join(data_dir,"Raw", "Raw_pictures")
    
    # Chemin où sont téléchergées les images de frames bar 
    frames_bar_path = os.path.join(data_dir, "Raw","Frame_bar_pictures")

    # Chemin où seront enregistrés les masques
    frame_masks_path = os.path.join(data_dir, "Raw","Frame_masks_pictures")

    # Chemins où seront enregistrés les images et masques finaux.
    processed_pictures_path = os.path.join(data_dir, "Processed", "Processed_pictures")

    processed_frame_masks_path = os.path.join(data_dir, "Processed", "Processed_frame_masks")
    
    return data_dir, json_path, pictures_path, frames_bar_path, frame_masks_path, processed_pictures_path, processed_frame_masks_path

def load_json(json_path):
    """load_json
    
    Loads data from a JSON file.
    
    Args:
        json_path (str): the path to the JSON file

    Returns:
        array :  a list of items contained in the JSON file
    """
    json_path = os.path.join(json_path, "Raw.json")

    if not os.path.exists(json_path):
        print("Mettre le fichier raw.json dans le dossier Data/Raw/Raw_json")
        return None

    with open(json_path, "r") as f:
        data = json.load(f)

    items = []
    for raw in data:
        objects = raw
        items.append(raw)

    return items

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
    #     print("Le dossier a été créé avec succès.")
    # else:
    #     print("Le dossier existe déjà.")
