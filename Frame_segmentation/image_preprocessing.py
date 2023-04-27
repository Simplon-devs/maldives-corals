from Frame_segmentation.path_creation import create_folder
import os
import requests
import json
from PIL import Image
import glob
import numpy as np
import cv2

def download_masks(items, frames_bar_path):
    """
    This function downloads mask images, only masks associated with frame_bar are downloaded,
    and they are saved in folders with the name of the associated object.
    """
    for item in items:
        # Vérifier s'il y a des objets "frame_bar" dans l'élément actuel
        frame_bar_objects = [obj for obj in item['Label']['objects'] if obj['value'] == 'frame_bar']
        if len(frame_bar_objects) > 0:
            image_id = item['External ID']
            # Chemin du dossier à créer
            pictures_frame_bar_path = os.path.join(frames_bar_path, image_id.replace(".jpg", ""))
            # Vérifier si le dossier existe déjà
            if not os.path.exists(pictures_frame_bar_path):
                create_folder(pictures_frame_bar_path)
                i = 0
                # Téléchargement seulement des masques frame_bar
                for obj in frame_bar_objects:
                    i += 1
                    image_url = obj['instanceURI']
                    response = requests.get(image_url)
                    if response.status_code == 200:
                        image_content = response.content
                        with open(os.path.join(pictures_frame_bar_path, f"frame_{i}.png"), "wb") as f:
                            f.write(image_content)
                        print(f"L'image a été enregistrée avec succès : frame_{i}.png")
                        print(item['External ID'])
                    else:
                        print("Impossible de télécharger l'image.")
            else:
                print(f"Le dossier '{pictures_frame_bar_path}' existe déjà. Aucune image ne sera téléchargée pour l'élément {item['External ID']}.")
        else:
            print(f"Aucun objet 'frame_bar' trouvé dans l'élément {item['External ID']}. Aucune image ne sera téléchargée.")

            
def download_images(items, pictures_path):
    """
    This function downloads images only for items that have associated frame_bar masks
    """
    for item in items:
        # Vérifier s'il y a des objets "frame_bar" dans l'élément actuel
        frame_bar_objects = [obj for obj in item['Label']['objects'] if obj['value'] == 'frame_bar']
        if len(frame_bar_objects) > 0:
            image_id = item['External ID']
            image_path = os.path.join(pictures_path, image_id)
            if os.path.exists(image_path):
                print(f"L'image {image_id} existe déjà dans le dossier. Passer à l'élément suivant.")
                continue
                
            url = item['Labeled Data']
            response = requests.get(url)
            if response.status_code == 200:
                image_content = response.content
                with open(image_path, "wb") as f:
                    f.write(image_content)
                    print(f"L'image a été enregistrée avec succès : {image_id}")
                    print(item['External ID'])
            else:
                print(f"Impossible de télécharger l'image {image_id}.")
        else:
            print(f"Aucun objet 'frame_bar' trouvé dans l'élément {item['External ID']}. Aucune image ne sera téléchargée.")


def superpose_masks(frames_bar_path, frame_masks_path):
    """superpose_masks
    
    the function combines several mask images into one superimposed image,
    which represents the intersection of the masks.
    This image is then saved in a specified folder.

    Args:
        pictures_path (str): path for the pictures
        masks_path (str): path for the masks
    """
    # Définir les noms des dossiers contenant les images à superposer
    image_folders = os.listdir(frames_bar_path)
    directory = frames_bar_path

    for image_folder in image_folders:
        # Définir le chemin du dossier contenant les images à superposer
        images_dir = os.path.join(directory, image_folder)

        # Charger les images
        images_png = []
        for i in range(1, 8):
            nom_de_fichier = os.path.join(images_dir, f"frame_{i}.png")
            if os.path.isfile(nom_de_fichier):
                image = Image.open(nom_de_fichier)
                images_png.append(image)

        # Superposer les images
        if images_png:
            image_superposee = images_png[0]
            for i in range(1, len(images_png)):
                image_superposee.alpha_composite(images_png[i])

            # Enregistrer l'image superposée
            mask_dir = frame_masks_path
            if not os.path.exists(mask_dir):
                os.makedirs(mask_dir)
            image_name = image_folder.replace(".jpg", "")
            image_superposee.save(os.path.join(mask_dir, f"{image_name}_mask.png"))


def prepare_pictures(frame_masks_path, pictures_path, processed_pictures_path, processed_frame_masks_path, target_width = 192):

    # Resize pictures and masks, adds padding, and stores them in out_path

    i, j = 0, 0
    target_height = 0.75 * target_width

    # Extraire les chemins de tous les fichiers d'image avec les extensions .jpg et .png dans pictures_path
    pics_paths = glob.glob(pictures_path + "/*.jpg")
    
    # Extraire les chemins de tous les fichiers de masque avec l'extension .png dans masks_path
    masks_paths = glob.glob(frame_masks_path + "/*.png")

    # Pictures
    
    for path in pics_paths:
        new_path = path.replace(pictures_path, processed_pictures_path).replace('.jpg', '.png')

        if new_path.replace(processed_pictures_path, frame_masks_path).replace('.png', '_mask.png') in masks_paths:

            image = Image.open(path)
            width, height = image.size
            new_height = int(height * target_width / width)
            size = target_width, new_height
            image.thumbnail(size, Image.ANTIALIAS)
            im = np.array(image, np.float32)

            padded = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)

            cv2.imwrite(new_path, padded)
            i += 1

    # Masks
    for path in masks_paths:
        new_path = path.replace(frame_masks_path, processed_frame_masks_path)
        
        image = Image.open(path)
        width, height = image.size
        new_height = int(height * target_width / width)
        size = target_width, new_height
        image = image.convert('L')
        image.thumbnail(size, Image.ANTIALIAS)
        im = np.array(image, np.float32)

        cv2.imwrite(new_path, 255*im)
        j += 1

    print("{} pictures processed\n{} masks processed".format(i, j))