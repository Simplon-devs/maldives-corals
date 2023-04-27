"""
This is the class that trains and uses the models needed to detect
individual coral fragments' positions on a structure.
For now it is made up of two components:
- an object detection model thats detects the position of individual
coral fragments on an image
- an image segmentation model that detects the structure on the picture
More models will be added later.
"""

import os
import time
from imageai.Detection.Custom import DetectionModelTrainer
from imageai.Detection.Custom import CustomObjectDetection
from typing import Iterable
from PIL import Image


from maldives_corals import utils
from maldives_corals.interface import CoralModelsInterface


class CoralsModels(CoralModelsInterface):

    def __init__(self):
        pass

    def fit_corals_detection(
        self,
        img: Iterable, 
        annot: Iterable, 
        start_from_pretrained=False
        ):
        """
        Fits the fragments detection model.
        Args:
            img: iterable of images as matrices (e.g. images opened with PIL).
            start_from_pretrained: if True, training will start from the last
            model trained. If False, a brand new model will be created.

            annot: iterable containing iterables of annotations, one for each 
            image. Each annotation contains a fragment's category (acropora, 
            bleached, etc...) as well as its bounding box's coordinates on 
            the image. Ex. ["acropora", x, y, width, height]. x, y, as well
            as width and heigth are relative coordinates (i.e. between 0 and 1)
            Example of annotations for one image:
            [["acropora", 0.25, 0.568, 0.02, 0.09],
             ["dead", 0.46, 0.49, 0.05, 0.04], ...]
        
        Returns None.
        """
        trainer = DetectionModelTrainer()
        trainer.setModelTypeAsYOLOv3()
        train_val_split = 0.8
        classes = ["acropora", "pocillopora", "dead", "bleached", "tag"]

        ###########################################################################
        # Converting the images and annotations to files so the model can use them 
        # in training
        ###########################################################################  

        print("Preparing training data...")
        # Creating the folders structure required by imageAI
        data_folder = f"data_{time.time()}"
        try: os.mkdir(data_folder)
        except FileExistsError: pass
        try: os.mkdir(f'{data_folder}/yolo')
        except FileExistsError: pass
        try: os.mkdir(f'{data_folder}/yolo/train')
        except FileExistsError: pass
        try: os.mkdir(f'{data_folder}/yolo/train/images')
        except FileExistsError: pass
        try: os.mkdir(f'{data_folder}/yolo/train/annotations')
        except FileExistsError: pass
        try: os.mkdir(f'{data_folder}/yolo/validation')
        except FileExistsError: pass
        try: os.mkdir(f'{data_folder}/yolo/validation/images')
        except FileExistsError: pass
        try: os.mkdir(f'{data_folder}/yolo/validation/annotations')
        except FileExistsError: pass

        data_split_index = int(train_val_split*len(img))
        print(data_split_index)

        train_img = img[:data_split_index]
        train_annot = annot[:data_split_index]

        val_img = img[:data_split_index]
        val_annot = annot[:data_split_index]

        current_img_id = 0
        for img_array, annots in zip(train_img, train_annot):
            im = Image.fromarray(img_array)
            im.save(f'{data_folder}/yolo/train/images/{current_img_id}.png')

            for a in annots:
                clean_annotation = ""
                class_index = classes.find(a[0])
                if class_index == -1: 
                    raise ValueError(f"Class '{a[0]} does not exist, please make sure you used the right spelling'")
                else:
                    clean_annotation += str(class_index)
                    clean_annotation += " "
                    clean_annotation += " ".join(a[1:])
                print(clean_annotation)

            current_img_id += 1

        trainer.setDataDirectory(data_directory="data")

        ###########################################################################
        # Training the model 
        ###########################################################################   
        if start_from_pretrained:
            trainer.setTrainConfig(object_names_array=classes, 
                            batch_size=10, 
                            num_experiments=200, 
                            train_from_pretrained_model="models/yolov3_data_last.pt")

        else:
            trainer.setTrainConfig(object_names_array=classes, 
                            batch_size=10, 
                            num_experiments=200, 
                            train_from_pretrained_model="models/yolov3.pt")
        trainer.trainModel()

        ###########################################################################
        # Moving the trained model to the 'models' folder and deleting the files
        # created for the training
        ########################################################################### 

    def detect_corals(
            self,
            img: Iterable
            ) -> list:
        """
        Detects corals on the images stored in parameter img.
        Args:
            img: iterable of images as matrices (e.g. images opened with PIL).

        Returns a list containing annotations for each 
        image. Each annotation contains a fragment's category (acropora, 
        bleached, etc...) as well as its bounding box's coordinates on 
        the image. Ex. ["acropora", x, y, width, height]. x, y, as well
        as width and heigth are relative coordinates (i.e. between 0 and 1)
        Example of annotations for one image:
        [["acropora", 0.25, 0.568, 0.02, 0.09],
        ["dead", 0.46, 0.49, 0.05, 0.04], ...]
        """
        # Préparer les images pour qu'elles soient utilisables par ImageAI
        # Faire la détection
        # Renvoyer une liste des annotations
        pass

    def fit_structure_detection(
            self,
            img: Iterable, 
            masks: Iterable
            ):
        """
        Fits the structure detection model, finding out where the structure
        is on the images. The structures' location are stored as masks.
        Args:
            img: iterable of images as matrices (e.g. images opened with PIL).

            masks: iterable of masks corresponding to each image as arrays
        
        Returns None
        """
        pass

    def detect_structure(
            self,
            img: Iterable
            ) -> list:
        """
        Detects the structure in the images stored in img and returns the
        corresponding masks.
        Args:
            img: iterable of images as matrices (e.g. images opened with PIL).

        Returns a list of masks corresponding to each image as arrays
        """
        pass

