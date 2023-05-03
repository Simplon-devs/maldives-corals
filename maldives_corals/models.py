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
import shutil
import time
import numpy as np
from imageai.Detection.Custom import DetectionModelTrainer
from imageai.Detection.Custom import CustomObjectDetection
from typing import Iterable
from PIL import Image


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
        epochs_count = 100
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

        train_img = img[:data_split_index]
        train_annot = annot[:data_split_index]

        val_img = img[data_split_index:]
        val_annot = annot[data_split_index:]

        current_img_id = 0
        for img_array, annots in zip(train_img, train_annot):
            im = Image.fromarray(img_array)
            im.save(f'{data_folder}/yolo/train/images/{current_img_id}.png')
            with open(f'{data_folder}/yolo/train/annotations/{current_img_id}.txt', 'w') as annot_file:
                annotations_lines = []
                for a in annots:
                    clean_annotation = ""
                    split_annotation = a.split(" ")
                    class_index = classes.index(split_annotation[0])
                    clean_annotation += str(class_index)
                    clean_annotation += " "
                    clean_annotation += " ".join([str(split_annotation[1]), 
                                                  str(split_annotation[2]), 
                                                  str(split_annotation[3]), 
                                                  str(split_annotation[4])])
                    annotations_lines.append(clean_annotation)
                for line in annotations_lines: annot_file.write(line + '\n')
                
            current_img_id += 1

        for img_array, annots in zip(val_img, val_annot):
            im = Image.fromarray(img_array)
            im.save(f'{data_folder}/yolo/validation/images/{current_img_id}.png')
            with open(f'{data_folder}/yolo/validation/annotations/{current_img_id}.txt', 'w') as annot_file:
                annotations_lines = []
                for a in annots:
                    clean_annotation = ""
                    split_annotation = a.split(" ")
                    class_index = classes.index(split_annotation[0])
                    clean_annotation += str(class_index)
                    clean_annotation += " "
                    clean_annotation += " ".join([str(split_annotation[1]), 
                                                  str(split_annotation[2]), 
                                                  str(split_annotation[3]), 
                                                  str(split_annotation[4])])
                    annotations_lines.append(clean_annotation)
                for line in annotations_lines: annot_file.write(line + '\n')
            
            current_img_id += 1


        trainer.setDataDirectory(data_directory=f'{data_folder}/yolo')

        ###########################################################################

        # Training the model (this can be VERY long)
        ###########################################################################   
        if start_from_pretrained:
            trainer.setTrainConfig(object_names_array=classes, 
                            batch_size=10, 
                            num_experiments=epochs_count, 
                            train_from_pretrained_model="models/yolov3_yolo_last.pt")

        else:
            trainer.setTrainConfig(object_names_array=classes, 
                            batch_size=10, 
                            num_experiments=epochs_count, 
                            train_from_pretrained_model="models/yolov3.pt")
        trainer.trainModel()

        ###########################################################################
        # Moving the trained model to the 'models' folder and deleting the files
        # created for the training
        ###########################################################################
        try: os.mkdir('models')
        except FileExistsError: pass
        try: os.mkdir('json')
        except FileExistsError: pass

        if os.path.isfile("models/yolov3_yolo_last.pt"):
            os.remove("models/yolov3_yolo_last.pt")
        if os.path.isfile("json/data_yolov3_detection_config.json"):
            os.remove("json/data_yolov3_detection_config.json")

        shutil.move(f'{data_folder}/yolo/models/yolov3_yolo_last.pt', 'models')
        shutil.move(f'{data_folder}/yolo/json/data_yolov3_detection_config.json', 'json')
        shutil.rmtree(data_folder)


    def detect_corals(
            self,
            img: Iterable,
            return_images=False
            ) -> list:
        """
        Detects corals on the images stored in parameter img.
        Args:
            img: iterable of images as matrices (e.g. images opened with PIL).
            return_images: if True, the function will return images wiht bounding 
            boxes instead of annotations.

        Returns a list containing annotations for each 
        image. Each annotation contains a fragment's category (acropora, 
        bleached, etc...) as well as its percentage probability and 
        its bounding box's coordinates on  the image. 
        Ex. ["acropora", probability, x, y, width, height]. x, y, as well
        as width and heigth are relative coordinates (i.e. between 0 and 1)
        Example of annotations for one image:
        [["acropora", 96.04, 0.25, 0.568, 0.02, 0.09],
        ["dead", 52,09, 0.46, 0.49, 0.05, 0.04], ...]
        """
        ###########################################################################
        # Converting the images arrays into files
        ########################################################################### 
        input_folder = f"input_{time.time()}"
        output_folder = f"output_{time.time()}"
        try: os.mkdir(input_folder)
        except FileExistsError: pass
        try: os.mkdir(output_folder)
        except FileExistsError: pass

        current_img_id = 0
        img_ids = []
        predictions = []

        for img_array in img:
            im = Image.fromarray(img_array)
            im.save(f'{input_folder}/{current_img_id}.png')
            img_ids.append(current_img_id)
            current_img_id += 1

        ###########################################################################
        # Detect coral fragments on the images
        ########################################################################### 

        
        detector = CustomObjectDetection()
        detector.setModelTypeAsYOLOv3()
        detector.setModelPath("models/yolov3_yolo_last.pt")
        detector.setJsonPath("json/data_yolov3_detection_config.json")
        detector.loadModel()


        if return_images:
            
            for file in os.listdir(input_folder):
                results = detector.detectObjectsFromImage(input_image=input_folder + '/' + file, 
                                                    minimum_percentage_probability=70,
                                                    output_image_path=output_folder + '/' + file)
            
            for file in os.listdir(output_folder):
                predictions.append(np.asarray(Image.open(output_folder + '/' + file)))
            

        else:

            for id in img_ids:
                detections = []
                with Image.open(f'{input_folder}/{id}.png') as input_img:
                    width, height = input_img.size

            results = detector.detectObjectsFromImage(input_image=f'{input_folder}/{id}.png', 
                                                        minimum_percentage_probability=70)
            for r in results:
                detections.append(
                    [
                    r["name"],
                    r["percentage_probability"],
                    r["box_points"][0]/width,
                    r["box_points"][1]/height,
                    (r["box_points"][2]-r["box_points"][0])/width,
                    (r["box_points"][3]-r["box_points"][1])/height
                ]
                )
            predictions.append(detections)

        shutil.rmtree(input_folder)
        shutil.rmtree(output_folder)
        return predictions

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

