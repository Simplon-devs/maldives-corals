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

import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Cropping2D, Conv2DTranspose, concatenate, Lambda
from keras import regularizers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import glob
import cv2
np.set_printoptions(threshold=np.inf)
import pandas as pd
import h5py
from keras.optimizers import *
from tensorflow import keras
from keras.metrics import MeanIoU

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
                    class_index = classes.index(a[0])
                    clean_annotation += str(class_index)
                    clean_annotation += " "
                    clean_annotation += " ".join([str(a[1]), str(a[2]), str(a[3]), str(a[4])])
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
                    class_index = classes.index(a[0])
                    clean_annotation += str(class_index)
                    clean_annotation += " "
                    clean_annotation += " ".join([str(a[1]), str(a[2]), str(a[3]), str(a[4])])
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
        try: os.mkdir('models')
        except FileExistsError: pass
        try: os.mkdir('json')
        except FileExistsError: pass

        shutil.move(f'{data_folder}/models/yolov3_data_last.pt', 'models')
        shutil.move(f'{data_folder}/json/data_yolov3_detection_config.json', 'json')
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
        detector.setModelPath("models/yolov3_data_last.pt")
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
        model = self.unet()
        learning_data, X_test, learning_labels, y_test = self.structure_data_augmentation()
        
        X, y = self.structure_feature(X, y)


        iou = MeanIoU(num_classes=2, name='iou')
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[iou])

        # Entraîner le modèle
        history = model.fit(learning_data, learning_labels, batch_size=64, epochs=2, validation_data=(X_test, y_test))

        #On évalue les performances du modèle sur l'ensemble de test
        score = model.evaluate(X_test, y_test, verbose=1)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        model.save("models/structure_model.h5")
        
                    
        
    def structure_data_augmentation(
        self,
        X, y : list
    ):
        X_train, X_test, y_train, y_test = self.structure_train_split_test(X, y)
        

        #ImageDataGenerator crée un générateur d'images augmentées en appliquant des transformations aléatoires telles que la rotation,
        #le décalage horizontal et vertical, le zoom et la retournement horizontal des images.
        datagen = ImageDataGenerator(
            rotation_range=20, # angle de rotation maximal en degrés
            width_shift_range=0.1, # fraction de la largeur totale pour le décalage horizontal
            height_shift_range=0.1, # fraction de la hauteur totale pour le décalage vertical
            zoom_range=0.1, # plage pour le zoom aléatoire
            horizontal_flip=True # retourner horizontalement l'image de manière aléatoire
        )

        # Ajuster l'augmentation de données aux images d'entraînement
        datagen.fit(X_train)

        # Augment the data
        #générateur d'images pour créer de nouvelles images en appliquant des transformations telles que la rotation, 
        #le décalage et le zoom, puis ajoute ces nouvelles images à l'ensemble d'apprentissage.
        #Le processus est répété jusqu'à ce que 20 lots d'images soient générés.
        batches = 0
        learning_data = X_train
        learning_labels = y_train
        for x_batch, y_batch in datagen.flow(X_train, y_train, batch_size=32):
            learning_data = np.concatenate((learning_data, x_batch), axis=0)
            learning_labels = np.concatenate((learning_labels, y_batch), axis=0)
            batches += 1
            if batches >= 20:
                # we need to break the loop by hand because
                # the generator loops indefinitely
                break
            
        return learning_data, X_test, learning_labels, y_test
        
    def structure_train_split_test(
        self,
        X: list,
        y: list
    ):
        X, y = self.structure_feature(X, y)
        # Diviser les données en ensembles d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        return X_train, X_test, y_train, y_test
        
        
    def structure_feature(
        self,
        images: list,
        masks: list
    ):
        images, masks = self.structure_data(images, masks)
        # Taille à garder
        size_to_keep = (144, 192, 3)

        #Le code extrait les images et les masques de la même taille que size_to_keep, puis normalise les masques en une représentation binaire
        X = np.array([image for image in images if image.shape == size_to_keep])
        y_255 = np.array([image for image in masks if image.shape == size_to_keep])
        y = y_255[:,:,:,:2]//255
        res = (1 - (y[:,:,:,1].astype(float))).astype(int)
        y[:,:,:,1] = res
        
        return X, y
    
    def structure_data(
            self,
            img : Iterable, 
            masks: Iterable
    ):
        images = []
        masks = []
        
        # Chargement des images
        for image_path in glob.glob(img):
            image = cv2.imread(image_path)
            images.append(image)

        # Chargement des masques
        for mask_path in glob.glob(masks):
            mask = cv2.imread(mask_path)            
            masks.append(mask)
        
        return images, masks
    
      
    def detect_structure(
            self,
            model_path,
            img: Iterable
            ) -> list:
        """
        Detects the structure in the images stored in img and returns the
        corresponding masks.
        Args:
            img: iterable of images as matrices (e.g. images opened with PIL).

        Returns a list of masks corresponding to each image as arrays
        """
        model = model.load(model_path)
        #On charge une nouvelle image
        new_image = cv2.imread(img)

        #On fait une prédiction sur l'image chargée avec le modèle
        prediction = model.predict(np.array([new_image]))[0]
        prediction_int = np.rint(prediction)

        #On affiche l'image prédite
        plt.imshow(prediction_int[:,:,1], cmap='gray')
        plt.show()
        
    # Création de l'architecture de l'UNET
    def unet(input_size=(144, 192, 3)):
        # Entrée
        inputs = keras.layers.Input(input_size)

        # Première couche de convolution
        conv1 = keras.layers.Conv2D(64, 3, activation='sigmoid', padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = keras.layers.Conv2D(64, 3, activation='sigmoid', padding='same', kernel_initializer='he_normal')(conv1)
        pool1 = keras.layers.MaxPooling2D(2)(conv1)
        pool1 = keras.layers.Dropout(0.3)(pool1)
        
        # Deuxième couche de convolution
        conv2 = keras.layers.Conv2D(128, 3, activation='sigmoid', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = keras.layers.Conv2D(128, 3, activation='sigmoid', padding='same', kernel_initializer='he_normal')(conv2)
        pool2 = keras.layers.MaxPooling2D(2)(conv2)
        pool2 = keras.layers.Dropout(0.3)(pool2)

        # Troisième couche de convolution
        conv3 = keras.layers.Conv2D(256, 3, activation='sigmoid', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = keras.layers.Conv2D(256, 3, activation='sigmoid', padding='same', kernel_initializer='he_normal')(conv3)
        pool3 = keras.layers.MaxPooling2D(2)(conv3)
        pool3 = keras.layers.Dropout(0.3)(pool3)

        # Quatrième couche de convolution
        conv4 = keras.layers.Conv2D(512, 3, activation='sigmoid', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = keras.layers.Conv2D(512, 3, activation='sigmoid', padding='same', kernel_initializer='he_normal')(conv4)
        pool4 = keras.layers.MaxPooling2D(2)(conv4)
        pool4 = keras.layers.Dropout(0.3)(pool4)

        # Bottleneck: 2 blocks of convolutional layers
        conv5 = keras.layers.Conv2D(1024, 3, activation='sigmoid', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = keras.layers.Conv2D(1024, 3, activation='sigmoid', padding='same', kernel_initializer='he_normal')(conv5)

        # Decoder: 4 blocks of transposed convolutional layers with concatenation and dropout in between
        up6 = keras.layers.Conv2DTranspose(512, 3, 2, padding='same')(conv5)
        up6 = keras.layers.concatenate([up6, conv4])
        up6 = keras.layers.Dropout(0.3)(up6)

        conv6 = keras.layers.Conv2D(512, 3, activation='sigmoid', padding='same', kernel_initializer='he_normal')(up6)
        conv6 = keras.layers.Conv2D(512, 3, activation='sigmoid', padding='same', kernel_initializer='he_normal')(conv6)

        up7 = keras.layers.Conv2DTranspose(256, 3, 2, padding='same')(conv6)
        up7 = keras.layers.concatenate([up7, conv3])
        up7 = keras.layers.Dropout(0.3)(up7)
        conv7 = keras.layers.Conv2D(256, 3, activation='sigmoid', padding='same', kernel_initializer='he_normal')(up7)
        conv7 = keras.layers.Conv2D(256, 3, activation='sigmoid', padding='same', kernel_initializer='he_normal')(conv7)

        up8 = keras.layers.Conv2DTranspose(128, 3, 2, padding='same')(conv7)
        up8 = keras.layers.concatenate([up8, conv2])
        up8 = keras.layers.Dropout(0.3)(up8)
        conv8 = keras.layers.Conv2D(128, 3, activation='sigmoid', padding='same', kernel_initializer='he_normal')(up8)
        conv8 = keras.layers.Conv2D(128, 3, activation='sigmoid', padding='same', kernel_initializer='he_normal')(conv8)
        
        up9 = keras.layers.Conv2DTranspose(64, 3, 2, padding='same')(conv8)
        up9 = keras.layers.concatenate([up9, conv1])
        up9 = keras.layers.Dropout(0.3)(up9)
        conv9 = keras.layers.Conv2D(64, 3, activation='sigmoid', padding='same', kernel_initializer='he_normal')(up9)
        conv9 = keras.layers.Conv2D(64, 3, activation='sigmoid', padding='same', kernel_initializer='he_normal')(conv9)
        
        # Output
        outputs = keras.layers.Conv2D(2, 1, padding="same", activation = "sigmoid")(conv9)

        model = Model(inputs, outputs)

        return model