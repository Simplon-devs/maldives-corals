# Import des libraries et modules
import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Cropping2D, Conv2DTranspose, concatenate, Lambda
from keras import regularizers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import glob
import cv2
import numpy as np
from PIL import Image
import os
np.set_printoptions(threshold=np.inf)
import pandas as pd
import h5py
from keras.optimizers import *
from tensorflow import keras
from keras.metrics import MeanIoU

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

model = unet()

# print(model.summary())

iou = MeanIoU(num_classes=2, name='iou')
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[iou])


data_dir = r'data/Frame_segmentation/'
out_path = data_dir + r'unet_padded/'
image_dir = out_path + '*.png'
mask_dir = data_dir + 'masks/*.png'

images = []
masks = []

# Chargement des images
for image_path in glob.glob(image_dir):
    image = cv2.imread(image_path)
    images.append(image)

# Chargement des masques
for mask_path in glob.glob(mask_dir):
    mask = cv2.imread(mask_path)
    
    # inverser les couleurs (N&B)
    # mask = cv2.bitwise_not(mask)
    
    masks.append(mask)

# Taille à garder
size_to_keep = (144, 192, 3)

#Le code extrait les images et les masques de la même taille que size_to_keep, puis normalise les masques en une représentation binaire
X = np.array([image for image in images if image.shape == size_to_keep])
y_255 = np.array([image for image in masks if image.shape == size_to_keep])
y = y_255[:,:,:,:2]//255
res = (1 - (y[:,:,:,1].astype(float))).astype(int)
print(y[:,:,:,1].shape)
y[:,:,:,1] = res

# plt.figure()
# plt.imshow(y[0,:,:,1], interpolation='none')
# plt.show()

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

print(train_dataset)

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
    
print(learning_data.shape)
print(learning_labels.shape)

# Entraîner le modèle avec l'augmentation de données
# history = model.fit(datagen.flow(X_train, y_train, batch_size=16), epochs=2, validation_data=(X_test, y_test))

# Entraîner le modèle
history = model.fit(learning_data, learning_labels, batch_size=64, epochs=2, validation_data=(X_test, y_test))

#On évalue les performances du modèle sur l'ensemble de test
score = model.evaluate(X_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#On sauvegarde le modèle pour une utilisation ultérieure
model.save("model_augment.h5")

#On charge le modèle sauvegardé
# model = load_model("model_good2.h5")

#On charge une nouvelle image
new_image = cv2.imread("data/Frame_segmentation/test/SH001H03201114.png")

#On fait une prédiction sur l'image chargée avec le modèle
prediction = model.predict(np.array([new_image]))[0]
prediction_int = np.rint(prediction)

#On affiche l'image prédite
plt.imshow(prediction_int[:,:,1], cmap='gray')
plt.show()