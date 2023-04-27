"""
This is the interface for the class that trains and uses the models needed to detect
individual coral fragments' positions on a structure.
For now it is made up of two components:
- an object detection model thats detects the position of individual
coral fragments on an image
- an image segmentation model that detects the structure on the picture
More models will be added later.
NOTE: this is an abstract class, see model.py for the
implementation.
"""
from abc import ABC, abstractclassmethod
from typing import Iterable

class CoralModelsInterface(ABC):

    @abstractclassmethod
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
        pass

    @abstractclassmethod
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
        pass

    @abstractclassmethod
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

    @abstractclassmethod
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

