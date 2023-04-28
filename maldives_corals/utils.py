"""
This file contains some helper functions used by the module's models,
many of them related to images and annotations processing.
"""

import os
import copy
import json
import cv2
import glob
import shutil
import numpy as np

from random import sample
from pylabel import importer
from PIL import Image


def coco_to_yolo(
        annotations_path: str, 
        img_path: str
        ):
    """
    Helper function that converts COCO image annotations to YOLO format
    so it can be used by the YOLO object detection model.
    """
    
    #Specify path to the coco.json file
    path_to_annotations = annotations_path
    #Specify the path to the images (if they are in a different folder than the annotations)
    path_to_images = img_path
    dataset = importer.ImportCoco(path_to_annotations, path_to_images=path_to_images, name="COCO_annot")
    dataset.path_to_annotations = path_to_annotations + "/yolo"
    dataset.export.ExportToYoloV5()

def equalize_img(
        img_folder: str, 
        output_folder: str
        ):
    """
    Equalizes images histograms so all images in a dataset have the same contrast.
    """

    processed_img = 0
    os.makedirs(output_folder)
    files = os.listdir(img_folder)
    for img in files:
        rgb_img = cv2.imread(img_folder + '/' + img)
        # convert from RGB color-space to YCrCb
        ycrcb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2YCrCb)
        # equalize the histogram of the Y channel
        ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])
        # convert back to RGB color-space from YCrCb
        equalized_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)
        cv2.imwrite(output_folder + '/' + img, equalized_img)
        processed_img += 1
        if processed_img % 100 == 0: 
            print(f"{processed_img} images processed")

def extract_filename(
        img: str
        ):
    """
    NOTE: depending on the annotations format you are using, this function
    might not be relevant to your use case. Check your annotations format 
    before using it.

    Extracts the name of an image based on its URL in annotations.
    """
    var1 = img.split("?", 1)[0]
    var1 = var1.split("-")[-1]
    return var1  


def add_new_annots(
        pictures_path: str, 
        annotation_path: str, 
        sloth_json_path: str, 
        new_pictures_path: str
        ):
    """
    NOTE: depending on the annotations format you are using, this function
    might not be relevant to your use case. Check your annotations format 
    before using it.

    Splits annotation into one file per picture, remove mentions to images
    that do not have any annotations and crops bounding boxes bigger than
    pictures.
    """

    data = json.load(open(sloth_json_path))

    i_it = 0

    a_count = 0
    i_count = 0

    for i_it in range(len(data)):
        
        keys = data[i_it].keys()
        
        for key in keys:
            print(key)

        pic = data[i_it]


        filename = pic["Labeled Data"]

        tail = extract_filename(filename)
        
        basename, extension = os.path.splitext(tail)

        
        im = cv2.imread(pictures_path + tail)
        print(pictures_path + tail)
        
        height, width = im.shape[:2]

        if(len(pic["Label"]["objects"])):
            i_count = i_count + 1
            
            new_dict = {"filename": tail, "width": width, "height": height, "annotations": []}

            for a_it in range(len(pic["Label"]["objects"])):
                a_count = a_count + 1
                try:
                    old_annot = pic["Label"]["objects"][a_it]["bbox"]
                except KeyError:
                    continue
                x, y, w, h = float(old_annot['left']), float(old_annot['top']), float(old_annot['width']), float(old_annot['height'])

                if(x < 0):
                    x = 0
                if(y < 0):
                    y = 0
                if(x + w > width):
                    w = width - x
                if(y + h > height):
                    h = height - y

                new_annot = {"class": pic["Label"]["objects"][a_it]["value"], 'x': x, 'y': y, 'height': h, 'width': w}
                new_dict['annotations'].append(new_annot)

            jsonString = json.dumps(new_dict, indent = 4)
            with open(annotation_path + basename + '.json', "w") as f:
                f.write(jsonString)

            if(pictures_path + tail != new_pictures_path + tail):
                shutil.copyfile(pictures_path + tail, new_pictures_path + tail)

    print("Added new annotations : {} annotations from {} images".format(a_count, i_count))

    

def create_coco(
        pictures_path: str, 
        annotations_path: str, 
        outputDir: str, 
        picture_subs: str
        ):
    """
    NOTE: depending on the annotations format you are using, this function
    might not be relevant to your use case. Check your annotations format 
    before using it.

    Converts individual annotations such as those generated by the function
    add_new_annots into two COCO files (train and val).
    Pictures in the original folder are NOT edited.
    Applies autocontrast & CLAHE to pictures and saves them in two subfolders (train
    and val).
    """


    trainDict = dict()
    trainDict["categories"] = [{"supercategory": "live_coral", "id": 0, "name": "acropora"},
                              {"supercategory": "live_coral", "id": 1, "name": "pocillopora"},
                              {"supercategory": "dead_coral", "id": 2, "name": "dead"},
                              {"supercategory": "live_coral", "id": 3, "name": "bleached"},
                              {"supercategory": "other", "id": 4, "name": "tag"}
                              ]
    valDict = copy.deepcopy(trainDict)
    
    a_count = 0
    i_count = 0
    
    train_images = list()
    val_images = list()
    
    train_annotations = list()
    val_annotations = list()

    shutil.rmtree(picture_subs[True])
    shutil.rmtree(picture_subs[False])
    os.mkdir(picture_subs[True])
    os.mkdir(picture_subs[False])

    files = glob.glob(annotations_path + '*.json')

    train = sample(range(0,len(files)),int(len(files)*0.8))
        
    annots={True:train_annotations,False:val_annotations}
    imgs={True:train_images,False:val_images}

    for filepath in files:
        data = json.load(open(filepath))
        image = {}
        image['file_name'] = data['filename']
        

        im = Image.open(pictures_path + image['file_name'])
        width, height = im.size
        scale = 1
        if width > 1440:
            scale = 1440 / width

        image['height'] = int(scale * data['height'])
        image['width'] = int(scale * data['width'])

        image['id'] = i_count
        imgs[i_count in train].append(image)

        r, g, b = im.split()
        # TURNING OFF AUTOCONTRAST !
        # r, g, b = ImageOps.autocontrast(r, cutoff = 1), ImageOps.autocontrast(g, cutoff = 1), ImageOps.autocontrast(b, cutoff = 1)
        im = Image.merge("RGB",[r, g, b])

        img_file = np.uint8(im)
        image_lab = cv2.cvtColor(img_file, cv2.COLOR_RGB2LAB)

        l_channel, a_channel, b_channel = cv2.split(image_lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l_channel)
        merged_channels = cv2.merge((cl, a_channel, b_channel))
        final_image = cv2.cvtColor(merged_channels, cv2.COLOR_LAB2BGR)

        size = (int(scale * width), int(scale * height))
        final_image = cv2.resize(final_image, size, interpolation=cv2.INTER_AREA)
        
        cv2.imwrite(picture_subs[i_count in train]+image['file_name'],final_image)

        a_it = 0
        for annot in data['annotations']:
            for value in trainDict["categories"]:
                annotation = {}
                if annot['class'] == value["name"]:
                    a_count = a_count + 1
                    annotation["iscrowd"] = 0
                    annotation["image_id"] = i_count
                    annotation["bbox"] = [scale * annot['x'], scale * annot['y'], scale * annot['width'], scale * annot['height']]
                    annotation["area"] = scale ** 2 * annot['width'] * annot['height']
                    annotation["category_id"] = value["id"]
                    annotation["ignore"] = 0
                    annotation["id"] = a_it
                    # annotation["segmentation"] = [[x1, y1, x1, (y1 + y2), (x1 + x2), (y1 + y2), (x1 + x2), y1]]                           

                    annots[i_count in train].append(annotation)
                    a_it += 1

        i_count += 1

    print("COCO annotations : Created {} annotations from {} images".format(a_count,i_count))
    
    trainDict["images"] = imgs[True]
    trainDict["annotations"] = annots[True]
    trainDict["type"] = "instances"
    jsonString = json.dumps(trainDict, indent = 4)
    with open(outputDir + "COCO_train.json", "w") as f:
        f.write(jsonString)
        
    valDict["images"] = imgs[False]
    valDict["annotations"] = annots[False]
    valDict["type"] = "instances"
    jsonString = json.dumps(valDict, indent = 4)
    with open(outputDir + "COCO_val.json", "w") as f:
        f.write(jsonString)