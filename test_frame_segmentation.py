from Frame_segmentation.path_creation import define_paths, load_json, create_folder, rename_files
from Frame_segmentation.image_preprocessing import download_masks, superpose_masks, download_images
import os
import pytest


def test_define_paths():

    data_dir, json_path, pictures_path, frames_bar_path, frame_masks_path = define_paths()

    assert data_dir == os.getcwd() + "/Data"
    assert json_path == os.path.join(data_dir,"Raw","Raw_json","raw.json")
    assert pictures_path == os.path.join(data_dir,"Raw", "Raw_pictures")
    assert frames_bar_path == os.path.join(data_dir, "Raw","Frame_bar_pictures")
    assert frame_masks_path == os.path.join(data_dir, "Raw","Frame_masks_pictures")


def test_create_folder():

    data_dir, json_path, pictures_path, frames_bar_path, frame_masks_path = define_paths()
    paths = [data_dir, json_path, pictures_path, frames_bar_path, frame_masks_path]

    for path in paths : 
        create_folder(path)
        assert os.path.exists(path)


def test_json_path():

    data_dir, json_path, pictures_path, frames_bar_path, frame_masks_path = define_paths()
    items = load_json(json_path)

    assert isinstance(items, list)


def test_download_mask():

    data_dir, json_path, pictures_path, frames_bar_path, frame_masks_path = define_paths()
    items = load_json(json_path)
    items_3 = items[0:3]

    download_masks(items_3, frame_masks_path)

    assert frame_masks_path is not None 


def test_superpose_masks():

    data_dir, json_path, pictures_path, frames_bar_path, frame_masks_path = define_paths()
    superpose_masks(frame_masks_path, frames_bar_path)

    assert frame_masks_path is not None 


def test_download_images():
    data_dir, json_path, pictures_path, frames_bar_path, frame_masks_path = define_paths()
    items = load_json(json_path)
    items_3 = items[0:3]
    download_images(items_3, pictures_path)

if __name__ == "__main__":

    test_define_paths()
    test_create_folder()
    test_json_path()
    test_download_mask()
    test_superpose_masks()
    test_download_images()


   