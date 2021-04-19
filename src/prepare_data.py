"""
Module responsible for preparing data to be fed through the neural net/VAE
"""

# give us pattern matching based file search
import glob as glob
# os management for file management
import os
# importing tdqm for pretty loading bar
from tqdm import tqdm


def prepare_dataset(root_path):
    """
    Find the data at the given path and split it into train/validate sets
    :param root_path: File path for data
    :return: train_data, validation_data
    """
    # get image dir folder
    image_dirs = os.listdir(root_path)
    # sort them. Likely isn't necessary but makes all_image_paths
    # in order later
    image_dirs.sort()
    print(len(image_dirs))
    print(image_dirs[:5])
    all_image_paths = []
    # go through all of the image_dirs
    for i in tqdm(range(len(image_dirs))):
        # get all of the image paths
        image_paths = glob.glob(f"{root_path}/{image_dirs[i]}/*")
        # sort them
        image_paths.sort()
        for image_path in image_paths:
            # add all of the in order image paths to all_image_paths
            all_image_paths.append(image_path)

    print(f"Total number of face images: {len(all_image_paths)}")
    # reserve the last 2000 images for validation data, everythign else
    # is training
    train_data = all_image_paths[:-2000]
    valid_data = all_image_paths[-2000:]
    print(f"Total number of training image: {len(train_data)}")
    print(f"Total number of validation image: {len(valid_data)}")
    return train_data, valid_data
