import pandas as pd
import os
from skimage.transform import resize
from skimage.io import imread
import numpy as np


def preprocess(path, remove=False, n=250):
    '''
    Preprocess the images taken from
    https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset?resource=download

    :param path: (str) The path to directory containing the benign, malignant, and normal directories
    :param remove: (bool) The original dataset has data that contain mask images. Setting this True removes the masks.
    :param n: (int) Image resize value.
    :return: (pandas.DataFrame) Pandas df containing flattened images in x and labels in y. Benign and malignant are
    both labeled with 1. To easily access the image, this can be done via df.iloc[:,:-1]
    '''

    folders = ["benign", "malignant", "normal"]
    # first remove masks if needed
    if remove:
        for folder in folders:
            for img in os.listdir(os.path.join(path, folder)):
                if "mask" in img:
                    os.remove(os.path.join(path, folder, img))
    # Now create pandas df
    flat_data = []
    targets = []
    for folder in folders:
        for img in os.listdir(os.path.join(path, folder)):
            img_arr = imread(os.path.join(path, folder, img), as_gray=True)
            resized = resize(img_arr, (n, n))
            flat_data.append(resized.flatten())
            targets.append(1 if folders.index(folder) <= 1 else 0)
    flat_data = np.array(flat_data)
    targets = np.array(targets)
    df = pd.DataFrame(flat_data)
    df["Target"] = targets
    return df
