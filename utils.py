import os
import rasterio as rio
import numpy as np
from tqdm import tqdm

def get_organized_dict_of_patches(folder_path: str):
    organized_dict = {}
    patch_files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]
    key = 1
    for patch_file in tqdm(patch_files):
        _,x,y = patch_file.split("_")
        y = y.split(".")[0]

        organized_dict[key] = rio.open(os.path.join(folder_path,patch_file)).read()
        key +=1
    return organized_dict

