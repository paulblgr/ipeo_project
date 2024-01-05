import os
import rasterio as rio
import numpy as np
from tqdm import tqdm
import torch
from dataset import PatchesDataset
import random
import pandas as pd

# Convert arrays each array of a patch dict into a tensor
'''def convert_to_tensors(input_dict):
    output_dict = {}
    for key, value in input_dict.items():
        if isinstance(value, np.ndarray):
            tensor_value = torch.tensor(value)
            output_dict[key] = tensor_value
    return output_dict'''

'''def reshape_tensors(input_dict):
    output_dict = {}
    for key, value in input_dict.items():
        reshaped_data = value.view(4, 128, 128)
        #output_dict[key] = T.ToPILImage()(reshaped_data)
        output_dict[key] = reshaped_data
    return output_dict'''

'''def get_organized_dict_of_patches(folder_path: str):
    organized_dict = {}
    patch_files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]
    for patch_file in tqdm(patch_files):
        _,x,y = patch_file.split("_")
        y = y.split(".")[0]
        key = (int(x),int(y))
        organized_dict[key] = torch.tensor(rio.open(os.path.join(folder_path,patch_file)).read())

    return organized_dict'''

def load_data(folder_path: str):
    patch_files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]
    images = []
    for patch_file in tqdm(patch_files, desc="Loading data"):
        _,x,y = patch_file.split("_")
        y = y.split(".")[0]
        patch = torch.tensor(rio.open(os.path.join(folder_path,patch_file)).read())
        image = {'x' : int(x), 'y' : int(y), 'patch' : patch, 'augmented' : False}
        images.append(image)
        
    return images

'''def get_patches_df(folder_path: str):
    patch_files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]
    x_pos = []
    y_pos = []
    patches = []
    for patch_file in tqdm(patch_files):
        _,x,y = patch_file.split("_")
        y = y.split(".")[0]
        x_pos.append(int(x))
        y_pos.append(int(y))
        
        patches.append(torch.tensor(rio.open(os.path.join(folder_path,patch_file)).read()))
        
    return pd.DataFrame({"x":x_pos, "y":y_pos, "patches":patches})'''



