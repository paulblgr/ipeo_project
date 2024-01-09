import os

import numpy as np
from tqdm import tqdm
import torch
import random
from unet_whole_model import *
import json
from PIL import Image

# Convert arrays each array of a patch dict into a tensor

def load_paths_data(folder_path: str):
    patch_files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]
    images = []
    for patch_file in tqdm(patch_files, desc="Loading data"):
        _,x,y = patch_file.split("_")
        y = y.split(".")[0]
        
        image = {'x' : int(x), 'y' : int(y), 'patch_path' : os.path.join(folder_path,patch_file), 'augmentation' : 'no_augmentation'}
        images.append(image)
    
    images = sorted(images, key = lambda img : (img['x'], img['y']) )

    return images

def save_to_tif(tensor, file_path):
    Image.fromarray((tensor.numpy()*255).astype(np.uint8)).save(file_path, format='TIFF')

def fill_test_record(test_records, modelname, model, train_data_name, other_datasets):
  test_records[modelname] = {}
  test_records[modelname][train_data_name] = {}

  best_epoch = model.get_best_epoch()
  f,a = model.get_best_f1_accuracy()

  test_records[modelname][train_data_name]['best_epoch'] = int(best_epoch)
  test_records[modelname][train_data_name]['f1'] = f
  test_records[modelname][train_data_name]['accuracy'] = a


  best_epoch_model = Model(modelname, lr = 9e-4)
  best_epoch_model.load_model(f"{modelname}_epoch_{best_epoch}")

  for dataset, name in zip(other_datasets):
      test_records[modelname][dataset.name] = {}
      test_records[modelname][dataset.name]['f1'],test_records[modelname][dataset.name]['accuracy']  = best_epoch_model.test_dataset(dataset)
  
  with open('test_records.json', 'w') as json_file:
    json.dump(test_records, json_file)
  
  return test_records