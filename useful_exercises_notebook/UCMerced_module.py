from torch.utils.data import Dataset

from PIL import Image

import os
import glob
import torchvision.transforms as T
import torch

# 
mean=torch.tensor([0.504, 0.504, 0.503])
std=torch.tensor([0.019 , 0.018, 0.018])

# normalize image [0-1] (or 0-255) to zero-mean unit standard deviation
normalize = T.Normalize(mean, std)
# we invert normalization for plotting later
std_inv = 1 / (std + 1e-7)
unnormalize = T.Normalize(-mean * std_inv, std_inv)

default_transform =  T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        normalize])

class UCMerced(Dataset):

    # mapping between label class names and indices
    LABEL_CLASSES = {
      'agricultural': 		  0,
      'airplane': 			    1,
      'baseballdiamond': 	  2,
      'beach': 				      3,
      'buildings': 			    4,
      'chaparral': 			    5,
      'denseresidential':   6,
      'forest': 				    7,
      'freeway': 				    8,
      'golfcourse': 			  9,
      'harbor': 				    10,
      'intersection': 		  11,
      'mediumresidential':  12,
      'mobilehomepark': 	  13,
      'overpass': 			    14,
      'parkinglot': 			  15,
      'river': 				      16,
      'runway': 				    17,
      'sparseresidential':  18,
      'storagetanks': 		  19,
      'tenniscourt': 			  20
    }

    # image indices to use for different splits
    SPLITS = {
      'train': list(range(0, 60)),    # use first 60 images of each class for training...
      'val':   list(range(61, 70)),   # ...images 61-70 for model validation...
      'test':  list(range(71, 100))   # ...and the rest for testing
    }

    def __init__(self, dataset_root='UCMerced_LandUse/Images', transforms=default_transform, split='train'):
        self.transforms = transforms

        # prepare data
        self.data = []                                  # list of tuples of (image path, label class)
        for labelclass in self.LABEL_CLASSES:
            # get images with correct index according to dataset split
            for imgIndex in self.SPLITS[split]:
                imgName = os.path.join(dataset_root, labelclass, f'{labelclass}{str(imgIndex).zfill(2)}.tif') 
                # example format: 'baseFolder/agricultural/agricultural07.tif'
                self.data.append((
                    imgName,
                    self.LABEL_CLASSES[labelclass]          # get index for label class
                ))


    def __len__(self):
        return len(self.data)


    def __getitem__(self, x):
        imgName, label = self.data[x]

        img = Image.open(imgName)
        if self.transforms is not None:
            img = self.transforms(img)
        return img, label
