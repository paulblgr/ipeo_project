import torch
from tqdm import tqdm
import torchvision.transforms as T
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import rasterio as rio
import os
import random

def transform_train_with_labels(image, label):
 
    # Geometric transformation applied to both image and label
    geometric_transforms =[T.RandomVerticalFlip(p=1),
                           T.RandomHorizontalFlip(p=1),
                           T.RandomRotation(degrees = (90,90)),
                           T.RandomRotation(degrees = (180,180)),
                           T.RandomRotation(degrees = (270,270))]
    
    geometric_transforms_names = ["vertical_flip", "horizontal_flip", "rotation_90", "rotation_180°", "rotation_270"]
    
    # Randomly select 3 transformations
    selected_transforms = random.sample(list(zip(geometric_transforms, geometric_transforms_names)), 3)

    # Extract the selected transformations and their names
    geometric_transforms, geometric_transforms_names = zip(*selected_transforms)
    geometric_transforms = list(geometric_transforms)
    geometric_transforms_names = list(geometric_transforms_names)

    new_images = [transform(image) for transform in geometric_transforms]
    new_labels = [transform(label) for transform in geometric_transforms]
    
    # Other transformation applied only to the image
    if random.random() < 0.5:
      other_transforms = [T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))]
      other_transforms_name = ["gaussian_blur"]
    else:
      other_transforms = []
      other_transforms_name = []
    
    new_images = new_images +  [transform(image) for transform in other_transforms]
    new_labels = new_labels + [label for _ in other_transforms]           
    return new_images, new_labels, geometric_transforms_names , other_transforms_name

def save_to_tif(tensor, file_path):
    Image.fromarray((tensor.numpy()*255).astype(np.uint8)).save(file_path, format='TIFF')

class PatchesDataset(torch.utils.data.Dataset):

    def __init__(self, images_paths, groundtruths_paths, dataset_name):
        assert len(images_paths) == len(groundtruths_paths)
        
        for img, gt in zip(images_paths, groundtruths_paths):
            assert (img['x'] == gt['x'] and img['y'] == gt['y'])
        
        self.name = dataset_name
        self.input = images_paths.copy()
        
        self.target = groundtruths_paths.copy()
        
        self.loaded_imgs = None
        self.loaded_gts = None

        self.means = None
        self.stds = None
        

    def get_rio_image(self, img_dict):
        return rio.open(img_dict['patch_path']).read()
    
    def get_tensor_image(self, img_dict):
        if 'patch_path' in img_dict:
          img_np = self.get_rio_image(img_dict)
          return torch.tensor(img_np)
        else: 
          return img_dict['patch'] 

    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        if self.loaded_imgs is None:
            raise Exception("Images not loaded")
        if self.loaded_gts is None:
            raise Exception("Groundtruths not loaded")
        
        return self.loaded_imgs[index], self.loaded_gts[index]
    
    def get_at_pos(self, x, y):
        res = []
        for i in range(len(self.input)):
            if self.input[i]['x'] == x and self.input[i]['y'] == y:
                img = self.get_tensor_image(self.input[i])
                gt = self.get_tensor_image(self.target[i])

                res_ = {
                    'img_dict' : self.input[i],
                    'img' : img,
                    'gt' : gt,
                    'gt_dict' : self.target[i]
                }

                res.append(res_)
        return res
    
    def plot_at_pos(self, x, y):
        imgs_labs_dicts = self.get_at_pos(x, y)
        images = [img_lab_dict['img'].numpy() for img_lab_dict in imgs_labs_dicts]
        image_tranformations = [img_lab_dict['img_dict']['augmentation'] for img_lab_dict in imgs_labs_dicts]

        labels = [img_lab_dict['gt'].numpy() for img_lab_dict in imgs_labs_dicts]
        labels_tranformations = [img_lab_dict['gt_dict']['augmentation'] for img_lab_dict in imgs_labs_dicts]

        rgb_images = [(img[:3] * 255).astype(np.uint8).transpose(1,2,0) for img in images]
        rgb_labels = [lab.astype(np.uint8).squeeze(0)  for lab in labels]


        labs = [Image.fromarray(lab) for lab in rgb_labels]

        imgs = [Image.fromarray(img) for img in rgb_images]

        _, axs = plt.subplots(2, len(imgs) , figsize=(len(imgs)*9, 15))

        if len(imgs) == 1:
            axs[0].imshow(imgs[0])
            axs[0].set_title(f"Image with {image_tranformations[0]}")
            axs[0].axis("off")
            axs[1].imshow(labs[0], cmap='gray')
            axs[1].set_title(f"Groundtruth with {labels_tranformations[0]}")
            axs[1].axis("off")
            plt.show()
            return

        for i in range(len(imgs)):
            axs[0,i].imshow(imgs[i])
            axs[0,i].set_title(f"Image with {image_tranformations[i]}")
            axs[0,i].axis("off")
            axs[1,i].imshow(labs[i], cmap='gray')
            axs[1,i].set_title(f"Groundtruth with {labels_tranformations[i]}")
            axs[1,i].axis("off")

        plt.show()

    def load_images_and_gts(self):
      if self.loaded_imgs is None: 
        self.loaded_imgs = [self.get_tensor_image(image) for image in tqdm(self.input, desc = "Loading images")]
      if self.loaded_gts is None:
        self.loaded_gts = [self.get_tensor_image(gt) for gt in tqdm(self.target,  desc = "Loading groundtruths")] 
      
      self.means, self.stds = self.compute_means_and_stds()

    def deload(self):
        self.loaded_imgs = None
        self.loaded_gts = None

    def get_images(self):
        if self.loaded_imgs is None:
            raise Exception("Images not loaded")
        return self.loaded_imgs
    
    def get_groundtruths(self):
        if self.loaded_gts is None:
            raise Exception("Groundtruths not loaded")
        return self.loaded_gts
    
    def compute_means_and_stds(self):
        if self.loaded_imgs is None:
            self.load_images_and_gts()
        stacked_tensor  = torch.stack(self.get_images(), axis = 0)
        means = torch.mean(stacked_tensor, axis =[0,2,3])
        stds = torch.std(stacked_tensor, axis = [0,2,3])
        
        return means, stds

    def augment(self, augmentation = transform_train_with_labels):
        all_new_imgs = []
        all_new_gts = []
        all_new_imgs_dict = []
        all_new_gts_dict = []
        
        self.load_images_and_gts()

        for img, gt, img_dict, gt_dict in tqdm(zip(self.loaded_imgs, self.loaded_gts, self.input, self.target), total = len(self.input), desc = "Augmenting dataset"): 

            new_imgs, new_gts, geometric_tranform, other_transform = augmentation(img, gt)
            
            new_imgs_dict = [{'x' : img_dict['x'], 
                         'y' : img_dict['y'],
                         'patch' : new_img ,
                         'augmentation' : augmentation}
                         for new_img, augmentation in zip(new_imgs, geometric_tranform + other_transform)]
            
            new_gts_dict = [{'x' : gt_dict['x'], 
                         'y' : gt_dict['y'],
                         'patch' : new_gt,
                         'augmentation' : augmentation} 
                         for new_gt, augmentation in zip(new_gts, geometric_tranform + ["no_augmentation"]*len(other_transform))]
            
            all_new_imgs.extend(new_imgs)
            all_new_gts.extend(new_gts)
            all_new_imgs_dict.extend(new_imgs_dict)
            all_new_gts_dict.extend(new_gts_dict)
        
        self.input.extend(all_new_imgs_dict)
        self.target.extend(all_new_gts_dict)
        self.loaded_imgs.extend(all_new_imgs)
        self.loaded_gts.extend(all_new_gts)