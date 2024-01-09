import torch
from tqdm import tqdm
import torchvision.transforms as T
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import rasterio as rio
import os
from utils import *

def transform_train_with_labels(image, label):
 
    # Geometric transformation applied to both image and label
    geometric_transforms =[T.RandomVerticalFlip(p=1),
                           T.RandomHorizontalFlip(p=1), #Je dirais ce serait bien d'ajouter rotation à 90° , 180° et 270°
                           T.RandomRotation(degrees = (90,90)),
                           T.RandomRotation(degrees = (180,180)),
                           T.RandomRotation(degrees = (270,270))]
    
    geometric_transforms_name = ["vertical_flip", "horizontal_flip", "rotation_90", "rotation_180°", "rotation_270"]
    
    new_images = [transform(image) for transform in geometric_transforms]
    new_labels = [transform(label) for transform in geometric_transforms]
    
    # Other transformation applied only to the image
    other_transforms = [T.GaussianBlur(kernel_size=3 , sigma = (0.1,0.5))]
    other_transforms_name = ["gaussian_blur"]
    new_images = new_images +  [transform(image) for transform in other_transforms]
    new_labels = new_labels + [label for _ in other_transforms]                   
    return new_images, new_labels, geometric_transforms_name , other_transforms_name

class PatchesDataset(torch.utils.data.Dataset):

    def __init__(self, images_paths, groundtruths_paths, dataset_name):
        assert len(images_paths) == len(groundtruths_paths)
        
        for img, gt in zip(images_paths, groundtruths_paths):
            assert (img['x'] == gt['x'] and img['y'] == gt['y'])
        
        self.name = dataset_name
        self.input = images_paths.copy()
        
        self.target = groundtruths_paths.copy()
        
        self.means, self.stds = self.compute_means_and_stds()
        self.normalize = T.Normalize(mean=self.means, std=self.stds)

    def get_rio_image(self, img):
        img_np = rio.open(img['patch_path']).read()
        if img['augmentation'] != 'no_augmentation':
           img_np = img_np/255
        return img_np
    
    def get_tensor_image(self, img):
        return torch.tensor(self.get_rio_image(img))

    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        return self.normalize(self.get_tensor_image(self.input[index])), self.get_tensor_image(self.target[index])
    
    def get_at_pos(self, x, y):
        res = []
        for i in range(len(self.input)):
            if self.input[i]['x'] == x and self.input[i]['y'] == y:
                res.append((self.input[i], self.target[i]))
        return res
    
    def plot_at_pos(self, x, y):
        imgs_labs = self.get_at_pos(x, y)
        images = [self.get_rio_image(img_lab[0]) for img_lab in imgs_labs]
        image_tranformations = [img_lab[0]['augmentation'] for img_lab in imgs_labs]

        labels = [self.get_rio_image(img_lab[1]) for img_lab in imgs_labs]
        labels_tranformations = [img_lab[1]['augmentation'] for img_lab in imgs_labs]

        rgb_images = [(img[:3] * 255).astype(np.uint8).transpose(1,2,0) for img in images]
        rgb_labels = [lab.astype(np.uint8).squeeze(0)  for lab in labels]


        labs = [Image.fromarray(lab) for lab in rgb_labels]

        imgs = [Image.fromarray(img) for img in rgb_images]

        _, axs = plt.subplots(2, len(imgs) , figsize=(len(imgs)*9, 15))

        for i in range(len(imgs)):
            axs[0,i].imshow(imgs[i])
            axs[0,i].set_title(f"Image with {image_tranformations[i]}")
            axs[0,i].axis("off")
            axs[1,i].imshow(labs[i], cmap='gray')
            axs[1,i].set_title(f"Groundtruth with {labels_tranformations[i]}")
            axs[1,i].axis("off")

        plt.show()
    

    def get_images(self):
        return [self.get_tensor_image(image) for image in tqdm(self.input)]
    
    def get_groundtruths(self):
        return [self.get_tensor_image(gt) for gt in tqdm(self.target)]
    
    def compute_means_and_stds(self):
        stacked_tensor  = torch.stack(self.get_images(), axis = 0)
        means = torch.mean(stacked_tensor, axis =[0,2,3])
        stds = torch.std(stacked_tensor, axis = [0,2,3])
        return means, stds

    def augment(self, augmentation = transform_train_with_labels):
        all_new_imgs = []
        all_new_gts = []

        augmented_data_path = f"glaciers_mapping_downsampled/{self.name}_augmented_data"
        augmented_data_path_gt = f"glaciers_mapping_downsampled/{self.name}_augmented_data_gt"
        os.makedirs(augmented_data_path, exist_ok = True)
        os.makedirs(augmented_data_path_gt, exist_ok = True)

        for img,gt in tqdm(zip(self.input, self.target), total = len(self.input), desc = "Augmenting dataset"): 

            new_imgs, new_gts, geometric_tranform, other_transform = augmentation(self.get_tensor_image(img),
                                                                                  self.get_tensor_image(gt))
            new_imgs_paths = [f"{augmented_data_path}/{augmentation}_{img['patch_path'].rsplit('/', 1)[-1]}" for augmentation in geometric_tranform + other_transform]
            new_gts_paths = [f"{augmented_data_path_gt}/{augmentation}_{gt['patch_path'].rsplit('/', 1)[-1]}" for augmentation in geometric_tranform + ["no_augmentation"]*len(other_transform)]

            for new_img, new_img_path in zip(new_imgs, new_imgs_paths):
                save_to_tif(new_img.permute(1,2,0), new_img_path)
                
            for new_gt, new_gt_path in zip(new_gts, new_gts_paths):
                save_to_tif(new_gt.squeeze(0), new_gt_path)

            new_imgs = [{'x' : img['x'], 
                         'y' : img['y'],
                         'patch_path' : new_img_path,
                         'augmentation' : augmentation}
                         for new_img_path, augmentation in zip(new_imgs_paths, geometric_tranform + other_transform)]
            
            new_gts = [{'x' : gt['x'], 
                         'y' : gt['y'],
                         'patch_path' : new_gt_path,
                         'augmentation' : augmentation} 
                         for new_gt_path, augmentation in zip(new_gts_paths, geometric_tranform + ["no_augmentation"]*len(other_transform))]
            
            all_new_imgs.extend(new_imgs)
            all_new_gts.extend(new_gts)
        
        self.input.extend(all_new_imgs)
        self.target.extend(all_new_gts)

        self.means, self.stds = self.compute_means_and_stds()
        self.normalize = T.Normalize(mean=self.means, std=self.stds)