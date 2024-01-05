import torch
from tqdm import tqdm
import torchvision.transforms as T
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def transform_train_with_labels(image, label):
    
    #rand = random.randint(1, 10)

    # Geometric transformations applied to both image and label
    '''geometric_transforms = T.Compose([
        T.RandomVerticalFlip(p=1),
        T.RandomHorizontalFlip(p=1),
        T.RandomResizedCrop((100, 100), scale=(0.08, 1.0), ratio=(0.9, 1.1)),
        T.Resize((128, 128)),
    ])'''

    geometric_transforms =[T.RandomVerticalFlip(p=1),
                           T.RandomHorizontalFlip(p=1), #Je dirais ce serait bien d'ajouter rotation à 90° , 180° et 270°
                           ]
    
    new_images = [transform(image) for transform in geometric_transforms]
    new_labels = [transform(label) for transform in geometric_transforms]
    # Other transformations applied only to the image
    '''other_transforms = T.Compose([
        T.RandomApply([T.GaussianBlur(kernel_size=3)]),
        T.ToTensor(), 
        normalize
    ])'''
    
    '''if (rand > 5):
        # Apply transformations to both image and label
        image, label = geometric_transforms(image), geometric_transforms(label)'''
    
    other_transforms = [T.GaussianBlur(kernel_size=3)]
    new_images = new_images +  [transform(image) for transform in other_transforms]
    new_labels = new_labels + [label for _ in other_transforms]                   
    return new_images, new_labels, geometric_transforms , other_transforms


class PatchesDataset(torch.utils.data.Dataset):

    def __init__(self, images, groundtruths):
        assert len(images) == len(groundtruths)
        
        for img, gt in zip(images, groundtruths):
            assert (img['x'] == gt['x'] and img['y'] == gt['y'])
        
        self.input = images.copy()
        
        self.target = groundtruths.copy()
        

        self.means, self.stds = self.compute_means_and_stds()

    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        return self.input[index]['patch'], self.target[index]['patch']
    
    def get_at_pos(self, x, y):
        res = []
        for i in range(len(self.input)):
            if self.input[i]['x'] == x and self.input[i]['y'] == y:
                res.append((self.input[i], self.target[i]))
        return res
    
    def plot_at_pos(self, x, y):
        imgs_labs = self.get_at_pos(x, y)
        images = [img_lab[0]['patch'].numpy() for img_lab in imgs_labs]
        image_tranformations = [img_lab[0]['augmention'] for img_lab in imgs_labs]

        labels = [img_lab[1]['patch'].numpy() for img_lab in imgs_labs]
        labels_tranformations = [img_lab[1]['augmention'] for img_lab in imgs_labs]

        rgb_images = [(img[:3] * 255).astype(np.uint8).transpose(1, 2, 0) for img in images]
        rgb_labels = [lab.astype(np.uint8).squeeze(0) * 255 for lab in labels]


        labs = [Image.fromarray(lab) for lab in rgb_labels]
        for i,lab in enumerate(labs):
            lab.save(f"test.png")

        imgs = [Image.fromarray(img) for img in rgb_images]

        _, axs = plt.subplots(2, len(imgs) , figsize=(len(imgs)*6, 10))

        for i in range(len(imgs)):
            axs[0,i].imshow(imgs[i])
            axs[0,i].set_title(f"Image with {image_tranformations[i]}")
            axs[0,i].axis("off")
            axs[1,i].imshow(labs[i], cmap='gray')
            axs[1,i].set_title(f"Label with {labels_tranformations[i]}")
            axs[1,i].axis("off")

        plt.show()


    def get_images(self):
        return [images['patch'] for images in self.input]
    
    def get_groundtruths(self):
        return [gt['patch'] for gt in self.target]
    
    def compute_means_and_stds(self):
        stacked_tensor  = torch.stack(self.get_images(), axis = 0)
        means = torch.mean(stacked_tensor, axis =[0,2,3])
        stds = torch.std(stacked_tensor, axis = [0,2,3])
        return means, stds

    def augment(self, augmentation = transform_train_with_labels):
        all_new_imgs = []
        all_new_gts = []
        for img,gt in tqdm(zip(self.input, self.target), total = len(self.input), desc = "Augmenting dataset"): 
            new_imgs, new_gts, geometric_tranform, other_transform = augmentation(img['patch'], gt['patch'])

            new_imgs = [{'x' : img['x'], 
                         'y' : img['y'],
                         'patch' : new_img,
                         'augmention' : augmentation} 
                         for new_img, augmentation in zip(new_imgs, geometric_tranform + other_transform)]
            
            new_gts = [{'x' : gt['x'], 
                         'y' : gt['y'],
                         'patch' : new_gt,
                         'augmention' : augmentation} 
                         for new_gt, augmentation in zip(new_gts, geometric_tranform + ["No augmentation"]*len(other_transform))]
            
            all_new_imgs.extend(new_imgs)
            all_new_gts.extend(new_gts)
        
        self.input.extend(all_new_imgs)
        self.target.extend(all_new_gts)

        self.means, self.stds = self.compute_means_and_stds()


    def preprocess(self):
        normalize = T.Normalize(mean=self.means, std=self.stds)
        for img in tqdm(self.input, total = len(self.input), desc = "Preprocessing dataset"):
            img['patch'] = normalize(img['patch'])
    
    def unnormalize(self):
        denormalize = T.Normalize(mean=-self.means/self.stds, std=1/self.stds)
        for img in tqdm(self.input, total = len(self.input), desc = "Unnormalizing dataset"):
            img['patch'] = denormalize(img['patch'])
        