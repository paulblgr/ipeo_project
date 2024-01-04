import torch
from tqdm import tqdm
import torchvision.transforms as T

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
    print(label)
    new_labels = [T.Compose([transform, T.ToTensor()])(label) for transform in geometric_transforms]
    # Other transformations applied only to the image
    '''other_transforms = T.Compose([
        T.RandomApply([T.GaussianBlur(kernel_size=3)]),
        T.ToTensor(), 
        normalize
    ])'''
    
    '''if (rand > 5):
        # Apply transformations to both image and label
        image, label = geometric_transforms(image), geometric_transforms(label)'''
    
    other_transforms = [T.RandomApply(T.GaussianBlur(kernel_size=3))]
    other_transforms = [T.Compose([transform, T.ToTensor()]) for transform in other_transforms]
    new_images = new_images +  [transform(image) for transform in other_transforms]
    new_labels = new_labels + [label for _ in other_transforms]
                                      
    return new_images, new_labels

class PatchesDataset(torch.utils.data.Dataset):

    def __init__(self, images, groundtruths):
        assert len(images) == len(groundtruths)
        for img, gt in zip(images, groundtruths):
            assert (img['x'] == gt['x'] and img['y'] == gt['y'])
        self.input = images
        self.target = groundtruths

    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        return self.input[index]['patch'], self.target[index]['patch']

    def get_images(self):
        return self.input
    
    def get_groundtruths(self):
        return self.target

    def augment(self, augmentation = transform_train_with_labels):
        all_new_imgs = []
        all_new_gts = []
        for img,gt in zip(self.input, self.target): 
            new_imgs = augmentation(img['patch'], gt['patch'])

            new_imgs = [{'x' : img['x'], 'y' : img['y'], 'patch' : new_img, 'augmented' : True} for new_img in new_imgs]
            new_gts = [{'x' : gt['x'], 'y' : gt['y'], 'patch' : new_gt, 'augmented' : True} for new_gt in new_gts]
            all_new_imgs.extend(new_imgs)
            all_new_gts.extend(new_gts)

        return self

