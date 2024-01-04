import torch


class Dataset(torch.utils.data.Dataset):

    def __init__(self, image, groundtruth):
        assert len(image) == len(groundtruth)
        self.input = image
        self.target = groundtruth

    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        return self.input[index], self.target[index]

    def get_images(self):
        return self.input

    def get_groundtruths(self):
        return self.target
