import torch
import torch.nn as nn
import numpy as np
import torch.utils.data
from dataset import *
import json
from unet_model import UNet
from torchmetrics.classification import BinaryF1Score as F1score
from tqdm import tqdm


class Model:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.unet = UNet(n_channels=4, n_classes=1)
        self.unet.to(self.device)
        self.optimizer = torch.optim.Adam(self.unet.parameters(), lr=3e-4)
        self.criterion = nn.BCEWithLogitsLoss()
        self.f1 = F1score().to(self.device)
        self.batch_size = 4
        self.all_histories = {
            "Train": {"Train loss": [], "F1_score": []},
            "Validation": {"Train loss": [], "F1_score": []},
        }
        self.model_path = "models/"

    def get_history(self):
        """Returns the training (and validation history if validation was used) of the current model.

        Returns:
            Dictionary: Dictionary containing all the losses and F1-scores of the training.
        """
        return self.all_histories

    def training_step(self, data, target):
        self.optimizer.zero_grad()
        raw_pred = self.unet(data)
        target = target.to(torch.float32)
        loss = self.criterion(raw_pred, target)
        raw_pred = raw_pred.squeeze(1)
        loss.backward()
        self.optimizer.step()

        return loss, raw_pred


    def train_epoch(self, data_loader):
        """Performs a training epoch using stochastic gradient descent and back propagation over the provided data loader.

        Args:
            data_loader (_type_): Data to be used for the training epoch

        Returns:
            _type_: Train loss and F1-score of the epoch
        """
        train_losses = []
        train_preds = []
        self.unet.train()
        for i, (data, target) in tqdm(enumerate(data_loader)):
            if i < 5 :
                data, target = data.to(self.device), target.to(self.device)
                loss, train_pred = self.training_step(data, target)
                train_losses.append(loss.cpu().detach().numpy())
                train_preds.append(train_pred)
                del data
                del target


        train_loss = float(np.stack(train_losses).mean())
        f1 =float(self.f1(torch.cat(train_preds), torch.cat(data_loader.dataset.get_groundtruths()) ).detach().cpu().item())
        torch.cuda.empty_cache()

        return train_loss, f1
    
    def evaluation_step(self, data, target):
        raw_pred = self.unet(data)
        target = target.to(torch.float32)
        loss = self.criterion(raw_pred, target)
        raw_pred = raw_pred.squeeze(1)
        return loss, raw_pred


    def validation_epoch(self, data_loader):
        """Performs a validation epoch on the provided data loader. Predicts the values using the current state of the model and computers the validation loss and validation F1-score.

        Args:
            data_loader (_type_): Data to use for validation epoch.

        Returns:
            _type_: Validation loss and validation F1-score
        """
        val_losses = []
        test_preds = []
        self.unet.eval()
        with torch.no_grad():
            for i , (data, target) in tqdm(enumerate(data_loader)):
                if i < 3 :
                    data, target = data.to(self.device), target.to(self.device)
                    loss, pred = self.evaluation_step(data, target)
                    val_losses.append(loss.cpu().detach().numpy())
                    test_preds.append(pred)
                    del data
                    del target

        val_loss = float(np.stack(val_losses).mean())

        f1 = float(self.f1(torch.cat(test_preds), torch.cat(data_loader.dataset.get_groundtruths())).detach().cpu().item())
        torch.cuda.empty_cache()
        
        return val_loss, f1

    def add_history(self, t, f, type):
        """Adds an input to the history of the model.

        Args:
            t (_type_): loss of the epoch
            f (_type_): f1-score of the epoch
            type (_type_): Train or validation epoch (default: "Train")
        """
        self.all_histories[type]["Train loss"].append(t)
        self.all_histories[type]["F1_score"].append(f)

    def train(self, train_set, val_set, num_epochs=10):
        """Trains the model using the provided data and target. Saves the history of the training.
        """
        

        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_set, batch_size=self.batch_size, shuffle=True
        )

        for epoch in range(num_epochs):
            t, f = self.train_epoch(train_loader)
            self.add_history(t, f, "Train")
            t, f = self.validation_epoch(val_loader)
            self.add_history(t, f, "Validation")
            self.save_model(f"model_epoch_{epoch}")
            print(
                "Epoch: {:d}/{:d} Train_loss: {:.5f}, Train_F1: {:.5f}, Val_loss: {:.5f}, Val_F1: {:.5f}".format(
                    epoch + 1,
                    num_epochs,
                    self.all_histories["Train"]["Train loss"][epoch],
                    self.all_histories["Train"]["F1_score"][epoch],
                    self.all_histories["Validation"]["Train loss"][epoch],
                    self.all_histories["Validation"]["F1_score"][epoch],
                )
            )
        return self.all_histories
    
    def predict(self, img):
        """Predicts the segmentation of the input image using the current state of the model.

        Args:
            img (_type_): Input image to be segmented in tensor format

        Returns:
            _type_: Segmentation of the input image in tensor format
        """
        img_input = img.unsqueeze(0)
        self.unet.eval()
        with torch.no_grad():
            pred = self.unet(img_input.to(self.device))
        
        pred = pred.squeeze(1)
        pred = torch.where(pred < 0.5, 0 ,1).cpu()

        return pred
    
    def plot_prediction(self, img_dict):
        """Plots the input image and it's segmentation.

        Args:
            img (_type_): Input image to be segmented in tensor format
        """
        img = img_dict[0]["patch"]
        gt = img_dict[1]["patch"].numpy()

        pred = self.predict(img)
        rgb_img = (img.numpy()[:3] * 255).astype(np.uint8).transpose(1, 2, 0)
        rgb_pred = pred.numpy().astype(np.uint8).squeeze(0) * 255
        rgb_gt = gt.astype(np.uint8).squeeze(0) * 255
        
        _ , axs = plt.subplots(1, 3, figsize=(15, 10))
        axs[0].imshow(rgb_img)
        axs[0].set_title("Image")
        axs[0].axis("off")
        axs[1].imshow(rgb_pred, cmap='gray')
        axs[1].set_title("Prediction")
        axs[1].axis("off")
        axs[2].imshow(rgb_gt, cmap='gray')
        axs[2].set_title("Groundtruth")
        axs[2].axis("off")
        plt.show()



    def save_model(self, model_name):
        """Saves the current state of the model. Automatically puts it in the correct folder using the proper file extensions.

        Args:
            model_name (_type_): Name of the model to use for save file.
        """
        path = self.model_path + "{:s}.{:s}"
        torch.save(self.unet.state_dict(), path.format(model_name, "pth"))
        json_dict = json.dumps(self.all_histories)
        file = open(path.format(model_name, "json"), "w")
        file.write(json_dict)
        file.close

    def load_model(self, model_name):
        """Loads a saved model and it's history into the current model. Overwrites any prior training and history of the current model.

        Args:
            model_name (_type_): Name of the model to load (automatic path finding as for save_model()).
        """
        path = path = self.model_path + "{:s}.{:s}"
        self.unet.load_state_dict(
            torch.load(
                path.format(model_name, "pth"),
                map_location=torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                ),
            )
        )
        self.unet.eval()
        with open(path.format(model_name, "json")) as file:
            data = file.read()
            self.all_histories = json.loads(data)