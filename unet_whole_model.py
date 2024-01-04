import torch
import torch.nn as nn
import numpy as np
import torch.utils.data
from dataset import Dataset
from helpers import load_all_images, prepare_train, prepare_test
import json
from unet_model import UNet
from torchmetrics.classification import BinaryF1Score as F1score
from tqdm import tqdm


class Model:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.unet = UNet(n_channels=4, n_classes=2)
        self.unet.to(self.device)
        self.optimizer = torch.optim.Adam(self.unet.parameters(), lr=3e-4)
        self.criterion = nn.CrossEntropyLoss()
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
        loss = self.criterion(raw_pred, target)
        loss.backward()
        self.optimizer.step()

        train_pred = raw_pred.argmax(1)

        return loss, train_pred


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
        for data, target in tqdm(data_loader):
            data, target = data.to(self.device), target.to(self.device)
            loss, train_pred = self.training_step(data, target)
            train_losses.append(loss.cpu().detach().numpy())
            train_preds.append(train_pred)
            del data
            del target


        train_loss = np.stack(train_losses).mean()
        f1 = self.f1(torch.cat(train_preds), data_loader[1]).detach().cpu().item()
        torch.cuda.empty_cache()

        return train_loss, f1
    
    def prediction_step(self, data, target):
        raw_pred = self.unet(data)
        loss = self.criterion(raw_pred, target)
        pred = raw_pred.argmax(1)

        return loss, pred


    def validation_epoch(self, data_loader):
        """Performs a validation epoch on the provided data loader. Predicts the values using the current state of the model and computers the validation loss and validation F1-score.

        Args:
            data_loader (_type_): Data to use for validation epoch.

        Returns:
            _type_: Validation loss and validation F1-score
        """
        val_losses = []
        preds = []
        self.unet.eval()
        with torch.no_grad():
            for (data, target) in tqdm(data_loader):
                data, target = data.to(self.device), target.to(self.device)
                loss, pred = self.prediction_step(data, target)
                val_losses.append(loss.cpu().detach().numpy())
                preds.append(pred)
                del data
                del target

        val_losses = np.stack(val_losses).mean()
        f1 = self.f1(torch.cat(preds), data_loader[1]).detach().cpu().item()
        torch.cuda.empty_cache()
        
        return val_losses, f1

    def add_history(self, t, f, type):
        """Adds an input to the history of the model.

        Args:
            t (_type_): loss of the epoch
            f (_type_): f1-score of the epoch
            type (_type_): Train or validation epoch (default: "Train")
        """
        self.all_histories[type]["Train loss"].append(t)
        self.all_histories[type]["F1_score"].append(f)

    def train(self, train_set, val_set, num_epochs=10, train_percent=0.9):
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
            t, f = self.train_epoch(val_loader)
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

    def import_data(self, train_path, label_path):
        """Loads the data and formats it according to the model's architecture's requirements.

        Args:
            train_path (_type_): Path to train image folder.
            label_path (_type_): Path to groundtruth image folder

        Returns:
            _type_: The formatted data ready to be used for the model training.
        """
        train_images = load_all_images(train_path)
        target_images = load_all_images(label_path)
        train_data = prepare_train(train_images)
        target_data = prepare_test(target_images)
        return train_data, target_data
