import torch
import torch.nn as nn
import torch.utils.data
from dataset import Dataset
from helpers import load_all_images, prepare_chunk_test, printProgressBar, model_to_raw
from data_modifiers import (
    import_train_data,
    augment_all_arrays,
    plt_to_model,
    split_images,
)
import json
from PIL import Image
from unet_group import unet_group
from torchmetrics.classification import BinaryF1Score as F1score


class Model:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.unet = unet_group()
        self.unet.to(self.device)
        self.optimizer = torch.optim.Adam(self.unet.parameters(), lr=3e-4)
        self.criterion = nn.BCELoss()
        self.f1 = F1score().to(self.device)
        self.batch_size = 1024
        self.all_histories = {
            "Train": {"Train loss": [], "F1_score": []},
            "Validation": {"Train loss": [], "F1_score": []},
        }
        self.model_path = "models/unet_group/"

    def get_history(self):
        """Returns the training (and validation history if validation was used) of the current model.

        Returns:
            Dictionary: Dictionary containing all the losses and F1-scores of the training.
        """
        return self.all_histories

    def predict(self, input):
        """Performs predictions on the input data using the current state of the model. Also applies thresholding the the output so that values are mapped to 0 or 1.

        Args:
            input (torch.Tensor): Tensor containing the data to predict.

        Returns:
            torch.Tensor: The tensor containing the predictions,
        """
        self.unet.eval()
        data_loader = torch.utils.data.DataLoader(input, self.batch_size, shuffle=False)
        predictions = list()
        for i, data in enumerate(data_loader, 1):
            predictions.append(self.unet.forward(data.to(self.device)).detach().cpu())
            printProgressBar(i, len(data_loader), "Predicting", "Complete", length=50)
        torch.cuda.empty_cache()
        predictions = torch.cat(predictions)
        predictions = torch.where(predictions > 0.5, 1, 0)
        return predictions

    def pred_and_true(self, data_loader):
        """Returns the predictions of the data inside the given data loader and their corresponding labels. Each having been thresholded.

        Args:
            data_loader (_type_): Data loader containing both the train and label

        Returns:
            _type_: The predictions and labels for the data loader.
        """
        predictions = self.predict_quick(data_loader.dataset.train_set())
        labels = data_loader.dataset.target_set()
        labels = torch.where(labels > 0.5, 1, 0)
        return predictions, labels

    def predict_quick(self, input):
        """Same as the predict(input) function but doesn't perform thresholding.

        Args:
            input (_type_): Data to be predicted

        Returns:
            _type_: Predicted labels.
        """
        self.unet.eval()
        data_loader = torch.utils.data.DataLoader(input, self.batch_size, shuffle=False)
        predictions = list()
        for i, data in enumerate(data_loader, 1):
            predictions.append(self.unet.forward(data.to(self.device)).detach().cpu())
            printProgressBar(i, len(data_loader), "Predicting", "Complete", length=50)
        torch.cuda.empty_cache()
        return torch.cat(predictions)

    def train_epoch(self, data_loader):
        """Performs a training epoch using stochastic gradient descent and back propagation over the provided data loader.

        Args:
            data_loader (_type_): Data to be used for the training epoch

        Returns:
            _type_: Train loss and F1-score of the epoch
        """
        train_loss = 0.0
        self.unet.train()
        for i, (data, target) in enumerate(data_loader, 1):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.unet(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            printProgressBar(
                i, len(data_loader), "Train progress", "Complete", length=50
            )

        del data
        del target
        torch.cuda.empty_cache()

        y_pred, y_true = self.pred_and_true(data_loader)
        y_pred, y_true = y_pred.to(self.device), y_true.to(self.device)
        f1 = self.f1(y_pred, y_true).detach().cpu().item()
        train_loss /= len(data_loader)
        return train_loss, f1

    def validation_epoch(self, data_loader):
        """Performs a validation epoch on the provided data loader. Predicts the values using the current state of the model and computers the validation loss and validation F1-score.

        Args:
            data_loader (_type_): Data to use for validation epoch.

        Returns:
            _type_: Validation loss and validation F1-score
        """
        val_loss = 0.0
        self.unet.eval()
        for i, (data, target) in enumerate(data_loader, 1):
            data, target = data.to(self.device), target.to(self.device)
            output = self.unet(data)
            loss = self.criterion(output, target)
            val_loss += loss.item()
            printProgressBar(
                i, len(data_loader), "Validation progress", "Complete", length=50
            )

        del data
        del target
        torch.cuda.empty_cache()

        y_pred, y_true = self.pred_and_true(data_loader)
        y_pred, y_true = y_pred.to(self.device), y_true.to(self.device)
        f1 = self.f1(y_pred, y_true).detach().cpu().item()
        val_loss /= len(data_loader)
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

    def train(self, train_data, train_target, num_epochs=10, train_percent=1):
        """Trains a model using the provided training set and labels over a certain number of epochs using a decidable percentage of the dataset.

        Args:
            train_data (_type_): Training set
            train_target (_type_): Label set
            num_epochs (int, optional): Number of epochs to perform training on. Defaults to 10.
            train_percent (int, optional): Percentage of the total training set to use (between 0 and 1). Defaults to 1.

        Returns:
            _type_: The history of losses and f1-scores of the model during all the epochs so far.
        """
        train_size = int(train_data.shape[0] * train_percent)
        train_set = Dataset(train_data[:train_size], train_target[:train_size])
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True
        )
        for epoch in range(num_epochs):
            t, f = self.train_epoch(train_loader)
            self.add_history(t, f, "Train")
            print(
                "Epoch: {:d}/{:d} Train_loss: {:.5f}, Train_F1: {:.5f}".format(
                    epoch + 1,
                    num_epochs,
                    self.all_histories["Train"]["Train loss"][epoch],
                    self.all_histories["Train"]["F1_score"][epoch],
                ),
                flush=True,
            )
        return self.all_histories

    def train_and_val(self, train_data, train_target, num_epochs=10, train_percent=0.9):
        """Performs a similar task to the train() function but also performs a validation epoch on whatever percentage of the training set was not used for training.

        Args:
            train_data (_type_): Training set
            train_target (_type_): Label set
            num_epochs (int, optional): Number of epochs to perform training and validation on. Defaults to 10.
            train_percent (float, optional): Percentage of data to use for training (must be less than 1). Defaults to 0.9.

        Returns:
            _type_: The history of losses and f1-scores of the model during all the epochs so far.
        """
        train_size = int(train_data.size(0) * train_percent)
        train_set = Dataset(train_data[:train_size], train_target[:train_size])
        val_set = Dataset(train_data[train_size:], train_target[train_size:])

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
            print(
                "Epoch: {:d}/{:d} Train_loss: {:.5f}, Train_F1: {:.5f}".format(
                    epoch + 1,
                    num_epochs,
                    self.all_histories["Train"]["Train loss"][epoch],
                    self.all_histories["Train"]["F1_score"][epoch],
                ),
                flush=True,
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
        state_dict = torch.load(path.format(model_name, "pth"))
        self.unet.load_state_dict(torch.load(path.format(model_name, "pth")))
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
        train_data = import_train_data(train_path)
        target_images = load_all_images(label_path)
        target_data = prepare_chunk_test(target_images)

        return train_data, target_data

    def prepare_test(self, all_images):
        img_size = (608, 608)
        augmented = augment_all_arrays(all_images.detach().cpu().numpy())
        chunks_numpy = split_images(augmented, img_size)
        chunks_model = plt_to_model(chunks_numpy)
        return chunks_model

    def rebuild_output(self, predictions):
        all_chunks = predictions.mul(255).to(torch.uint8)
        all_chunks = all_chunks.permute(0, 2, 3, 1)
        all_images = self.rebuild_all_images(all_chunks)
        return all_images.cpu().detach().numpy()

    def rebuild_all_images(self, chunk_tensor):
        chunk_pixel_size = 16
        image_pixel_size = 608
        img_chunk_width = image_pixel_size // chunk_pixel_size
        img_chunk_size = img_chunk_width**2
        assert chunk_tensor.shape[0] % img_chunk_size == 0
        all_images = list()
        for i in range(int(chunk_tensor.shape[0] / img_chunk_size)):
            all_images.append(self.rebuild_image_from_chunks(chunk_tensor, i))
        return torch.stack(all_images)

    def rebuild_image_from_chunks(self, chunk_tensor, start_index):
        chunk_pixel_size = 16
        image_pixel_size = 608
        img_chunk_width = image_pixel_size // chunk_pixel_size
        img_chunk_size = img_chunk_width**2
        start = start_index * img_chunk_size
        assert chunk_tensor.shape[0] >= start + img_chunk_size
        columns = list()
        complete = list()
        colum_index = 0
        row_index = 0
        for i in range(start, start + img_chunk_size):
            columns.append(
                chunk_tensor[start + (row_index * img_chunk_width) + colum_index]
            )
            colum_index += 1
            if (i + 1) % img_chunk_width == 0:
                complete.append(torch.cat(columns))
                columns.clear()
                colum_index = 0
                row_index += 1
        return torch.cat(complete, dim=1)
