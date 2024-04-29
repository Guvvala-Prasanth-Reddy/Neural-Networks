import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from utils.consts import *
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from torchvision import datasets, transforms
from utils.spectrograms import generate_spectrograms



class CNN(pl.LightningModule):
    def __init__(self, input_size, hidden_size, output_size,  kernel_size_1 , kernel_size_2 , dropout_rate=0.2):
        super(CNN , self).__init__()

        self.conv_layer_1 = nn.Conv2d(in_channels=3 , out_channels=16 , kernel_size= kernel_size_1)
        self.max_pooling = nn.MaxPool2d(kernel_size=2)
        self.conv_layer_2 = nn.Conv2d(in_channels=16 , out_channels=32 ,kernel_size= kernel_size_2 )
        self.avg_pooling = nn.AvgPool2d( kernel_size= 2)

        self.fc1 = nn.Linear(32*248*623 , 128)
        self.fc2 = nn.Linear(128 , 32)
        self.fc3 = nn.Linear(32 ,10 )
        self.relu = nn.ReLU()
        self.cost = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax()

        self.dropout = nn.Dropout()

    def forward(self, x):
        x = self.max_pooling(self.relu(self.conv_layer_1(x)))
        # print(x.shape , "conv 1")
        x = self.avg_pooling(self.relu(self.conv_layer_2(x)))
        print(x.shape , "conv 2")
        x = torch.flatten(x  , 1)
        print(x.shape)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        # print(x.shape)
        return x

    def evaluate(self, batch, stage=None):
        x, y = batch
        # print(x , y)
        y_hat = self.forward(x)
        loss = self.cost(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y).float().mean()
        if stage:
            self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=True, logger=True) #
            self.log(f"{stage}_acc", acc,  on_epoch=True, prog_bar=True, logger=True)
        return loss

    def training_step(self, batch, batch_idx):
        return(self.evaluate(batch, "train"))

    def validation_step(self, batch, batch_idx):
        return(self.evaluate(batch, "val"))

    def test_step(self, batch, batch_idx):
        return(self.evaluate(batch, "test"))

    def on_train_epoch_end(self):
        print(f"Train Loss: {self.trainer.callback_metrics['train_loss'].item()}, Train Accuracy: {100 * self.trainer.callback_metrics['train_acc'].item():.2f}%")

    def on_validation_epoch_end(self):
        print(f"Validation Loss: {self.trainer.callback_metrics['val_loss'].item()}, Validation Accuracy: {100 * self.trainer.callback_metrics['val_acc'].item():.2f}%")

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001, weight_decay=0.01)
        return optimizer

if __name__ == '__main__':
    torch.cuda.empty_cache()
# Load and preprocess data
    wine = load_wine()
    transform = transforms.Compose([transforms.ToTensor()])
    # generate_spectrograms('data/train/')
    X = datasets.ImageFolder( 'feature_files/' , transform = transform)

    print(X.classes , "classes")

    # scaler = StandardScaler()

    # X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    # X_train = scaler.fit_transform(X_train)
    # X_test_val = scaler.transform(X_test_val)
    # X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.5, random_state=42, stratify=y_test_val)

    # Create PyTorch datasets and dataloaders
    # train_dataset = TensorDataset(torch.tensor(X, dtype=torch))
    # val_dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
    # test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

    batch_size=4
    train_loader = DataLoader(X, batch_size=batch_size, shuffle=True , num_workers=4)
    val_loader = DataLoader(X, batch_size=batch_size , num_workers=4)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size)
    for images, labels in train_loader:
        print("Label data type:", labels.dtype)  # Check if labels are Long type
        break

    # Initialize MLP model
    input_size = 0
    hidden_size = 64
    output_size = 50  # Number of classes

    model = CNN(input_size, hidden_size, output_size , kernel_size_1= 5 , kernel_size_2= 3 )
    print(model)

    # Train the model using PyTorch Lightning Trainer
    trainer = pl.Trainer(max_epochs=50,logger=CSVLogger(save_dir="logs/") , log_every_n_steps=10)
    trainer.fit(model, train_loader, val_loader)

    # Test the model
    # test1 = trainer.test(model, dataloaders=test_loader)
