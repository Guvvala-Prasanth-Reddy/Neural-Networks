import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, random_split, Subset
from utils.consts import *
import pytorch_lightning as pl
from torchvision import datasets, transforms 
from PIL import Image
import sys
import os
import pandas as pd
import shutil
from ray import tune, train
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
import ray.cloudpickle as pickle
from pathlib import Path
import tempfile

# seed rng for reproducible results
torch.manual_seed(42)

def transform_image(path):
    # Open an image file, ensuring it is read in RGBA mode
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
    
class CNN(pl.LightningModule):
    def __init__(self, kernel_size_1, kernel_size_2, hidden_size, lr, weight_decay, dropout_rate=0.2):
        super(CNN , self).__init__()

        # define components of convolutional architecture
        self.conv_layer_1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=kernel_size_1, padding=1)
        self.max_pooling = nn.MaxPool2d(kernel_size=2)
        self.conv_layer_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=kernel_size_2, padding=1)
        self.avg_pooling = nn.AvgPool2d(kernel_size=2)

        # define components of MLP architecture
        fc1_input_size = (((512 + 3 - kernel_size_1) // 2) + 3 - kernel_size_2) // 2
        self.fc1 = nn.Linear(32 * (fc1_input_size ** 2), hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 10)
        self.relu = nn.ReLU()
        self.cost = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax()
        self.dropout_0 = nn.Dropout(0.3)
        self.dropout = nn.Dropout(dropout_rate)

        # set the lr and weight decay here to be used when configuring optimizer
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, x):
        x = self.max_pooling(self.relu(self.conv_layer_1(x)))
        x = self.max_pooling(self.relu(self.conv_layer_2(x)))      

        x = torch.flatten(x, 1)

        x = self.dropout_0(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

    def evaluate(self, batch, stage=None):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.cost(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y).float().mean()
        if stage:
            self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=True, logger=True)
            self.log(f"{stage}_acc", acc,  on_epoch=True, prog_bar=True, logger=True)
        return loss

    def training_step(self, batch, batch_idx):
        return(self.evaluate(batch, "train"))

    def validation_step(self, batch, batch_idx):
        return(self.evaluate(batch, "val"))

    def test_step(self, batch, batch_idx):
        return(self.evaluate(batch, "test"))

    def on_train_epoch_end(self):
        train_acc = 100 * self.trainer.callback_metrics['train_acc'].item()
        train_loss = self.trainer.callback_metrics['train_loss'].item()
        print(f"Train Loss: {train_loss}, Train Accuracy: {train_acc:.2f}%")

    def save(self, val_acc):
        torch.save(self.state_dict() , f'models/models-{val_acc}')

    def on_validation_epoch_end(self):
        val_acc = 100 * self.trainer.callback_metrics['val_acc'].item()
        val_loss = 100 * self.trainer.callback_metrics['val_loss'].item()
        print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_acc:.2f}%")
        self.log('val_loss', val_loss, prog_bar=False)
        self.log('val_acc', val_acc, prog_bar=False)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

def run_train(config, train_sampler, val_sampler, X):
    """ Use this function to perform a hyperparameter search when unsure about
        the correct values to use
    """

    # resample train and validation data using specific batch size
    train_loader = DataLoader(X, batch_size=config['batch_size'], sampler=train_sampler, num_workers=4)
    val_loader = DataLoader(X, batch_size=config['batch_size'], sampler=val_sampler, num_workers=4)

    model = CNN(config['kernel_size_1'], 
                config['kernel_size_2'],
                config['hidden_size'],
                lr=config['lr'],
                weight_decay=config['weight_decay'],
                dropout_rate=config['dropout_rate'])
    trainer = pl.Trainer(precision="16-mixed", max_epochs=10, logger=False, enable_checkpointing=False)
    trainer.fit(model, train_loader, val_loader)

    # sometimes a ./checkpoints/ directory is created with files > 1 GB, deleting
    # it if it appears to avoid a disk quota exceeded error
    if os.path.isdir('checkpoints'):
        shutil.rmtree('checkpoints')

    # the lines below are boilerplate for registering the accuracy and loss of a
    # hyperparameter set with raytune
    checkpoint_data = {
            "epoch": 10,
            "net_state_dict": model.state_dict(),
        }
    with tempfile.TemporaryDirectory() as checkpoint_dir:
        data_path = Path(checkpoint_dir) / "data.pkl"
        with open(data_path, "wb") as fp:
            pickle.dump(checkpoint_data, fp)

        checkpoint = Checkpoint.from_directory(checkpoint_dir)
        train.report(
            {
                "loss": trainer.callback_metrics['val_loss'].item(), 
                "accuracy": trainer.callback_metrics['val_acc'].item()
            },
            checkpoint=checkpoint,
        )

if __name__ == '__main__':
    torch.cuda.empty_cache()
    
    #generate_spectrograms('data/train/')
    #generate_spectrograms_kaggle('data/test/')
    X = datasets.ImageFolder(os.path.join(Path.cwd(), 'feature_files'),
                             transform=transforms.Compose([transforms.ToTensor()]), 
                             loader=transform_image)
    X_kaggle = datasets.ImageFolder(os.path.join(Path.cwd(), 'feature_files'), 
                                    transform=transforms.Compose([transforms.ToTensor()]),
                                    loader=transform_image)

    # Split the dataset into train and test sets, stratified by labels
    train_indices, val_indices = train_test_split(
        range(len(X)),
        test_size=0.2,
        stratify= [label for _, label in X],
        random_state=42
    )

    # Create data loaders for train and test sets
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

    hyperparameter_set = {
        'kernel_size_1': tune.grid_search([2, 5, 7]),
        'kernel_size_2': tune.grid_search([2, 5, 7]),
        'batch_size': tune.grid_search([8, 16, 32]),
        'hidden_size': tune.grid_search([32, 64, 128]),
        'dropout_rate': tune.uniform(0.1, 0.5),
        'lr': tune.loguniform(1e-4, 1e-1),
        'weight_decay': tune.loguniform(1e-6, 1e-2)
    }

    scheduler = ASHAScheduler(
        grace_period=1,
        reduction_factor=2)

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(lambda x: run_train(x, train_sampler, val_sampler, X)),
            resources={"cpu": 3, "gpu": 1}
        ),
        tune_config=tune.TuneConfig(
            metric="accuracy",
            mode="max",
            scheduler=scheduler,
            num_samples=1,
        ),
        param_space=hyperparameter_set,
    )
    results = tuner.fit()

    best_result = results.get_best_result("accuracy", "max")

    print("Best trial final validation loss: {}".format(
        best_result.metrics["loss"]))
    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation accuracy: {}".format(
        best_result.metrics["accuracy"]))

    # print('Creating kaggle file')
    # target_predicted = []
    # model.eval()
    # with torch.no_grad(): 
    #     for inputs, targets in test_loader:
    #         predicted_outputs = model(inputs)
    #         _, predicted = torch.max(predicted_outputs.data, 1)
    #         target_predicted.extend(np.take(X.classes, predicted.tolist()))

    # df = pd.DataFrame()
    # df['id'] = os.listdir('data/test/')
    # df['class'] = target_predicted
    # print(target_predicted)
    # df.to_csv('outputs.csv', index=False)