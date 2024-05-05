import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split , KFold , StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset , random_split , Subset
from utils.consts import *
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from torchvision import datasets, transforms 
from utils.spectrograms import generate_spectrograms , generate_spectrograms_kaggle
from PIL import Image
from pytorch_lightning.callbacks import EarlyStopping , ModelCheckpoint
import torchvision.models as models
import os

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, random_split, Subset
from utils.consts import *
import pytorch_lightning as pl
from torchvision import datasets, transforms 
from utils.spectrograms import generate_spectrograms , generate_spectrograms_kaggle , generate_spectrograms_validation
from PIL import Image
from pytorch_lightning.callbacks import EarlyStopping , ModelCheckpoint

### Flags and treshold ###
Accuracy_flag = False
Treshold = 80.00
validation_accuracies = []
train_accuracies = []
import sys
import os
import pandas as pd
import shutil
import ray
from ray import tune, train
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
import ray.cloudpickle as pickle
from pathlib import Path
import tempfile


def transform_image(path):
    # Open an image file, ensuring it is read in RGBA mode
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
    
class Transfer(pl.LightningModule):
    def __init__(self, kernel_size_1, kernel_size_2, hidden_size, lr, weight_decay, num_classes ,  dropout_rate=0.2, freeze = True ):
        super(Transfer , self).__init__()
        self.accuracy_flag = False
        self.model= models.vgg16(pretrained = True)


        if freeze:
            for parameter in self.model.features.parameters():
                parameter.requires_grad = False
        self.model.classifier =  nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
            nn.Linear(4096, 1024),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, num_classes)
        )
        
        self.cost = nn.CrossEntropyLoss()
        self.lr = lr
        self.weight_decay = weight_decay
        

    def forward(self, x):
        return self.model(x)
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
    
    def save(self , val_acc):
        torch.save(self.state_dict() , f'models/models-cnn-{val_acc}')
        train_acc = 100 * self.trainer.callback_metrics['train_acc'].item()
        train_loss = self.trainer.callback_metrics['train_loss'].item()
        print(f"Train Loss: {train_loss}, Train Accuracy: {train_acc:.2f}%")

    def save(self, val_acc):
        torch.save(self.state_dict() , f'models/models-{val_acc}')

    def on_validation_epoch_end(self):
        val_acc = 100*self.trainer.callback_metrics['val_acc'].item()
        if( val_acc > Treshold ):
            validation_accuracies.append(val_acc)
            if( len(validation_accuracies) > 5 ):
                self.accuracy_flag = True
                self.trainer.should_stop = True
                self.save(val_acc)
        print(f"Validation Loss: {self.trainer.callback_metrics['val_loss'].item()}, Validation Accuracy: {val_acc:.2f}%")
        val_acc = 100 * self.trainer.callback_metrics['val_acc'].item()
        val_loss = 100 * self.trainer.callback_metrics['val_loss'].item()
        print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_acc:.2f}%")
        self.log('val_loss', val_loss, prog_bar=False)
        self.log('val_acc', val_acc, prog_bar=False)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

def run_train(config, train_sampler, validation_sampler, X , Y):
    """ Use this function to perform a hyperparameter search when unsure about
        the correct values to use
    """
    train_index = []
    validation_index = []
    for i in range(10):
        train_idx,_ = train_sampler[i]
        validation_idx , _ = validation_sampler[i]
        train_index.extend(train_idx)
        validation_index.extend(validation_idx)

    # resample train and validation data using specific batch size
    train_loader = DataLoader(X, batch_size=config['batch_size'], sampler=train_index, num_workers=4)
    val_loader = DataLoader(Y, batch_size=config['batch_size'], sampler=validation_index, num_workers=4)

    model = Transfer(config['kernel_size_1'], 
                config['kernel_size_2'],
                config['hidden_size'],
                lr=config['lr'],
                weight_decay=config['weight_decay'],
                dropout_rate=config['dropout_rate'] , num_classes = 10)
    trainer = pl.Trainer(precision="16-mixed", max_epochs=10, logger=False, enable_checkpointing=False)
    trainer.fit(model, train_loader, val_loader)





if __name__ == '__main__':
    torch.cuda.empty_cache()
    transform = transforms.Compose([transforms.ToTensor() ])
    X = datasets.ImageFolder(os.path.join(Path.cwd(), feature_files),
                             transform=transforms.Compose([transforms.ToTensor()]), 
                             loader=transform_image)
    Y = datasets.ImageFolder( os.path.join(Path.cwd(), validation_dir) , transform = transform , loader = transform_image)
    X_kaggle = datasets.ImageFolder(os.path.join(Path.cwd(), 'feature_files'), 
                                    transform=transforms.Compose([transforms.ToTensor()]),
                                    loader=transform_image)
    num_of_folds = 10
    labels =  np.array([x[1] for x in X.imgs])
    kfold_cross_validation = StratifiedKFold(n_splits = num_of_folds , shuffle = True)
    print(X)
    labels_validation = np.array([x[1] for x in Y.imgs])
    train_splits = list(kfold_cross_validation.split(X , labels ))
    validation_splits = list(kfold_cross_validation.split( Y , labels_validation))

    ray.init(
        configure_logging=True,
        logging_level='info',
        log_to_driver=False
    )

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
            tune.with_parameters(lambda x: run_train(x, train_splits, validation_splits, X , Y)),
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