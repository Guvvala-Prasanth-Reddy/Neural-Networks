import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from utils.consts import *
import pytorch_lightning as pl
from torchvision import datasets, transforms 
from utils.spectrograms import transform_image, generate_spectrograms, generate_spectrograms_kaggle
import os
import pandas as pd
import ray
from ray import tune, train
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler
from pathlib import Path
import tempfile
from utils.kaggle_utils import prediction
import matplotlib.pyplot as plt
import sys

# seed rng for reproducible results
torch.manual_seed(42)
    
class CNN(pl.LightningModule):
    """ Class which represents our custom CNN architecture
    """

    def __init__(self, kernel_size_1, kernel_size_2, hidden_size, lr, weight_decay, dropout_rate=0.2):
        """ Initializes a CNN object

            Parameters:
                kernel_size_1: the size of the kernel used by the first convolutional layer
                kernel_size_2: the size of the kernel used by the second convolutional layer
                hidden_size: the size of the hidden layers of the MLP classifier used after
                    our convolutional layers
                lr: learning rate
                weight_decay: a penalty applied to large weights to enforce regularization
                dropout_rate: a fraction controlling how many of the network nodes are affected
                    by dropout
        """
        
        super(CNN , self).__init__()
        # define components of convolutional architecture
        self.conv_layer_1 = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=kernel_size_1, padding=1)
        self.max_pooling = nn.MaxPool2d(kernel_size=2)
        self.conv_layer_2 = nn.Conv2d(in_channels=5, out_channels=10, kernel_size=kernel_size_2, padding=1)
        self.avg_pooling = nn.AvgPool2d(kernel_size=2)

        # define components of MLP architecture
        fc1_input_size = (((512 + 3 - kernel_size_1) // 2) + 3 - kernel_size_2) // 2
        self.fc1 = nn.Linear(10 * (fc1_input_size ** 2), hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 10)
        self.relu = nn.ReLU()
        self.cost = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax()
        self.dropout_0 = nn.Dropout(dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)

        # set the lr and weight decay here to be used when configuring optimizer
        self.lr = lr
        self.weight_decay = weight_decay

        # create lists to hold accuracies for plotting
        self.val_acc_list = []
        self.train_acc_list = []

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

        y_hat = None
        if stage == 'val':
            with torch.no_grad():
                self.eval()
                y_hat = self.forward(x)
            self.train()
        else:
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
        self.train_acc_list.append(self.trainer.callback_metrics['train_acc'].item())
    
    def save(self , val_acc):
        torch.save(self.state_dict() , f'models/models-cnn-{val_acc}')
        train_acc = 100 * self.trainer.callback_metrics['train_acc'].item()
        train_loss = self.trainer.callback_metrics['train_loss'].item()
        print(f"Train Loss: {train_loss}, Train Accuracy: {train_acc:.2f}%")

    def save(self, val_acc):
        torch.save(self.state_dict() , f'models/models-{val_acc}')

    def on_validation_epoch_end(self):
        self.log('val_loss', self.trainer.callback_metrics['val_loss'], prog_bar=False)
        self.log('val_acc', self.trainer.callback_metrics['val_acc'], prog_bar=False)
        self.val_acc_list.append(self.trainer.callback_metrics['val_acc'].item())

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


def run_train(config, X, train_sampler, valid_sampler, kaggle_X=None) -> None:
    """ Use this function to perform a hyperparameter search when unsure about
        the correct values to use

        Parameters:
            config: a library of hyperparameter values used by Ray Tune
            X: the spectrogram dataset
            train_sampler: a sampler in the indices of X which are to be
                used for training
            val_sampler: a sampler on the indices of X which are to be used
                for validation
            kaggle_X: the kaggle spectrogram dataset if a prediction file is
                to be generated; None otherwise
    """

    # resample train and validation data using specific batch size
    train_loader = DataLoader(X, batch_size=config['batch_size'], sampler=train_sampler, num_workers=4)
    val_loader = DataLoader(X, batch_size=config['batch_size'], sampler=valid_sampler, num_workers=4)

    num_epochs = 20

    model = CNN(config['kernel_size_1'], 
                config['kernel_size_2'],
                config['hidden_size'],
                lr=config['lr'],
                weight_decay=config['weight_decay'],
                dropout_rate=config['dropout_rate'])
    trainer = pl.Trainer(precision='16-mixed', max_epochs=num_epochs, logger=False, enable_checkpointing=False)
    trainer.fit(model, train_loader, val_loader)

    # if a kaggle dataset is provided, create a prediction file after training model
    if kaggle_X is not None:
        df = pd.DataFrame()
        df['id'] = os.listdir(testing_data_path)
        df['class'] = prediction(model, kaggle_X, X.classes)
        df.to_csv('outputs.csv', index=False)

        # plot train vs validation acc for each epoch
        print(model.val_acc_list)
        plt.figure()
        plt.plot(list(range(num_epochs)), model.train_acc_list[-num_epochs:], marker='o', label='Train Accuracy')
        plt.plot(list(range(num_epochs)), model.val_acc_list[-num_epochs:], marker='o', label='Validation Accuracy')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Train vs. Validation Accuracy (CNN)')
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(Path.cwd(), 'CNN', 'training_vs_valid_acc.png'))

    # otherwise, continue training with Ray Tune
    else:
        # the lines below are boilerplate for registering the accuracy and loss of a
        # hyperparameter set with Ray Tune
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            checkpoint = Checkpoint.from_directory(checkpoint_dir)
            train.report(
                {
                    'loss': trainer.callback_metrics['val_loss'].item(), 
                    'accuracy': trainer.callback_metrics['val_acc'].item()
                },
                checkpoint=checkpoint,
            )

if __name__ == '__main__':
    torch.cuda.empty_cache()

    # only create spectrograms if we detect they are not already present
    if not os.path.isdir(training_spectorgram_path):
        print('Creating training spectrograms. Please be patient...')
        generate_spectrograms(training_data_path)
    if not os.path.isdir(kaggle_spectrogram_path):
        print('Creating training spectrograms. Please be patient...')
        generate_spectrograms_kaggle(testing_data_path)

    # load image datasets
    transform = transforms.Compose([transforms.ToTensor()])
    X = datasets.ImageFolder(os.path.join(Path.cwd(), training_spectorgram_path),
                             transform=transforms.Compose([transforms.ToTensor()]), 
                             loader=transform_image)
    X_kaggle = datasets.ImageFolder(os.path.join(Path.cwd(), kaggle_spectrogram_path), 
                                    transform=transforms.Compose([transforms.ToTensor()]),
                                    loader=transform_image)

    # split the dataset into stratified train and valid splits
    train_indices, val_indices = train_test_split(
        range(len(X)),
        test_size=0.2,
        stratify= [label for _, label in X],
        random_state=42
    )

    # create data loaders for train and validate splits
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

    ray.init(
        configure_logging=True,
        logging_level='info',
        log_to_driver=False
    )

    # define the hyperparameters to test and their ranges
    hyperparameter_set = {
        'kernel_size_1': tune.grid_search([5, 7, 10]),
        'kernel_size_2': tune.grid_search([5, 7, 10]),
        'batch_size': tune.grid_search([16, 32]),
        'hidden_size': tune.grid_search([128, 256]),
        'dropout_rate': tune.uniform(0.3, 0.5),
        'lr': tune.loguniform(1e-3, 1e-1),
        'weight_decay': tune.loguniform(1e-5, 1e-3)
    }

    # define ASHA scheduler to be used by Ray Tune
    scheduler = ASHAScheduler(
        grace_period=1,
        reduction_factor=2
    )

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(lambda x: run_train(x, X, train_sampler, val_sampler)),
            resources={'cpu': 3, 'gpu': 1}
        ),
        tune_config=tune.TuneConfig(
            metric='accuracy',
            mode='max',
            scheduler=scheduler,
            num_samples=10,
        ),
        param_space=hyperparameter_set,
    )
    results = tuner.fit()

    # after Ray Tune finishes, display the results of best configuration
    best_result = results.get_best_result('accuracy', 'max')
    print('Best trial config: {}'.format(best_result.config))
    print('Best trial final validation accuracy: {}'.format(
        best_result.metrics['accuracy']))

    # create kaggle file with the configuration of the best model
    print('\n\n\nModel training complete. Generating kaggle file with best configuration...\n\n')
    run_train(best_result.config, X, train_sampler, val_sampler, kaggle_X=X_kaggle)
