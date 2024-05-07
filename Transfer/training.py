import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.consts import *
import pytorch_lightning as pl
from torchvision import datasets, transforms 
import torchvision.models as models
import os
from torchvision import datasets, transforms 
from utils.spectrograms import transform_image, generate_spectrograms, generate_spectrograms_kaggle
import ray
from ray import tune, train
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler
from pathlib import Path
from sklearn.model_selection import train_test_split
import tempfile
import pandas as pd
from utils.kaggle_utils import prediction
import matplotlib.pyplot as plt

# seed rng for reproducible results
torch.manual_seed(42)

class Transfer(pl.LightningModule):
    """ Class to represent our transfer learning model
    """

    def __init__(self, hidden_layer, lr, weight_decay, num_classes, dropout_rate=0.2, freeze=True):
        """ Initializes a CNN object

            Parameters:
                hidden_size: the size of the hidden layers of the MLP classifier used after
                    our convolutional layers
                lr: learning rate
                weight_decay: a penalty applied to large weights to enforce regularization
                dropout_rate: a fraction controlling how many of the network nodes are affected
                    by dropout
                num_classes: the number of classes in our dataset
                freeze: indicates whether to use VGG16's pre-trained layers
        """

        super(Transfer , self).__init__()
        self.accuracy_flag = False
        self.model = models.vgg16(pretrained=True)

        if freeze:
            for parameter in self.model.features.parameters():
                parameter.requires_grad = False
        self.model.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, hidden_layer),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_layer, hidden_layer),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_layer, num_classes)
        )
        
        self.cost = nn.CrossEntropyLoss()
        self.lr = lr
        self.weight_decay = weight_decay

        # create lists to hold accuracies for plotting
        self.val_acc_list = []
        self.train_acc_list = []
        

    def forward(self, x):
        return self.model(x)

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
        self.train_acc_list.append(self.trainer.callback_metrics['train_acc'].item())
        print(f'Train accuracy: {self.trainer.callback_metrics["train_acc"].item()}')
    
    def save(self , val_acc):
        torch.save(self.state_dict() , f'models/models-cnn-{val_acc}')
        train_acc = 100 * self.trainer.callback_metrics['train_acc'].item()
        train_loss = self.trainer.callback_metrics['train_loss'].item()
        print(f"Train Loss: {train_loss}, Train Accuracy: {train_acc:.2f}%")

    def save(self, val_acc):
        torch.save(self.state_dict() , f'models/models-{val_acc}')

    def on_validation_epoch_end(self):
        self.log('val_loss', self.trainer.callback_metrics['val_acc'], prog_bar=False)
        self.log('val_acc', self.trainer.callback_metrics['val_loss'], prog_bar=False)
        self.val_acc_list.append(self.trainer.callback_metrics['val_acc'].item())
        print(f'Validation accuracy: {self.trainer.callback_metrics["val_acc"].item()}')

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

def run_train(config, X, training_sampler, validation_sampler, kaggle_X=None) -> None:
    """ Use this function to perform a hyperparameter search when unsure about
        the correct values to use
    """

    # resample train and validation data using specific batch size
    train_loader = DataLoader(X, batch_size=config['batch_size'], sampler=training_sampler, num_workers=4)
    val_loader = DataLoader(X, batch_size=config['batch_size'], sampler=validation_sampler, num_workers=4)

    num_epochs = 20

    model = Transfer(config['hidden_size'],
                     lr=config['lr'],
                     weight_decay=config['weight_decay'],
                     dropout_rate=config['dropout_rate'], 
                     num_classes=10)
    trainer = pl.Trainer(precision='16-mixed', max_epochs=num_epochs, logger=False, enable_checkpointing=False)
    trainer.fit(model, train_loader, val_loader)

    # if a kaggle dataset is provided, create a prediction file after training model
    if kaggle_X is not None:
        df = pd.DataFrame()
        df['id'] = os.listdir(testing_data_path)
        df['class'] = prediction(model, kaggle_X, X.classes)
        df.to_csv('outputs.csv', index=False)

        # plot train vs validation acc for each epoch
        plt.figure()
        plt.plot(list(range(num_epochs)), model.train_acc_list[-num_epochs:], marker='o', label='Train Accuracy')
        plt.plot(list(range(num_epochs)), model.val_acc_list[-num_epochs:], marker='o', label='Validation Accuracy')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Train vs. Validation Accuracy (Transfer Learning)')
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(Path.cwd(), 'Transfer', 'training_vs_valid_acc.png'))
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
    
    # load spectrogram dataset
    transform = transforms.Compose([transforms.ToTensor() ])
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


    hyperparameter_set = {
        'batch_size': tune.grid_search([16, 32]),
        'hidden_size': tune.grid_search([128, 256, 512]),
        'dropout_rate': tune.uniform(0.25, 0.5),
        'lr': tune.loguniform(1e-3, 1e-1),
        'weight_decay': tune.loguniform(1e-5, 1e-3)
    }

    scheduler = ASHAScheduler(
        grace_period=1,
        reduction_factor=2)

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

    best_result = results.get_best_result('accuracy', 'max')

    print('Best trial config: {}'.format(best_result.config))
    print('Best trial final validation accuracy: {}'.format(
        best_result.metrics['accuracy']))
    
    # create kaggle file with the configuration of the best model
    print('\n\n\nModel training complete. Generating kaggle file with best configuration...\n\n')
    run_train(best_result.config, X, train_sampler, val_sampler, kaggle_X=X_kaggle)
