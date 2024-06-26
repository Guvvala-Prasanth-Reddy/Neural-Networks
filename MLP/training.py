import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import recall_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import os
import ray
from ray import tune, train
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler
import tempfile
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict
from utils.consts import *


class MLP(nn.Module):
    """ This class represents our chosen MLP architecture.
    """

    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.2):
        """ Initializes a MLP object

        Parameters
            input_size: the size of the MLP input layer
            hidden_size: the size of the hidden layers of the MLP classifier used after
                our convolutional layers
            output_size: the size of the output layer of the MLP
            dropout_rate: a fraction controlling how many of the network nodes are affected
                by dropout
        """

        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.cost = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x


def train_mlp_kaggle(config: Dict[str, any]) -> None:
    """ Use this function to train a model on a set of parameters suggested by raytune.
        A kaggle prediction file will be created for each epoch and a train vs. validation
        accuracy plot will be generated as well.
    """

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])

    # Initialize MLP model
    input_size = X_train.shape[1]
    output_size = len(np.unique(y_train))  # Number of classes

    model = MLP(input_size, config['hidden_size'], output_size, dropout_rate=config['dropout_rate'])

    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

    # create lists to hold train and validation accuracies for plotting
    train_acc_list = []
    valid_acc_list = []

    # create lists to hold recall and f1 score over epochs for plotting
    f1_score_list_train = []
    f1_score_list_valid = []
    recall_score_list_train = []
    recall_score_list_valid = []

    # training loop
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        combined_preds = []
        combined_targets = []
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = model.cost(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += targets.size(0)
            correct_train += (predicted == targets).sum().item()

            combined_preds.extend([pred_tensor.tolist() for pred_tensor in predicted])
            combined_targets.extend([target_tensor.tolist() for target_tensor in targets])

        train_accuracy = correct_train / total_train
        train_acc_list.append(train_accuracy)
        recall_score_list_train.append(recall_score(combined_targets, combined_preds, average='weighted'))
        f1_score_list_train.append(f1_score(combined_targets, combined_preds, average='weighted'))
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss / len(train_loader):.4f}, Train Accuracy: {100 * train_accuracy:.2f}%")

        # validation loop
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        combined_preds = []
        combined_targets = []
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = model.cost(outputs, targets)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_val += targets.size(0)
                correct_val += (predicted == targets).sum().item()

                combined_preds.extend([pred_tensor.tolist() for pred_tensor in predicted])
                combined_targets.extend([target_tensor.tolist() for target_tensor in targets])

        val_accuracy = correct_val / total_val
        valid_acc_list.append(val_accuracy)
        recall_score_list_valid.append(recall_score(combined_targets, combined_preds, average='weighted'))
        f1_score_list_valid.append(f1_score(combined_targets, combined_preds, average='weighted'))
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss / len(val_loader):.4f}, Validation Accuracy: {100 * val_accuracy:.2f}%")

        # simulate kaggle predictions here
        kaggle_preds_list = []
        with torch.no_grad():
          for inputs in test_loader:
            outputs = model(inputs[0])
            _, predicted = torch.max(outputs, 1)
            kaggle_preds_list.extend(predicted.tolist())
        for pred_idx in range(len(kaggle_preds_list)):
          for class_name, class_idx in class_map.items():
            if class_idx == kaggle_preds_list[pred_idx]:
              kaggle_preds_list[pred_idx] = class_name
              break
        pd.concat([kaggle_file_ids, pd.DataFrame(kaggle_preds_list, columns=['class'])], axis=1).to_csv(os.path.join(Path.cwd(), 'MLP', 'MLP-kaggle-preds.csv'), index=False)

    # plot train vs validation acc for each epoch
    plt.figure()
    plt.plot(list(range(num_epochs)), train_acc_list, marker='o', label='Train Accuracy')
    plt.plot(list(range(num_epochs)), valid_acc_list, marker='o', label='Validation Accuracy')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Train vs. Validation Accuracy (MLP)')
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(Path.cwd(), 'MLP', 'training_vs_valid_acc.png'))

    # print recall and f1 score
    plt.figure()
    plt.plot(list(range(num_epochs)), recall_score_list_train, marker='o', linestyle='dashed', color='blue', label='Train Recall Score')
    plt.plot(list(range(num_epochs)), recall_score_list_valid, marker='o', color='blue', label='Validation Recall Score')
    plt.plot(list(range(num_epochs)), f1_score_list_train, marker='o', linestyle='dashed', color='orange', label='Train F1 Score')
    plt.plot(list(range(num_epochs)), f1_score_list_valid, marker='o', color='orange', label='Validation F1 Score')
    plt.legend()
    plt.xlabel('Epoch')
    plt.title('Train vs. Validation Recall and F1 Scores (MLP)')
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(Path.cwd(), 'MLP', 'recall_and_f1.png'))


def train_mlp_raytune(config: Dict[str, any]) -> None:
    """ Function to perform hyperparameter searching on our MLP model using Ray Tune.

        Parameters:
            config: a dictionary of values being tested by Ray Tune for this specific
                training run
    """

    # fill the dataloaders according to the batch size being tested by Ray Tune
    batch_size=config['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # initialize MLP model
    input_size = X_train.shape[1]
    hidden_size = config['hidden_size']
    output_size = len(np.unique(y_train))
    model = MLP(input_size, hidden_size, output_size, dropout_rate=config['dropout_rate'])

    # define optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

    # Ttaining loop
    val_accuracy = 0
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = model.cost(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += targets.size(0)
            correct_train += (predicted == targets).sum().item()

        # validation loop
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = model.cost(outputs, targets)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_val += targets.size(0)
                correct_val += (predicted == targets).sum().item()
        val_accuracy = correct_val / total_val

        # save model checkpoints for use by Ray Tune
        with tempfile.TemporaryDirectory() as checkpoint_dir:

            checkpoint = Checkpoint.from_directory(checkpoint_dir)
            train.report(
                {"loss": val_loss, "accuracy": val_accuracy},
                checkpoint=checkpoint,
            )


if __name__ == '__main__':
    """ Main function which train the MLP via Ray Tune hyperparameter search
        and then creates a kaggle prediction file using the best performing
        model
    """

    # seed RNG for reproducible results
    torch.manual_seed(42)

    # Load and preprocess data
    X_train = np.load(os.path.join(processed_data_path, 'X_train.npy'))
    X_val = np.load(os.path.join(processed_data_path, 'X_test.npy'))
    y_train = np.load(os.path.join(processed_data_path, 'y_train.npy'))
    y_val = np.load(os.path.join(processed_data_path, 'y_test.npy'))
    X_kaggle = np.load(os.path.join(processed_data_path, 'X_kaggle.npy'))
    kaggle_file_ids = pd.read_csv(os.path.join(processed_data_path, 'kaggle_file_order.csv'))

    # combine classes to create mapping from genres to integers
    y_combined = np.append(y_train, y_val, axis=0)
    class_map = dict()
    for class_idx, class_name in enumerate(np.unique(y_combined)):
        class_map[class_name] = class_idx
    mapped_classes = np.array([class_map[value] for value in y_combined])
    y_train = mapped_classes[:len(y_train)]
    y_val = mapped_classes[len(y_train):]

    # Create PyTorch datasets and dataloaders
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_kaggle, dtype=torch.float32))

    num_epochs = 10

    # specify hyperparameters for Ray Tune to search
    hyperparameter_set = {
        'batch_size': tune.grid_search([8, 16, 32, 64]),
        'hidden_size': tune.grid_search([64, 128, 256]),
        'dropout_rate': tune.uniform(0.3, 0.5),
        'learning_rate': tune.loguniform(1e-3, 1e-1),
        'weight_decay': tune.loguniform(1e-5, 1e-2)
    }

    # create ASHA scheduler to be used by Ray Tune
    scheduler = ASHAScheduler(
        max_t=num_epochs,
        grace_period=1,
        reduction_factor=2)
    
    ray.init(
        configure_logging=True,
        logging_level='info',
        log_to_driver=False
    )

    # run the Ray Tune fitting
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_mlp_raytune),
            resources={'cpu': 2, 'gpu': 1}
        ),
        tune_config=tune.TuneConfig(
            metric='accuracy',
            mode='max',
            scheduler=scheduler,
            num_samples=15,
        ),
        param_space=hyperparameter_set,
    )
    results = tuner.fit()

    # print metrics on the best hyperparameter configuration
    best_result = results.get_best_result('accuracy', 'max')
    print('Best trial final validation loss: {}'.format(
        best_result.metrics['loss']))
    print('Best trial config: {}'.format(best_result.config))
    print('Best trial final validation accuracy: {}'.format(
        best_result.metrics['accuracy']))

    # now generate kaggle predictions using the config of the best performing model
    print('\n\n\nModel training complete. Generating kaggle file with best configuration...\n\n')
    train_mlp_kaggle(best_result.config)
