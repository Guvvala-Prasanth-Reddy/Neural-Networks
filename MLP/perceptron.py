import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import time
from utils.consts import *

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.2):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.cost = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Load and preprocess data
#wine = load_wine()
#X, y = wine.data, wine.target
#scaler = StandardScaler()
#X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
#X_train = scaler.fit_transform(X_train)
#X_test_val = scaler.transform(X_test_val)

# skip all the above because it's done in our data processing pipeline?

# read our data instead...
X_train = np.load( processed_data+ 'X_train.npy')
X_test_val = np.load(processed_data + 'X_test.npy')
y_train = np.load(processed_data + 'y_train.npy')
y_test_val = np.load(processed_data + 'y_test.npy')

print('Shape of X_train: ' + str(X_train.shape))
print('Shape of X_test: ' + str(X_test_val.shape))
print('Shape of y_train: ' + str(y_train.shape))
print('Shape of y_test: ' + str(y_test_val.shape))

# convert our labels from strings to integers since tensors don't like strings

# create a mapping from string labels to numerical labels
train_label_map = {label: index for index, label in enumerate(np.unique(y_train))}
test_label_map = {label: index for index, label in enumerate(np.unique(y_test_val))}

# convert string labels to numerical labels using the mapping
y_train = np.array([train_label_map[label] for label in y_train])
y_test_val = np.array([test_label_map[label] for label in y_test_val])

print('Shape of y_train: ' + str(y_train.shape))
print('Shape of y_test: ' + str(y_test_val.shape))

X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.5, random_state=42, stratify=y_test_val)

# Create PyTorch datasets and dataloaders
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

# begin hyperparameter search here
# tested hyperparameters:
#   batch size: [8, 6, 32, 64]
#   hidden layer width: [16, 32, 64, 128]
#   learning rate: [0.1, 0.01, 0.001, 0.0001]
#   weight decay [0, 0.01, 0.1]

for batch_size in [8, 16, 32, 64]:
    for hidden_size in [16, 32, 64, 128]:
        for lr in [0.0001, 0.001, 0.01, 0.1]:
            for weight_decay in [0, 0.01, 0.1]:

                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size)
                test_loader = DataLoader(test_dataset, batch_size=batch_size)

                # Initialize MLP model
                input_size = X_train.shape[1]
                output_size = len(np.unique(y_train))  # Number of classes

                model = MLP(input_size, hidden_size, output_size)
                print(model)

                # Define optimizer and loss function
                optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

                print('Training beginning with the following parameter set (10 epochs, 1 hidden layer):')
                print(f'\tbatch size={batch_size}, hidden layer size={hidden_size}, learning rate={lr}, weight decay={weight_decay}\n')
                t0 = time.time()

                # Training loop
                num_epochs = 10
                for epoch in range(num_epochs):
                    # Training
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

                    train_accuracy = correct_train / total_train
                    print(f"\tEpoch {epoch+1}/{num_epochs}, Train Loss: {train_loss / len(train_loader):.4f}, Train Accuracy: {100 * train_accuracy:.2f}%")

                    # Validation
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
                    print(f"\tEpoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss / len(val_loader):.4f}, Validation Accuracy: {100 * val_accuracy:.2f}%")

                    elapsed_time = time.time() - t0
                    print(f'Parameter set training and validation completed in {elapsed_time} seconds\n\n')