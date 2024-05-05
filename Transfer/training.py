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

### Flags and treshold ###
Accuracy_flag = False
Treshold = 80.00
validation_accuracies = []
train_accuracies = []

def transform_image(path):
    # Open an image file, ensuring it is read in RGBA mode
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
    
class Transfer(pl.LightningModule):
    def __init__(self, input_size, hidden_size, output_size,  kernel_size_1 , kernel_size_2 , num_classes ,  dropout_rate=0.5  , freeze = True ):
        super(Transfer , self).__init__()
        self.accuracy_flag = False
        self.model= models.vgg16(pretrained = True)
        last_layer_features = self.model.classifier[(len(self.model.classifier)-1)].in_features


        if freeze:
            for parameter in self.model.features.parameters():
                parameter.requires_grad = False
        self.model.classifier[-1] = nn.Linear(last_layer_features , num_classes)
        # self.conv_layer_1 = nn.Conv2d(in_channels=3 , out_channels= 32 , kernel_size= kernel_size_1 , padding = 1 )
        # self.max_pooling = nn.MaxPool2d(kernel_size=2 )
        # self.conv_layer_2 = nn.Conv2d(in_channels=32 , out_channels= 64 ,kernel_size= kernel_size_2 , padding = 1 )
        # self.avg_pooling = nn.AvgPool2d( kernel_size= 2)
        # # self.conv_layer_3 = nn.Conv2d(in_channels= 32 , out_channels= 64 , kernel_size= 3 , stride = 1 , padding = 2)
        # # self.conv_layer_4 = nn.Conv2d(in_channels = 64 , out_channels = 128 , kernel_size = 3 , stride = 1 , padding = 2)
        # # self.conv_layer_5 = nn.Conv2d(in_channels = 128 , out_channels = 256 , kernel_size = kernel_size_2)





        # self.fc1 = nn.Linear(64*128*128 , 256 )
        # self.fc2 = nn.Linear(256,  10)
        # self.relu = nn.ReLU()
        self.cost = nn.CrossEntropyLoss()
        # self.softmax = nn.Softmax()
        # self.dropout_0 = nn.Dropout(0.3)
        # self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        return self.model(x)
        # print(x.shape , "input shape")
        # x = self.max_pooling(self.relu(self.conv_layer_1(x)))
        # # print(x.shape , "conv 1")
        # x = self.max_pooling(self.relu(self.conv_layer_2(x)))
        # # print(x.shape , "conv 2")
        # # x = self.max_pooling(self.relu(self.conv_layer_3(x)))
        # # x = self.max_pooling(self.relu(self.conv_layer_4(x)))
        # # x = self.max_pooling(self.relu(self.conv_layer_5(x)))

        # # print(x.shape , "conv 3")
        # x = torch.flatten(x  , 1)
        # # print(x.shape)
        # x = self.dropout_0(x)
        # x = self.relu(self.fc1(x))
        # x = self.dropout(x)
        # x = self.fc2(x)
        # x = self.relu(self.fc2(x))
        # x = self.dropout(x)
        # x = self.fc3(x)
        # x = self.dropout(x)
        # x = self.fc4(x)
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
    
    def save(self , val_acc):
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

    def configure_optimizers(self):
        parameters_training = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        optimizer = optim.Adam(parameters_training, lr=0.001, weight_decay=0.00001)
        return optimizer

def run_train(training_indices , validation_indices , X ,  batch_size , model):
    k_fold_train  , k_fold_validation = Subset(X , training_indices ) , Subset(X , validation_indices)
    train_loader , val_loader = DataLoader( k_fold_train ,  batch_size = batch_size , shuffle = True) , DataLoader(k_fold_validation , batch_size = batch_size , shuffle = True)
    trainer = pl.Trainer( precision = 16,max_epochs=5,logger=CSVLogger(save_dir="logs/") , log_every_n_steps=10)
    trainer.fit(model, train_loader, val_loader)





if __name__ == '__main__':
    torch.cuda.empty_cache()
# Load and preprocess data
    wine = load_wine()
    transform = transforms.Compose([ transforms.Resize((224, 224)) ,transforms.ToTensor() ])
    

    # This went horrible , transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    #  std=[0.229, 0.224, 0.225])]
    #Normalization source :- https://discuss.pytorch.org/t/why-image-datasets-need-normalizing-with-means-and-stds-specified-like-in-transforms-normalize-mean-0-485-0-456-0-406-std-0-229-0-224-0-225/187818/2
    # generate_spectrograms('data/train/')
    # generate_spectrograms_validation('data/train/')
    Validation = datasets.ImageFolder( validation_dir , transform = transform , loader = transform_image)
    X = datasets.ImageFolder( 'feature_files/' , transform = transform , loader = transform_image)
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
    
    batch_size=32
    # train_size = int(0.8 * len(X))
    # val_size = len(X) - train_size
    # train_dataset, val_dataset = random_split(X, [train_size, val_size])
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True , num_workers=4)
    # val_loader = DataLoader(train_dataset, batch_size=batch_size , num_workers=4 )
    # test_loader = DataLoader(test_dataset, batch_size=batch_size)
    # for images, labels in train_loader:
    #     print("Label data type:", labels.dtype)  # Check if labels are Long type
    #     break

    # Initialize MLP model
    input_size = 0
    hidden_size = 64
    output_size = 50  # Number of classes

    # model = CNN(input_size, hidden_size, output_size , kernel_size_1= 3, kernel_size_2= 3 )
    model = Transfer(input_size, hidden_size, output_size , kernel_size_1= 3, kernel_size_2= 3, num_classes = 10 )
    print(model)

    # Train the model using PyTorch Lightning Trainer
    # trainer = pl.Trainer( precision = 16,max_epochs=100,logger=CSVLogger(save_dir="logs/") , log_every_n_steps=10)
    # trainer.fit(model, train_loader, val_loader)

    # Test the model
    # test1 = trainer.test(model, dataloaders=test_loader)
    num_of_folds = 10
    labels =  np.array([x[1] for x in X.imgs])
    kfold_cross_validation = StratifiedKFold(n_splits = num_of_folds , shuffle = True)
    print(X)
    labels_validation = np.array(x[1] for x in Validation.imgs)

    splits = kfold_cross_validation.split(X , labels )
    # validation_splits = kfold_cross_validation( validation , labels_validation)
    # combination = (splits[0] , validation_splits[1])

    for training_indices , validation_indices in splits :
            run_train( training_indices , validation_indices , X , batch_size , model)
    generate_spectrograms_kaggle('data/test/')  