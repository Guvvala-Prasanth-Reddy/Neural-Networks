from  CNN.training import CNN , transform_image 
from utils.spectrograms import generate_spectrograms_kaggle
from utils.consts import *
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import os
import torch
from PIL import Image
import numpy as np
import pandas as pd

class UnlabelledImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        """
        Args:
            image_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image


def load_model(model_path , input_size , hidden_size , output_size  ):
    model = model = CNN(input_size, hidden_size, output_size , kernel_size_1= 3, kernel_size_2= 3 )
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def prediction(model , X_test , classes):
    target_actual  = []
    target_predicted = []
    loader = DataLoader(X_test, batch_size=32, shuffle=False)
    with torch.no_grad(): 
        for inputs in loader:
            predicted_outputs = model(inputs)
            _, predicted = torch.max(predicted_outputs.data, 1)
            # target_actual.extend( actual_targets.view(actual_targets.size(0), -1).tolist() )
            target_predicted.extend(  np.take(classes , predicted.tolist()))
            print( np.take(classes , predicted.tolist() ))
    return target_predicted

def test_model():
    input_size = 0
    hidden_size = 64
    output_size = 10
    transform = transforms.Compose([transforms.ToTensor() ])
    X = datasets.ImageFolder( 'feature_files/' , transform = transform , loader = transform_image)
    classes = X.classes
    X_test = UnlabelledImageDataset( kaggle_pred_dir , transform = transform , )
    model = load_model('models/models-accuracy100.0' ,  input_size , hidden_size , output_size)
    predicted = prediction(model , X_test , classes)
    files = os.listdir('data/test/')
    df = pd.DataFrame()
    df['id'] = files
    df['class'] = predicted
    df.to_csv('outputs.csv' , index = False)


if __name__ == '__main__':
    test_model()