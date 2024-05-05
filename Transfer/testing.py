from Transfer.training import Transfer
from CNN.testing import UnlabelledImageDataset,prediction
from utils.consts import *
from torchvision import datasets, transforms
from  CNN.training import transform_image
import torch 
import os
import pandas as pd
def load_model(model_path , input_size , hidden_size , output_size  ):
    model =  Transfer(input_size, hidden_size, output_size , kernel_size_1= 3, kernel_size_2= 3 , num_classes=10)
    model.load_state_dict(torch.load(model_path))
    # model= model.load_from_checkpoint(model_path)
    model.eval()
    return model

def test_model():
    input_size = 0
    hidden_size = 64
    output_size = 50 
    # generate_spectrograms_kaggle('data/test/')
    transform = transforms.Compose([transforms.Resize((224, 224)) ,transforms.ToTensor() ])
    X = datasets.ImageFolder( 'feature_files/' , transform = transform , loader = transform_image)
    classes = X.classes
    X_test = UnlabelledImageDataset( kaggle_pred_dir , transform = transform , )
    model = load_model('models/models-95.3125' ,  input_size , hidden_size , output_size)
    predicted = prediction(model , X_test , classes)
    files = os.listdir('data/test/')
    df = pd.DataFrame()
    df['id'] = files
    df['class'] = predicted
    df.to_csv('outputs.csv' , index = False)



if __name__ == '__main__':
    test_model()