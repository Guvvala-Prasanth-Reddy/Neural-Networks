import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from utils.consts import *
import shutil
from PIL import Image

def transform_image(path):
    # Open an image file, ensuring it is read in RGBA mode
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def generate_spectrograms( path  ):
    directory = path
    sub_directories = os.listdir(path)
    if( '.DS_Store' in sub_directories):
        sub_directories.remove('.DS_Store')
    if( os.path.exists(feature_files) != True):
         os.mkdir(feature_files)
    for sub_directory in sub_directories:
        files = []
        
        if sub_directory.startswith('.DS') != True :
            files  = os.listdir(os.path.join(directory , sub_directory ))
        if( os.path.exists(os.path.join(feature_files , sub_directory))):
                    shutil.rmtree(os.path.join(feature_files , sub_directory))
        os.makedirs( feature_files +"/"+ sub_directory)
            
        print(sub_directory)
        for file in files:
            if( file.endswith('.au')):
                audio, sample_rate = librosa.load(os.path.join(directory , sub_directory , file))
                stft_audio = librosa.stft(audio, n_fft = frameSize, hop_length = hopSize)
                # print(stft_audio.shape , 'stft')
                y_audio = np.abs(stft_audio) ** 2
                # abs =  plot_spectrogram(y_audio, sample_rate, hopSize , "linear" , file)
                # abs.savefig( os.path.join( feature_files  , sub_directory ,  f"{file}-abs.png"))
                # plt.close(abs)
                y_log_audio = librosa.power_to_db(y_audio , ref = np.max)
                # print(y_log_audio.shape)
                
                amplitude_to_db =  plot_spectrogram(y_log_audio, sample_rate, hopSize, "log" ,file , sub_directory)                
                
                plt.close(amplitude_to_db)

def generate_spectrograms_kaggle(path):
    files = os.listdir(path)
    directory = kaggle_pred_dir
    if( os.path.exists(directory) != True):
        os.mkdir(directory)
    for file in files:
            if( file.endswith('.au')):
                audio, sample_rate = librosa.load(os.path.join(path , file))
                stft_audio = librosa.stft(audio, n_fft = frameSize, hop_length = hopSize)
                y_audio = np.abs(stft_audio) ** 2
                
                # abs =  plot_spectrogram(y_audio, sample_rate, hopSize , "linear" , file)
                # abs.savefig( os.path.join( feature_files  , sub_directory ,  f"{file}-abs.png"))
                # plt.close(abs)
                y_log_audio = librosa.power_to_db(y_audio , ref = np.max)
                # print(y_log_audio.shape)
                amplitude_to_db = plot_spectrogram(y_log_audio, sample_rate, hopSize, "log" ,file)                
                amplitude_to_db.savefig(os.path.join(kaggle_pred_dir , f"{file}.png"))
                plt.close(amplitude_to_db)

def generate_spectrograms_validation(path):
    directory = path
    sub_directories = os.listdir(path)
    if( '.DS_Store' in sub_directories):
        sub_directories.remove('.DS_Store')
    if( os.path.exists(validation_dir) != True):
         os.mkdir(validation_dir)
    for sub_directory in sub_directories:
        files = []
        
        if sub_directory.startswith('.DS') != True :
            files  = os.listdir(os.path.join(directory , sub_directory ))
        if( os.path.exists(os.path.join(validation_dir , sub_directory))):
                    shutil.rmtree(os.path.join(validation_dir , sub_directory))
        os.makedirs( validation_dir +"/"+ sub_directory)
            
        print(sub_directory)
        for file in files:
            if( file.endswith('.au')):
                audio, sample_rate = librosa.load(os.path.join(directory , sub_directory , file))
                stft_audio = librosa.stft(audio, n_fft = frameSize, hop_length = hopSize)
                # print(stft_audio.shape , 'stft')
                y_audio = np.abs(stft_audio) ** 2
                # abs =  plot_spectrogram(y_audio, sample_rate, hopSize , "linear" , file)
                # abs.savefig( os.path.join( feature_files  , sub_directory ,  f"{file}-abs.png"))
                # plt.close(abs)
                y_log_audio = librosa.power_to_db(y_audio , ref = np.max)
                # print(y_log_audio.shape)
                
                amplitude_to_db =  plot_spectrogram(y_log_audio, sample_rate, hopSize, "log" ,file , sub_directory , mode ="validation")
                amplitude_to_db.savefig(os.path.join(validation_dir , sub_directory , f"{file}.png"))                
                plt.close(amplitude_to_db)



def plot_spectrogram(y, sr, hop_length, y_axis , file_name   ,sub_directory='' , mode = "training"):

    print(y.shape)

    frames = y.shape[1]
    if( mode == "validation"):
        figure , axis =  plt.subplots(figsize = (5.12,5.12))
        image  = librosa.display.specshow(y, sr = sr, hop_length = hop_length ,  x_axis = "time", y_axis = y_axis , cmap='viridis')
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_xlabel('')
        axis.set_ylabel('')
        figure.tight_layout(pad=0)
        return figure
    if( sub_directory == ''):
        figure , axis =  plt.subplots(figsize = (5.12,5.12))
        image  = librosa.display.specshow(y, sr = sr, hop_length = hop_length ,  x_axis = "time", y_axis = y_axis , cmap='viridis')
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_xlabel('')
        axis.set_ylabel('')
        figure.tight_layout(pad=0)
        return figure

    frames_per_file = frames//3
    for i in range(3):
        frame_starting = i*frames_per_file
        frame_ending = (i+1)*frames_per_file
        figure , axis =  plt.subplots(figsize = (5.12,5.12))
        image  = librosa.display.specshow(y[: , frame_starting : frame_ending], sr = sr, hop_length = hop_length ,  x_axis = "time", y_axis = y_axis , cmap='viridis')
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_xlabel('')
        axis.set_ylabel('')
        figure.tight_layout(pad=0)
        if sub_directory != '':
            figure.savefig(os.path.join(feature_files ,  sub_directory , f"{file_name}-part-{i}-amp-db.png"))
            plt.close(figure)
        else:
            return figure
    
    
    
    
    # figure.colorbar(image , format="%+2.f" , ax = axis )
    # axis.set_title(f'{file_name}')
   

if __name__ == "__main__":
    generate_spectrograms('data/train/')

