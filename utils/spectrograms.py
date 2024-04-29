import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from utils.consts import *
import shutil

def generate_spectrograms( path ):
    directory = path
    sub_directories = os.listdir(path)
    if( '.DS_Store' in sub_directories):
        sub_directories.remove('.DS_Store')
    if( os.path.exists(path) != True):
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
                y_audio = np.abs(stft_audio) ** 2
                abs =  plot_spectrogram(y_audio, sample_rate, hopSize , "linear" , file)
                abs.savefig( os.path.join( feature_files  , sub_directory ,  f"{file}-abs.png"))
                plt.close(abs)
                y_log_audio = librosa.power_to_db(y_audio)
                amplitude_to_db =  plot_spectrogram(y_log_audio, sample_rate, hopSize, "log" ,file)                
                amplitude_to_db.savefig(os.path.join(feature_files ,  sub_directory , f"{file}-amp-db.png"))
                plt.close(amplitude_to_db)


def plot_spectrogram(y, sr, hop_length, y_axis , file_name ):
    figure , axis =  plt.subplots(figsize = (25,10))
    image  = librosa.display.specshow(y, sr = sr, hop_length = hop_length)
    
    # figure.colorbar(image , format="%+2.f" , ax = axis )
    # axis.set_title(f'{file_name}')
    axis.set_xticks([])
    axis.set_yticks([])
    figure.tight_layout(pad=0)
    return figure

if __name__ == "__main__":
    generate_spectrograms('data/train/')

