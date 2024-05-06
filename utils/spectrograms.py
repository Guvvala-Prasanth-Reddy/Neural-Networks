import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from utils.consts import *
import shutil
from PIL import Image
import sys

# frame size to use when generating STFTs
frame_size = 2048
# hop size to use when generating STFTs
hop_size = 256

def transform_image(path: str) -> Image:
    """ Returns an Image object from the path to an image file

        Parameters:
            path: the path to an image file

        Returns:
            the Image object corresponding to the provided path
    """

    with open(path, 'rb') as f:
        return Image.open(f).convert('RGB')
    

def generate_spectrograms(dataset_path: str) -> None:
    """ Function used to create spectrograms using a training dataset of audio files

        Parameters:
            dataset_path: path to an audio dataset where files are arranged
                in folders by classes
    """
    
    # perform folder setup
    sub_directories = os.listdir(dataset_path)
    if '.DS_Store' in sub_directories:
        sub_directories.remove('.DS_Store')
    if not os.path.exists(training_spectorgram_path):
         os.mkdir(training_spectorgram_path)

    # iterate over class folders
    for sub_directory in sub_directories:
        files = []
        
        if sub_directory.startswith('.DS') != True:
            files = os.listdir(os.path.join(dataset_path, sub_directory ))
        if os.path.exists(os.path.join(training_spectorgram_path, sub_directory)):
                    shutil.rmtree(os.path.join(training_spectorgram_path , sub_directory))
        os.makedirs(os.path.join(training_spectorgram_path, sub_directory))
            
        # iterate over all files per class
        for filename in files:
            if filename.endswith('.au'):
                audio, sample_rate = librosa.load(os.path.join(dataset_path, sub_directory, filename))
                stft_audio = librosa.stft(audio, n_fft=frame_size, hop_length=hop_size)
                y_audio = np.abs(stft_audio) ** 2
                y_log_audio = librosa.power_to_db(y_audio, ref=np.max)
                plot_spectrogram(y_log_audio, sample_rate, hop_size, 'log', os.path.join(training_spectorgram_path, sub_directory, f'{filename}-amp-db.png'))                


def generate_spectrograms_kaggle(path: str) -> None:
    """ Creates a folder of spectrograms using the provided path to a dataset of test
            (unlabeled) audio files

        Parameters:
            path: the path to the test dataset of audio files
    """
    
    if not os.path.exists(kaggle_spectrogram_path):
        os.mkdir(kaggle_spectrogram_path)

    for filename in os.listdir(path):
        if filename.endswith('.au'):
            audio, sample_rate = librosa.load(os.path.join(path, filename))
            stft_audio = librosa.stft(audio, n_fft=frame_size, hop_length=hop_size)
            y_audio = np.abs(stft_audio) ** 2
            
            y_log_audio = librosa.power_to_db(y_audio, ref=np.max)
            plot_spectrogram(y_log_audio, sample_rate, hop_size, 'log', os.path.join(kaggle_spectrogram_path, 'kaggle', f'{filename}.png'))


def plot_spectrogram(y: np.ndarray, sr: int, hop_length: int, y_axis, file_path: str, mode:str='training') -> plt.Figure:
    """ This function plots a spectrogram using the provided audio samples and 

        Parameters:
            y: an array of audio samples
            sr: the sample rate of the audio data
            hop_length: the hop length used on the audio data
            y_axis: the y-axis scale, either 'log' or 'linear'
            file_path: the path to which the generated spectrogram file should be saved
            mode: 

    """

    figure, axis = plt.subplots(figsize=(5.12, 5.12))
    librosa.display.specshow(y, sr=sr, hop_length=hop_length, x_axis='time', y_axis=y_axis, cmap='viridis')
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_xlabel('')
    axis.set_ylabel('')
    figure.tight_layout(pad=0)

    filepath_dir = os.path.split(file_path)[0]
    if not os.path.isdir(filepath_dir):
        os.makedirs(filepath_dir)

    figure.savefig(file_path)
    plt.close(figure)


if __name__ == "__main__":
    """ Main method for testing
    """

    generate_spectrograms_kaggle(testing_data_path)

