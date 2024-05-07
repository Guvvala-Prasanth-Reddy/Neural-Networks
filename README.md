# Neural-Networks

## Description

UNM CS 529 Project 3: Creation of Multilayer Perceptron (MLP) and Convolutional Neural Network (CNN) classifiers using Pytorch and Pytorch-Lightning libraries respectively.

## Instructions for Use

### Install Dependencies

Create a Python virtual environment
```bash
python -m venv YOURVENV
```

Activate the environment by running this command on Windows...
```bash
YOURENV/Scripts/activate
```
or this command on Linux/MacOS
```bash
source ./YOURENV/bin/activate
```

Install required dependecies with the command below (all platforms):
```bash
pip install -r requirements.txt
```

Note that some of the dependencies require Python 3.10 to install correctly and will not install on more recent versions of Python.

### Train MLP

1. Specify values for the variables below in `utils/consts.py`:
   - `training_data_path`: the path to your training data directory (audio files)
   - `testing_data_path`: the path to your kaggle testing data directory (audio files)

2. Run `python -m MLP.training` from the top level directory.
- A hyperparameter search will be started with Ray Tune; this will take a while.
- After training is complete, the best performing model will be used to generate a predictions file in `MLP/`.

### Train CNN

1. Specify values for the variables below in `utils/consts.py`:
   - `training_data_path`: the path to your training data directory (audio files)
   - `testing_data_path`: the path to your kaggle testing data directory (audio files)

2. Run `python -m CNN.training` from the top level directory.
- A hyperparameter search will be started with Ray Tune; this will take a while.
- After training is complete, the best performing model will be used to generate a predictions file in `CNN/`.

### Train Transfer Learning

1. Specify values for the variables below in `utils/consts.py`:
   - `training_data_path`: the path to your training data directory (audio files)
   - `testing_data_path`: the path to your kaggle testing data directory (audio files)

2. Run `python -m Transfer.training` from the top level directory.
- A hyperparameter search will be started with Ray Tune; this will take a while.
- After training is complete, the best performing model will be used to generate a predictions file in `Transfer/`.

### Code Manifest
| File Name | Description |
| --- | --- |
| `CNN/training.py` | This file contains the training algorithm for the CNN. |
| `MLP/training.py` | This file contains the training algorithm for the MLP.  |
| `Transfer/training.py` | This file contains the training algorithm for our transfer learning model.  |
| `utils/consts.py` | This file has constants used throughout the library.  |
| `utils/spectrograms.py` | This file contains our functions to convert audio files to spectrograms. |
| `requirements.txt` | This file contains our project's Python dependencies. |


## Developer Contributions

Prasanth Reddy Guvvala
- Implemented CNN architecture.
- Implemented spectrogram transforms.
- Implemented transfer learning.
- Implemented data augmentation.

Thomas Fisher
- Implemented MLP architecture.
- Implemented Ray Tune hyperparameter searching.
- Implemented model evaluation.
- Implemented report framework.

## kaggle Submission

Leaderboard position 11 achieved with accuracy 68% on May 6th (team name: Fisher & Guvvala).
