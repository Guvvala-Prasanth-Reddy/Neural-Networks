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

### Train MLP

1. Specify values for the variables below in `utils/consts.py`:
   - `training_data_path`: the path to your training data directory (audio files)
   - `testing_data_path`: the path to your kaggle testing data directory (audio files)

2. Run `python -m MLP.training` from the top level directory.
- How to address audio data?
- A hyperparameter search will be started with Ray Tune; this will take a while.
- After training is complete, a model with best validation accuracy will be saved to `models/trained_model`.

3. Generate predictions on your test data using the trained model: `python -m MLP.testing`.
- A file will be generated containing kaggle predictions and saved as `kaggle_predictions.csv` in the top level directory.

### Train CNN

1. Specify values for the variables below in `utils/consts.py`:
   - `training_data_path`: the path to your training data directory (audio files)
   - `testing_data_path`: the path to your kaggle testing data directory (audio files)

2. Run `python -m CNN.training` from the top level directory.
- Spectrograms will be automatically generated for your training data; this will take a short while.
- A hyperparameter search will be started with Ray Tune; this will take a long while.
- After training is complete, a model with best validation accuracy will be saved to `models/trained_model`.

3. Generate predictions on your test data using the trained model: `python -m CNN.testing`.
- A file will be generated containing kaggle predictions and saved as `kaggle_predictions.csv` in the top level directory.

### Code Manifest
| File Name | Description |
| --- | --- |
| `CNN/training.py` | This file contains the training algorithm for the CNN. |
| `CNN/testing.py` | This file generates kaggle predictions using a trained CNN model.  |
| `MLP/training.py` | This file contains the training algorithm for the MLP.  |
| `MLP/testing.py` | This file generates kaggle predictions using a trained CNN model.  |
| `Transfer/testing.py` | This file contains the training algorithm for our transfer learning CNN.  |
| `Transfer/training.py` | This file generates kaggle predictions using a trained transfer learning CNN.  |
| `utils/consts.py` | This file has constants used throughout the library.  |
| `utils/spectrograms.py` | This file contains our functions to convert audio files to spectrograms. |
| `requirements.txt` | This file contains our project's Python dependencies. |


## Developer Contributions

Prasanth Reddy Guvvala
- Implemented CNN architecture.
- Implemented spectrogram transforms.
- Implemented transfer learning.
- Impelemented data augmentation.

Thomas Fisher
- Implemented MLP architecture.
- Implemented Ray Tune hyperparameter searching.
- Implemented model evaluation.
- Implemented report framework.

## kaggle Submission

Leaderboard position XX achieved with accuracy YY on May 6th (team name: Fisher & Guvvala).
