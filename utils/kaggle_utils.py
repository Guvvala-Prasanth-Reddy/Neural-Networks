from typing import List
import torch
import numpy as np
from torch.utils.data import DataLoader


def prediction(model, X_test, classes) -> List[str]:
    """ File returns predicted targets of the provided features
        using the provided model

        Parameters:
            model: a trained model
            X_test: features belonging to unlabeled data
            classes: the possible classes in the dataset
    """
    
    target_predicted = []
    loader = DataLoader(X_test, batch_size=32, shuffle=False)
    with torch.no_grad(): 
        for inputs in loader:
            predicted_outputs = model(inputs)
            _, predicted = torch.max(predicted_outputs.data, 1)
            target_predicted.extend(np.take(classes, predicted.tolist()))
    return target_predicted