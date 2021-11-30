#################################################
# ML Model Functions and Class
#################################################
import numpy as np

import tensorflow
import joblib
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_

def init_weights(m):
    try:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    except Exception:
        return
        
class MLP(torch.nn.Module):
    # define model elements
    # n_inputs = 64
    def __init__(self, n_inputs):
        super(MLP, self).__init__()    
        self.base_model = torch.nn.Sequential(
            torch.nn.Linear(n_inputs, 544),
            torch.nn.ReLU(),
            torch.nn.Dropout(.2),
            torch.nn.Linear(544, 272),
            torch.nn.Dropout(.2),
            torch.nn.ReLU(),
            torch.nn.Linear(272, 136),
            torch.nn.Dropout(.2),
            torch.nn.ReLU(),
            torch.nn.Linear(136, 68),
            torch.nn.Dropout(.2),
            torch.nn.ReLU(),
            torch.nn.Linear(68, 1),
            ).to(torch.device("cpu"))
        self.base_model.apply(init_weights)

    # forward propagate input
    def forward(self, X):
        X = X.to(torch.device("cpu"))
        # input to first hidden layer
        X = self.base_model(X)
        return X
