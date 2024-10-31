import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import fftpack
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz, resample

import torch
from torch.utils.data import Dataset, DataLoader
from glob import glob
import os

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader, random_split
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import time

x = torch.tensor(np.random.random((2, 12, 14))).to(torch.float32)

class MaxPoolTransformerModel(nn.Module):
    def __init__(self, input_dim, cnn_out_channels, transformer_dim, seq_len, output_dim):
        super(MaxPoolTransformerModel, self).__init__()
        
        self.maxpool1d = nn.MaxPool1d(2)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=cnn_out_channels, nhead=1), num_layers=1
        )
        self.fcClass = nn.Sequential(nn.Flatten(), nn.Linear(84, 2))
    
    def forward(self, x):         
        x = self.maxpool1d(x)
        x = self.transformer(x)
        x = x.view(x.shape[0], -1) 
        classed = self.fcClass(x)
        
        return classed
    
input_dim = 12        
cnn_out_channels = 7
seq_len = 66              
output_dim = 500    

model = MaxPoolTransformerModel(input_dim, cnn_out_channels, cnn_out_channels, seq_len, output_dim)
input_tensor = x

start_time = time.time()
output_tensor = model(input_tensor)
output_tensor_new = torch.argmax(output_tensor, axis = 1)
print(output_tensor_new, time.time() - start_time) 

