# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 18:13:31 2025

@author: jiaxx213
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

import pickle
import sys

#from functions_gamma import Gamma_Coeff
from functions_basic import data_split_path, LITT_path

#class Gamma_Coeff(nn.Module):
#...
#########################
# The core code is hidden due to copyright issue
# Contact the author for full version
#########################
    
class Dataset_Container(Dataset):
    def __init__(self, input_list):
        n_data = len(input_list)
        self.n_data = n_data
        
        common_len = len(input_list[0])
        self.data_list = [[] for _ in range(n_data)]
        for i, item_data in enumerate(input_list):
            self.data_list[i] = item_data
            assert(len(item_data) == common_len)
            
        self.common_len = common_len

    def __getitem__(self, idx):
        idx_data_list = []
        for item_data in self.data_list:
            idx_data_list.append(item_data[idx])
        return idx_data_list


#class Data_Manager:
#...
#########################
# The core code is hidden due to copyright issue
# Contact the author for full version
#########################


