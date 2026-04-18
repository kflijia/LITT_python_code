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

class Gamma_Coeff(nn.Module):
    def __init__(self, lengths, hidden_size):
        super().__init__()
        self.lengths = lengths
        self.n_seqs = len(lengths)
        self.seq_idxs = range(self.n_seqs)
        self.hidden_size = hidden_size
        device = lengths.device
        
        self.W_params = nn.ParameterList([nn.Parameter(torch.empty(seq_len, hidden_size).to(device))  for seq_len in self.lengths])
        self.b_params = nn.ParameterList([nn.Parameter(torch.empty(seq_len, hidden_size).to(device))  for seq_len in self.lengths])
        
        # initialize
        for i, seq_len in enumerate(self.lengths):
            self.W_params[i].data.uniform_(0.0, 0.01)
            self.b_params[i].data.zero_()

    def forward(self, time_sequences, indices):
        device = time_sequences[0].device
        out_coeffs = []
        for i, seq_idx in enumerate(indices):
            t_seq = time_sequences[i]
            W_param = self.W_params[seq_idx]
            b_param = self.b_params[seq_idx]
            x_gamma = t_seq * W_param + b_param  # x_time broadcast to (seq_len, hidden_size)
            gamma_zeros = torch.zeros_like(x_gamma).to(device)
            coeff = torch.exp(-torch.max(gamma_zeros, x_gamma))    # 0.0 < coeff <=1.0
            out_coeff = 1 - coeff    # 1.0 > out_coeff >= 0.0
            out_coeffs.append(out_coeff)
        
        return out_coeffs
    
    # --- loading and get W & b ---
    def get_gamma_state(self, in_indices=None):
        if in_indices is None:
            in_indices = self.seq_idxs
        
        gamma_dict = self.state_dict()
        gamma_keys = list(gamma_dict.keys())
        dict_half_len = int(len(gamma_dict)/2)
        W_keys = gamma_keys[:dict_half_len]
        b_keys = gamma_keys[-dict_half_len:]
        copy_W_dict = {}
        copy_b_dict = {}

        for i in in_indices:
            W_k = W_keys[i]
            b_k = b_keys[i]
            copy_W_dict[W_k] = gamma_dict[W_k].detach().clone()
            copy_b_dict[b_k] = gamma_dict[b_k].detach().clone()
        gamma_state_dict = {'W_state': copy_W_dict, 'b_state': copy_b_dict}
        return gamma_state_dict
    
    def load_gamma_state(self, in_dict, in_indices=None):
        W_dict = in_dict['W_state']
        b_dict = in_dict['b_state']
        if in_indices is None:
            in_indices = self.seq_idxs
            
        assert(len(W_dict) == len(b_dict) and len(W_dict) == len(in_indices))
        for i, idx in enumerate(in_indices):
            assert(str(idx) in list(W_dict.keys())[i])
            self.W_params[idx].data.copy_(list(W_dict.values())[i])
            self.b_params[idx].data.copy_(list(b_dict.values())[i])
        return
    
    def compare_gamma_state(self, in_dict):
        init_W_dict = in_dict['W_state']
        init_b_dict = in_dict['b_state']
        cur_gamma_dict = self.state_dict()
        
        W_tags = [not torch.equal(cur_gamma_dict[key], value)  for key, value in init_W_dict.items()]
        b_tags = [not torch.equal(cur_gamma_dict[key], value)  for key, value in init_b_dict.items()]
        W_rate = sum(W_tags)/len(W_tags)
        b_rate = sum(b_tags)/len(b_tags)
        return W_rate, b_rate
    
    # --- inner computing between W & b and coeff ---
    def apply_coeff(self, coeffs, time_sequences, indices):
        for i, seq_idx in enumerate(indices):
            if torch.is_tensor(seq_idx):
                seq_idx = seq_idx.item()
            
            x_coeff = 1- coeffs[i]
            x_gamma = -torch.log(x_coeff) # >=0  (seq_len, hidden_size)
            t_seq = time_sequences[i]   # (seq_len, 1)
            assert(t_seq.size(0)==x_gamma.size(0))
            
            for j, x_t in enumerate(t_seq):
                if x_t == 0:
                    self.b_params[seq_idx][j].data.copy_(x_gamma[j])
                    self.W_params[seq_idx][j].data.copy_(torch.zeros(self.hidden_size))
                else:
                    self.W_params[seq_idx][j].data.copy_(x_gamma[j] / x_t)
                    self.b_params[seq_idx][j].data.copy_(torch.zeros(self.hidden_size))
        return
    
    def zero_coeff(self, indices):
        with torch.no_grad():
            for seq_idx in indices:
                self.W_params[seq_idx] = torch.zeros_like(self.W_params[seq_idx])
                self.b_params[seq_idx] = torch.zeros_like(self.b_params[seq_idx])


    
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


class Data_Manager:
    def __init__(self, sequences, time_sequences, oc_targets, hidden_size, n_batch, device):
        assert(len(sequences) == len(time_sequences))
        self.sequences = [seq.to(device) for seq in sequences]
        self.time_sequences = [t_seq.to(device) for t_seq in time_sequences]
        self.merged_sequences = [torch.cat((seq, time_sequences[idx]), dim=1).to(device) for idx, seq in enumerate(sequences)]

        self.lengths = [len(seq) for seq in sequences]
        self.lengths = torch.tensor(self.lengths).to(device)
        self.max_len = max(self.lengths)
        self.n_seqs = torch.tensor(len(sequences)).to(device)
        self.seq_idxs = torch.tensor(range(self.n_seqs)).to(device)
        
        self.input_size = sequences[0].shape[1]
        self.hidden_size = hidden_size
        self.outcomes = list(oc_targets.keys())
        self.outcome_size = len(self.outcomes)
        self.device = device
        
        self.targets = []
        self.target_masks = []
        tmp = torch.stack([oc_targets[oc] for oc in self.outcomes], dim=1).to(device)
        targets = [tmp[i] for i in range(tmp.shape[0])]
        for i, tg in enumerate(targets):
            mask = torch.ones_like(tg)
            mask[tg>9999] = 0
            self.target_masks.append(mask)
            tg[tg>9999] = 0
            tg = tg/self.max_len    # change days to rate
            self.targets.append(tg)
        
        self.gamma_coeff = Gamma_Coeff(self.lengths, hidden_size)
        self.gamma_coeff.to(device)
        self.n_batch = n_batch
        
        last_day_rates = []
        for i, seq_len in enumerate(self.lengths):
            my_day = self.time_sequences[i][seq_len-1]
            my_day_rate = my_day/self.max_len
            last_day_rates.append(my_day_rate)
        self.last_day_rates = torch.tensor(last_day_rates).to(device)

    def load_full(self):
        self.train_indices = np.array([idx for idx in range(self.n_seqs)])
        self.test_indices = np.array([])
        self.split_fname_prex = "full_"
        
        print("Data Manager: full training loaded.")
        self.init_flipping()

    def load_split(self, in_seed, in_split_rate):
        assert(in_split_rate<1 and in_split_rate>0)
        perc = str(int(in_split_rate*100))
        split_fname = os.path.join(data_split_path, f'split_{perc}[{str(in_seed)}]_data.pkl')
        if not os.path.exists(split_fname):
            sys.exit(f"split file '{split_fname}' not exist!")
        
        with open(split_fname, 'rb') as f:
            split_save = pickle.load(f)
        self.train_indices = split_save['train_indices']
        self.test_indices = split_save['test_indices']
        self.split_fname_prex = f"split_{perc}[{str(in_seed)}]_"
        
        print(f"Data Manager: split {in_seed} loaded.")
        self.init_flipping()

    def init_flipping(self):
        self.flip_idxs = []
        for i, tg_msk in enumerate(self.target_masks):
            if i not in self.train_indices:
                continue
            if not all(tg_msk==0):
                self.flip_idxs.append(i)
        
        self.n_flip = len(self.flip_idxs)
        self.flip_lengths = torch.tensor([self.lengths[idx] for idx in self.flip_idxs]).to(self.device)
        self.flip_gamma_coeff = Gamma_Coeff(self.flip_lengths, self.hidden_size)
        self.flip_gamma_coeff.to(self.device)
        print(f"Data Manager: {self.n_flip} / {len(self.target_masks)} label flipped.")
        return

    def get_DataLoader(self, mode_name):
        if mode_name == 'LITT':
            data_list = [self.time_sequences, self.sequences, self.last_day_rates, self.seq_idxs, self.lengths, self.targets, self.target_masks]
            collate_fn = self.collate_fn_LITT

        elif mode_name == 'Event':
            data_list = [self.time_sequences, self.sequences, self.last_day_rates, self.seq_idxs, self.lengths, self.target_masks]
            collate_fn = self.collate_fn_Event

        else:
            sys.exit("Wrong model name:",mode_name)
            
        dataset = Dataset_Container(data_list)
        train_dataset = Subset(dataset, self.train_indices)
        train_dataloader = DataLoader(train_dataset, batch_size=self.n_batch, shuffle=True, collate_fn=collate_fn)
        
        if len(self.test_indices)>0:
            test_dataset = Subset(dataset, self.test_indices)
            test_dataloader = DataLoader(test_dataset, batch_size=self.n_batch, shuffle=True, collate_fn=collate_fn)
        else:
            test_dataloader = None
        return train_dataloader, test_dataloader
    
    def get_Flip_DataLoader(self, flip_event_tag = False):
        self.flip_event_tag = flip_event_tag
        
        flip_seq_idxs = torch.tensor(range(self.n_flip)).to(self.device)
        flip_sequences = [self.sequences[idx] for idx in self.flip_idxs]
        flip_time_sequences = [self.time_sequences[idx] for idx in self.flip_idxs]
        flip_last_day_rates = [self.last_day_rates[idx] for idx in self.flip_idxs]
        #flip_lengths = [self.lengths[idx] for idx in self.flip_idxs]
        flip_targets = [torch.zeros_like(self.targets[idx]) for idx in self.flip_idxs]
        flip_target_masks = [torch.zeros_like(self.target_masks[idx]) for idx in self.flip_idxs]
        
        data_list = [flip_time_sequences, flip_sequences, flip_last_day_rates, flip_seq_idxs, self.flip_lengths, flip_targets, flip_target_masks]
        dataset = Dataset_Container(data_list)
        flip_dataset = Subset(dataset, flip_seq_idxs)
        flip_dataloader = DataLoader(flip_dataset, batch_size=self.n_batch, shuffle=True, collate_fn=self.collate_fn_Flip)
        return flip_dataloader
    
    def collate_fn_LITT(self, batch):
        time_sequences, sequences, last_day_rates, seq_idxs, lengths, targets, target_masks = zip(*batch)
        
        coeff_params = self.gamma_coeff(time_sequences, seq_idxs)
        
        if self.gamma_coeff.training:
            for param in coeff_params:
                assert(param.requires_grad==True)
        
        padded_coeff = pad_sequence(coeff_params, batch_first=True, padding_value=0.0)
        packed_coeff = pack_padded_sequence(padded_coeff, lengths, batch_first=True, enforce_sorted=False)
        
        padded_X = pad_sequence(sequences, batch_first=True, padding_value=0.0)
        packed_X = pack_padded_sequence(padded_X, lengths, batch_first=True, enforce_sorted=False)
        
        assert(all(packed_coeff.sorted_indices == packed_X.sorted_indices))
        
        device = last_day_rates[0].device
        last_day_rates = torch.tensor(last_day_rates).to(device)
        padded_targets = pad_sequence(targets, batch_first=True, padding_value=0)
        padded_target_masks = pad_sequence(target_masks, batch_first=True, padding_value=0)
        
        return packed_coeff, packed_X, last_day_rates, seq_idxs, padded_targets, padded_target_masks

    def collate_fn_Flip(self, batch):
        time_sequences, sequences, last_day_rates, seq_idxs, lengths, targets, target_masks = zip(*batch)
        
        coeff_params = self.flip_gamma_coeff(time_sequences, seq_idxs)
        if self.flip_gamma_coeff.training:
            for param in coeff_params:
                assert(param.requires_grad==True)
        
        padded_coeff = pad_sequence(coeff_params, batch_first=True, padding_value=0.0)
        packed_coeff = pack_padded_sequence(padded_coeff, lengths, batch_first=True, enforce_sorted=False)
        
        padded_X = pad_sequence(sequences, batch_first=True, padding_value=0.0)
        packed_X = pack_padded_sequence(padded_X, lengths, batch_first=True, enforce_sorted=False)
        
        assert(all(packed_coeff.sorted_indices == packed_X.sorted_indices))
        
        device = last_day_rates[0].device
        last_day_rates = torch.tensor(last_day_rates).to(device)
        padded_targets = pad_sequence(targets, batch_first=True, padding_value=0)
        padded_target_masks = pad_sequence(target_masks, batch_first=True, padding_value=0)
        
        if not self.flip_event_tag: # training Flip
            return packed_coeff, packed_X, last_day_rates, seq_idxs, padded_targets, padded_target_masks
        else:                       # Flip Event analysis
            return time_sequences, packed_coeff, coeff_params, packed_X, last_day_rates, seq_idxs, padded_target_masks


    def collate_fn_Event(self, batch):
        time_sequences, sequences, last_day_rates, seq_idxs, lengths, target_masks = zip(*batch)
        
        coeff_params = self.gamma_coeff(time_sequences, seq_idxs)
        padded_coeff = pad_sequence(coeff_params, batch_first=True, padding_value=0.0)
        packed_coeff = pack_padded_sequence(padded_coeff, lengths, batch_first=True, enforce_sorted=False)
        
        padded_X = pad_sequence(sequences, batch_first=True, padding_value=0.0)
        packed_X = pack_padded_sequence(padded_X, lengths, batch_first=True, enforce_sorted=False)
        
        assert(all(packed_coeff.sorted_indices == packed_X.sorted_indices))
        
        device = last_day_rates[0].device
        last_day_rates = torch.tensor(last_day_rates).to(device)
        padded_target_masks = pad_sequence(target_masks, batch_first=True, padding_value=0)
        
        return time_sequences, packed_coeff, coeff_params, packed_X, last_day_rates, seq_idxs, padded_target_masks



