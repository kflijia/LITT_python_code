# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 22:04:45 2025

@author: jiaxx213
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import os
import numpy as np

import pickle
import sys
import matplotlib.pyplot as plt

from functions_basic import save_path, result_path, LITT_path
from functions_basic import get_performance, print_grad_graph, inspect_optimizer_grads

#from functions_tSNE import Plotter_tSNE_2D
#from functions_PCA import Plotter_PCA_2D
import warnings
from sklearn.exceptions import UndefinedMetricWarning


class Cell_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Weight matrices for 4 gates: input, forget, cell, output
        self.W_x = nn.Parameter(torch.empty(input_size, hidden_size * 4))
        self.W_h = nn.Parameter(torch.empty(hidden_size, hidden_size * 4))
        self.b_x = nn.Parameter(torch.empty(hidden_size * 4))
        self.b_h = nn.Parameter(torch.empty(hidden_size * 4))
        
        # Xavier uniform initialization for weights
        nn.init.xavier_uniform_(self.W_x)
        nn.init.xavier_uniform_(self.W_h)
        
        # Initialize biases
        with torch.no_grad():
            # Zero all biases initially
            self.b_x.zero_()
            self.b_h.zero_()
            # Set forget gate bias to 1.0 (indices hidden_size:2*hidden_size)
            self.b_x[self.hidden_size:2*self.hidden_size].fill_(1.0)
            self.b_h[self.hidden_size:2*self.hidden_size].fill_(1.0)

    def forward(self, x, hidden):
        h_prev, c_prev = hidden
        gates = torch.matmul(x, self.W_x) + self.b_x + torch.matmul(h_prev, self.W_h) + self.b_h
        input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, dim=-1)
        
        
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)
        cell_gate = torch.tanh(cell_gate)
        output_gate = torch.sigmoid(output_gate)
        
        c_next = forget_gate * c_prev + input_gate * cell_gate
        h_next = output_gate * torch.tanh(c_next)
        return h_next, c_next
    
    def compare_cell_state(self, state_dict1):
        state_dict2 = self.state_dict()
        tag_arr = []
        tag_arr = [not torch.equal(state_dict1[key].data, state_dict2[key].data) for key in state_dict1]
        T_rate = sum(tag_arr)/len(tag_arr)
        return T_rate

        
#class Model_LITT(nn.Module):
#...
#########################
# The core code is hidden due to copyright issue
# Contact the author for full version
#########################





class Trainer_LITT():
    def __init__(self, data_manager):
        super().__init__()

        input_size = data_manager.input_size
        hidden_size = data_manager.hidden_size
        outcome_size = data_manager.outcome_size
        device = data_manager.device
        
        self.gamma_coeff = data_manager.gamma_coeff
        
        self.n_seqs = data_manager.n_seqs
        self.max_len = data_manager.max_len
        self.fname_prex = data_manager.split_fname_prex
        self.outcomes = data_manager.outcomes
        self.train_indices = data_manager.train_indices

        self.model_LITT = Model_LITT(input_size, hidden_size, outcome_size)
        self.model_LITT.to(device)
        print("")
        print(f"Model LITT_Trainer: input_size = {input_size}, hidden_size = {hidden_size}, outcome_size = {outcome_size}")
        
        # Separate optimizers
        self.optimizer_mask = optim.Adam(
            self.model_LITT.mask_head.parameters(),
            lr=0.03, betas=(0.5, 0.999)
        )
        self.scheduler_mask = StepLR(self.optimizer_mask, step_size=15, gamma=0.5)  # Reduce lr by 50% every 15 epochs
        
        parameter_list = nn.ParameterList(list(self.model_LITT.cell.parameters()) + list(self.model_LITT.value_head.parameters()))
        parameter_list.extend(self.gamma_coeff.W_params)
        parameter_list.extend(self.gamma_coeff.b_params)
        self.optimizer_result = optim.Adam(
            parameter_list,
            lr=0.01, betas=(0.5, 0.999)
        )
        self.scheduler_result = StepLR(self.optimizer_result, step_size=15, gamma=0.5)
        
        # Loss functions
        self.loss_mask = nn.BCEWithLogitsLoss()  # For mask head
        self.loss_result = nn.MSELoss()   # For final output
        
        # for flipping
        self.n_flip = data_manager.n_flip
        self.flip_gamma_coeff = data_manager.flip_gamma_coeff

    def loading(self):
        os.makedirs(LITT_path, exist_ok=True)
        
        checkpoint_fname = os.path.join(LITT_path, self.fname_prex + 'LITT_checkpoint.pkl')
        loss_trace_fname = os.path.join(LITT_path, self.fname_prex + 'LITT_loss_trace.npy')
        
        self.checkpoint_fname = checkpoint_fname
        self.loss_trace_fname = loss_trace_fname
        
        if os.path.exists(checkpoint_fname):
            print(f"Loading checkpoint from {checkpoint_fname} ...")
            with open(checkpoint_fname, 'rb') as f:
                checkpoint = pickle.load(f)
            assert(self.model_LITT.input_size == checkpoint['input_size'])
            assert(self.model_LITT.hidden_size == checkpoint['hidden_size'])
            assert(self.model_LITT.outcome_size == checkpoint['outcome_size'])
            
            self.model_LITT.cell.load_state_dict(checkpoint['cell_state_dict'])
            self.model_LITT.mask_head.load_state_dict(checkpoint['mask_state_dict'])
            self.model_LITT.value_head.load_state_dict(checkpoint['value_state_dict'])
            self.gamma_coeff.load_gamma_state(checkpoint['gamma_state_dict'])
            
            self.optimizer_mask.load_state_dict(checkpoint['optimizer_mask_state_dict'])
            self.scheduler_mask.load_state_dict(checkpoint['scheduler_mask_state_dict'])
            self.optimizer_result.load_state_dict(checkpoint['optimizer_res_state_dict'])
            self.scheduler_result.load_state_dict(checkpoint['scheduler_res_state_dict'])
            
            self.start_epoch = checkpoint['start_epoch']
            self.max_epoch = checkpoint['max_epoch']
            self.best_loss = checkpoint['train_loss']
            
            self.pat_loss_trace = np.load(loss_trace_fname, allow_pickle=True).item()
            print(f"== Loaded model_LITT: epoch {self.start_epoch}, train_loss {self.best_loss:.4f} ==")
            return True
        else:
            self.start_epoch = 0
            self.max_epoch = 0
            self.best_loss = float('inf')

            self.pat_loss_trace = {idx: [] for idx in range(self.n_seqs)}
            np.save(self.loss_trace_fname, self.pat_loss_trace)
            print("== Initialized model_LITT ==")
            return False
            
    def saving(self):
        checkpoint = {
            'start_epoch': self.start_epoch,
            'max_epoch': self.max_epoch,
            'train_loss': self.best_loss,
            'input_size': self.model_LITT.input_size,
            'outcome_size': self.model_LITT.outcome_size,
            'hidden_size': self.model_LITT.hidden_size,

            'cell_state_dict': self.model_LITT.cell.state_dict(),
            'mask_state_dict': self.model_LITT.mask_head.state_dict(),
            'value_state_dict': self.model_LITT.value_head.state_dict(),
            'gamma_state_dict': self.gamma_coeff.get_gamma_state(),
            
            'optimizer_mask_state_dict': self.optimizer_mask.state_dict(),
            'scheduler_mask_state_dict': self.scheduler_mask.state_dict(),
            'optimizer_res_state_dict': self.optimizer_result.state_dict(),
            'scheduler_res_state_dict': self.scheduler_result.state_dict()
        }
        
        with open(self.checkpoint_fname, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        np.save(self.loss_trace_fname, self.pat_loss_trace)
        return
    
    def check_requires_grad(self):
        # all trainable including cell
        self.model_LITT.train()
        for param in self.model_LITT.parameters():
            assert(param.requires_grad == True)
        
        self.gamma_coeff.train()
        for idx in self.gamma_coeff.seq_idxs:
            self.gamma_coeff.W_params[idx].requires_grad = True
            self.gamma_coeff.b_params[idx].requires_grad = True
        
        torch.autograd.set_detect_anomaly(True)
        return

    def run_training(self, n_epochs, train_dataloader, debug_level = 0, flipping_tag = False):
        #...
        #########################
        # The core code is hidden due to copyright issue
        # Contact the author for full version
        #########################
        return

    def run_eval(self, in_dataloader, in_split_str, save_tag = True, print_tag = False, in_opt_th = None):
        #...
        #########################
        # The core code is hidden due to copyright issue
        # Contact the author for full version
        #########################
        return
    
    #---------------------------------------------
    #----------        flipping       ------------
    #---------------------------------------------
    def flip_load(self):
        self.model_LITT.eval()
        self.gamma_coeff.eval()
        for param in self.model_LITT.cell.parameters():
            param.requires_grad = False
        for param in self.model_LITT.value_head.parameters():
            param.requires_grad = False
        for param in self.model_LITT.mask_head.parameters():
            param.requires_grad = False
        
        self.flip_gamma_coeff.train()
        for param in self.flip_gamma_coeff.parameters():
            param.requires_grad = True
        for idx in self.flip_gamma_coeff.seq_idxs:
            self.flip_gamma_coeff.W_params[idx].requires_grad = True
            self.flip_gamma_coeff.b_params[idx].requires_grad = True
        
        
        dummy_param = nn.Parameter(torch.empty(0)) # 0-element tensor
        trainable_params = list(filter(lambda p: p.requires_grad, self.model_LITT.mask_head.parameters()))
        if not trainable_params:
            trainable_params = [dummy_param]
        
        self.flip_optimizer_mask = optim.Adam(
            trainable_params, # should be no change
            lr=0.03, betas=(0.5, 0.999)
        )
        self.flip_scheduler_mask = StepLR(self.flip_optimizer_mask, step_size=15, gamma=0.5)
        
        flip_parameter_list = nn.ParameterList(list(self.model_LITT.cell.parameters()) + list(self.model_LITT.value_head.parameters()))
        flip_parameter_list.extend(self.flip_gamma_coeff.W_params)
        flip_parameter_list.extend(self.flip_gamma_coeff.b_params)
        self.flip_optimizer_result = optim.Adam(
            filter(lambda p: p.requires_grad, flip_parameter_list), # should only W_params and b_params change
            lr=0.01, betas=(0.5, 0.999)
        )
        self.flip_scheduler_result = StepLR(self.flip_optimizer_result, step_size=15, gamma=0.5)
        
        
        self.flip_checkpoint_fname = os.path.join(LITT_path, self.fname_prex + 'LITT_Flip_checkpoint.pkl')
        self.flip_loss_trace_fname = os.path.join(LITT_path, self.fname_prex + 'LITT_Flip_loss_trace.npy')
        
        if os.path.exists(self.flip_checkpoint_fname):
            with open(self.flip_checkpoint_fname, 'rb') as f:
                checkpoint = pickle.load(f)
            
            self.flip_gamma_coeff.load_gamma_state(checkpoint['flip_gamma_state_dict'])
            self.flip_optimizer_mask.load_state_dict(checkpoint['flip_optimizer_mask_state_dict'])
            self.flip_scheduler_mask.load_state_dict(checkpoint['flip_scheduler_mask_state_dict'])
            self.flip_optimizer_result.load_state_dict(checkpoint['flip_optimizer_res_state_dict'])
            self.flip_scheduler_result.load_state_dict(checkpoint['flip_scheduler_res_state_dict'])
            self.flip_max_epoch = checkpoint['flip_max_epoch']
            self.flip_best_loss = checkpoint['flip_train_loss']
            self.flip_pat_loss_trace = np.load(self.flip_loss_trace_fname, allow_pickle=True).item()
            print(f"== Loaded Flipping: flip_max_epoch {self.flip_max_epoch}, flip_train_loss {self.flip_best_loss:.4f} ==")
            return True
        else:
            self.flip_idxs = self.flip_gamma_coeff.seq_idxs  # range(n_flip)
            self.flip_max_epoch = 0
            self.flip_start_epoch = 0
            self.flip_best_loss = float('inf')
            self.flip_pat_loss_trace = {idx: [] for idx in range(self.n_flip)}
            return False

    def flip_save(self):
        checkpoint = {
            'flip_max_epoch': self.flip_max_epoch,
            'flip_train_loss': self.flip_best_loss,
            'flip_gamma_state_dict': self.flip_gamma_coeff.get_gamma_state(),
            'flip_optimizer_mask_state_dict': self.flip_optimizer_mask.state_dict(),
            'flip_scheduler_mask_state_dict': self.flip_scheduler_mask.state_dict(),
            'flip_optimizer_res_state_dict': self.flip_optimizer_result.state_dict(),
            'flip_scheduler_res_state_dict': self.flip_scheduler_result.state_dict()
        }
        
        with open(self.flip_checkpoint_fname, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        np.save(self.flip_loss_trace_fname, self.flip_pat_loss_trace)
        return
    
    def flip_switch(self):
        self.optimizer_mask, self.flip_optimizer_mask = self.flip_optimizer_mask, self.optimizer_mask
        self.scheduler_mask, self.flip_scheduler_mask = self.flip_scheduler_mask, self.scheduler_mask
        self.optimizer_result, self.flip_optimizer_result = self.flip_optimizer_result, self.optimizer_result
        self.scheduler_result, self.flip_scheduler_result = self.flip_scheduler_result, self.scheduler_result
        self.gamma_coeff, self.flip_gamma_coeff = self.flip_gamma_coeff, self.gamma_coeff
        self.train_indices, self.flip_idxs = self.flip_idxs, self.train_indices
        self.best_loss, self.flip_best_loss = self.flip_best_loss, self.best_loss
        self.max_epoch, self.flip_max_epoch = self.flip_max_epoch, self.max_epoch
        self.start_epoch, self.flip_start_epoch = self.flip_start_epoch, self.start_epoch
        self.pat_loss_trace, self.flip_pat_loss_trace = self.flip_pat_loss_trace, self.pat_loss_trace
        return
    
    def run_flipping(self, n_epochs, flip_dataloader, debug_level = 0):
        flip_trained = self.flip_load()
        
        if not flip_trained:
            print("== Start Flip Training ==")
            torch.autograd.set_detect_anomaly(True)

            self.flip_switch()
            self.run_training(n_epochs, flip_dataloader, debug_level, True)
            self.flip_switch()
            
            self.flip_save()
            print("== Done Flip Training ==")
        return
            
            



        

        








