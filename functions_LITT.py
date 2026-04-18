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

        
class Model_LITT(nn.Module):
    def __init__(self, input_size, hidden_size, outcome_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.outcome_size = outcome_size
        
        self.cell = Cell_LSTM(input_size, hidden_size)
        
        # Value head (for numeric y)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, outcome_size),
            nn.Sigmoid()  # Values in [0, 1] multiplying with max_len
        )
        # Mask head (outputs logits, no sigmoid)
        self.mask_head = nn.Linear(self.hidden_size+1, outcome_size)  # Logits for BCEWithLogitsLoss
        
        # Initialize parameters
        linear_layer = self.value_head[0]
        nn.init.kaiming_normal_(linear_layer.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(linear_layer.bias)
        
        
    def forward(self, packed_coeff, packed_X, last_day_rates, record_state = False):
        x = packed_X.data
        batch_sizes = packed_X.batch_sizes
        unsorted_indices = packed_X.unsorted_indices
        #sorted_indices = packed_X.sorted_indices
        
        #sorted_gammas = gamma_params[sorted_indices, :, :]
        x_coeffs = 1 - packed_coeff.data    # 0.0 < coeff <=1.0
        
        batch_size = batch_sizes[0]
        h = torch.zeros(batch_size, self.hidden_size).to(x.device)
        c = torch.zeros(batch_size, self.hidden_size).to(x.device)
        
        if record_state:
            cell_states = [[] for _ in range(batch_size)]  # List of lists for each sequence
            hidden_states = [[] for _ in range(batch_size)]
        
        idx = 0
        for t, bs in enumerate(batch_sizes):
            #print("=== debug t=",t,"bs=",bs)
            x_t = x[idx:idx + bs] # shape (bs, input_size)
            h_t = h[:bs]
            c_t = c[:bs]
            
            h_t, c_t = self.cell(x_t, (h_t, c_t))
            
            # --- LITT gate coeff ---
            #gamma_t = x_gamma[idx:idx + bs] # shape (bs, hidden_size)
            #gamma_t = sorted_gammas[:bs, t] # shape (bs, hidden_size) 
            #gamma_t_zeros = torch.zeros_like(gamma_t).to(x.device)
            # 0.0 < coeff_t <=1.0
            #coeff_t = torch.exp(-torch.max(gamma_t_zeros, gamma_t))  # shape (bs, hidden_size) 
            
            coeff_t = x_coeffs[idx:idx + bs]  # shape (bs, hidden_size) 
            
            # # only for record, 1.0 > rec_coeff_t >=0
            # rec_coeff_t = 1 - coeff_t  # shape (bs, hidden_size) 
            # for i in range(bs):
            #     gamma_coeffs[i].append(rec_coeff_t[i].clone().detach())
            
            # --- apply gamma ---
            c_t = coeff_t * c_t
            
            if record_state:
                for i in range(bs):
                    cell_states[i].append(c_t[i].clone())
                    hidden_states[i].append(h_t[i].clone())

            # --- update ---
            h = torch.cat([h_t, h[bs:]], dim=0) if bs < h.size(0) else h_t.clone()
            c = torch.cat([c_t, c[bs:]], dim=0) if bs < c.size(0) else c_t.clone()
            idx += bs
        
        if record_state:
            cell_states = [torch.stack(cell_states[i], dim=0) for i in unsorted_indices]    # [seq_len, hidden_size] for each elemant
            hidden_states = [torch.stack(hidden_states[i], dim=0) for i in unsorted_indices]

        # Use final hidden state for prediction
        final_h = h[unsorted_indices]  # Reorder to original batch order
        #y_value = self.max_len * self.value_head(final_h)  # (batch_size, output_size), in [0,max_len]
        y_value = self.value_head(final_h)
        
        last_day_rates = last_day_rates.unsqueeze(1)
        y_mask_input = torch.cat([final_h, last_day_rates], dim=1)
        y_mask_logits = self.mask_head(y_mask_input)
        y_mask = torch.sigmoid(y_mask_logits)   # (batch_size, output_size), in [0,1]
        
        # Final output: mask * values
        y_pred = y_mask * y_value  # (batch_size, output_size)
        
        if record_state:
            return y_mask, y_value, y_pred, y_mask_logits, [cell_states, hidden_states]
        else:
            return y_mask, y_value, y_pred, y_mask_logits




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
        if not flipping_tag:
            self.check_requires_grad()
        
        if len(range(self.max_epoch, n_epochs)) == 0:
            max_epoch = self.max_epoch
            print(f"LITT Training Skipped! n_epochs = {n_epochs} is smaller than max_epoch+1 = {max_epoch+1}")
            return


        for epoch in range(self.start_epoch, n_epochs):
            if debug_level>=1:
                print(f"LITT Epoch {epoch}/{n_epochs-1} ")
            else:
                print(f"LITT Epoch {epoch}/{n_epochs-1}:", end='')

            train_loss = 0.0
            train_loss_mask = 0.0
            train_loss_res = 0.0
            batch_idx = 0
            
            init_gammas = self.gamma_coeff.get_gamma_state(self.train_indices)
            lr_change_message = ''

            # batch gamma_params is gamma_coeff's output is in data_manager with padded_T input 
            # batch packed_X is x for model_LITT input
            for packed_coeff, packed_X, last_day_rates, seq_idxs, target_values, target_masks in train_dataloader:
                # Zero gradients for both optimizers
                self.optimizer_mask.zero_grad()
                self.optimizer_result.zero_grad()
                
                _, _, out_results, out_mask_logits = self.model_LITT(packed_coeff, packed_X, last_day_rates)
                
                # Compute losses
                # --- individual losses for records only ---
                masks_pred = out_mask_logits.clone().detach() 
                masks_label = target_masks.clone().detach()
                res_pred = out_results.clone().detach()
                res_label = target_values.clone().detach()
                for i, idx in enumerate(seq_idxs):
                    idx = idx.item()
                    my_loss_m = self.loss_mask(masks_pred[i], masks_label[i])   # (seq_len, hidden_size)
                    my_loss_r = self.loss_result(res_pred[i], res_label[i])
                    my_loss = my_loss_m + my_loss_r
                    self.pat_loss_trace[idx].append(my_loss)
                
                # --- overall loss for backward propogation ---
                mask_loss = self.loss_mask(out_mask_logits, target_masks)
                res_loss = self.loss_result(out_results, target_values)
                days_loss = torch.sqrt(res_loss).item() * self.max_len
                
                # Combined loss (adjust weights as needed) Backward pass and optimize
                loss = mask_loss + res_loss
                loss.backward()
                
                self.optimizer_mask.step()
                self.optimizer_result.step()
                if batch_idx == 0:
                    mask_lr_prev = self.optimizer_mask.param_groups[0]['lr']
                    self.scheduler_mask.step()
                    mask_lr_new = self.scheduler_mask.get_last_lr()[0]
                    if mask_lr_new != mask_lr_prev:
                        lr_change_message += (f"          *** optimizer_mask Learning rate changed from {mask_lr_prev:.6f} to {mask_lr_new:.6f} at epoch {epoch} *** \n")
                    
                    result_lr_prev = self.optimizer_result.param_groups[0]['lr']
                    self.scheduler_result.step()
                    result_lr_new = self.scheduler_result.get_last_lr()[0]
                    if result_lr_new != result_lr_prev:
                        lr_change_message += (f"          *** optimizer_result Learning rate changed from {result_lr_prev:.6f} to {result_lr_new:.6f} at epoch {epoch} *** ")
                
                if debug_level>=2:
                    for i, idx in enumerate(seq_idxs):
                        idx = idx.item()
                        if len(self.pat_loss_trace[idx])>1:
                            short_losses = [f"{num:.4f}" for num in self.pat_loss_trace[idx]]
                            short_trace = ", ".join(short_losses)
                            print(f"        idx = {idx} Loss_trace = [{short_trace}] ")
                if debug_level>=1:
                    print(f"    Batch {batch_idx}: Loss = {loss.item():.4f} mask {mask_loss.item():.4f} + result {res_loss.item():.4f}  days {days_loss:.2f}")
                batch_idx += 1
                train_loss += loss.item()
                train_loss_mask += mask_loss.item()
                train_loss_res += res_loss.item()
            
            W_rate, b_rate = self.gamma_coeff.compare_gamma_state(init_gammas)
            
            avg_train_loss = train_loss / len(train_dataloader)
            avg_train_loss_mask = train_loss_mask / len(train_dataloader)
            avg_train_loss_res = train_loss_res / len(train_dataloader)
            avg_train_loss_res_days = np.sqrt(avg_train_loss_res) * self.max_len
            print(f"  Loss (avg) = {avg_train_loss:.4f}  mask {avg_train_loss_mask:.4f}  + result {avg_train_loss_res:.4f} days {avg_train_loss_res_days:.2f} ", end='')
            print(f" gamma changed: W = {W_rate*100:.1f}%, b = {b_rate*100:.1f}% ", end='')
            
            if avg_train_loss < self.best_loss:
                self.best_loss = avg_train_loss
                self.start_epoch = epoch + 1
                self.max_epoch = self.start_epoch
                if not flipping_tag:
                    self.saving()
                    print("  -- LITT Model saved with trained gamma.")
                else:
                    print("")
            else:
                print("")
            
            if len(lr_change_message)>0:
                print(lr_change_message)
            
        self.max_epoch = epoch + 1
        if not flipping_tag:
            self.saving()
        return

    def run_eval(self, in_dataloader, in_split_str, save_tag = True, print_tag = False, in_opt_th = None):
        #warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        
        assert(in_split_str in ['testing', 'training'])
        test_label_fname = os.path.join(result_path, in_split_str, self.fname_prex + 'LITT_test_label.npy')
        
        y_values_arr = {'pred':[], 'label':[]}
        y_days_arr = {'pred':[], 'label':[], 'last':[]}
        y_masks_arr = {'pred':[], 'label':[]}
        self.model_LITT.eval()
        self.gamma_coeff.eval()
        max_len = self.max_len.cpu()
        with torch.no_grad():
            for packed_coeff, packed_X, last_day_rates, seq_idxs, target_values, target_masks in in_dataloader:
                out_masks, _, out_values, _ = self.model_LITT(packed_coeff, packed_X, last_day_rates)
                for i, idx in enumerate(seq_idxs):
                    #idx = idx.item()
                    mask_pred = out_masks[i].cpu()   # (outcome_size)
                    mask_label = target_masks[i].cpu()
                    value_pred = out_values[i].cpu()
                    value_label = target_values[i].cpu()
                    day_pred = value_pred * max_len
                    day_label = value_label * max_len
                    day_last = last_day_rates[i].cpu() * max_len
                    y_values_arr['label'].append(value_label)
                    y_values_arr['pred'].append(value_pred)
                    y_days_arr['label'].append(day_label)
                    y_days_arr['pred'].append(day_pred)
                    y_days_arr['last'].append(day_last)
                    y_masks_arr['label'].append(mask_label)
                    y_masks_arr['pred'].append(mask_pred)
         
            y_dict = {'values':y_values_arr, 'days':y_days_arr, 'masks':y_masks_arr}
            y_oc_df, opti_th_arr = get_performance(y_dict, self.outcomes, 'LITT', print_tag, in_opt_th)
            
            if save_tag:
                np.save(test_label_fname, y_oc_df)
                
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
            
            



        

        








