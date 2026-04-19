# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 16:56:16 2026

@author: jiaxx213
"""

import sys
import random
import pickle
        
import csv
import string

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np
import pandas as pd
import os
import math
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import kurtosis, mode
from sklearn.neighbors import KernelDensity
from scipy.interpolate import interp1d

from functions_data import Gamma_Coeff
from functions_basic import var_list, save_path, data_split_path, LITT_path, load_sourcedata
from functions_Path import disp_dict

LITT_path = './'

class Flipping_Analyzer():
    def __init__(self, recur_dict, path_list):  #, data_manager, recur_dict, path_list):
        super().__init__()
        self.event_fname = os.path.join(LITT_path, 'event_data.pkl')
        self.fname_prex = 'full_'
        #self.fname_prex = data_manager.split_fname_prex  # full_
        self.recur_dict = recur_dict
        self.path_list = path_list
        
        self.n_flip = 835 # data_manager.n_flip
        #self.flip_idxs = data_manager.flip_idxs
        #self.flip_lengths = data_manager.flip_lengths
        #self.flip_gamma_coeff = data_manager.flip_gamma_coeff
        self.flip_checkpoint_fname = os.path.join(LITT_path, self.fname_prex + 'LITT_Flip_checkpoint.pkl')
        
        #self.data_manager = data_manager
        #self.gamma_coeff = data_manager.gamma_coeff
        self.n_seqs = 3276 #data_manager.n_seqs
        self.max_len = 489 #data_manager.max_len
        
    def loading(self):
        #  load_events
        with open(self.event_fname, 'rb') as f:
            loaded_event_df = pickle.load(f)
        
        disp_columns = {}
        event_df_step = loaded_event_df['event_df_step']
        event_df_day = loaded_event_df['event_df_day']
        for col in event_df_step.columns:
            split_list = col.split('_')
            evt = '_'.join(split_list[:-1])
            
            disp_col = disp_dict[evt]+'_'+split_list[-1]
            disp_columns[col] = disp_col
        event_df_step.rename(columns=disp_columns, inplace=True)
        event_df_day.rename(columns=disp_columns, inplace=True)
        
        self.event_df_step = event_df_step
        self.event_df_day = event_df_day
        self.events = list(self.event_df_day.columns)
        
        # convert disp key for recur_dict
        def inner_convert_disp(cur_pointer):
            keys = list(cur_pointer.keys())
            if len(keys)==0:
                return
            for k in keys:
                 split_list = k.split('_')
                 evt = '_'.join(split_list[:-1])
                 disp_k = disp_dict[evt]+'_'+split_list[-1]
                 cur_pointer[disp_k] = cur_pointer.pop(k)
                 inner_convert_disp(cur_pointer[disp_k]['next'])
            return
        inner_convert_disp(self.recur_dict)
        
        #  load timing
        timing_fname = os.path.join(LITT_path, self.fname_prex + 'LITT_timing.pkl')
        print(f"Loading {self.fname_prex}timing from {timing_fname} ...")
        with open(timing_fname, 'rb') as f:
            self.all_timing_list, self.all_label_list = pickle.load(f)
    
        # get or load flip_timing
        flip_timing_fname = os.path.join(LITT_path, self.fname_prex + 'LITT_flipped_timing.pkl')
        if os.path.exists(flip_timing_fname):
            print(f"Loading existing flipped_timing from {flip_timing_fname} ...")
            with open(flip_timing_fname, 'rb') as f:
                self.flip_timing_list = pickle.load(f)
        
        #----- for no data loading -----
        with open("lengths.txt", "r") as f:
            lengths_list = [int(line.strip()) for line in f]
        
        with open("flip_lengths.txt", "r") as f:
            flip_lengths_list = [int(line.strip()) for line in f]
        
        self.flip_idxs = []
        self.flip_lengths = []
        for idx, idx_labels in enumerate(self.all_label_list):
            if not all(idx_labels==0):
                self.flip_idxs.append(idx)
                self.flip_lengths.append(lengths_list[idx])
        assert(self.n_flip == len(self.flip_idxs))
        for i, f_len in enumerate(self.flip_lengths):
            assert(f_len==len(self.flip_timing_list[i]))
        return
        #--------------------------------
        
        
        
        
        print(f"Compute flipped timing ... \n Loading checkpoint from {self.flip_checkpoint_fname} ...")
        with open(self.flip_checkpoint_fname, 'rb') as f:
            checkpoint = pickle.load(f)
            
        self.flip_gamma_coeff.load_gamma_state(checkpoint['flip_gamma_state_dict'])
        self.flip_gamma_coeff.eval()
        
        flip_dataloader = self.data_manager.get_Flip_DataLoader(True)
        self.flip_timing_list = [np.nan for i in range(self.n_flip)]
        
        with torch.no_grad():
            # seq_idxs follows reset range(n_flip)
            for time_sequences, _, list_coeff_seq, _, _, seq_idxs, target_masks in flip_dataloader:
                for i, t_seq in enumerate(time_sequences):
                    idx = seq_idxs[i]
                    full_coeff_seq = list_coeff_seq[i]
                    avg_coeff_seq = []
                    for coeff in full_coeff_seq:
                        avg_coeff_seq.append(torch.mean(coeff))
                    avg_coeff_seq = torch.tensor(avg_coeff_seq).unsqueeze(-1)
                    
                    coeff_seq = avg_coeff_seq[1:]   # (seq_len-1, 1)  the first coeff is 0.0
                    gamma_seq = -torch.log(1-coeff_seq)
                    
                    t_seq = t_seq.cpu().numpy().flatten()
                    gamma_seq = gamma_seq.cpu().numpy().flatten()
                    
                    delta_t_seq = t_seq[1:] - t_seq[:-1]
                    reltv_delta_t_seq = delta_t_seq * gamma_seq
                    flip_reltv_t_seq = [0]
                    
                    for j in range(len(reltv_delta_t_seq)):
                        j_reltv_t = sum(reltv_delta_t_seq[:(j+1)])
                        flip_reltv_t_seq.append(float(j_reltv_t))
                    
                    my_true_idx = self.flip_idxs[idx]
                    my_timing_df = self.all_timing_list[my_true_idx]
                    my_t_seq = my_timing_df['t_seq']
                    assert(all(my_t_seq==t_seq))
                    my_reltv_t_seq = my_timing_df['reltv_t_seq']
                    
                    timing_df = {'t_seq':t_seq, 'reltv_t_seq':my_reltv_t_seq, 'flip_t_seq':flip_reltv_t_seq}
                    timing_df = pd.DataFrame(timing_df)
                    self.flip_timing_list[idx] = timing_df

        with open(flip_timing_fname, 'wb') as f:
            pickle.dump(self.flip_timing_list, f)
            
            
    def do_analysis(self):
        flip_path_dict = {}
        for i, timing_df in enumerate(self.flip_timing_list):
            true_idx = self.flip_idxs[i]
            path_df = pd.DataFrame(columns=['orig', 'relv', 'flip'])
            event_row = self.event_df_step.iloc[true_idx]
            
            
            clean_row = event_row[pd.notna(event_row)]
            sorted_row = clean_row.sort_values()
            for evt, step in sorted_row.items():
                if evt in self.events:
                    if step not in timing_df.index:
                        print(f"true_idx = {true_idx}")
                        print(f"event_row = {event_row}")
                        print(f"evt = {evt}, step={step}, timing_df={timing_df}")
                    
                    path_df.loc[evt] = list(timing_df.loc[step])
            flip_path_dict[true_idx] = path_df

        self.flip_path_dict = flip_path_dict
        
        all_path_dict = {}
        for idx in range(len(self.event_df_step)):
            idx_row = self.event_df_step.iloc[idx]
            my_timing_df = self.all_timing_list[idx]
            path_df = pd.DataFrame(columns=['orig', 'relv'])
            
            clean_row = idx_row[pd.notna(idx_row)]
            sorted_row = clean_row.sort_values()
            for evt, step in sorted_row.items():
                if evt in self.events:
                    
                    if step not in my_timing_df.index:
                        print(f"idx = {idx}")
                        print(f"idx_row = {idx_row}")
                        print(f"evt = {evt}, step={step}, my_timing_df={my_timing_df}")
                    
                    
                    path_df.loc[evt] = list(my_timing_df.loc[step]) # 't_seq', 'reltv_t_seq'
            all_path_dict[idx] = path_df
        
        self.all_path_dict = all_path_dict
        # for idx, df in flip_path_dict.items():
        #     if not all(all_path_dict[idx].values.flatten() == df[['orig', 'relv']].values.flatten()):
        #         print(f"all_path_dict[idx] = {all_path_dict[idx]}")
        #         print(f"df[['orig', 'relv']] = {df[['orig', 'relv']]}")
        
        return

    def get_path_idxs(self, path_evt_list):
        cur_idxs = []
        cur_pointer = self.recur_dict
        for evt in path_evt_list:
            cur_idxs = cur_pointer[evt]['true_idxs']
            cur_pointer = cur_pointer[evt]['next']
        assert(len(cur_idxs)>0)
        
        pos_idxs = []
        neg_idxs = []
        slc_flip_idxs = []
        for idx in cur_idxs:
            if any(self.all_label_list[idx]) >0:
                pos_idxs.append(idx)
            else:
                neg_idxs.append(idx)
            if idx in self.flip_idxs:
                slc_flip_idxs.append(idx)
        
        return (pos_idxs, neg_idxs), slc_flip_idxs
        
    def plot_relv_changes(self, path_info, plt_events):
        path_id, path_step = path_info
        plt_path_df = self.path_list[path_id]
        plt_path = list(plt_path_df['evt'])
        if path_step>0:
            plt_path = plt_path[:path_step]
        (pos_idxs, neg_idxs), _ = self.get_path_idxs(plt_path)
        print(f"positive {len(pos_idxs)}, negative {len(neg_idxs)}")
        plt_idxs = {'pos':pos_idxs, 'neg':neg_idxs}
        #plt_events = ['anthracyclins_1', 'anthracyclins_2', 'anthracyclins_3', 'targeted_1', 'targeted_2', 'targeted_3', 'radiation_1', 'radiation_2', 'radiation_3']
        
        plt_dict = {}
        for evt in plt_events:
            plt_dict[evt] = {'pos':[], 'neg':[]}
        
        for typ in ['pos', 'neg']:
            idxs_arr = plt_idxs[typ]
            for idx in idxs_arr:
                path_df = self.all_path_dict[idx]  # {'t_seq':t_seq, 'reltv_t_seq':reltv_t_seq}
                for evt in plt_events:
                    if evt in path_df.index:
                        evt_row = path_df.loc[evt]
                        # if evt_row['relv'] == np.inf:
                        #     evt_change = 1.0
                        # else:
                        evt_change = (evt_row['relv']-evt_row['orig']) #/ evt_row['orig']
                        plt_dict[evt][typ].append(evt_change)
        
        fig, axes = plt.subplots(3, 3, figsize=(12, 10))
        colors = {'pos':"crimson", 'neg':"royalblue"}
        for i, evt in enumerate(plt_events):
            ax = axes[int(i/3), i%3]
            cur_plt_data = plt_dict[evt]
            for typ, values in cur_plt_data.items():
                sns.kdeplot(
                    values, 
                    fill=True, 
                    ax=ax, 
                    label=typ, 
                    color=colors[typ],
                    bw_adjust=0.1,
                    alpha=0.3
                )
            # Formatting
            ax.set_title(f"{evt}: Relative Change")#, fontweight='bold')
            ax.set_ylabel("Density")
            ax.set_xlabel("Days")
            ax.legend(loc='upper right')
            if i < 6: ax.set_xlabel("") # Hide x-label for top two to reduce clutter
        plt.tight_layout()
        plt.show()
        return
    
    def plot_flip_changes(self, path_info, plt_events):
        path_id, path_step = path_info
        plt_path_df = self.path_list[path_id]
        plt_path = list(plt_path_df['evt'])
        if path_step>0:
            plt_path = plt_path[:path_step]
            
        _, slt_flip_idxs = self.get_path_idxs(plt_path)
        #plt_events = ['anthracyclins_1', 'anthracyclins_2', 'anthracyclins_3', 'targeted_1', 'targeted_2', 'targeted_3', 'radiation_1', 'radiation_2', 'radiation_3']
        plt_dict = {}
        for evt in plt_events:
            plt_dict[evt] = []
            
        for idx in slt_flip_idxs:
            path_df = self.flip_path_dict[idx]
            for evt in plt_events:
                if evt in path_df.index:
                    evt_row = path_df.loc[evt]
                    # if evt_row['relv'] == 0:
                    #     continue
                    # if evt_row['relv'] == np.inf:
                    #     if evt_row['flip'] == np.inf:
                    #         evt_change = 0.0
                    #     else:
                    #         evt_change = -1.0
                    # else:
                    
                    evt_change = (evt_row['flip'] - evt_row['relv']) #/ evt_row['relv']
                    plt_dict[evt].append(evt_change)
        
        fig, axes = plt.subplots(3, 3, figsize=(12, 10))
        for i, evt in enumerate(plt_events):
            ax = axes[int(i/3), i%3]
            cur_plt_data = plt_dict[evt]
            sns.kdeplot(
                cur_plt_data, 
                fill=True, 
                ax=ax, 
                label=evt, 
                color="royalblue",
                bw_adjust=0.1,
                alpha=0.3
            )
            # Formatting
            ax.set_title(f"{evt} {len(cur_plt_data)}: Flip Change")#, fontweight='bold')
            ax.set_ylabel("Density")
            ax.set_xlabel("Days")
            
            #ax.legend(loc='upper right')
            if i < 6: ax.set_xlabel("") # Hide x-label for top two to reduce clutter
        plt.tight_layout()
        plt.show()
        return
    

    def plot_compare(self, path_info, plt_events, in_param):
        path_id, path_step = path_info
        plt_path_df = self.path_list[path_id]
        plt_path = list(plt_path_df['evt'])
        if path_step>0:
            plt_path = plt_path[:path_step]
        
        _, slt_flip_idxs = self.get_path_idxs(plt_path)
        plt_dict = {}
        typ_list = ['orig', 'relv', 'flip']
        for typ in typ_list:
            plt_dict[typ] = {evt:[] for evt in plt_events}
        
        for idx in slt_flip_idxs:
            path_df = self.flip_path_dict[idx]
            for evt in plt_events:
                if evt in path_df.index:
                    evt_row = path_df.loc[evt]
                    for typ in typ_list:
                        plt_dict[typ][evt].append(evt_row[typ])
                    
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        colors = ["royalblue", "crimson", "forestgreen", 
                  "#FF8C00", "#9400D3", "#2F4F4F"] # Dark Orange, Dark Violet, Dark Slate Gray
        
        n_pat = len(slt_flip_idxs)
        plt_title_list = [r"$\mathbf{Original}$", r"$\mathbf{Relative}$", r"$\mathbf{Counterfactual}$"]

        for i, plt_key in enumerate(typ_list):
            ax = axes[i]
            cur_plt_data = plt_dict[plt_key]
            num_arr = []
            for j, (evt, values) in enumerate(cur_plt_data.items()):
                sns.kdeplot(
                    values, 
                    fill=True, 
                    ax=ax, 
                    label=evt, 
                    color=colors[j],
                    bw_adjust=in_param,
                    alpha=0.3
                )
                num_arr.append(len(values))
        
            # Formatting
            #ax.set_title(f"Analysis: {plt_key} {len(slt_flip_idxs)} patients [{num_arr}]", fontweight='bold')
            ax.set_title(plt_title_list[i] + f" Time Distribution ({n_pat} patients)")
            ax.set_ylabel("Density")
            ax.legend(loc='upper right')
            if i < 2: ax.set_xlabel("") # Hide x-label for top two to reduce clutter
        
        ax.set_xlabel("Days")
        
        x_min = min([ax.get_xlim()[0] for ax in axes])
        x_max = max([ax.get_xlim()[1] for ax in axes])
        y_min = min([ax.get_ylim()[0] for ax in axes])
        y_max = max([ax.get_ylim()[1] for ax in axes])
        for ax in axes:
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
        
        plt.tight_layout()
        plt.show()
        
        return
        
        
        
        
        
        
        
        
        