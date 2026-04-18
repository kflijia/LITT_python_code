# -*- coding: utf-8 -*-
"""
Created on Sun Feb 22 15:24:55 2026

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


disp_dict = {'anthracyclins':'Chemo', 'targeted':'Targt', 'radiation':'Radia',
             'metformin':'M_Mtf','insulin':'M_Ins', 
             'sulfonylurea':'M_Sul', # second line after insulin
             'sglt2_inhib':'M_SGL',  # benifit cardiovascular but pressure to kidney
             'glp1_recpt':'M_GLP', # mimic the natural hormone GLP-1, stimulating insulin release
             'ezetimibe': 'M_Ezt', # inhibit cholesterol absorption in intestine; reduce cardiovascular risk)
             'pcsk9_inhib':'M_PCS', # lower LDL "bad" cholesterol, well-tolerated, limited side effect
             'fibrates':'M_Fib', # raise "good" HDL cholesterol; reduce heart disease risk
             'bile_acid':'M_BAS', # lower bad LDL cholesterol by binding to bile acids
             'niacin':'M_VB3', # essential water-soluble nutrient; raising "good" HDL and lowering "bad" LDL
             'diuretics':'M_Diu', # Water pill; help kidneys, treat hert failure
             'hyperlipidemia':'Dx_HL', 'diabetes':'Dx_DB', 'hypertension':'Dx_HT', 'obesity':'Dx_Ob', 'smoking':'DxSmk', 'ckd':'DxCKD'}


class path_cluster():
    def __init__(self, main_idx, main_len):
        self.main_idx = main_idx
        self.main_len = main_len
        
        self.queues = {'left':pd.DataFrame(columns = ['idx', 'dist', 'merge', 'len'])}
        self.queues['right'] = self.queues['left'].copy()
        self.dists = {'left': np.inf, 'right': np.inf}
        
        self.border_queues = {'left':pd.DataFrame(columns = ['idx', 'root', 'dist', 'merge', 'len'])}
        self.border_queues['right'] = self.border_queues['left'].copy()
    
    def get_idxs(self):
        all_idxs = [self.main_idx] + list(self.queues['right']['idx']) + list(self.border_queues['right']['idx']) + list(self.queues['left']['idx']) + list(self.border_queues['left']['idx'])
        all_idxs = np.array(all_idxs).astype(int)
        return all_idxs
    
    def insert(self, in_pair, idx_info):
        idx_root, idx = in_pair
        if idx_root == self.main_idx:
            return self.insert_main(idx, idx_info)
        
        elif idx_root in list(self.queues['right']['idx']):
            return self.insert_side('right', in_pair, idx_info)
        elif idx_root in list(self.queues['left']['idx']):
            return self.insert_side('left', in_pair, idx_info)
        
        elif idx_root in list(self.border_queues['right']['idx']):
            return self.insert_border('right', in_pair, idx_info)
        elif idx_root in list(self.border_queues['left']['idx']):
            return self.insert_border('left', in_pair, idx_info)
        
        else:
            raise ValueError(f"Insert Error [main_idx={self.main_idx}] [in_pair={in_pair}]")
        
    def insert_main(self, idx, idx_info):
        in_dist, in_merge, in_len = idx_info
        if in_dist <= self.dists['right']:
            side = 'right'
        elif in_dist <= self.dists['left']:
            side = 'left'
        else:
            return False
            #new_cluster = path_cluster(idx, in_len)
            #return new_cluster
        
        queue_len = len(self.queues[side])
        self.queues[side].loc[queue_len] = [idx, in_dist, in_merge, in_len]
        self.dists[side] = in_dist
        return True

    def insert_side(self, side, in_pair, idx_info):
        idx_root, idx = in_pair
        in_dist, in_merge, in_len = idx_info
        side_df = self.queues[side]
        root_row = side_df[side_df['idx']==idx_root].index[0]
        root_dist = side_df.loc[root_row]['dist']
        border_df = self.border_queues[side]
        if root_dist == self.dists[side]:
            if len(border_df)==0:
                last_row = len(side_df)-1
                side_df.loc[[root_row,last_row]] = side_df.loc[[last_row,root_row]].values
                border_df.loc[0] = [idx, idx_root, in_dist, in_merge, in_len]
                return True
            else:
                border_row = len(border_df)-1
                if border_df.loc[border_row]['root']==idx_root and in_dist <= border_df.loc[border_row]['dist']:
                    border_df.loc[border_row+1] = [idx, idx_root, in_dist, in_merge, in_len]
                    return True
        
        # pop idx_root from side
        new_cluster = path_cluster(idx_root, side_df.loc[root_row]['len'])
        self.queues[side] = side_df[side_df['idx']!=idx_root].reset_index(drop=True)
        pop_border_idxs = [idx_root]
        pop_border_rows = []
        for r in range(len(border_df)):
            r_val = border_df.loc[r]
            if r_val['root'] in pop_border_idxs:
                pop_border_idxs.append(r_val['idx'])
                pop_border_rows.append(r)
                new_cluster.insert((r_val['root'], r_val['idx']), (r_val['dist'], r_val['merge'], r_val['len']))
        self.border_queues[side] = border_df.drop(labels=pop_border_rows).reset_index(drop=True)
        
        new_cluster.insert(in_pair, idx_info)
        
        #print(f"++++ [self.main_idx] insert_side +++ pop new cluster: {idx_root}, [{new_cluster.get_idxs()}]")
        
        return new_cluster

    def insert_border(self, side, in_pair, idx_info):
        idx_root, idx = in_pair
        in_dist, in_merge, in_len = idx_info
        border_df = self.border_queues[side]
        root_row = border_df[border_df['idx']==idx_root].index[0]
        root_row_val = border_df.loc[root_row]
        last_row = len(border_df)-1
        last_row_val = border_df.loc[last_row]
        
        if last_row_val['root']==idx_root and in_dist <= last_row_val['dist']:
            border_df.loc[last_row+1] = [idx, idx_root, in_dist, in_merge, in_len]
            return True
        
        if root_row_val['root'] == last_row_val['root'] and root_row_val['dist'] == last_row_val['dist']:
            border_df.loc[[root_row,last_row]] = border_df.loc[[last_row,root_row]].values
            border_df.loc[last_row+1] = [idx, idx_root, in_dist, in_merge, in_len]
            return True
        
        # pop idx_root from border
        new_cluster = path_cluster(idx_root, border_df.loc[root_row]['len'])
        pop_border_idxs = [idx_root]
        pop_border_rows = [root_row]
        for r in range(len(border_df)):
            r_val = border_df.loc[r]
            if r_val['root'] in pop_border_idxs:
                pop_border_idxs.append(r_val['idx'])
                pop_border_rows.append(r)
                new_cluster.insert((r_val['root'], r_val['idx']), (r_val['dist'], r_val['merge'], r_val['len']))
        self.border_queues[side] = border_df.drop(labels=pop_border_rows).reset_index(drop=True)
        
        new_cluster.insert(in_pair, idx_info)
        #print(f"++++ [self.main_idx] insert_border +++ pop new cluster: {idx_root}, [{new_cluster.get_idxs()}]")
        return new_cluster
    
    def suggest_root(self, idx, path_dist_table):
        for side in ['right', 'left']:
            border_df = self.border_queues[side]
            side_df = self.queues[side]
            if len(border_df)>0:
                last_row = len(border_df)-1
                last_row_val = border_df.loc[last_row]
                idx_root = int(last_row_val['root'])
                root_dist = path_dist_table[idx_root, idx]
                if root_dist <= last_row_val['dist'] and root_dist>0:
                    return (idx_root, root_dist)
            if len(side_df)>=0:
                idx_root = int(self.main_idx)
                root_dist = path_dist_table[idx_root, idx]
                if root_dist <= self.dists[side] and root_dist>0:
                    return (idx_root, root_dist)
        return None
    
    def print_side(self, side, path_merge_dict, path_dist_table):
        print_dict = {}
        # -1: full item
        # 0: " ", 1:"   | ", 2:" └> ", 3:" ─┘ ", 4:"──", 5:" |   "
        #                 -2:" ┌>", -3:" ─┐ " 
        fold_arrow = 2
        fold_line = 3
        if side=='left':
            fold_arrow = -fold_arrow
            fold_line = -fold_line
        
        side_df = self.queues[side]
        border_df = self.border_queues[side][['idx', 'dist', 'merge', 'len']]
        
        if not border_df.empty:
            print_df = pd.concat([side_df, border_df]).reset_index(drop=True)
        else:
            print_df = side_df
        
        for i in print_df.index:
            i_info = np.array(print_df.loc[i])
            idx, header_len, merge_end, path_len = int(i_info[0]), int(i_info[1]), i_info[2], int(i_info[3])
            if i>0:
                last_idx = int(print_df.loc[i-1]['idx'])
                if path_dist_table[last_idx, idx] > header_len:
                    header_len = path_dist_table[last_idx, idx]
                    
            print_dict[idx] = []
            for l in range(path_len):
                if l < header_len-1:
                    print_dict[idx].append(0)
                elif l == header_len-1:
                    print_dict[idx].append(fold_arrow)
                    for p in range(i-1,-1,-1):
                        p_idx = print_df.loc[p]['idx']
                        p_print = print_dict[p_idx]
                        if len(p_print) >= l+1 and p_print[l]==0:
                            print_dict[p_idx][l] = 1
                        elif len(p_print) < l+1:
                            print_dict[p_idx].extend([0]*(l-len(p_print)) + [1])
                        else:
                            break
                    
                elif l == merge_end:
                    l_merge_to = path_merge_dict[idx][0][2]
                    for m in range(int(l_merge_to-l)):
                        print_dict[idx].append(4)
                    print_dict[idx].append(fold_line)
                    for p in range(i-1,-1,-1):
                        p_idx = print_df.loc[p]['idx']
                        p_print = print_dict[p_idx]
                        if len(p_print) >= l_merge_to+1 and p_print[l_merge_to]==0:
                            print_dict[p_idx][l_merge_to] = 5
                        elif len(p_print) < l_merge_to+1:
                            print_dict[p_idx].extend([0]*(l_merge_to-len(p_print)) + [5])
                        else:
                            break
                        
                elif l > merge_end:
                    continue
                else:
                    print_dict[idx].append(-1)
        
        return print_dict
        
    def print_cluster(self, path_merge_dict, path_dist_table):
        left_dict = self.print_side('left', path_merge_dict, path_dist_table)
        right_dict = self.print_side('right', path_merge_dict, path_dist_table)
        full_dict = left_dict | right_dict
        full_dict[self.main_idx] = [-1] * int(self.main_len)

        ordered_idxs = list(self.border_queues['left']['idx'])[::-1] + list(self.queues['left']['idx'])[::-1] + [self.main_idx] + list(self.queues['right']['idx']) + list(self.border_queues['right']['idx']) 
        ordered_idxs = np.array(ordered_idxs).astype(int)
        full_print_dict = {idx:full_dict[idx] for idx in ordered_idxs}
        return full_print_dict


class path_collector():
    def __init__(self, path_list, atol_arr=None):
        self.rtol = 1e-05
        if atol_arr is None:
            self.atol_arr = None
        else:
            self.atol_arr = np.array(atol_arr)
            # cleaned_path_list = []
            # for path in path_list:
            #     repeated = False
            #     for c_path in cleaned_path_list:
            #         if self.equal_path(path, c_path):
            #             repeated = True
            #             break
            #     if not repeated:
            #         cleaned_path_list.append(path)
            # path_list = cleaned_path_list
        
        n_path = len(path_list)
        
        self.path_list = path_list
        path_dist_table = np.zeros([n_path, n_path])
        for i in range(n_path):
            path_dist_table[i,i] = len(path_list[i])
            for j in range(i+1, n_path):
                ij_dist = self.find_header_len(path_list[i], path_list[j])
                path_dist_table[i,j] = path_dist_table[j,i] = ij_dist
        self.n_path = n_path
        self.path_dist_table = path_dist_table
        
        self.path_ends_dict = {}
        self.path_merge_dict = {}
        for idx, path in enumerate(path_list):
            self.path_ends_dict[idx] = len(path)-1
            self.path_merge_dict[idx] = []
        
        self.cluster_pool = []
        self.idxs_in_pool = {}
    
    def find_header_len(self, path_1_df, path_2_df):
        path_1 = list(path_1_df['evt'])
        path_2 = list(path_2_df['evt'])
        if path_1 == path_2:
            return len(path_1)
        
        for s, key in enumerate(path_1):
            if not path_2[s] == key:
                break
        return s

    def equal_path(self, df_1, df_2):
        if len(df_1) != len(df_2):
            return False
        for s in range(len(df_1)):
            step_1 = df_1.loc[s]
            step_2 = df_2.loc[s]
            if not self.equal_end(step_1, step_2):
                return False
        return True

    def equal_end(self, end_1, end_2):
        if not end_1['evt']==end_2['evt']:
            return False
        values_1 = np.array(end_1[1:].values.astype(float).tolist())
        values_2 = np.array(end_2[1:].values.astype(float).tolist())
        if self.atol_arr is None:
            return np.allclose(values_1, values_2, rtol=self.rtol)
        else:
            return np.allclose(values_1, values_2, atol=self.atol_arr)
            
    def in_pool(self, in_idx):
        return in_idx in self.idxs_in_pool.keys()
    
    def insert_single(self, idx):
        #print(f"insert_single [{idx}]")
        if self.in_pool(idx):
            return
        idx_merges = self.path_merge_dict[idx]
        merge_insert_res = False
        if len(idx_merges)>0:
            for m, idx_merge in enumerate(idx_merges):
                m_root = idx_merge[1]
                if not self.in_pool(m_root):
                    self.insert_single(m_root)
                m_merge = idx_merge[0]
                merge_insert_res = self.insert_pair((m_root, idx), m_merge)
                if merge_insert_res is True:
                    break
        
        if merge_insert_res is False:
            # no merges, or merge insert failed
            # if self.count_list[idx] > 1000:
            #     new_cluster = path_cluster(idx, len(self.path_list[idx]))
            #     self.cluster_pool.append(new_cluster)
            #     self.refresh_idxs_in_pool()
            #     return
            
            idx_len = len(self.path_list[idx])
            insert_list = []
            for cluster_id, cluster in enumerate(self.cluster_pool):
                suggest_res = cluster.suggest_root(idx, self.path_dist_table)
                if suggest_res is not None:
                    c_root, c_dist = suggest_res
                    c_idxs = cluster.get_idxs()
                    c_size = len(c_idxs)
                    c_tuple = (cluster_id, c_root, c_dist, c_size)
                    insert_list.append(c_tuple)
            sorted_list = sorted(insert_list, key=lambda x: (x[2], x[3]), reverse=True)
    
            insert_res = None
            for c_option in sorted_list:
                c_id, c_root, c_dist, _ = c_option
                idx_info = (c_dist, np.inf, idx_len)
                insert_res = self.cluster_pool[c_id].insert((c_root, idx), idx_info)
                # if idx == 123:
                #     print(f"insert_res = {insert_res}")
                if insert_res is True:
                    break
            if insert_res is None:
                new_cluster = path_cluster(idx, len(self.path_list[idx]))
                self.cluster_pool.append(new_cluster)
            elif type(insert_res).__name__ == 'path_cluster':
                self.cluster_pool.append(insert_res)
        
        # # re-insert main_idx only clusters
        # re_insert_list = []
        # for cluster in self.cluster_pool:
        #     if len(cluster.get_idxs())==1 and cluster.main_idx!= idx:
        #         re_idx = cluster.main_idx
        #         re_insert_list.append(re_idx)
        #         del self.idxs_in_pool[re_idx]
        
        # if len(re_insert_list)>0:
        #     cluster_pool = [c for c in self.cluster_pool if c.main_idx not in re_insert_list]
        #     self.cluster_pool = cluster_pool
        #     for re_idx in re_insert_list:
        #         #self.path_merge_dict[re_idx] = []
        #         self.insert_single(re_idx)
        
        self.refresh_idxs_in_pool()
        return


    def refresh_idxs_in_pool(self):
        idxs_in_pool = {}
        for cluster_id, cluster in enumerate(self.cluster_pool):
            for idx in cluster.get_idxs():
                idxs_in_pool[idx] = cluster_id
        self.idxs_in_pool = idxs_in_pool

        return
    
    def insert_pair(self, in_pair, idx_merge):
        #print(f"insert_pair [{in_pair}]")
        root_idx, idx = in_pair
        idx_len = len(self.path_list[idx])
        idx_dist = self.path_dist_table[root_idx, idx]
        idx_info = (idx_dist, idx_merge, idx_len)
        cluster_id = self.idxs_in_pool[root_idx]
        insert_res = self.cluster_pool[cluster_id].insert((root_idx, idx), idx_info)
        if insert_res is False:
            return False
        
        if type(insert_res).__name__ == 'path_cluster':
            self.cluster_pool.append(insert_res)
        self.refresh_idxs_in_pool()
        
        return True
    
    def recurr_merge(self, in_idxs):
        cand_idxs = [idx for idx in in_idxs if self.path_ends_dict[idx]>=0]
        merge_dict = {}
        merged_idx_list = []
        
        while len(cand_idxs)>0:
            i = cand_idxs.pop(0)
            if i in merged_idx_list:
                continue
            path_end = self.path_list[i].loc[self.path_ends_dict[i]]
            for j in cand_idxs:
                if j in merged_idx_list:
                    continue
                pointer_end = self.path_list[j].loc[self.path_ends_dict[j]]
                if self.equal_end(path_end, pointer_end):
                    if i in merge_dict.keys():
                        merge_dict[i].append(j)
                        merged_idx_list.append(j)
                    else:
                        merge_dict[i] = [j]
                        merged_idx_list.extend([i,j])
        
        if len(merge_dict)==0:
            return in_idxs
        
        for _, (key, vals) in enumerate(merge_dict.items()):
            key_end = self.path_ends_dict[key]
            for v in vals:
                self.path_merge_dict[v].insert(0, (self.path_ends_dict[v], key, key_end))

            merge_idxs = [key] + vals
            for idx in merge_idxs:
                self.path_ends_dict[idx] = self.path_ends_dict[idx] - 1
            
            left_idx_list = self.recurr_merge(merge_idxs)
            for idx in left_idx_list:
                self.insert_single(idx)
                # if idx != key:
                #     if not self.in_pool(key):
                #         self.insert_single(key)
                #     idx_merge = self.path_merge_dict[idx][0][0]
                #     self.insert_pair((key, idx), idx_merge)

        single_idx_list = [idx for idx in in_idxs if idx not in merged_idx_list]
        return single_idx_list
        
    def print_paths(self):
        for c_id, cluster in enumerate(self.cluster_pool):
            c_print_dict = cluster.print_cluster(self.path_merge_dict, self.path_dist_table)
            alert_str = ''
            print(f"==== cluster #{c_id} ==== \n")
            for idx in list(c_print_dict.keys()):
                path_print = c_print_dict[idx]
                path_df = self.path_list[idx]       # columns=['evt', 'kurt', 'size', 'rate', 'true_size']
                #path_count = self.count_list[idx]
                #path_str = f"[{idx:3d}]({path_count:4d}): "
                path_str = f"[{idx:3d}]: "
                for l, val in enumerate(path_print):
                    if val == 0:
                        path_str += ' ' * 33 #29
                    elif val == 1:
                        path_str += ' ' * 27 + "│" + " " * 5  # ' ' * 23 + "│" + " " * 5
                    elif val == 2:
                        path_str += ' ' * 27 + "└───> "  #' ' * 23 + "└───> "
                    elif val == -2:
                        path_str += ' ' * 27 + "┌───> "  #' ' * 23 + "┌───> "
                    elif abs(val) == 3:
                        my_merge = self.path_merge_dict[idx][0]
                        if not (my_merge[2]==l):
                            alert_str += f"!!!!! Alert [{idx}] l = {l}, path_merge_dict[idx]={self.path_merge_dict[idx]}  \n"
                        path_str = path_str[:-3]
                        if val>0:
                            path_str += f"───>───┘ [{my_merge[1]:3d}, {my_merge[2]:d}]" + ' ' * 19 #15
                        else:
                            path_str += f"───>───┐ [{my_merge[1]:3d}, {my_merge[2]:d}]" + ' ' * 19 #15 
                    elif val == 4:
                        path_str = path_str[:-3] + '─' * 33 + path_str[-3:]  #path_str[:-3] + '─' * 29 + path_str[-3:]
                    elif val == 5:
                        path_str += ' ' * 4 + "│" + " " * 28  #24
                    else:
                        row = list(path_df.iloc[l])
                        path_str += f"{row[0]}({row[1]:>6.2f};{row[4]:>4d}/{row[2]:>4d};{int(row[3]*100):>3d}%)"
                        if l < len(path_df)-1:
                            path_str += "-> "
                        else:
                            path_str += "   "
                print(path_str)
                
            print("")
            if len(alert_str)>0:
                print(alert_str)
        return

LITT_path = './'

class Path_Analyzer():
    def __init__(self): #, data_manager):
        super().__init__()
        self.event_fname = os.path.join(LITT_path, 'event_data.pkl')
        self.timing_fname = os.path.join(LITT_path, 'full_LITT_timing.pkl')
        self.flip_timing_fname = os.path.join(LITT_path, 'full_LITT_flip_timing.pkl')
        #self.timing_fname = os.path.join(LITT_path, data_manager.split_fname_prex + 'LITT_timing.pkl')
        #self.flip_timing_fname = os.path.join(LITT_path, data_manager.split_fname_prex + 'LITT_flip_timing.pkl')
        #self.data_manager = data_manager
        
        # gamma_coeff and flip_gamma_coeff guarenteed loaded
        #self.gamma_coeff = data_manager.gamma_coeff
        #self.flip_gamma_coeff = data_manager.flip_gamma_coeff
        
        self.n_seqs = 3276 #data_manager.n_seqs
        self.max_len = 489 #data_manager.max_len
        self.n_flip = 835  #data_manager.n_flip
        #self.flip_idxs = data_manager.flip_idxs
        
    def loading(self):
        
        # load events
        with open(self.event_fname, 'rb') as f:
            loaded_event_df = pickle.load(f)
        event_df_step = loaded_event_df['event_df_step']
        event_df_day = loaded_event_df['event_df_day']
        disp_columns = {}
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
        
        #----- for no data loading -----
        return
        
        
        # # load trained timing
        # with open(self.timing_fname, 'rb') as f:
        #     self.timing_list, self.label_list = pickle.load(f)
            
        # load trained flipped timing
        if os.path.exists(self.flip_timing_fname):
            print(f"Loading existing timing from {self.flip_timing_fname} ...")
            with open(self.flip_timing_fname, 'rb') as f:
                self.flip_timing_list, self.flip_label_list = pickle.load(f)
        else:
            self.flip_gamma_coeff.eval()
            
            train_dataloader_Event = self.data_manager.get_Flip_DataLoader(flip_event_tag = True)
            self.flip_timing_list = [np.nan for i in range(self.n_flip)]
            self.flip_label_list = [np.nan for i in range(self.n_flip)]
            
            with torch.no_grad():
                # training timing only
                for time_sequences, _, list_coeff_seq, _, _, seq_idxs, target_masks in train_dataloader_Event:
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
                        reltv_t_seq = [0]
                        
                        for j in range(len(reltv_delta_t_seq)):
                            j_reltv_t = sum(reltv_delta_t_seq[:(j+1)])
                            # if j_reltv_t==float('Inf'):
                            #     cut_len = len(reltv_t_seq)
                            #     t_seq = t_seq[:cut_len] 
                            #     break
                            # else:
                            reltv_t_seq.append(float(j_reltv_t))
                        
                        bi_label = target_masks[i].cpu().numpy()
                        timing_df = {'t_seq':t_seq, 'reltv_t_seq':reltv_t_seq}
                        timing_df = pd.DataFrame(timing_df)
                        self.flip_timing_list[idx] = timing_df
                        self.flip_label_list[idx] = bi_label

            with open(self.flip_timing_fname, 'wb') as f:
                pickle.dump((self.flip_timing_list, self.flip_label_list), f)
            
    def select_path(self, recur_dict, slt_keys = [], kurt_bar = 6):
        path_list = []
        path_body_idxs_list = []
        self.recur_dict = recur_dict
        self.unique_evts = []
        if len(slt_keys)==0:
            slt_keys = list(disp_dict.keys())
        def get_path_recur(cur_pointer, cur_path_df):
            next_evts = list(cur_pointer['next'].keys())
            if len(next_evts)==0:
                path_list.append(cur_path_df)
                idxs_arr = np.array(cur_pointer['idxs'])
                body_tags = np.array(cur_pointer['body_tags'])
                body_idxs = idxs_arr[body_tags]
                path_body_idxs_list.append(body_idxs)
                return
            
            for i, evt in enumerate(next_evts):
                evt_split = evt.split('_')
                evt_name = '_'.join(evt_split[:-1])
                evt_count = evt_split[-1]
                disp_evt = str(disp_dict[evt_name]) + '_' + evt_count
                # select partual paths
                if evt_name not in slt_keys:
                    continue
                # ---
                evt_kurt = cur_pointer['next'][evt]['kurt']
                evt_size = len(cur_pointer['next'][evt]['idxs'])
                evt_pos_tags = cur_pointer['next'][evt]['pos_tags']
                evt_rate = float(np.round(sum(evt_pos_tags)/evt_size,3))
                evt_true_size = len(cur_pointer['next'][evt]['true_idxs'])
                if np.isnan(evt_kurt) or evt_size==0 or evt_true_size==0 or evt_kurt<kurt_bar:
                    continue
                if disp_evt not in self.unique_evts:
                    self.unique_evts.append(disp_evt)
                # ---
                evt_list = [disp_evt, evt_kurt, evt_size, evt_rate, evt_true_size]
                evt_path_df = cur_path_df.copy()
                evt_path_df.loc[len(evt_path_df)] = evt_list
                # ---
                get_path_recur(cur_pointer['next'][evt], evt_path_df)
            
            
        recur_header = {'next':recur_dict}
        init_df = pd.DataFrame(columns=['evt', 'kurt', 'size', 'rate', 'true_size'])
        get_path_recur(recur_header, init_df)
        self.path_list = path_list
        self.path_body_idxs_list = path_body_idxs_list
        return path_list


    def compact_path(self, atol_arr=None, pre_evts=[]):
        path_pool = path_collector(self.path_list, atol_arr)
        
        n_path = len(self.path_list)
        ordered_idxs = list(range(n_path))
        
        order_dict = {}
        for idx, path in enumerate(self.path_list):
            order_dict[idx] = len(path)
        ordered_dict = dict(sorted(order_dict.items(), key=lambda item: item[1], reverse=True))
        ordered_idxs = list(ordered_dict.keys())
        
        pre_len = len(pre_evts)
        if pre_len>0:
            separator = "|"
            pre_path_str = separator.join(map(str, pre_evts))
            pre_ordered_idxs = []
            for idx in ordered_idxs:
                idx_path = self.path_list[idx]['evt'].tolist()
                
                idx_path_str = separator.join(map(str, idx_path))
                if idx_path_str.find(pre_path_str)>=0:
                #idx_pre = self.path_list[idx]['evt'].tolist()[:pre_len]
                #if list(idx_pre) == list(pre_evts):
                    pre_ordered_idxs.append(idx)
            ordered_idxs = pre_ordered_idxs

        single_idxs = path_pool.recurr_merge(ordered_idxs)
        for idx in single_idxs:
            path_pool.insert_single(idx)
        
        print(f"n_path = {len(ordered_idxs)}")
        print(f"n_cluster = {len(path_pool.cluster_pool)}")
        path_pool.print_paths()
        
        return path_pool



    def compare_timing(self):
        return
