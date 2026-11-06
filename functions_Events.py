# -*- coding: utf-8 -*-
"""
Created on Sun Jan 25 14:01:49 2026

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

demo_vars = ['Age', 'Race']

numeric_vars = ['ldl', 'hdl', 'hba1c', 'troponin', 'bnp', 'wbc', 'ab_gluc', 'ab_creat', 'weight', 'height', 'sbp', 'dbp', 'bmi']

binary_vars = ['anthracyclins','targeted','radiation','metformin','insulin', 'sulfonylurea', 'sglt2_inhib', 'glp1_recpt', 'ezetimibe', 'pcsk9_inhib', 'fibrates', 'bile_acid', 'niacin', 'diuretics',
           'hyperlipidemia', 'diabetes', 'hypertension', 'obesity', 'smoking', 'ckd']


event_setting = {'anthracyclins':[1,2,3],'targeted':[1,2,3],'radiation':[1,2,3],
                 'metformin':[1],'insulin':[1], 'sulfonylurea':[1], 'sglt2_inhib':[1], 'glp1_recpt':[1], 'ezetimibe':[1], 'pcsk9_inhib':[1], 'fibrates':[1], 'bile_acid':[1], 'niacin':[1], 'diuretics':[1],
                 'hyperlipidemia':[1,2,3], 'diabetes':[1,2,3], 'hypertension':[1,2,3], 'obesity':[1,2,3], 'smoking':[1,2,3], 'ckd':[1,2,3]}

disp_dict = {'anthracyclins':'Chemo', 'targeted':'Targt', 'radiation':'Radia',
             'metformin':'Md_Metf','insulin':'Md_Insl', 
             'sulfonylurea':'Md_Sulf', # second line after insulin
             'sglt2_inhib':'Md_SGLT2',  # benifit cardiovascular but pressure to kidney
             'glp1_recpt':'Md_GLP1', # mimic the natural hormone GLP-1, stimulating insulin release
             'ezetimibe': 'Md_Ezt', # inhibit cholesterol absorption in intestine; reduce cardiovascular risk)
             'pcsk9_inhib':'Md_PCSK9', # lower LDL "bad" cholesterol, well-tolerated, limited side effect
             'fibrates':'Md_Fibr', # raise "good" HDL cholesterol; reduce heart disease risk
             'bile_acid':'Md_BAS', # lower bad LDL cholesterol by binding to bile acids
             'niacin':'Md_VB3', # essential water-soluble nutrient; raising "good" HDL and lowering "bad" LDL
             'diuretics':'Md_Diu', # Water pill; help kidneys, treat hert failure
             'hyperlipidemia':'Dx_HL', 'diabetes':'Dx_DB', 'hypertension':'Dx_HT', 'obesity':'Dx_Ob', 'smoking':'Dx_Smk', 'ckd':'Dx_CKD'}

orig_dict = {val:key for key, val in disp_dict.items()}
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

LITT_path = './'

class Event_Analyzer():
    def __init__(self): #, data_manager):
        super().__init__()
        #self.event_fname = os.path.join(data_split_path, 'event_data.pkl')
        self.event_fname = os.path.join(LITT_path, 'event_data.pkl')
        #self.data_manager = data_manager
        #self.gamma_coeff = data_manager.gamma_coeff
        self.n_seqs = 3276 #data_manager.n_seqs
        self.max_len = 489 #data_manager.max_len
        
    def load_events(self):
        if not os.path.exists(self.event_fname):
            self.get_events()
        
        with open(self.event_fname, 'rb') as f:
            loaded_event_df = pickle.load(f)
        self.event_df_step = loaded_event_df['event_df_step']
        self.event_df_day = loaded_event_df['event_df_day']
        self.events = list(self.event_df_day.columns)
        return
    
    def get_events(self):
        sys.exit("No source data! Please load "+self.event_fname)
        sequences = self.data_manager.sequences
        time_sequences = self.data_manager.time_sequences
        
        event_arr = []
        for _, (key, value) in enumerate(event_setting.items()):
            assert(0 not in value)
            for v in value:
                event_arr.append(key+'_'+str(int(v)))

        event_df_step = pd.DataFrame(columns=event_arr)
        event_df_day = pd.DataFrame(columns=event_arr)
        
        for i in range(self.data_manager.n_seqs):
            event_df_step.loc[i] = np.nan
            event_df_day.loc[i] = np.nan
            tensor_x = sequences[i].cpu()
            df_x = pd.DataFrame(tensor_x.numpy(), columns=var_list)
            
            tensor_time = time_sequences[i].cpu()
            arr_time = np.array(tensor_time.numpy().flatten(), dtype=np.int32)
            for _, (key, counts) in enumerate(event_setting.items()):
                pos_steps = np.where(np.array(df_x[key])==1)[0]
                for c in counts:
                    if c > len(pos_steps):
                        break
                    step_value = pos_steps[c-1]
                    day_value = arr_time[step_value]
                    df_col = key+'_'+str(int(c))
                    
                    event_df_step.loc[i, df_col] = int(step_value)
                    event_df_day.loc[i, df_col] = int(day_value)
        
        # print summary
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 150)
        print('Step Table Summary:')
        step_non_zero_counts = (pd.notna(event_df_step)).sum()
        print(pd.DataFrame(step_non_zero_counts).transpose())

        print('Day Table Summary:')
        day_non_zero_counts = (pd.notna(event_df_day)).sum()
        print(pd.DataFrame(day_non_zero_counts).transpose())
        
        data_to_save = {
            'event_df_step': event_df_step,
            'event_df_day': event_df_day
        }
        with open(self.event_fname, 'wb') as f:
            pickle.dump(data_to_save, f)
    
    def load_timing(self):
        #...
        #########################
        # The core code is hidden due to copyright issue
        # Contact the author for full version
        #########################
        

    def do_analyze(self, log_out=True):
        self.load_events()
        self.load_timing()
        
        # training valid only
        event_df_reltv_day = pd.DataFrame(columns=self.event_df_day.columns, index=self.event_df_day.index)
        
        binary_labels = [np.nan for i in range(self.n_seqs)]
        
        for idx in self.train_indices:
            label_arr = self.label_list[idx]
            label = any(label_arr==1)
            binary_labels[idx] = label
            
            event_row = self.event_df_step.loc[idx]
            timing_df = self.timing_list[idx]
            for key, step in event_row.items():
                if step > 0:
                    reltv_day = timing_df.loc[step]['reltv_t_seq']
                    event_df_reltv_day.loc[idx, key] = reltv_day
        
        self.event_df_reltv_day = event_df_reltv_day
        binary_labels = np.array(binary_labels)
        
        pat_idxs = self.train_indices

        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 300)
        
        np.seterr(divide='ignore')
        
        # ----- start ------
        
        
        recur_queue = []
        recur_dict = {}
        recur_queue.append((recur_dict, [], pat_idxs))
        n_round = 0
        while len(recur_queue)>0:
            cur_task = recur_queue.pop(0) # pointer and a pair of lists
            cur_pointer, cur_pre_path, cur_true_idxs = cur_task
            event_eval_df, event_idxs_dict = self.eval_events(cur_pre_path, cur_true_idxs)
            if log_out:
                print(f"Recurr Round {n_round}: pre_path = {cur_pre_path}, true_idxs_size = {len(cur_true_idxs)}")
                print("    event_eval_df is:")
                print(event_eval_df)
                print("-----------------------------------")
            
            event_recoder = {}
            for key, value in event_setting.items():
                event_recoder[key] = 0
            for evt in event_eval_df.columns:
                evt_kurt = event_eval_df.loc['kurt', evt]
                evt_idxs = list(event_idxs_dict[evt]['idxs'])
                evt_body_tags = list(event_idxs_dict[evt]['tags'])
                evt_true_idxs = list(event_idxs_dict[evt]['true_idxs'])
                evt_true_size = event_eval_df.loc['true_size', evt]
                evt_labels = list(binary_labels[evt_idxs])
                cur_pointer[evt] = {'next':{}, 'kurt':evt_kurt, 'idxs':evt_idxs, 'body_tags':evt_body_tags, 'pos_tags':evt_labels, 'true_idxs':evt_true_idxs}
                
                evt_split = evt.split('_')
                event = '_'.join(evt_split[:-1])
                count = int(evt_split[-1])
                if evt_true_size>0 and count > event_recoder[event] and evt_kurt>6:
                    recur_queue.append((cur_pointer[evt]['next'], cur_pre_path+[evt], list(evt_true_idxs)))
                    event_recoder[event] = count
            n_round += 1
        #self.print_path(recur_dict)
        #self.print_tree(recur_dict)
        
        #self.disp_3heatmap([("radiation",1)], [("anthracyclins",1)], [("targeted",1)])

        return recur_dict
            
            
    def print_path(self, recur_dict):
        
        def path_recur(cur_pointer, cur_evt, cur_kurt, cur_print):
            if len(cur_pointer['next']) == 0:
                print(f"{cur_print} {cur_evt}({cur_kurt:.2f}).")
            else:
                my_print = f"{cur_print} {cur_evt}({cur_kurt:.2f}) ->"
                for evt in list(cur_pointer['next'].keys()):
                    path_recur(cur_pointer['next'][evt], evt, cur_pointer['next'][evt]['kurt'], my_print)

        for evt in list(recur_dict.keys()):
            path_recur(recur_dict[evt], evt, recur_dict[evt]['kurt'], "")

    def print_tree(self, recur_dict):

        def tree_recur(cur_pointer, cur_evt_list, cur_print="", is_last=True):
            cur_evt, cur_kurt, cur_size, cur_rate = cur_evt_list
            cur_evt_split = cur_evt.split('_')
            evt_name = '_'.join(cur_evt_split[:-1])
            evt_count = cur_evt_split[-1]
            disp_evt = str(disp_dict[evt_name]) + '_' + evt_count
            print(cur_print + ("└── " if is_last else "├── ") + f"{disp_evt}(kurt={cur_kurt:.2f} size={cur_size} rate={cur_rate*100:.1f}%).")
            my_print = cur_print + ("    " if is_last else "│   ")
            next_evts = list(cur_pointer['next'].keys())
            for i, evt in enumerate(next_evts):
                evt_kurt = cur_pointer['next'][evt]['kurt']
                evt_size = len(cur_pointer['next'][evt]['idxs'])
                #evt_body_tags = cur_pointer['next'][evt]['body_tags']
                #evt_rate = float(np.round(sum(evt_body_tags)/evt_size,3))
                evt_pos_tags = cur_pointer['next'][evt]['pos_tags']
                evt_rate = float(np.round(sum(evt_pos_tags)/evt_size,3))
                evt_list = [evt, evt_kurt, evt_size, evt_rate]
                tree_recur(cur_pointer['next'][evt], evt_list, my_print, i == len(next_evts) - 1)
        
        print("Timeline")
        for evt in list(recur_dict.keys()):
            evt_kurt = recur_dict[evt]['kurt']
            evt_size = len(recur_dict[evt]['idxs'])
            evt_pos_tags = recur_dict[evt]['pos_tags']
            evt_rate = float(np.round(sum(evt_pos_tags)/evt_size,3))
            tree_recur(recur_dict[evt], [evt, evt_kurt, evt_size, evt_rate])

    def get_corre_df(self, in_rout, slt_evts=[]):
        cond_events = {}
        
        for _, (key, value) in enumerate(event_setting.items()):
            cond_events[key] = 0
            
        cur_idxs = self.train_indices
        for pair in in_rout:
            key, value = pair
            cond_events[key] = value
            evt = key+'_'+str(int(value))
            evt_distrib = np.array(list(self.event_df_reltv_day.loc[cur_idxs][evt]))
            evt_filter = ~np.isnan(evt_distrib) & ~np.isinf(evt_distrib)
            evt_idxs = np.array(cur_idxs)[evt_filter]
            cur_idxs = evt_idxs

        evt_list = []
        for key in cond_events.keys():
            candi_counts = [ val for val in event_setting[key] if val > cond_events[key] ]
            if len(candi_counts) > 0:
                key_count = int(np.min(candi_counts))
                evt_list.append(disp_dict[key]+'_'+str(int(key_count)))
                
        if len(slt_evts)>0:
            filtered_evts = [evt for evt in evt_list if evt in slt_evts]
            evt_list = filtered_evts

        n_evt = len(evt_list)
        corr_matrix = np.zeros((n_evt, n_evt))

        for i in range(n_evt):
            for j in range(i+1, n_evt):
                evt1 = evt_list[i]
                evt2 = evt_list[j]
                df = pd.DataFrame({
                    evt1: list(self.event_df_reltv_day.loc[cur_idxs][evt1]),
                    evt2: list(self.event_df_reltv_day.loc[cur_idxs][evt2])
                })
                df_complete = df.dropna()
                assoc = float(df_complete.corr().loc[evt1,evt2])
                corr_matrix[i, j] = assoc
                corr_matrix[j, i] = assoc
        
        np.fill_diagonal(corr_matrix, 1)
        df_corr = pd.DataFrame(corr_matrix, index=evt_list, columns=evt_list)
        return df_corr
    
    def disp_3heatmap(self, rout_0, rout_1, rout_2, slt_evts):
        #slt_evts = ['targeted_1', 'metformin_1', 'insulin_1', 'sulfonylurea_1', 'glp1_recpt_1', 'diuretics_1', 'hyperlipidemia_1', 'diabetes_1', 'hypertension_1', 'obesity_1', 'smoking_1', 'ckd_1']
        #slt_evts = ['sglt2_inhib_1', 'glp1_recpt_1', 'ezetimibe_1', 'fibrates_1', 'bile_acid_1', 'niacin_1']
        #slt_evts = ['diuretics_1', 'hyperlipidemia_1', 'diabetes_1', 'hypertension_1', 'obesity_1', 'smoking_1', 'ckd_1']

        corr_df_list = []
        for rout in [rout_0, rout_1, rout_2]:
            rout_corr_df = self.get_corre_df(rout, slt_evts)
            old_names = list(rout_corr_df.columns)
            new_names = [disp_dict[evt] if (evt in disp_dict) else evt for evt in old_names]
            rout_corr_df.columns = new_names
            rout_corr_df.index = new_names
            corr_df_list.append(rout_corr_df)
        
        
        title_arr = ['Given 1st Radiation', 'Given 1st Chemotherapy', 'Given 1st Targeted-Therapy  ']
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)  # 1 row, 3 columns
        for i, ax in enumerate(axes):
            corr_df = corr_df_list[i]
            mask = np.zeros_like(corr_df, dtype=bool)
            np.fill_diagonal(mask, True)  # True = masked (hidden)
            
            sns.heatmap(
                corr_df,
                ax = ax,
                annot=True,                # show numbers in cells
                fmt=".2f",                 # 2 decimal places
                cmap="coolwarm",           # red-blue diverging colormap
                vmin=-1, vmax=1,           # for correlation-like values
                center=0,
                mask=mask,
                square=True,
                linewidths=0.5,
                cbar=False,
                annot_kws={"size": 8}
            )
            for j in range(len(corr_df)):
                ax.add_patch(plt.Rectangle((j, j), 1, 1, fill=True, color='darkgrey', alpha=0.9, zorder=2))
        
            ax.set_title(title_arr[i], fontsize=14, pad=12)
            ax.tick_params(axis='both', which='major', labelsize=11)
            
            # Rotate x-ticks if labels are long
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=11)
            plt.setp(ax.get_yticklabels(), rotation=0, fontsize=11)
        
        # Optional: shared colorbar (recommended for 3 heatmaps)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        fig.colorbar(axes[0].collections[0], cax=cbar_ax, label="Association Strength", 
                     orientation="vertical", aspect=30)
        cbar_ax.yaxis.label.set_size(13)  # colorbar label font size
        
        # Overall figure title
        fig.suptitle("Association Heatmaps for Three Different Conditional Scenarios", fontsize=16, y=1.02)
        
        #plt.tight_layout(rect=[0, 0, 0.9, 1])  # leave space for colorbar
        plt.show()


    def check_hist_path(self, in_idx, in_evt, pre_path):
        separator = "|"
        pat_row = self.event_df_step.loc[in_idx]
        pat_row = pat_row[pd.notna(pat_row)]
        pat_path = pat_row.sort_values().index.tolist()
        pat_str_path = separator.join(map(str, pat_path))
        evt_loc = -1
        for i, evt in enumerate(pre_path):
            if pat_str_path.find(evt) > evt_loc:
                evt_loc = pat_str_path.find(evt)
            else:
                return False
        if pat_str_path.find(in_evt) > evt_loc:
            return True
        else:
            return False
            

    def eval_events(self, pre_path, idxs_slc):
        np.seterr(divide='ignore', invalid='ignore')
        
        event_arr = list(self.event_df_step.columns)

        eval_df = pd.DataFrame(columns=event_arr, index=['kurt', 'size', 'body_rate', 'true_size'])
        idxs_dict = {}

        idxs_slc = list(idxs_slc)
        assert(len(idxs_slc) > 4)
        
        if len(pre_path)>0:
            last_evt = pre_path[-1]
            evt_zero = self.event_df_reltv_day.loc[idxs_slc][last_evt]
        else:
            evt_zero = [0 for i in idxs_slc]
        
        
        for evt in event_arr:
            evt_distrib = np.array(list(self.event_df_reltv_day.loc[idxs_slc][evt] - evt_zero))
            evt_filter = ~np.isnan(evt_distrib) & ~np.isinf(evt_distrib) & ~(evt_distrib<0)
            evt_idxs = np.array(idxs_slc)[evt_filter]
            evt_distrib = evt_distrib[evt_filter]
            evt_size = len(evt_distrib)
            if evt_size>20:
                evt_kurt, evt_body_tag = self.get_kurt(evt_distrib)
                #print(f" event = {evt}")
                if evt_kurt<3:
                    continue
                eval_df.loc['kurt', evt] = evt_kurt
                eval_df.loc['size', evt] = int(evt_size)    # original size
                eval_df.loc['body_rate', evt] = float(np.round(sum(evt_body_tag)/evt_size,3))
                
                body_idxs = np.array(evt_idxs[evt_body_tag])
                true_idxs = []
                for idx in body_idxs:
                    if self.check_hist_path(idx, evt, pre_path):
                        true_idxs.append(idx)
                true_idxs = np.array(true_idxs)
                idxs_dict[evt] = {'idxs':evt_idxs, 'tags':evt_body_tag, 'true_idxs':true_idxs}
                eval_df.loc['true_size', evt] = len(true_idxs)
        rm_evts = []
        for evt in event_arr:
            if all(pd.isna(eval_df[evt])):
                rm_evts.append(evt)
                if evt in idxs_dict.keys():
                    del idxs_dict[evt]
        rm_evts = eval_df.drop(columns=rm_evts)
        rm_evts = rm_evts.sort_values(by=['kurt','true_size'], ascending=False, axis=1)
        return rm_evts, idxs_dict
    
    def get_kurt(self, dist):
        normal_kurt = kurtosis(dist, fisher=True)
        mu = np.mean(dist)
        sigma = np.std(dist)
        body_tag_arr = np.ones_like(dist)==1
        
        #print(f"kurtosis = {normal_kurt}, mu = {mu}, sigma = {sigma},", end='')

        if normal_kurt > 3:    # leptokurtic → heavier tails → use stricter threshold
            z_tail_threshold = 3.0
        elif normal_kurt < -0.5:  # platykurtic → lighter tails → more liberal
            z_tail_threshold = 2.0
        else:
            z_tail_threshold = 2.5  # normal-ish
    
        z_scores = np.abs((dist - mu) / sigma)
        body_tag_arr[z_scores >= z_tail_threshold] = False
        
        return normal_kurt, body_tag_arr
    
    
    def disp_kurt_KDE(self, recur_dict, in_rout):
        cur_next = recur_dict
        for pair in in_rout:
            key, value = pair
            evt = key+'_'+str(int(value))
            cur_pointer = cur_next[evt]
            cur_next = cur_pointer['next']
        
        full_idxs = list(cur_pointer['idxs'])
        dist = np.array(list(self.event_df_reltv_day.loc[full_idxs][evt]))
        
        normal_kurt = cur_pointer['kurt']
        body_tag_arr = cur_pointer['body_tags']
        z_tail_threshold = 3.0
        
        
        kde_bw = np.ceil(0.3*np.std(dist))
        kde = KernelDensity(bandwidth=kde_bw).fit(dist[:, np.newaxis])
        x_grid = np.linspace(dist.min() - 0.5 * dist.std(), dist.max() + 0.5 * dist.std(), 1000)[:, np.newaxis]
        log_dens = kde.score_samples(x_grid)
        density = np.exp(log_dens)
        mode_val = x_grid[np.argmax(density)][0]
        
        plt.figure(figsize=(14, 6))
        plt.hist(dist[body_tag_arr == True], bins=40, alpha=0.6, color='skyblue', label=f'Body (z < {z_tail_threshold})')
        plt.hist(dist[body_tag_arr == False], bins=40, alpha=0.6, color='salmon', label=f'Tail (z ≥ {z_tail_threshold})')
        plt.axvline(mode_val, color='green', linestyle='--', lw=2, label=f'Estimated mode = {mode_val:.2f}')
        
        plt.title(f'Distribution (excess kurtosis = {normal_kurt:.3f})\nTail vs Body classification')
        plt.xlabel('Value')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        print(f"Excess kurtosis: {normal_kurt:.4f}")
        print(f"Number of tail samples: {np.sum(body_tag_arr == False)} ({np.sum(body_tag_arr == False)/len(body_tag_arr)*100:.1f}%)")
        print(f"Number of body samples: {np.sum(body_tag_arr == True)}")
        
    
    
        
        
