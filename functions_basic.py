# -*- coding: utf-8 -*-
"""
Created on Sun Jul  6 13:36:32 2025

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

from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_curve
from sklearn.metrics import precision_score, recall_score, f1_score

import warnings
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)



# in_dict = {'ldl':0, 'hdl':1, 'hba1c':2, 'troponin':3, 'bnp':4, 'wbc':5, ###'hr':6, 'ab_gluc':7, 'ab_creat':8, 'weight':9, 'height':10, 'sbp':11, 'dbp':12, 'bmi':13,
#            'anthracyclins':14,'targeted':15,'radiation':16,'metformin':17,'insulin':18, 'sulfonylurea':19, 'sglt2_inhib':20, 'glp1_recpt':21, 'ezetimibe':22, 'pcsk9_inhib':23, 'fibrates':24, 'bile_acid':25, 'niacin':26, 'diuretics':27,
#            'hyperlipidemia':28, 'diabetes':29, 'hypertension':30, 'obesity':31, 'smoking':32, 'ckd':33,
#            'Age':34, 'Race':35} #, 'RecordID':36}

var_list = ['ldl', 'hdl', 'hba1c', 'troponin', 'bnp', 'wbc', 'ab_gluc', 'ab_creat', 'weight', 'height', 'sbp', 'dbp', 'bmi',
           'anthracyclins','targeted','radiation','metformin','insulin', 'sulfonylurea', 'sglt2_inhib', 'glp1_recpt', 'ezetimibe', 'pcsk9_inhib', 'fibrates', 'bile_acid', 'niacin', 'diuretics',
           'hyperlipidemia', 'diabetes', 'hypertension', 'obesity', 'smoking', 'ckd',
           'Age', 'Race'] #, 'RecordID':36}
var_dict = {s: i for i, s in enumerate(var_list)}
dim_x = len(var_list)

#full_outcomes = ['HF','IS','ARR','AM','CA']


csv_data_path = './data_csv_files/BC_csv_data'
oc_db_fname = './data_csv_files/BC_outcomes_inDays.csv'
alldata_fname = './data_csv_files/BC_data_all.csv'


data_split_path = './data_splits/'
save_path = './BC_T2_training_save/'
result_path = './BC_test_save/'

LITT_path = './BC_LITT/'

def load_sourcedata():
    dataset_fname = os.path.join(data_split_path, 'sequence_data.pkl')
    
    if not os.path.exists(dataset_fname):
        print('\n\n=== start data pre-processing to generate '+dataset_fname+' ===')
        if not os.path.isdir(csv_data_path):
            n_split_f = separate_alldata(alldata_fname, csv_data_path)
            print('  %d patient .csv files are split.' % (n_split_f))
            #sys.exit(f"input csv file path '{csv_data_path}' not exist!")
        if not os.path.exists(oc_db_fname):
            sys.exit(f"input outcome file (from database) '{oc_db_fname}' not exist!")
        print('  compacting all .csv to '+dataset_fname+' ---')
        pat_x_describe = preprocess_csv_data(csv_data_path, oc_db_fname, dataset_fname, var_list)
        print(pat_x_describe['length'])
        print('=== data pre-processing done ===\n\n')
    
    with open(dataset_fname, 'rb') as f:
        loaded_data = pickle.load(f)
    return loaded_data
    

def exist_split(in_seed, split_rate):
    perc = str(int(split_rate*100))
    fname = os.path.join(data_split_path, f'split_{perc}[{str(in_seed)}]_data.pkl')
    return os.path.exists(fname)

def load_split(in_seed, split_rate):
    assert(split_rate<1 and split_rate>0)
    perc = str(int(split_rate*100))
    fname = os.path.join(data_split_path, f'split_{perc}[{str(in_seed)}]_data.pkl')
    if not os.path.exists(fname):
        sys.exit(f"split file '{fname}' not exist!")
    
    with open(fname, 'rb') as f:
        split_save = pickle.load(f)
    train_indices = split_save['train_indices']
    test_indices = split_save['test_indices']
    return train_indices, test_indices

def get_all_seeds():
    filenames = os.listdir(data_split_path)
    split_files = [f for f in filenames if f.startswith('split_') and f.endswith('_data.pkl')]
    exist_seeds = []
    for _, f in enumerate(split_files):
        f_seed = int(f.split("_")[1][:-1].split("[")[1])
        exist_seeds.append(f_seed)
    return exist_seeds

def random_new_seed():
    exist_seeds = get_all_seeds()
    # Randomly pick one
    candidates = np.array([x for x in range(20, 400) if x not in exist_seeds])
    new_seed = np.random.choice(candidates)
    return new_seed


def create_split(dataset, in_seed, split_rate):
    # "split_perc[seed]_data.pkl"
    assert(split_rate<1 and split_rate>0)
    n_total = len(dataset)
    
    perc = str(int(split_rate*100))
    # Set random seed for reproducibility
    torch.manual_seed(in_seed)
    
    n_train = int(n_total * split_rate)
    n_test = n_total - n_train
    train_dataset, test_dataset = random_split(dataset, [n_train, n_test])
    train_indices = train_dataset.indices
    test_indices = test_dataset.indices
    
    new_f = os.path.join(data_split_path, f'split_{perc}[{str(in_seed)}]_data.pkl')
    split_save = {
        'train_indices': train_indices,
        'test_indices': test_indices,
        'rng_state': torch.get_rng_state(),
        'np_rng_state': np.random.get_state()
    }
    with open(new_f, 'wb') as f:
        pickle.dump(split_save, f)
    # with open(filename, 'rb') as f:
    #     split_save = pickle.load(f)
    print(f"New Seed = {in_seed}  with file {new_f} created.")
    return


# =======================================================================================


def separate_alldata(reader_fname, writer_path, delimiter=',', keep_headers=True):
    if os.path.isdir(writer_path):
        os.rmdir(writer_path)
    os.mkdir(writer_path)
    print('split output folder created')
    writer_fname_template='%s.csv'
    writer_fname = '' # initialize
    reader = csv.reader(open(reader_fname, 'r'), delimiter=delimiter)
    headers = next(reader, None);
#    headers = headers[:3]
    headers = ['Time', 'Parameter', 'Value']
    buffer = []
    for i, row in enumerate(reader):
        cur_pid = row[3]
        if row[1] == 'RecordID':
            pid = str(int(float(row[2])))
            assert pid == cur_pid, 'Error! pid=%s, cur_pid=%s' % (pid, cur_pid)
            writer_fname = os.path.join(
                writer_path,
                writer_fname_template % cur_pid
            )
            cur_f = open(writer_fname, 'a', newline='')
            cur_writer = csv.writer(cur_f, delimiter=delimiter)
            if keep_headers:
                cur_writer.writerow(headers)
            if len(buffer)>0:
                new_buffer = []
                for i, b_row in enumerate(buffer):
                    b_pid = b_row[3]
                    if b_pid==cur_pid:
                        cur_writer.writerow(b_row[:3])
                    else:
                        new_buffer.append(b_row)
                buffer = new_buffer
        else:
            if str(cur_pid) not in writer_fname:
                buffer.append(row)
            else:
                cur_writer.writerow(row[:3])
    assert len(buffer)==0, 'something wrong! buffer ='+buffer
    
    return len(os.listdir(writer_path))



#string_to_index = {s: i for i, s in enumerate(string_list)}
def preprocess_csv_data(import_data_path, import_oc_path, out_path, in_list):
    # 
    in_dict = {s: i for i, s in enumerate(in_list)}
    # csv files
    filenames = os.listdir(import_data_path)
    filenames.sort(key=lambda x: float(x[:-4]))
    
    oc_df = pd.read_csv(import_oc_path)
    oc_df.set_index('RecordID', inplace=True)
    oc_names = oc_df.columns.tolist()
    
    dim_x = len(in_dict)
    pat_stats_table = []
    
    sequences = []
    time_sequences = []
    mask_sequences = []
    targets = {s: [] for _, s in enumerate(oc_names)}
    pid_list = []
    seq_lengths = []
    for _, fname in enumerate(filenames):
        pid = int(fname[:-4])
        pid_list.append(pid)
        pid_oc = oc_df.loc[pid]
        for _, s in enumerate(oc_names):
            the_oc = pid_oc[s] if pid_oc[s]>=0 else float(10000)
            targets[s].append(the_oc)
        
        fname = os.path.join(import_data_path, fname)
        df = pd.read_csv(fname)
        #df['Time'] = pd.to_timedelta(df['Time'], unit="D")
        grouped_data = df.groupby('Time')
        n_groups = grouped_data.ngroups # sequence length
        #group_sizes = grouped_data.size().values # array int
        group_names = list(grouped_data.groups.keys())
        data_x = np.zeros((n_groups, dim_x))
        data_time = np.zeros((n_groups, 1))
        data_mask = np.zeros((n_groups, dim_x))
        for i, name in enumerate(group_names):
            data_time[i] = int(name)
            df_group = grouped_data.get_group(name)
            for _, j in enumerate(df_group.index):
                col_idx = in_dict[df_group.at[j, 'Parameter']]
                value = df_group.at[j, 'Value']
                data_x[i, col_idx] = value
                data_mask[i, col_idx] = 1
    
        arr_x = np.array(data_x)
        masked_x = np.ma.masked_equal(arr_x, 0.0)  # Masks zeros
        non_zero_means = masked_x.mean(axis=0)  # Mean of non-zeros per column
        seq_lengths.append(n_groups)
        max_time = np.max(data_time) # last_day
        arr_x_means = np.append(non_zero_means.data, [n_groups, max_time]) # addup length and last_day
        pat_stats_table.append(arr_x_means)
        
        # non_zero_counts = np.count_nonzero(arr_x, axis=0)
        # non_zero_percentages = np.round((non_zero_counts / n_groups) * 100)
        # pat_stats_table.append(non_zero_percentages)
        
        tensor_x = torch.tensor(data_x, dtype=torch.float32)
        tensor_time = torch.tensor(data_time, dtype=torch.float32)
        tensor_mask = torch.tensor(data_mask, dtype=torch.int)
        
        sequences.append(tensor_x)
        time_sequences.append(tensor_time)
        mask_sequences.append(tensor_mask)
    
    for _, s in enumerate(oc_names):
        targets[s] = torch.tensor(targets[s], dtype=torch.float32)
    
    pat_stats_df = pd.DataFrame(pat_stats_table, columns=in_list+['length', 'last_day'], index=pid_list)
    pd.set_option('display.max_columns', None)
    print(pat_stats_df.describe(include='all'))
    last_days = list(pat_stats_df['last_day'])
    
    data_to_save = {
        'sequences': sequences,
        'time_sequences': time_sequences,
        'mask_sequences': mask_sequences,
        'targets': targets,
        'pid_list': pid_list,
        'seq_lengths': seq_lengths,
        'last_days': last_days,
        'pat_stats_df': pat_stats_df
    }
    
    # out_path is like 'sequence_data.pkl'
    with open(out_path, 'wb') as f:
        pickle.dump(data_to_save, f)
    
    # # Optional: Verify by loading
    # with open('sequence_data.pkl', 'rb') as f:
    #     loaded_data = pickle.load(f)
    
    return pat_stats_df


# =======================================================================================
# Standard LSTM for sequential output in [0, 1] with packed input
class StandardLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, packed_X):
        # Packed input: Process only valid time steps
        packed_output, (h_n, c_n) = self.lstm(packed_X)
        output, _ = pad_packed_sequence(packed_output, batch_first=True, padding_value=0.0)

        # Predict scalar for each time step
        y_pred = self.output_layer(output)  # Shape: (batch_size, max_seq_len, output_size)
        
        return y_pred
    
# =======================================================================================
# for baseline models

def Flatting_SequenceDataset(dataset, in_indices):
    max_seq_len = dataset.max_len()
    input_size = dataset.sequences[0].size(-1)  # Assuming input_size is 3 based on previous context
    sequences = [dataset.sequences[i] for i in in_indices]
    time_sequences = [dataset.time_sequences[i] for i in in_indices]
    targets = [dataset.targets[i] for i in in_indices]
    
    # For flattened sequences
    flattened_sequences = []
    for seq, idx in enumerate(sequences):
        padded_seq = torch.zeros(max_seq_len, input_size)
        seq_len = seq.size(0)
        padded_seq[:seq_len, :] = seq[:seq_len, :]
        flattened_seq = padded_seq.view(-1).numpy() # Flatten to (max_seq_len * input_size)
        flattened_seq = np.append(flattened_seq, time_sequences[idx].numpy())
        flattened_sequences.append(flattened_seq)
    
    # Convert to numpy arrays
    X_train = np.array(flattened_sequences)  # Shape: (n_train, max_seq_len * input_size)
    y_train = np.array(targets)  # Shape: (n_train,)
    
    return X_train, y_train


# =======================================================================================

# for test output 

def find_optimal_threshold(precision, recall, thresholds):
    """
    Find the optimal threshold that maximizes (precision * recall) / (precision + recall).

    :param precision: Array of precision values.
    :param recall: Array of recall values.
    :param thresholds: Array of threshold values.
    :return: A tuple containing the optimal threshold and the corresponding maximum score.
    """
    # Calculate (precision * recall) / (precision + recall) for each pair of precision and recall
    # Avoid division by zero
    scores = (precision * recall) / (precision + recall + 1e-10)

    # Find the index of the maximum score
    optimal_idx = np.argmax(scores)

    # Find the optimal threshold corresponding to this index
    optimal_threshold = thresholds[optimal_idx]

    return optimal_threshold, scores[optimal_idx]

import warnings
from sklearn.exceptions import UndefinedMetricWarning

def get_performance(y_dict, oc_arr, model_name, print_tag, in_opt_th):
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    
    criterion = nn.MSELoss()    # For avg over all oc
    criterion_oc = nn.MSELoss(reduction='none')    # For each oc
    
    # --- array format ---
    for i in ['values', 'days', 'masks']:
        for j in ['pred', 'label']:
            y_dict[i][j] = np.array(y_dict[i][j], dtype = float)
    
    
    label_days = torch.tensor(y_dict['days']['label'])
    pred_days = torch.tensor(y_dict['days']['pred'])
    loss = criterion(pred_days, label_days)
    loss = torch.sqrt(loss)
    print(f"{model_name} RMSE Loss (avg) = {loss.item():.2f} days   ")
    
    loss_oc = criterion_oc(pred_days, label_days)
    loss_oc = torch.sqrt(loss_oc)
    loss_oc = loss_oc.mean(dim=0)
    for i, oc in enumerate(oc_arr):
        print(f"        Outcome {oc} = {loss_oc[i].item():.2f} days ")
    
    y_dict['days']['label'] = np.array(y_dict['days']['label'], dtype = int)
    y_dict['days']['last'] = np.array(y_dict['days']['last'], dtype = int)
    
     # --- exclude no 1 oc for binary---
    label_masks = y_dict['masks']['label']
    oc_disp_arr = []
    oc_hide_arr = []
    for i, oc in enumerate(oc_arr):
        if np.sum(label_masks[:,i]) > 12:
            oc_disp_arr.append(oc)
        else:
            oc_hide_arr.append(oc)
    if len(oc_hide_arr) > 0:
        print(f"  ** Outcome(s) {oc_hide_arr} removed from binary performance evaluation due to limited positive num")

    def inner_perform(pred_masks):
    # --- masks ---
        label_masks = y_dict['masks']['label']
        #pred_masks = y_dict['masks']['pred']
        
        if in_opt_th is None:
            opt_th_arr = []
            print("Optimal Thresholds for Binary Labels: \n             ", end='')
            for i, oc in enumerate(oc_disp_arr):
                precision_out, recall_out, thresholds_out = precision_recall_curve(label_masks[:,i], pred_masks[:,i])
                thresholds_out = np.append(thresholds_out, 1)
                optimal_threshold, _ = find_optimal_threshold(precision_out, recall_out, thresholds_out)
                opt_th_arr.append(optimal_threshold)
                print(f"{oc} = {optimal_threshold:.4f} ", end='')
            print("")
        else:
            opt_th_arr = in_opt_th
            assert(len(opt_th_arr)==len(oc_disp_arr))
            print("Using the given optimal thresholds for Binary Labels: ", end='')
            for i, oc in enumerate(oc_disp_arr):
                print(f"{oc} = {opt_th_arr[i]:.4f} ", end='')
            print("")
        
        y_dict['masks']['label'] = np.array(y_dict['masks']['label'], dtype = int)
    
        # --- by outcomes ---    
        oc_df_save = {}
        oc_auc_arr = []
        oc_precision_arr = []
        oc_recall_arr = []
        oc_f1_arr = []
        for i, oc in enumerate(oc_disp_arr):
            optimal_threshold = opt_th_arr[i]
            oc_label_masks = np.array(label_masks[:,i], int)
            oc_pred_masks = pred_masks[:,i]    # numpy float
            oc_pred_masks_binary = np.array(np.where(pred_masks[:,i] > optimal_threshold, 1, 0), int)
            
            oc_label_days = y_dict['days']['label'][:,i]    # numpy int
            oc_pred_days = y_dict['days']['pred'][:,i]    # numpy float
            oc_last_day = y_dict['days']['last'][i]    # numpy int
            
            oc_df = pd.DataFrame({
                    oc+' label days': oc_label_days,   # numpy int
                    oc+' last days': oc_last_day,   # numpy int
                    oc+' pred days': oc_pred_days,    # numpy float
                    oc+' label mask': oc_label_masks,   # numpy int
                    oc+' pred mask': oc_pred_masks,    # numpy float
                    oc+' pred mask binary': oc_pred_masks_binary   # numpy int
                })
            oc_df_save[oc] = oc_df
            
            oc_auc_arr.append(roc_auc_score(oc_label_masks, oc_pred_masks_binary))
            oc_precision_arr.append(precision_score(oc_label_masks, oc_pred_masks_binary))
            oc_recall_arr.append(recall_score(oc_label_masks, oc_pred_masks_binary))
            oc_f1_arr.append(f1_score(oc_label_masks, oc_pred_masks_binary))
            
        auc_avg = np.mean(oc_auc_arr)
        precision_avg = np.mean(oc_precision_arr)
        recall_avg = np.mean(oc_recall_arr)
        f1_avg = np.mean(oc_f1_arr)
        print(f"{model_name}", " Binary Label AUC(avg): {:.4f},  Precision(avg): {:.4f},  Recall(avg): {:.4f},  Test F1(avg): {:.4f} ".format(auc_avg, precision_avg, recall_avg, f1_avg))
        for i, oc in enumerate(oc_disp_arr):
            print(f"        Outcome {oc}: AUC={oc_auc_arr[i]:.4f}  Precision={oc_precision_arr[i]:.4f}  Recall={oc_recall_arr[i]:.4f}  F1={oc_f1_arr[i]:.4f}")
        
        
        if print_tag:
            pd.options.display.float_format = '{:,.4f}'.format
            df_arr = []
            for _, oc in enumerate(oc_disp_arr):
                df_arr.append(oc_df_save[oc])
            displayed_df = pd.concat(df_arr, axis=1)
            print(displayed_df.head(100))
        
        return oc_df_save, opt_th_arr
    
    # print("#------------------------ using value -----------------------------#")
    # pred_masks = y_dict['values']['pred']
    # oc_df_save, opt_th_arr = inner_perform(pred_masks)
    
    print("#------------------------ using mask -----------------------------#")
    pred_masks = y_dict['masks']['pred']
    oc_df_save, opt_th_arr = inner_perform(pred_masks)
    
    
    
    return oc_df_save, opt_th_arr

# =======================================================================================


def print_grad_graph(tensor, prefix="", depth=0, max_depth=3):
    """
    Print the gradient computation graph for a tensor.
    
    Args:
        tensor: Tensor to inspect (e.g., loss).
        prefix: String prefix for indentation.
        depth: Current recursion depth.
        max_depth: Maximum depth to traverse.
    """
    if tensor.grad_fn is None:
        print(f"{prefix}No grad_fn (leaf tensor or no computation)")
        return
    if depth >= max_depth:
        print(f"{prefix}... (max depth reached)")
        return
    print(f"{prefix}grad_fn: {tensor.grad_fn}")
    for next_fn in tensor.grad_fn.next_functions:
        if next_fn[0] is not None:
            print_grad_graph(next_fn[0].variable, prefix + "  ", depth + 1, max_depth)

def inspect_optimizer_grads(optimizer):
    """
    Print gradient information for optimizer parameters.
    
    Args:
        optimizer: Optimizer to inspect.
    """
    print("\nOptimizer Parameter Gradients:")
    for group_idx, group in enumerate(optimizer.param_groups):
        for param_idx, param in enumerate(group['params']):
            if param.grad is not None:
                print(f"  Group {group_idx}, Param {param_idx}:")
                print(f"    Shape: {param.shape}, Grad Norm: {torch.norm(param.grad).item():.4f}")
                print(f"    Requires Grad: {param.requires_grad}")
            else:
                print(f"  Group {group_idx}, Param {param_idx}: No gradient computed")
