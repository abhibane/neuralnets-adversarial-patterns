# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 12:03:10 2018

@author: AEB6KOR
"""


import numpy as np
import matplotlib.pyplot as plt
import pickle
import os, errno
from scipy.special import expit as activation_function
from scipy.stats import truncnorm
from scipy import stats
from sklearn.metrics import confusion_matrix
import copy
import time
import datetime as dt
from scipy.spatial import distance
from rbm import RBM
from hopfield_nw import HopfieldNet

def getProbDistribution(list_vals, max_bound, min_bound, bucket_count):
    # max_bound = np.max(list_vals)
    # min_bound = np.min(list_vals)
    
    bucket_size = (max_bound - min_bound)/bucket_count
        
    bkt_frequency = {}
    
    bkt_low = min_bound
    bkt_high = 0.0
    
    for i in range(bucket_count):
        bkt_high = bkt_low + bucket_size
        
        bkt_frequency[bkt_high] = 0
        
        bkt_low = bkt_high
    
    for i in range(len(list_vals)):
        for k in bkt_frequency:
            if list_vals[i] <= k:
                bkt_frequency[k] = bkt_frequency[k] + 1
                break
                
    bkt_pdf = []
    bkt_cdf = []
    bkt_vals = []
    
    cum_sum = 0.0
    obs_count = len(list_vals)
    
    for k in bkt_frequency:
        prob_val = bkt_frequency[k]/obs_count
        cum_sum = cum_sum + prob_val
        
        bkt_pdf.append(prob_val)
        bkt_cdf.append(cum_sum)
        bkt_vals.append(k)
        
    return bkt_vals, bkt_pdf, bkt_cdf

def kldiv(p, q):
    if len(p) != len(q):
        print('Distributions do not match! Exiting')
        return -1
    
    kldiv = 0.0
    eps = 0.0001
    
    for i in range(len(p)):
        plog = 0.0
        qlog = 0.0
        
        if p[i] > 0:
            plog = np.log(p[i])
        else:
            plog = np.log(eps)
            
        if q[i] > 0:
            qlog = np.log(q[i])
        else:
            qlog = np.log(eps)
            
        kldiv = kldiv + p[i] * (plog - qlog)
        
    return kldiv

if __name__ == '__main__':
    start_time = time.time()
    
    linColMap = {}
    linColMap[0] = 'k-'
    linColMap[1] = 'r-'
    linColMap[2] = 'b-'
    linColMap[3] = 'g-'
    linColMap[4] = 'c-'
    linColMap[5] = 'm-'
    linColMap[6] = 'y-'
    linColMap[7] = 'b--'
    linColMap[8] = 'g--'
    linColMap[9] = 'r--'
    linColMap[10] = 'k--'
    linColMap[11] = 'c--'
    linColMap[12] = 'm--'
    linColMap[13] = 'y--'
  
    read_dir_name = "SimulationResults/Pattern_Angles_20180827/"
    file_name_str = "patternset1to5_series.txt"
    file_name = read_dir_name + file_name_str
    
    text_array = np.genfromtxt(file_name_str, delimiter=' ')
    
    text_array_float = text_array.astype(np.float)
  
    print('Array Shape:', text_array_float.shape)
    print('Dtype:', text_array_float.dtype)
    
    text_array_float[text_array_float > 0] = 1
    text_array_float[text_array_float <= 0] = 0
    
    max_bound = np.max(text_array_float)
    min_bound = np.min(text_array_float)
    
    print('Max :', np.max(text_array_float))
    print('Min :', np.min(text_array_float))
    
    #print(text_array_float[0])
    
    neuron_count = 784
 
    hidden_nodes = 784
  
    r = RBM(num_visible = neuron_count, num_hidden = hidden_nodes)
    
    training_data = np.copy(text_array_float)
    
    print('Training Data Shape:', training_data.shape)
    
    rbm_wt_vec_list, wt_rec_interval = r.train(training_data, max_epochs = 5000)
    
    print('RBM Weights', r.weights.shape)
    
    dateStr = dt.date.today().strftime("%Y%m%d")
    dirName = "SimulationResults/Pattern_MisClassification_" + dateStr + "/"
     
    if not os.path.exists(dirName):
        os.makedirs(dirName)

    min_noise = 0.1
    max_noise = 1.0
    noise_int = 0.025
    
# =============================================================================
#     print('Original Image')
#     
#     img = text_array_float[0].reshape((28,28))
#     plt.imshow(img, cmap="Greys")
#     plt.show()
#     
# =============================================================================
    noise_ratio = min_noise
    
    patt_miscl_noise = {}
    
    sim_count = 10
    
    patt_class_err = {}
    patt_noise_series = {}
    
    for idx in range(1, 11):
        idx_start = 5 * (idx - 1)
        idx_end = 5 * idx
        patt_idx_list = np.array(text_array_float[idx_start:idx_end])
        
        if idx not in patt_class_err:
            patt_class_err[idx - 1] = []
            patt_noise_series[idx - 1] = []
            
        noise_ratio = min_noise
        
        while noise_ratio < max_noise:
            flip_count = int(noise_ratio * neuron_count)
            
            miscl_count = 0
            
            for src_idx in range(patt_idx_list.shape[0]):
                for sim_iter in range(sim_count):
                    idx_list = np.random.choice(neuron_count,flip_count,replace=False)
                    
                    user = np.copy(text_array_float[src_idx])
            
                    user[idx_list] = (user[idx_list] + 1) % 2
                    
                    user_list = []
                    user_list.append(user)
                    recalled_vector = r.run_hidden(r.run_visible(np.array(user_list)))
                    
                    ham_dist_vals = []
                    
                    for i in range(text_array_float.shape[0]):
                        hamming_dist = np.count_nonzero(text_array_float[i] != recalled_vector)
                        
                        ham_dist_vals.append(hamming_dist)
                        
                    ham_dist_min_val = np.min(ham_dist_vals)
                    ham_dist_min_idx = np.argmin(ham_dist_vals)
                    
                    if (src_idx != ham_dist_min_idx):
                        miscl_count = miscl_count + 1
            
            patt_class_err[idx - 1].append(miscl_count/50.0)
            patt_noise_series[idx - 1].append(noise_ratio)
            
            noise_ratio = noise_ratio + noise_int
    
    print('Min. Noise Levels')
    print(patt_class_err)
    
    interval_count = int((max_noise - min_noise)/noise_int)
    fileStr = 'Patt_Misclassification_Noise_P' + str(text_array_float.shape[0]) + '_RBM_h' + str(hidden_nodes) + '_Int' + str(interval_count)
    fileName = dirName + fileStr
    fig = plt.figure()
    fig.set_size_inches(18.5, 10.5)
    
    for patt_idx in patt_class_err:
        plt.plot(patt_noise_series[patt_idx], patt_class_err[patt_idx], linColMap[patt_idx % len(linColMap.keys())], label=str(patt_idx))
        print('Pattern:', patt_idx, ', ', patt_noise_series[patt_idx][2], ': ', patt_class_err[patt_idx][2], ', ', patt_noise_series[patt_idx][3], ': ', patt_class_err[patt_idx][3], ', ', patt_noise_series[patt_idx][4], ': ', patt_class_err[patt_idx][4])

    plt.legend()
    plt.xlabel('Noise Level')
    plt.ylabel('Misclassification Probability')
    plt.title('Misclassification Probability vs Noise Level')
    fig.savefig(fileName)

    end_time = time.time()
    print('Total Time: ', (end_time - start_time)/60.0, ' mins')