# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 14:14:13 2018

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
#from rbm import RBM
#from hopfield_nw import HopfieldNet

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
# =============================================================================
#   training_data = np.array([[1,1,1,0,0,0],[1,0,1,0,0,0],[1,1,1,0,0,0],[0,0,1,1,1,0], [0,0,1,1,0,0],[0,0,1,1,1,0]])
#   training_data = np.array([[0,1,0,1,1,1,0,1,0],[1,0,0,0,1,0,0,0,1]])
#   
# =============================================================================
    
# =============================================================================
#   with open("neural_network_mnist/data/mnist/pickled_mnist.pkl", "br") as fh:
#         data = pickle.load(fh)
#         
#   train_imgs = data[0]
#   test_imgs = data[1]
#   train_labels = data[2]
#   test_labels = data[3]
#   train_labels_one_hot = data[4]
#   test_labels_one_hot = data[5]
#   image_size = 28 # width and length
#   no_of_different_labels = 10 #  i.e. 0, 1, 2, 3, ..., 9
#   image_pixels = image_size * image_size
#   
#   image_count = 20 #len(train_imgs)
#   image_list = []
#   
#   for i in range(image_count):
#       image_list.append(train_imgs[i])
# 
# =============================================================================
# =============================================================================
#   text_file = open("patternset1to5_series.txt", "r")
#   lines = text_file.readlines()
# 
#   vect_list = []
# 
#   for line in lines:
#       vect_list.append(line.split(' '))
# 
# #vect_list[vect_list == '-1'] = 0
# #vect_list[vect_list == '1'] = 1
# 
# # print(vect_list[0])
#       
#   text_list = []
#   
#   for i in range(len(vect_list)):
#       text_list.append([])
#       for j in range(len(vect_list[i]) - 1):
#           text_list[i].append(vect_list[i][j])
# 
#   text_array = np.array(text_list)
# 
#   print('Array Shape:', text_array.shape)
#   print('Dtype:', text_array.dtype)
# 
#   text_array_float = text_array.astype(np.float)
# 
# =============================================================================
# =============================================================================
#   for i in range(text_array.shape[0]):
#       for j in range(text_array.shape[1]):
#           # print('1,1:', text_array[1][1])
#           if text_array[i][j] == '-1':
#               text_array[i][j] = 0
#           elif text_array[i][j] == '1':
#               text_array[i][j] = 1
#               
# =============================================================================
#  print('Array:', text_array)
              
  # text_array_float = text_array.astype(np.float)
# =============================================================================
#   for i in range(text_array_float.shape[0]):
#       for j in range(text_array_float.shape[1]):
#           #print('1,1:', text_array[1][1])
#           if text_array_float[i][j] == -1:
#               text_array_float[i][j] = 0
#           elif text_array_float[i][j] == 1:
#               text_array_float[i][j] = 1
#             
#   print('Min:', np.min(text_array_float))
#   print('Max:', np.max(text_array_float))
#   print('Dtype:', text_array_float.dtype)
# =============================================================================
  
# =============================================================================
#   for i in range(10):
#      print('Image ', i)
#      fileName = 'digit_' + str(i)
#      fig = plt.figure()
#      img = text_array_float[i].reshape((28,28))
#      plt.imshow(img, cmap="Greys")
#      plt.show()
#      fig.savefig(fileName)
#   
# =============================================================================
# =============================================================================
#   
#   train_frac = 0.5
#   
#   train_img_count = text_array_float.shape[0] #int(train_frac * image_count)
#   
#   train_image_list = []
#   test_image_list = []
#   
#   for i in range(train_img_count):
#       train_image_list.append(text_array_float[i])
#       
#   for i in range(train_img_count):
#       test_image_list.append(text_array_float[i])
#       
#   raw_data_train = np.array(train_image_list)
#     
#   neuron_count = 784
#  
#   # Train RBM
#   training_data = np.copy(raw_data_train)
#   training_data[training_data <= 0] = 0
#   training_data[training_data > 0] = 1
#   
#   hidden_nodes = 784
# =============================================================================
  
# =============================================================================
#   r = RBM(num_visible = neuron_count, num_hidden = hidden_nodes)
#   r.train(training_data, max_epochs = 5000)
#   print('RBM Weights')
#   #print(r.weights)
#   print(r.weights.shape)
#   
#   ham_dist_vals = []
#           
#   for i in range(train_img_count):
#       # Recall with RBM
#       user = np.copy(train_image_list[i])
#       user[user <= 0] = int(0)
#       user[user > 0 ] = int(1)
#       
#       user_list = []
#       user_list.append(user)
#       recalled_vector = r.run_hidden(r.run_visible(np.array(user_list)))
#       
#       recalled_vector[recalled_vector <= 0] = int(0)
#       recalled_vector[recalled_vector > 0] = int(1)
#       
#       hamming_dist = np.count_nonzero(np.array(user_list) != recalled_vector)
#       
#       ham_dist_vals.append(hamming_dist)
#       
#   print('Recalled Mean Hamming Dist: ', np.mean(ham_dist_vals))
#   print('Recalled Max Hamming Dist: ', np.max(ham_dist_vals))
#   print('Recalled Min Hamming Dist: ', np.min(ham_dist_vals))
#   
#   dateStr = dt.date.today().strftime("%Y%m%d")
#   dirName = "SimulationResults/RBM_Hopfield_Comp-" + dateStr + "/"
#   
#   if not os.path.exists(dirName):
#     os.makedirs(dirName)
#   
#   bkt_vals, bkt_pdf, bkt_cdf = getProbDistribution(ham_dist_vals, np.max(ham_dist_vals), np.min(ham_dist_vals), 20)
#   
#   fileStr = 'PDF_HamDist_' + str(neuron_count) + '_RBM_h' + str(hidden_nodes)
#   fileName = dirName + fileStr
#   fig = plt.figure()
#   fig.set_size_inches(18.5, 10.5)
#   plt.plot(bkt_vals, bkt_pdf)
#   x1, x2, y1, y2 = plt.axis()
#   plt.axis((x1, x2, y1, 1.0))
#   # plt.legend()
#   plt.xlabel('Hamming Distance')
#   plt.ylabel('Probability')
#   plt.show()
#   fig.savefig(fileName)
#   
#   rbm_wt_vector = r.weights.reshape((r.weights.shape[0] * r.weights.shape[1], 1))
#   
#   rw_max_val = np.max(rbm_wt_vector)
#   rw_min_val = np.min(rbm_wt_vector)
#   
#   print('RBM Weights Max:', rw_max_val, ', Min:', rw_min_val)
#   
#   max_bound = rw_max_val
#   min_bound = rw_min_val
# =============================================================================
  
# =============================================================================
#   if(rw_max_val > max_bound):
#       max_bound = rw_max_val
#       
#   if(rw_min_val < min_bound):
#       min_bound = rw_min_val
# =============================================================================
  neuron_count = 784
  pdfs=[]
  dateStr = dt.date.today().strftime("%Y%m%d")
  dirName = "SimulationResults/Hopfield_Individual-" + dateStr + "/"
  if not os.path.exists(dirName):
      os.makedirs(dirName)
  for itr in range (10) : 
      name = 'weight' + str(itr) + '.txt' 
      text_file = open( name, "r")
      lines = text_file.readlines()
      vect_list = []
      for line in lines:
          vect_list.append(line.split(' '))
      text_list = []
  
      for i in range(len(vect_list)):
          text_list.append([])
          for j in range(len(vect_list[i]) - 1):
              text_list[i].append(vect_list[i][j])

      text_array = np.array(text_list)
      
      print('Array Shape:', text_array.shape)
      print('Dtype:', text_array.dtype)
    
      text_array_float = text_array.astype(np.float)
      
      hop_wt_vector = text_array_float.reshape((text_array_float.shape[0] * text_array_float.shape[1], 1))
      
      hop_max_val = np.max(hop_wt_vector)
      hop_min_val = np.min(hop_wt_vector)
      
      #if(hop_max_val > max_bound):
      max_bound = hop_max_val
      
     # if(hop_min_val < min_bound):
      min_bound = hop_min_val
    
      #bkt_vals, rbm_pdf, rbm_cdf = getProbDistribution(rbm_wt_vector, max_bound, min_bound, bucket_count = 50)
      
      bkt_vals, hopf_pdf, hopf_cdf = getProbDistribution(hop_wt_vector, max_bound, min_bound, bucket_count = 50)
     
      
      fileStr = 'CDF_Hopf_weights' + str(itr) #+ '_RBM_h' + str(hidden_nodes)
      fileName = dirName + fileStr
      fig = plt.figure()
      fig.set_size_inches(18.5, 10.5)
      plt.plot(bkt_vals, hopf_cdf, 'r-', label='Hopfield Weights CDF')
      #plt.plot(bkt_vals, rbm_cdf, 'b-', label='RBM Weights CDFs')
      x1, x2, y1, y2 = plt.axis()
      plt.axis((x1, x2, y1, 1.0))
      plt.legend()
      plt.xlabel('Weight Values')
      plt.ylabel('Probability')
      plt.show()
      fig.savefig(fileName)
      
      fileStr = 'PDF_Hopf_weights' + str(itr) # + '_RBM_h' + str(hidden_nodes)
      fileName = dirName + fileStr
      fig = plt.figure()
      fig.set_size_inches(18.5, 10.5)
      plt.plot(bkt_vals, hopf_pdf, 'r-', label='Hopfield Weights PDF')
      #plt.plot(bkt_vals, rbm_pdf, 'b-', label='RBM Weights PDFs')
      x1, x2, y1, y2 = plt.axis()
      plt.axis((x1, x2, y1, 1.0))
      plt.legend()
      plt.xlabel('Weight Values')
      plt.ylabel('Probability')
      plt.show()
      fig.savefig(fileName)
          
  
# =============================================================================
#       kl_div_val = kldiv(hopf_pdf, hopf_pdf)
#       
#       #print('RBM Weights Count:', rbm_wt_vector.shape)
#       print('Hopfield Weights Count:', hop_wt_vector.shape)
#       
#       #print('RBM Hidden Nodes:', hidden_nodes)
#       
#       print('KL Div (HopF - RBM):', kl_div_val)
#       
#       kl_div_val = kldiv(hopf_pdf, hopf_pdf)
#       
#       print('KL Div (RBM - HopF):', kl_div_val)
#       
#       
# =============================================================================
