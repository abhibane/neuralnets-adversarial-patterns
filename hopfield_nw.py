# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 18:47:30 2018

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
import scipy.io
from random import * 
import math
#from mnist import MNIST

class HopfieldNet(object):

    def __init__(self, num_units):
        self.num_units = num_units
        self.w = np.zeros((num_units, num_units))
        self.b = np.zeros(num_units)

    def store(self, data):
        data = data.reshape(data.shape[0], 1)
        #print(data)
        #activations = np.dot(data, data.T)
        #activations = np.outer(data,data)
        activations = np.dot(data,data.T)
        np.fill_diagonal(activations, 0)  # because there are no connections to itself
        #print(activations)
        self.w += activations
        self.b += data.ravel()
     
    def learn(self, learn_data):
        k = len(learn_data[0])
        hopfield = np.zeros([k,k])
        for in_data in learn_data: 
            np_arr = np.matrix(in_data)
            lesson = np_arr.T*np_arr
            np.fill_diagonal(lesson, 0)
            hopfield = hopfield + lesson
        self.w = np.copy(hopfield)
        return hopfield

    def get_energy(self, data):
        # first let's again compute product of activations
        data = data.reshape(data.shape[0], 1)
        activations = np.float32(np.dot(data, data.T))
        np.fill_diagonal(activations, 0)
        # then multiply each activation by a weight elementwise
        activations *= self.w
        # total energy consists of weight and bias term
        weight_term = np.sum(activations) / 2  # divide by 2, because we've counted neurons twice
        bias_term = np.dot(self.b, data)[0]
        return - bias_term - weight_term
    
    def iterativeLearning(self,neuron_count, activity, data):
        itr = list(range(len(data)))
        shuffle(itr)
        tempWeight = np.copy(self.w )
        for p in range(len(data)) : 
            testData = np.copy(data[itr.pop()])
            processingData = np.copy(testData)
            origData = np.copy(testData)
            learning = 1  
            #while learning ==1 :
            recallPattern = hopfield.recall(testData)
            if (np.count_nonzero(origData != recallPattern)==0):
                learning = 0
            else :
                   nitr = list(range(neuron_count))
                   shuffle(nitr)
                   count=0;
                   for i in range(neuron_count):
                        #energyOld = - 0.5 * np.matmul(np.matmul(self.w,testData.T).T,testData.T)
                        energyOld = - 0.5 * np.dot(np.dot(self.w,testData.T).T,testData.T);
                        #print('Energy Before Flip ', energyOld)
                        temp = nitr.pop()
                        tempData = testData[temp]
                        if (testData[temp]==1):
                            testData[temp]=-1
                        else:
                            testData[temp]=1
                        energyNew = - 0.5 * np.dot(np.dot(self.w,testData.T).T,testData.T)
                        #energyNew = - 0.5 * np.dot(np.dot(self.w,testData),testData)    
                        #print('Energy After Flip ', energyNew)
                        if((energyNew) > (energyOld)):
                            processingData[temp] = tempData
                            testData[temp] = tempData
                        else:
                            processingData[temp]=testData[temp]                            
                            count = count +1 
                   
                   tempA = (np.outer(testData,testData)-activity)
                   tempB = (np.outer(origData,origData)-activity)
                   tempC = (0.5/neuron_count)
                   deltaW = tempC*(tempA-tempB)
                   self.w = np.add(self.w,deltaW)
                   
                            
                        
                        #testData[temp] = tempData
# =============================================================================
#                     energyOri = - 0.5 * np.matmul(np.matmul(self.w,origData.T).T,origData.T);
#                     energyDiff = math.sqrt(math.pow(energyOri,2)-math.pow(energyOld,2))
#                     if (energyDiff == 0 and sum(origData==testData)==784):
#                         learning = 0;
#                         plt.imshow(testData.reshape(28,28))
# =============================================================================
        tempW = self.w
        plt.figure()
        plt.imshow(tempW)
        plt.show()
        sum(tempWeight==self.w)   
        return
             
 
    def recall(self, data):
        
# =============================================================================
#         for _ in range(5):
#             restored_data = np.dot(self.w, data)
#             restored_data[restored_data <0] = -1 
#             restored_data[restored_data >0 ] = 1
# =============================================================================
        #return restored_data #, activation_function(restored_data)
        data = np.copy(data)
        idx = range(len(data))
        # make 10 passes through the data
        for _ in range(10):
            for i in range(len(data)):
                data[i] = 1.0 if np.dot(self.w[i],data) > 0 else -1.0
        return data
    
    def logistic(self, x):
        return 1.0 / (1 + np.exp(-x))
    
    def plot_images(images, title, no_i_x, no_i_y=3):
        fig = plt.figure(figsize=(10, 15))
        fig.canvas.set_window_title(title)
        images = np.array(images).reshape(-1,28, 28)
        images = np.pad(
            images, ((0, 0), (1, 1), (1, 1)), 'constant', constant_values=-1)
        for i in range(no_i_x):
            for j in range(no_i_y):
                ax = fig.add_subplot(no_i_x, no_i_y, no_i_x * j + (i + 1))
                ax.matshow(images[no_i_x * j + i], cmap="gray")
                plt.xticks(np.array([]))
                plt.yticks(np.array([]))
    
                if j == 0 and i == 0:
                    ax.set_title("Real")
                elif j == 0 and i == 1:
                    ax.set_title("Distorted")
                elif j == 0 and i == 2:
                    ax.set_title("Reconstructed")
    
if __name__ == '__main__':
# =============================================================================
#    training_data = np.array([[1,1,1,0,0,0],[1,0,1,0,0,0],[1,1,1,0,0,0],[0,0,1,1,1,0], [0,0,1,1,0,0],[0,0,1,1,1,0]])
#    training_data = np.array([[0,1,0,1,1,1,0,1,0],[1,0,0,0,1,0,0,0,1]])
#    
#    raw_data_train = np.array([[0,0,1,0,0,0,0,1,0,0,1,1,1,1,1,0,0,1,0,0,0,0,1,0,0],
#                              [1,0,0,0,1,0,1,0,1,0,0,0,1,0,0,0,1,0,1,0,1,0,0,0,1],
#                              [1,1,1,1,1,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0],
#                              [1,1,1,1,-1,1,-1, -1, -1, 1, 1, 1, 1, 1, -1, 1, -1, -1, -1, -1, 1,-1, -1, -1, -1],
#                              [1, -1, -1, -1, 1, -1, 1, -1, 1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1,-1, -1, -1, 1, -1, -1 ]
#                              ])
#      
#    training_data = np.copy(raw_data_train) 
#    test_data = np.copy(raw_data_train)
#    training_data[training_data <= 0] = -1
#    training_data[training_data > 0] = 1
#    test_data[test_data <= 0] = -1
#    test_data[test_data > 0] = 1
#   
#    neuron_count = 25
#    hopfield = HopfieldNet(neuron_count)
# =============================================================================

  
# =============================================================================
#   
# =============================================================================
 # =============================================================================
# =============================================================================
       with open("neural_network_mnist/data/mnist/pickled_mnist.pkl", "br") as fh:
                 data = pickle.load(fh) 
                 train_imgs = data[0]
                 test_imgs = data[1]
                 train_labels = data[2]
                 test_labels = data[3]
                 train_labels_one_hot = data[4]
                 test_labels_one_hot = data[5]
                 image_size = 28 # width and length
                 no_of_different_labels = 10 #  i.e. 0, 1, 2, 3, ..., 9
                 image_pixels = image_size * image_size 
                 image_count = 50 #len(train_imgs)
                 image_list = []
                 for i in range(image_count):
                     image_list.append(train_imgs[i])
                 raw_data_train = np.array(image_list)
       training_data = np.copy(raw_data_train)
       test_data = np.copy(raw_data_train)
       training_data[training_data <= 0] = -1
       training_data[training_data > 0] = 1
       test_data[test_data <= 0] = -1
       test_data[test_data > 0] = 1
       neuron_count = 784
       hopfield = HopfieldNet(neuron_count)
 
# =============================================================================
#        emnist_data = MNIST(path='gzip\\', return_type='numpy')
#        emnist_data.select_emnist(data_type)
#        x_orig, y_orig = emnist_data.load_training()
#        raw_data_train = np.array(x_orig)
#        training_data = np.copy(raw_data_train)
#        test_data = np.copy(raw_data_train)
#        training_data[training_data <= 0] = -1
#        training_data[training_data > 0] = 1
#        test_data[test_data <= 0] = -1
#        test_data[test_data > 0] = 1
#        neuron_count = 784
#        hopfield = HopfieldNet(neuron_count)
#        image_count = 5 
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
#     mat = scipy.io.loadmat('alphabet.mat')
#     training_data = np.copy(mat)
#     training_data[training_data<=0]=-1
#     training_data[training_data>0] =1
#     neuron_count = 400
#     hopfield = HopfieldNet(neuron_count)
# =============================================================================
  ##print('Weights')
  ##print(hopfield.w)
  ## Hebbian Learning Algorithm  
       
       index = np.array([21,40,5,7,2,35,18,15,46,22])
       index_list = []
       for i in range (10):
          index_list.append(train_imgs[index[i]])
       raw_data_train = np.array(index_list)
       index_data = np.copy(raw_data_train)
       index_data[index_data <= 0] = -1
       index_data[index_data > 0] = 1
       np.savetxt("pattern.txt",(index_data),fmt='%d')
          
       activation = 0 
       #hopfield.learn(training_data)
       for i in range(len(training_data)):
              #hopfield.store(training_data[i])
              activation += sum(training_data[i]==1)
       activation = (1/(image_count*neuron_count))*activation 
       
       #hopfield.iterativeLearning(neuron_count, activation, training_data)
       #hopfield.w = hopfield.w /neuron_count
       for i in range(len(index_data)): 
         print('Activation', np.count_nonzero(test_data[i] == 1))
         #recalled_vector = hopfield.recall(test_data[i])
         print('Index :', i )
         img = index_data[i].reshape((28,28))
         plt.imshow(img, cmap="Greys")
         plt.show()
# =============================================================================
#          img = recalled_vector.reshape((28,28))
#          plt.imshow(img, cmap="Greys")
#          plt.show()
# =============================================================================
       
              
              
        
      #print('Pattern ', i, end=' ')
      ##print('Pattern ', i, ': ', training_data[i]
    

   #hopfield.w = hopfield.w /neuron_count
   
  #print('Weights')
  #print(hopfield.w)
    #success = 0.0 
    
       
         
    
# =============================================================================
#      hopfield.plot_images(output_data, "Reconstructed Data", 10)
#      plt.show()
#                     
# =============================================================================
# =============================================================================
# # =============================================================================
#       for i in range(10):
#          img = training_data[i].reshape((28,28))
#          plt.imshow(img, cmap="Greys")
#          plt.show()
# # =============================================================================
# =============================================================================
# =============================================================================
#         if np.array_equal(training_data[i], recalled_vector):
#             success += 1.0
# =============================================================================
    #print("Accuracy of the network is %f" % ((success/len(training_data)) * 100))
      
# =============================================================================
#   
#  user = np.array([0,0,1,0,0,0,0,1,0,0,1,1,1,1,1,0,0,1,0,0,0,0,1,0,0])
#  #user[user==0] = -1 
#   user = raw_data_train[0]
#   user[user==0]=-1
#   #print('Recall Pattern: ', user)
#   #print(hopfield.get_energy(user))
#   recalled_vector = hopfield.recall(user)
#   ##print(recalled_vector)
#   #recalled_vector[recalled_vector <= 0] = 0
#   #recalled_vector[recalled_vector > 0] = 1
#   #print(recalled_vector)
#   
#   print('Hamming distance:', np.count_nonzero(user != recalled_vector))
#   
#   user = np.array([1,0,0,0,1,0,1,0,1,0,0,0,1,0,0,0,1,0,1,0,1,0,0,0,1])
#   user = raw_data_train[1]
#   user[user==0]=-1
#   #print('Recall Pattern: ', user)
#   #print(hopfield.get_energy(user))
#   recalled_vector = hopfield.recall(user)
#   ##print(recalled_vector)
#   #recalled_vector[recalled_vector < 0] = 0
#   #recalled_vector[recalled_vector > 0] = 1
#   #print(recalled_vector)
#   
#   print('Hamming distance:', np.count_nonzero(user != recalled_vector))
#   
#   user = np.array([1,1,1,1,1,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0])
#   user = raw_data_train[2]
#   user[user==0]=-1
#   #print('Recall Pattern: ', user)
#   #print(hopfield.get_energy(user))
#   recalled_vector = hopfield.recall(user)
#   ##print(recalled_vector)
#   #recalled_vector[recalled_vector < 0] = 0
#   #recalled_vector[recalled_vector > 0] = 1
#   #print(recalled_vector)
#   
#   print('Hamming distance:', np.count_nonzero(user != recalled_vector))
#   
     #plt.scatter(hopfield.w[0][:], hopfield.w[:][0])
# 
# =============================================================================
