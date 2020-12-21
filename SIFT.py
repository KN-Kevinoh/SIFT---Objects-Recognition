#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: kevin
"""
# import all necessaries libraries

import cv2
import pandas as pd
import numpy as np
from numpy import sum, sqrt, multiply
import os
import utils
#import pickle

# define treshold  

_THRESHOLD = 0.8

# define neighbours

K_NEIGHBOURS = 5

# sift descriptor

def sift(img):
    
    # create sift object
    sift = cv2.xfeatures2d.SIFT_create()
    """
    # get image  keys points and descriptors
    # by detecting keys points and computing descriptors
    """
    img_keys_points, img_descriptors = sift.detectAndCompute(img, None) 
    
    return img_keys_points, img_descriptors


# retrieve all sift descriptors over dataset and save

def compute_sift_on_dataset(data = None, labels = None, save_name = None):
    #check if current path file not exists before
    if not (os.path.isfile("data_descriptors/" + save_name + ".pkl")):
        
        # dataframe for descriptors
        df_container = pd.DataFrame(columns=("obj_name","nb_kp", "descriptors"))
        # tmp dictionary data
        dic = dict()
        
        for img in data:
            kp, desc =  sift(img[0])
            dic["obj_name"] = labels[img[1]]
            dic["nb_kp"] = len(kp)
            dic["descriptors"] = desc
            
            # add element in df_container
            df_container = df_container.append(dic, ignore_index=True)
            
        # save descriptors like pickle file or csv file
        df_container.to_pickle("data_descriptors/" + save_name + ".pkl")
        df_container.to_csv("data_descriptors/" + save_name + ".csv",  encoding='utf-8', index=False)
        
"""
#   Match similarity given sift descriptor of test image 
#   over our learning training dataset.
#   for a giving train image compute distance of each descriptor
#   in test image over all descriptors of target train set image.
#   Match similarity of test descriptor using Threshold: 
#   absolute distance, ratio between the closest and the second
#   closer
""" 

# normalize vector
def l2_normalize(x):
 return x / sqrt(sum(multiply(x, x)))

"""
# compute  distance over each descriptor 
# on train image and apply Threshold correspondance method,
# get ratio as closest distance and closer distance
"""
def euclidian_distance(test_descriptor, train_img_descriptor):
    
    test_descriptor = np.matrix(test_descriptor).T
    test_descriptor = l2_normalize(test_descriptor)
   
    # compute ratio of closest distance and closer distance
    list_distances = list()
    for target_desc in train_img_descriptor:
        target_desc = np.matrix(target_desc).T
        target_desc = l2_normalize(target_desc)
        dist = target_desc - test_descriptor
        dist = sum(multiply(dist,dist))
       
        dist = sqrt(dist)
        list_distances.append(dist)
        
    list_distances.sort()
    ratio = list_distances[0] / list_distances[1]
    return ratio
        
        

def findCosineSimilarity(test_descriptor, train_img_descriptor):
    test_descriptor = test_descriptor.reshape((test_descriptor.shape[0],1))
     
    list_distances = list()
    for index, target_desc in enumerate(train_img_descriptor):
        target_desc = target_desc.reshape((target_desc.shape[0],1))
        
        a = np.matmul(np.transpose(target_desc), test_descriptor)
        b = np.sum(np.multiply(target_desc, target_desc))
        c = np.sum(np.multiply(test_descriptor, test_descriptor))
        dist = 1 - (a / (np.sqrt(b) * np.sqrt(c)))
        
        list_distances.append(dist)
       
    list_distances.sort()
    ratio = list_distances[0] / list_distances[1]
    return ratio
     
        
"""
# Match one test descriptor in test set over train set.
# @return array matching
"""
def match_target_descriptor(test_img_descriptor = None, train_set = None):
    
    tab_desc_ratio = np.zeros(shape = (len(train_set), len(test_img_descriptor)))
    
    for i, train_img_descriptor in enumerate(train_set):
       
        for j, desc in enumerate(test_img_descriptor):
            ratio = euclidian_distance(desc, train_img_descriptor)
            tab_desc_ratio[i, j] = ratio
            
    return tab_desc_ratio
        

"""
# match_test_train match all test set sift descriptor over train set
"""  

def match_test_train(test_set = None, train_set = None, save_name = None):
    
    list_match = list() 
    i = 0
    for test_img_descriptor in test_set:
        desc_test_match = match_target_descriptor(test_img_descriptor, train_set)
        list_match.append(desc_test_match)
        print(i+1)
        i+=1
      
    #if not save_name is None:
     #   np.savez_compressed("macth_descriptor/" + save_name + ".npz", array_1 =  np.asarray(list_match))
     
    return list_match


# use threading to overcome computation time
def match_test_train_threading(test_set = None, train_set = None, save_name = None):
      
    list_match = list() 
        
    #initialize thread schedule
    i = 0
   
    for test_img_descriptor in test_set:
        a = utils.ThreadSchedule(target=match_target_descriptor,args=(test_img_descriptor,train_set))
        #print(a)
        list_match.insert(i, a)
        print(list_match[i])
        i += 1
        

    # start all thread
    for j in range(len(list_match)):
        #print(i)
        list_match[j].start()
        print("--start task ",j)
       
    # waiting result of each test sift descriptor
    for j in range(len(list_match)):
        list_match[j] = list_match[j].join()
        print("**End task ",j)
    """   
    if not save_name is None:      
       
       with open('macth_descriptor/save_match.txt', 'wb') as fp:
           pickle.dump(list_match, fp)
       
       np.savetxt("macth_descriptor/" + save_name + ".txt", list_match)
    """
    return list_match

"""
# Evaluate match test set.
# @return array matching
------------
# Ici nous déduisons la liste de scores de chaque image test par rapport
# au images du train set
"""
def evaluate_matching(results_match = None,train_set = None, threshold = _THRESHOLD):
    list_scores = list()
    for index, tab_ratio in enumerate(results_match):
        # match_desc, a tab_ratio img test sur chaque img train
        #for i, tab_ratio in enumerate(match_desc):
            
        tab_ratio = (tab_ratio < threshold)
        
        """
        # sum each line size of each corresponding image test where the
        # ratio satisfied the condition , then divide size of current
        # image train. As a result we get a list of score of
        # target test image over each train image
        """
        
        tab_scores = [np.sum(sub_list)/ len(train_set[j]) 
                     for sub_list, j in  zip(tab_ratio, range(len(train_set)))]
        
        list_scores.append(tab_scores)
            
        #resultats_match[index].reshape(1, len(train_set))
    
    #np.savetxt("macth_descriptor/scoresObjtCat.txt", results_match)
    
    return list_scores

"""
# predict test image class from k neighbours defined based on each each test 
# test image score list
"""
def predict(list_test_scores = None, list_labels = None, labels_train = None,  k_neighbours = K_NEIGHBOURS):
    """
        here it will be a question of taking the list of scores, 
        making a correspondence with the labels and take the labels 
        that best represent
    """
    # convert for array to easy retrieve with numpy
    labels = np.asarray(list_labels, dtype = str)
    labels_train = np.asarray(labels_train, dtype = int)
    test_set_predict = list()
    
    for index, list_scores in enumerate(list_test_scores):
        
         # order list an invert
         #list_scores = np.asarray(list_scores)
         #list_scores.reshape(1, len(list_scores))
         
         # sort and invert with index
         order_desc = np.argsort(list_scores)[::-1]
         
         # get k first best scores
         list_k_neighbours = order_desc[:k_neighbours]
         
         # get labels (integer) corresponding on train
         neighbours_lab = labels_train[list_k_neighbours]
         
         # get each lablels (string)
         list_k_neighbours_labels = labels[neighbours_lab]
         
         # compute labels occurrences
         neighbours_labels, counts_occurences = np.unique(
             list_k_neighbours_labels, return_counts=True)
         
         # order by index of most occurencies
         counts_occurences = np.argsort(counts_occurences)[::-1]
         
         predict_label = neighbours_labels[counts_occurences[0]]
         
         # get predict num class which is the most represent
         pred_num = list_labels.index(predict_label)
         
         test_set_predict.append(pred_num)
            
    return test_set_predict
    
    
# predict_labels and true_labels are  list of integers
 
def confusion_matrix(predict_labels = None, true_labels = None, labels = None):
    
    cf = np.zeros((len(labels), len(labels)))
    
    for i in range(len(true_labels)):
        i = int(i)
        pred = predict_labels[i]
        true_lab =  true_labels[i]
        
        cf[pred, true_lab] += 1
       
    #draw confusion matrix
    utils.make_confusion_matrix(cf,categories=labels, cmap='binary')


############
# for real test, get matching descriptor for the winner object on train 
# NB: train_desc is retrieve through predicted labels    
    
def get_match_desc_index(test_descriptor = None, train_img_descriptor = None):
    test_descriptor = test_descriptor.reshape((test_descriptor.shape[0],1))
    test_descriptor = l2_normalize(test_descriptor)
   
    # compute ratio of closest distance and closer distance
    list_distances = list()
    for index, target_desc in enumerate(train_img_descriptor):
        target_desc = target_desc.reshape((target_desc.shape[0],1))
        target_desc = l2_normalize(target_desc)
        dist = target_desc - test_descriptor
        dist = sum(multiply(dist,dist))
       
        dist = sqrt(dist)
        list_distances.append(dist)
    
    # get position of min dist
    index_min = np.argmin(list_distances, axis=0)
    
    return index_min

def create_DMatch(list_ratio_img_test = None,test_img_desc = None,
                                   train_img_desc = None, 
                                   pred_num = None, 
                                   threshold = _THRESHOLD):
    
    matches = list()
    tmp = list_ratio_img_test[0][pred_num]
    tmp =  (tmp < threshold)
    
    print("***********Nombre de ratio valides : {} *********\n {}".format(np.sum(tmp), tmp))

    for i, is_true in enumerate(tmp):
        
        if is_true:
            j = get_match_desc_index(test_img_desc[i], train_img_desc) 
            #list_desc_matches.append([test_img_desc[i], train_img_desc[j]])
            matches.append(cv2.DMatch(i,j,0))
    
    return matches
            

    
