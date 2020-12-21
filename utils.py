#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: kevin
"""
import cv2
import os
import pandas as pd
import numpy as np
from threading import  Thread
import seaborn as sns
from matplotlib import pyplot as plt



# convert image to pgm format using ImageMagick default install in linux
def image_to_gray(input_img = None):
    img = cv2.imread(input_img, 0)
    img = cv2.medianBlur(img, 3)
    img = cv2.resize(img, (128,128), interpolation = cv2.INTER_AREA)
    return img


# Load dataset objectsCategories of 101 classes
def load_dataset_objectCategories(path):
    
    if os.path.isfile("data/101_ObjectCategories/dataset.pkl"):
        data = pd.read_pickle("data/101_ObjectCategories/dataset.pkl")
        train = pd.read_pickle("data/101_ObjectCategories/train_set.pkl")
        test = pd.read_pickle("data/101_ObjectCategories/test_set.pkl")
        labels = np.loadtxt('data/101_ObjectCategories/labels.txt', dtype=str)
        
    else:
        data = pd.DataFrame(columns=("images","labels"))
        train = pd.DataFrame(columns=("images","labels"))
        test = pd.DataFrame(columns=("images","labels"))
        labels = list()
       
        for folder in os.listdir(path):
            
            labels.append(folder)
            path_folder = path + folder +"/"
            #i = 0
            for filename in os.listdir(path_folder):  
                
                path_file = path_folder + filename
                img = image_to_gray(path_file)
                dic = dict()
                dic["images"] = img
                dic["labels"] = labels.index(folder)
                
                # add image in data and annotated label index
                data = data.append(dic, ignore_index=True)
                """
                i += 1
                if i > 5:
                    break
                """
                
        # shuffle and split into train and test 
        data = data.sample(frac=1, random_state=42).reset_index(drop=True)
        
        train = data.iloc[:round(len(data)/2)+1, :]
        test = data.iloc[round(len(data)/2)+1:, :]
        
        # save data, train and test 
        data.to_pickle("data/101_ObjectCategories/dataset.pkl")
        train.to_pickle("data/101_ObjectCategories/train_set.pkl")
        test.to_pickle("data/101_ObjectCategories/test_set.pkl")
        
        # save labels
        np.savetxt('data/101_ObjectCategories/labels.txt', labels, fmt='%s')
       
    print("101_objectsCategories {} images, {} labels: {} train and {} test set."
          .format(len(data), len(labels), len(train), len(test)))
    
    return train, test, labels
            
    


# Load dataset coil of 100 classe

def load_dataset_coil(path):
    
    if os.path.isfile("data/coil-100/dataset.pkl"):
        data = pd.read_pickle("data/coil-100/dataset.pkl")
        train = pd.read_pickle("data/coil-100/train_set.pkl")
        test = pd.read_pickle("data/coil-100/test_set.pkl")
        labels = np.loadtxt('data/coil-100/labels.txt', dtype=str)
        
    else:
        data = pd.DataFrame(columns=("images","labels"))
        train = pd.DataFrame(columns=("images","labels"))
        test = pd.DataFrame(columns=("images","labels"))
        labels = list()
    
        for filename in os.listdir(path): 
            
            path_folder = path + "/"
            
            # add objet_num in labels if not exists
            obj = filename.split('__')
            
            if obj[0] not in labels:
                labels.append(obj[0])
                
            path_file = path_folder + filename
            img = image_to_gray(path_file)
            dic = dict()
            dic["images"] = img
            dic["labels"] = labels.index(obj[0])
            
            # add image in data and annotated label index
            data = data.append(dic, ignore_index=True)
        
        # shuffle and split into train and test 
        data = data.sample(frac=1, random_state=42).reset_index(drop=True)
        
        train = data.iloc[:round(len(data)/2)+1, :]
        test = data.iloc[round(len(data)/2)+1:, :]
        
        #train = data.iloc[:300, :]
        #test = data.iloc[300:600, :]
        
        # save data, train and test 
        data.to_pickle("data/coil-100/dataset.pkl")
        train.to_pickle("data/coil-100/train_set.pkl")
        test.to_pickle("data/coil-100/test_set.pkl")
        
        # save labels
        np.savetxt('data/coil-100/labels.txt', labels, fmt='%s')
   
    print("coil-100 {} images, {} labels: {} train and {} test set."
          .format(len(data), len(labels), len(train), len(test)))
   
    return train, test, labels


"""
######### implements thread
"""

import sys
from builtins import super    # https://stackoverflow.com/a/30159479

if sys.version_info >= (3, 0):
    _thread_target_key = '_target'
    _thread_args_key = '_args'
    _thread_kwargs_key = '_kwargs'
else:
    _thread_target_key = '_Thread__target'
    _thread_args_key = '_Thread__args'
    _thread_kwargs_key = '_Thread__kwargs'

class ThreadSchedule(Thread):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._return = None

    def run(self):
        target = getattr(self, _thread_target_key)
        if not target is None:
            self._return = target(
                *getattr(self, _thread_args_key),
                **getattr(self, _thread_kwargs_key)
            )

    def join(self, *args, **kwargs):
        super().join(*args, **kwargs)
        return self._return
    
#--------------

############## Draw confusion matrix

#@reuse from  Dennis Trimarchi github
def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):
    

    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cf) / float(np.sum(cf))

        #if it is a binary confusion matrix, show some more stats
        if len(cf)==2:
            #Metrics for Binary Confusion Matrices
            precision = cf[0,0] / np.sum(cf[:,0])
            recall    = cf[0,0] / np.sum(cf[0,:])
            print(cf[0,0], np.sum(cf[1,:]))
            f1_score  = 2*precision*recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy,precision,recall,f1_score)
            print(stats_text)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""


    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize==None:
        #Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False


    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)
    
    if title:
        plt.title(title)