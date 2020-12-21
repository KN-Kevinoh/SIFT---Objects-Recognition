#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 18:24:41 2020

@author: kevin & Jerémie
"""
import utils
import SIFT as mysift

import argparse 
import pandas as pd
import numpy as np
import cv2

from matplotlib import pyplot as plt

parser = argparse.ArgumentParser(description="Object recognition using SIFT descriptor", 
                                 prog="object_recognition", usage='%(prog)s [options]')
parser.add_argument("--path", metavar="img_path" , 
                    type=str, nargs='+', help="")

parser.parse_args()

def foo(a):
    bar = 2*a
    print(bar)
    return

if __name__ == "__main__": 
    
    parser = argparse.ArgumentParser(description="Object recognition using SIFT descriptor", 
                                     prog="object_recognition")
    parser.add_argument("--path" , type=str,
                        help="Enter absolute path of your image")

    args = parser.parse_args()
    path_img = args.path

    # load image
    img = utils.image_to_gray(str(path_img))
    print("\n********************")
    print("* image :\n\n ", img )
    
    # load references descriptors
    train_bd = pd.read_pickle("data_descriptors/train_objDB_descriptors.pkl")
    
    # load train dataset
    train_set_img = pd.read_pickle("data/101_ObjectCategories/train_set.pkl")
    
    list_labels =  np.loadtxt('data/101_ObjectCategories/labels.txt', dtype=np.str)
    list_labels = list_labels.tolist()
    print("\n********************")
    print("* labels : ", len(list_labels),"\n\n", list_labels)
    
    # Extract descriptors
    kp_1, desc_1 =  mysift.sift(img)
    
    print("\n********************")
    print("\n* descriptors : ", desc_1.shape,"\n\n",desc_1)
    tab_desc_ratio =  mysift.match_target_descriptor(desc_1, train_bd["descriptors"].values.tolist())
    
    print("\n********************")
    print("\n*tab ratio on train: ",tab_desc_ratio.shape,"\n\n",tab_desc_ratio)
    
    list_match =list()
    list_match.append(tab_desc_ratio)
    list_scores =  mysift.evaluate_matching(list_match, train_bd["descriptors"].values.tolist())
    print("\n********************")
    print("\n* List of scores on train : ({},{}) \n".format(len(list_scores),len(list_scores[0])), list_scores)

    pred =  mysift.predict(list_scores, list_labels, train_set_img["labels"].values.tolist())
    print("\n********************")   
    pred = int(pred[0])
    print("* predict result: {}: {}".format(pred, list_labels[pred]))
    print("\n********************") 
    
    # get all images from predict label 
    df = train_set_img.loc[train_set_img['labels'] == pred]
    print("\n*******rows images train predict*************", len(df)) 
    print(df)
    print("\n********************") 
    
    # choose one randomly and make matching
    df = df['images'].values.tolist()
    img_predict_class = df[np.random.randint(len(df))]
    
    print("\n********************") 
    print("*image_predict: \n\n", img_predict_class)
    print("\n********************") 
    
    pk_2, desc_2 =  mysift.sift(img_predict_class)
    print("\n********************") 
    print("descriptors predict image : ", len(desc_2),"\n\n", desc_2)
    print("\n********************") 
    
    
    DMatches =  mysift.create_DMatch(
        list_ratio_img_test = list_match,
        test_img_desc = desc_1,
        train_img_desc = desc_2, 
        pred_num = pred)
    print("\n********************") 
    print("\n*DMATCH : \n\n",DMatches) 
    print("\n********************") 
    # draw matching and plot
    
    result = cv2.drawMatches(img, kp_1, img_predict_class, pk_2, DMatches, None, flags=2)
    plt.imshow(result)
    plt.title(list_labels[pred], loc='center')
    plt.savefig("images/resultats/test.jpg")
    plt.show() 
    