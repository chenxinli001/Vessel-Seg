# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 19:40:38 2019

@author: wenaoma
"""
###################################################
#
#   Script to
#   - Calculate prediction of the test dataset
#   - Calculate the parameters to evaluate the prediction
#
##################################################

#Python
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
import sys
sys.path.insert(0, './lib2/')
# help_functions.py
from help_functions2 import *
# extract_patches.py
from extract_patches2 import recompone
from extract_patches2 import kill_border
from extract_patches2 import pred_only_FOV
from extract_patches2 import get_data_testing_overlap
from pre_processing2 import my_PreProc




def Evalution_drive(preImg):
  
    #========= CONFIG FILE TO READ FROM =======

    #===========================================
    #run the training on invariant or local

    
    #original test images (for FOV selection)

    DRIVE_test_imgs_original = './data/DRIVE_datasets_training_testing/DRIVE_dataset_imgs_test.hdf5'
    test_imgs_orig = load_hdf5(DRIVE_test_imgs_original)
    full_img_height = test_imgs_orig.shape[2]
    full_img_width = test_imgs_orig.shape[3]
    #the border masks provided by the DRIVE
    DRIVE_test_border_masks = './data/DRIVE_datasets_training_testing/DRIVE_dataset_borderMasks_test.hdf5'
    test_border_masks = load_hdf5(DRIVE_test_border_masks)
    # dimension of the patches
    patch_height = 64
    patch_width = 64
    #the stride in case output with average
    stride_height = 5 
    stride_width = 5
    assert (stride_height < patch_height and stride_width < patch_width)
    #model name
#    name_experiment = config.get('experiment name', 'name')
#    path_experiment = './' +name_experiment +'/'
    #N full images to be predicted
    #Imgs_to_test = int(config.get('testing settings', 'full_images_to_test'))
    #Grouping of the predicted images
    N_visual = 1
    #====== average mode ===========
    average_mode = True
    
    

    patches_imgs_test = None
    new_height = None
    new_width = None
    masks_test  = None
    patches_masks_test = None
    if average_mode == True:
        patches_imgs_test, new_height, new_width, masks_test = get_data_testing_overlap(
            DRIVE_test_imgs_original = DRIVE_test_imgs_original,  #original
            DRIVE_test_groudTruth = './data/DRIVE_datasets_training_testing/DRIVE_dataset_groundTruth_test.hdf5',  #masks
            Imgs_to_test = 20,
            patch_height = patch_height,
            patch_width = patch_width,
            stride_height = stride_height,
            stride_width = stride_width
        )

    
    
    

    pred_imgs = None
    orig_imgs = None
    gtruth_masks = None
    pred_imgs = preImg
    if average_mode == True:
        #pred_imgs = recompone_overlap(pred_patches, new_height, new_width, stride_height, stride_width)# predictions
        orig_imgs = my_PreProc(test_imgs_orig[0:pred_imgs.shape[0],:,:,:])    #originals
        gtruth_masks = masks_test  #ground truth masks
    else:
        #pred_imgs = recompone(pred_patches,13,12)       # predictions
        orig_imgs = recompone(patches_imgs_test,13,12)  # originals
        gtruth_masks = recompone(patches_masks_test,13,12)  #masks
    # apply the DRIVE masks on the repdictions #set everything outside the FOV to zero!!
    kill_border(pred_imgs, test_border_masks)  #DRIVE MASK  #only for visualization
    ## back to original dimensions
    orig_imgs = orig_imgs[:,:,0:full_img_height,0:full_img_width]
    pred_imgs = pred_imgs[:,:,0:full_img_height,0:full_img_width]
    
    gtruth_masks = gtruth_masks[:,:,0:full_img_height,0:full_img_width]


    #visualize results comparing mask and prediction:
    assert (orig_imgs.shape[0]==pred_imgs.shape[0] and orig_imgs.shape[0]==gtruth_masks.shape[0])
    N_predicted = orig_imgs.shape[0]
    group = N_visual
    assert (N_predicted%group==0)
    for i in range(int(N_predicted/group)):
        orig_stripe = group_images(orig_imgs[i*group:(i*group)+group,:,:,:],group)
        masks_stripe = group_images(gtruth_masks[i*group:(i*group)+group,:,:,:],group)
        pred_stripe = group_images(pred_imgs[i*group:(i*group)+group,:,:,:],group)


    
    #test_border_masks = test_border_masks1
    #gtruth_masks = segLabel
    #====== Evaluate the results

    #predictions only inside the FOV
    y_scores, y_true = pred_only_FOV(pred_imgs,gtruth_masks, test_border_masks)  #returns data only inside the FOV

    #Area under the ROC curve
    fpr, tpr, thresholds = roc_curve((y_true), y_scores)
    AUC_ROC = roc_auc_score(y_true, y_scores)
    # test_integral = np.trapz(tpr,fpr) #trapz is numpy integration


    
    #Precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    precision = np.fliplr([precision])[0]  #so the array is increasing (you won't get negative AUC)
    recall = np.fliplr([recall])[0]  #so the array is increasing (you won't get negative AUC)



    
    #Confusion matrix
    threshold_confusion = 0.5

    y_pred = np.empty((y_scores.shape[0]))
    for i in range(y_scores.shape[0]):
        if y_scores[i]>=threshold_confusion:
            y_pred[i]=1
        else:
            y_pred[i]=0
    confusion = confusion_matrix(y_true, y_pred)

    accuracy = 0
    if float(np.sum(confusion))!=0:
        accuracy = float(confusion[0,0]+confusion[1,1])/float(np.sum(confusion))

    specificity = 0
    if float(confusion[0,0]+confusion[0,1])!=0:
        specificity = float(confusion[0,0])/float(confusion[0,0]+confusion[0,1])

    sensitivity = 0
    if float(confusion[1,1]+confusion[1,0])!=0:
        sensitivity = float(confusion[1,1])/float(confusion[1,1]+confusion[1,0])

    precision = 0
    if float(confusion[1,1]+confusion[0,1])!=0:
        precision = float(confusion[1,1])/float(confusion[1,1]+confusion[0,1])
   
    #Jaccard similarity index
    #jaccard_index = jaccard_similarity_score(y_true, y_pred, normalize=True)

    
    #F1 score
    #F1_score = f1_score(y_true, y_pred, labels=None, average='binary', sample_weight=None)

    
    return AUC_ROC,accuracy,specificity,sensitivity


#preImg = np.zeros((20,1,584,565), np.float32)
#preImg[:,0,:,:] = np.load('ProMap0.npy')[:,2,:,:]

#AUC_ROC,accuracy,specificity,sensitivity = Evalution_drive(preImg)
#print("AUC:"+str(AUC_ROC))
#print("accuracy:"+str(accuracy))
#print("specificity:"+str(specificity))
#print("sensitivity:"+str(sensitivity))
