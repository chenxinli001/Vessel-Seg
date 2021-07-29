# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 15:22:26 2019

@author: wenaoma
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 15:04:10 2018

@author: wenaoma
"""

from torch.utils import data
import numpy as np
from PIL import Image  
import torch
import torch.autograd as autograd
from torchvision import transforms as T
from config import DefaultConfig
from torch.utils.data import DataLoader
import os
import cv2
from Tools.ImageResize import imageResize, creatMask, cropImage
from lib.Utils import *
from loss import CrossEntropyLoss2d,multiclassLoss
from skimage import morphology
from lib.Utils import *
from models.ResUnet_small import ResUnet,ResUnet_illum,ResUnet_illum_tran,ResUnet_illum_tran_trad,ResUnet_illum_tran_trad_conv,ResUnet_illum_tran_trad_conv2,ResUnet_illum_tran_trad_conv3
from sklearn import metrics
from Tools.FakePad import fakePad
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
import tensorlayer as tl
from VesselSegProbMap.VesselSegmentation_ProbMap import VesselProMap
import natsort

def extract_ordered_overlap_trad(img, patch_h, patch_w,stride_h,stride_w,ratio):
    img_h = img.shape[0]  #height of the full image
    img_w = img.shape[1] #width of the full image
    assert ((img_h-patch_h)%stride_h==0 and (img_w-patch_w)%stride_w==0)
    N_patches_img = ((img_h-patch_h)//stride_h+1)*((img_w-patch_w)//stride_w+1)  #// --> division between integers
    patches = np.empty((N_patches_img, patch_h//ratio, patch_w//ratio, 2))
    iter_tot = 0   #iter over the total number of patches (N_patches)
    for h in range((img_h-patch_h)//stride_h+1):
        for w in range((img_w-patch_w)//stride_w+1):
            patch = img[h*stride_h:(h*stride_h)+patch_h, w*stride_w:(w*stride_w)+patch_w, :]
            patch = cv2.resize(patch,(patch_h//ratio, patch_w//ratio))
            patches[iter_tot]=patch
            iter_tot +=1   #total
    assert (iter_tot==N_patches_img)
    return patches  #array with all the img divided in patches

def paint_border_overlap_trad(img, patch_h, patch_w, stride_h, stride_w):
    img_h = img.shape[0]  #height of the full image
    img_w = img.shape[1] #width of the full image
    leftover_h = (img_h-patch_h)%stride_h  #leftover on the h dim
    leftover_w = (img_w-patch_w)%stride_w  #leftover on the w dim
    if (leftover_h != 0):  #change dimension of img_h
        tmp_full_imgs = np.zeros((img_h+(stride_h-leftover_h),img_w, 2))
        tmp_full_imgs[0:img_h,0:img_w, :] = img
        img = tmp_full_imgs
    if (leftover_w != 0):   #change dimension of img_w
        tmp_full_imgs = np.zeros((img.shape[0], img_w+(stride_w - leftover_w), 2))
        tmp_full_imgs[0:img.shape[0], 0:img_w, :] = img
        img = tmp_full_imgs
    return img

def GetMask(k):
    if k<=9:
        ImgName = './data/AV_DRIVE/test/mask/0' + str(k) + '_test_mask.png'

    elif k<=20:
        ImgName = './data/AV_DRIVE/test/mask/' + str(k) + '_test_mask.png'

    Img0 = cv2.imread(ImgName)
    Mask = np.zeros((Img0.shape[0],Img0.shape[1]),np.float32)
    Mask[Img0[:,:,2]>0] = 1
    
    return Mask 

def get_patch_trad(batch_size, patch_size, train_data1, train_data2,train_data3,label_data):
    data1 = np.zeros((batch_size, 3, patch_size, patch_size), np.float32)
    data2 = np.zeros((batch_size, 3, patch_size, patch_size), np.float32)
    data3 = np.zeros((batch_size, 2, patch_size, patch_size), np.float32)
    data4 = np.zeros((batch_size, 2, patch_size//2, patch_size//2), np.float32)
    data5 = np.zeros((batch_size, 2, patch_size//4, patch_size//4), np.float32)
    label = np.zeros((batch_size, 3, patch_size, patch_size), np.float32)
    #z = np.random.randint(0,20)
    for j in range(batch_size):
        x = np.random.randint(0,train_data1.shape[2]-patch_size+1)
        y = np.random.randint(0,train_data1.shape[3]-patch_size+1)
        random_size = np.random.randint(0,3)
        z = np.random.randint(0,20)
        choice = np.random.randint(0,5)
        #PatchNum = np.random.randint(0,train_data.shape[2])
        if random_size == 0:
            data_mat_3 = np.zeros((3, patch_size, patch_size), np.float32)
            data_mat_1 = train_data1[z,:,x:x+patch_size,y:y+patch_size]
            data_mat_2 = train_data2[z,:,x:x+patch_size,y:y+patch_size]
            data_mat_3[0:2,:,:] = train_data3[z,:,x:x+patch_size,y:y+patch_size]
            label_mat = label_data[z,:,x:x+patch_size,y:y+patch_size]                        
            #label[j,:,:,:] = label_data[z,:,PatchNum,:,:]
        elif random_size ==1:     
            data_mat_3 = np.zeros((3, patch_size, patch_size), np.float32)
            data_mat_1 = np.transpose(cv2.resize(np.transpose(train_data1[z,:,x:x+96,y:y+96],(1,2,0)), (patch_size,patch_size)),(2,0,1))
            data_mat_2 = np.transpose(cv2.resize(np.transpose(train_data2[z,:,x:x+96,y:y+96],(1,2,0)), (patch_size,patch_size)),(2,0,1))
            data_mat_3[0:2,:,:] = np.transpose(cv2.resize(np.transpose(train_data3[z,:,x:x+96,y:y+96],(1,2,0)), (patch_size,patch_size)),(2,0,1))
            label_mat = np.transpose(cv2.resize(np.transpose(label_data[z,:,x:x+96,y:y+96],(1,2,0)), (patch_size,patch_size)),(2,0,1)) 
        else:
            data_mat_3 = np.zeros((3, patch_size, patch_size), np.float32)
            data_mat_1 = np.transpose(cv2.resize(np.transpose(train_data1[z,:,x:x+128,y:y+128],(1,2,0)), (patch_size,patch_size)),(2,0,1))
            data_mat_2 = np.transpose(cv2.resize(np.transpose(train_data2[z,:,x:x+128,y:y+128],(1,2,0)), (patch_size,patch_size)),(2,0,1))
            data_mat_3[0:2,:,:] = np.transpose(cv2.resize(np.transpose(train_data3[z,:,x:x+128,y:y+128],(1,2,0)), (patch_size,patch_size)),(2,0,1))
            label_mat = np.transpose(cv2.resize(np.transpose(label_data[z,:,x:x+128,y:y+128],(1,2,0)), (patch_size,patch_size)),(2,0,1))
        data_mat_1, data_mat_2, data_mat_3,label_mat = data_aug2(data_mat_1, data_mat_2, data_mat_3,label_mat, choice)
        data1[j,:,:,:] = data_mat_1
        data2[j,:,:,:] = data_mat_2
        data3[j,:,:,:] = data_mat_3[0:2,:,:]
        data4[j,:,:,:] = np.transpose(cv2.resize(np.transpose(data_mat_3[0:2,:,:],(1,2,0)), (patch_size//2,patch_size//2)),(2,0,1))
        data5[j,:,:,:] = np.transpose(cv2.resize(np.transpose(data_mat_3[0:2,:,:],(1,2,0)), (patch_size//4,patch_size//4)),(2,0,1))
        label[j,:,:,:] = label_mat   
    return data1, data2,data3,data4,data5,label
        


def get_patch(batch_size, patch_size, train_data1, train_data2, label_data):
    data1 = np.zeros((batch_size, 3, patch_size, patch_size), np.float32)
    data2 = np.zeros((batch_size, 3, patch_size, patch_size), np.float32)
    label = np.zeros((batch_size, 3, patch_size, patch_size), np.float32)
    #z = np.random.randint(0,20)
    for j in range(batch_size):
        x = np.random.randint(0,train_data1.shape[2]-patch_size+1)
        y = np.random.randint(0,train_data1.shape[3]-patch_size+1)
        random_size = np.random.randint(0,3)
        z = np.random.randint(0,20)
        choice = np.random.randint(0,5)
        #PatchNum = np.random.randint(0,train_data.shape[2])
        if random_size == 0:
            data_mat_1 = train_data1[z,:,x:x+patch_size,y:y+patch_size]
            data_mat_2 = train_data2[z,:,x:x+patch_size,y:y+patch_size]
            label_mat = label_data[z,:,x:x+patch_size,y:y+patch_size]                        
            #label[j,:,:,:] = label_data[z,:,PatchNum,:,:]
        elif random_size ==1:            
            data_mat_1 = np.transpose(cv2.resize(np.transpose(train_data1[z,:,x:x+96,y:y+96],(1,2,0)), (patch_size,patch_size)),(2,0,1))
            data_mat_2 = np.transpose(cv2.resize(np.transpose(train_data2[z,:,x:x+96,y:y+96],(1,2,0)), (patch_size,patch_size)),(2,0,1))
            label_mat = np.transpose(cv2.resize(np.transpose(label_data[z,:,x:x+96,y:y+96],(1,2,0)), (patch_size,patch_size)),(2,0,1)) 
        else:
            data_mat_1 = np.transpose(cv2.resize(np.transpose(train_data1[z,:,x:x+128,y:y+128],(1,2,0)), (patch_size,patch_size)),(2,0,1))
            data_mat_2 = np.transpose(cv2.resize(np.transpose(train_data2[z,:,x:x+128,y:y+128],(1,2,0)), (patch_size,patch_size)),(2,0,1))
            label_mat = np.transpose(cv2.resize(np.transpose(label_data[z,:,x:x+128,y:y+128],(1,2,0)), (patch_size,patch_size)),(2,0,1))
        data_mat_1, data_mat_2, label_mat = data_aug(data_mat_1, data_mat_2, label_mat, choice)
        data1[j,:,:,:] = data_mat_1
        data2[j,:,:,:] = data_mat_2
        label[j,:,:,:] = label_mat    
    return data1, data2, label
    



    

def modelEvalution(i,net,savePath,loss_all):
    ArteryPredAll = np.zeros((20, 1, 584, 565), np.float32)
    VeinPredAll = np.zeros((20, 1, 584, 565), np.float32)
    VesselPredAll = np.zeros((20, 1, 584, 565), np.float32)
    LabelArteryAll = np.zeros((20, 1, 584, 565), np.float32)
    LabelVeinAll = np.zeros((20, 1, 584, 565), np.float32)
    LabelVesselAll = np.zeros((20, 1, 584, 565), np.float32)
    ProMap = np.zeros((20, 3, 584, 565), np.float32)
    LabelMap = np.zeros((20, 3, 584, 565), np.float32)
    MaskAll = np.zeros((20, 1, 584, 565), np.float32)
    Vessel = VesselProMap('./data/AV_DRIVE/test/images')

    for k in range(20):
        ArteryPred,VeinPred,VesselPred,LabelArtery,LabelVein,LabelVessel,Mask = GetResult(net,k+1,Vessel)
        ArteryPredAll[k,:,:,:] = ArteryPred
        VeinPredAll[k,:,:,:] = VeinPred
        VesselPredAll[k,:,:,:] = VesselPred
        LabelArteryAll[k,:,:,:] = LabelArtery
        LabelVeinAll[k,:,:,:] = LabelVein
        LabelVesselAll[k,:,:,:] = LabelVessel
        #Mask  = GetMask(k+1)
        MaskAll[k,:,:,:] = Mask
        cv2.imwrite("./data/save_image/Artery1/"+str(k)+".png",ArteryPred[0]*255)
        cv2.imwrite("./data/save_image/Vein1/"+str(k)+".png",VeinPred[0]*255)
        cv2.imwrite("./data/save_image/Vessel1/"+str(k)+".png",VesselPred[0]*255)
        
        #print (k)
    ProMap[:,0,:,:] = ArteryPredAll[:,0,:,:]
    ProMap[:,1,:,:] = VeinPredAll[:,0,:,:]
    ProMap[:,2,:,:] = VesselPredAll[:,0,:,:]
    LabelMap[:,0,:,:] = LabelArteryAll[:,0,:,:]
    LabelMap[:,1,:,:] = LabelVeinAll[:,0,:,:]
    LabelMap[:,2,:,:] = LabelVesselAll[:,0,:,:]
    np.save("./ProMap.npy",ProMap)
    np.save("./Label.npy",LabelMap)
    ArteryAUC,ArteryAcc,ArterySp,ArterySe,VeinAUC,VeinAcc,VeinSp,VeinSe = Evalution_AV(ArteryPredAll,VeinPredAll,LabelArteryAll,LabelVeinAll, MaskAll)
    #VeinAUC,VeinAcc,VeinSp,VeinSe = Evalution(VeinPredAll,LabelVeinAll, MaskAll)
    VesselAUC,VesselAcc,VesselSp,VesselSe = Evalution(VesselPredAll,LabelVesselAll, MaskAll)
    
    
    print("=========================DRIVE=============================")
    print("The {} step ArteryAcc is:{}".format(i,ArteryAcc))
    print("The {} step ArterySens is:{}".format(i,ArterySe))
    print("The {} step ArterySpec is:{}".format(i,ArterySp))
    print("The {} step ArteryAUC is:{}".format(i,ArteryAUC))
    print("-----------------------------------------------------------")
    print("The {} step VeinAcc is:{}".format(i,VeinAcc))
    print("The {} step VeinSens is:{}".format(i,VeinSe))
    print("The {} step VeinSpec is:{}".format(i,VeinSp))
    print("The {} step VeinAUC is:{}".format(i,VeinAUC))
    print("-----------------------------------------------------------")
    print("The {} step VesselAcc is:{}".format(i,VesselAcc))
    print("The {} step VesselSens is:{}".format(i,VesselSe))
    print("The {} step VesselSpec is:{}".format(i,VesselSp))
    print("The {} step VesselAUC is:{}".format(i,VesselAUC))
    
    
    if not os.path.exists(savePath):
         file_w = open(savePath,'w')
    file_w = open(savePath,'r+')       
    file_w.read()
    file_w.write("=========================DRIVE=============================" + '\n' +
                 "The {} step ArteryAcc is:{}".format(i,ArteryAcc) + '\n' +
                 "The {} step ArterySens is:{}".format(i,ArterySe) + '\n' +
                 "The {} step ArterySpec is:{}".format(i,ArterySp) + '\n' +
                 "The {} step ArteryAUC is:{}".format(i,ArteryAUC) + '\n' +
                 "-----------------------------------------------------------" + '\n' +
                 "The {} step VeinAcc is:{}".format(i,VeinAcc) + '\n' +
                 "The {} step VeinSens is:{}".format(i,VeinSe) + '\n' +
                 "The {} step VeinSpec is:{}".format(i,VeinSp) + '\n' +
                 "The {} step VeinAUC is:{}".format(i,VeinAUC) + '\n' +
                 "-----------------------------------------------------------" + '\n' 
                 "The {} step VesselAcc is:{}".format(i,VesselAcc) + '\n'
                 "The {} step VesselSens is:{}".format(i,VesselSe) + '\n'
                 "The {} step VesselSpec is:{}".format(i,VesselSp) + '\n'
                 "The {} step VesselAUC is:{}".format(i,VesselAUC) + '\n') 
    file_w.close()
    
    ArteryPredAll = np.zeros((15, 1, 800, 1200), np.float32)
    VeinPredAll = np.zeros((15, 1, 800, 1200), np.float32)
    VesselPredAll = np.zeros((15, 1, 800, 1200), np.float32)
    LabelArteryAll = np.zeros((15, 1, 800, 1200), np.float32)
    LabelVeinAll = np.zeros((15, 1, 800, 1200), np.float32)
    LabelVesselAll = np.zeros((15, 1, 800, 1200), np.float32)
    MaskAll = np.zeros((15, 1, 800, 1200), np.float32)
    Vessel = VesselProMap('./data/HRF_RevisedAVLabel/images/test')

    for k in range(21,36):
        ArteryPred,VeinPred,VesselPred,LabelArtery,LabelVein,LabelVessel,Mask = GetResult(net,k,Vessel)
        ArteryPredAll[k-21,:,:,:] = cv2.resize(ArteryPred[0],(1200,800))
        VeinPredAll[k-21,:,:,:] = cv2.resize(VeinPred[0],(1200,800))
        VesselPredAll[k-21,:,:,:] = cv2.resize(VesselPred[0],(1200,800))
        LabelArteryAll[k-21,:,:,:] = LabelArtery
        LabelVeinAll[k-21,:,:,:] = LabelVein
        LabelVesselAll[k-21,:,:,:] = LabelVessel        
        MaskAll[k-21,:,:,:] = cv2.resize(Mask[0].astype(np.float32),(1200,800))
    ArteryAUC,ArteryAcc,ArterySp,ArterySe,VeinAUC,VeinAcc,VeinSp,VeinSe = Evalution_AV(ArteryPredAll,VeinPredAll,LabelArteryAll,LabelVeinAll, MaskAll)
    #VeinAUC,VeinAcc,VeinSp,VeinSe = Evalution(VeinPredAll,LabelVeinAll, MaskAll)
    VesselAUC,VesselAcc,VesselSp,VesselSe = Evalution(VesselPredAll,LabelVesselAll, MaskAll)
    
    
    print("=============================HRF===========================")
    print("The {} step ArteryAcc is:{}".format(i,ArteryAcc))
    print("The {} step ArterySens is:{}".format(i,ArterySe))
    print("The {} step ArterySpec is:{}".format(i,ArterySp))
    print("The {} step ArteryAUC is:{}".format(i,ArteryAUC))
    print("-----------------------------------------------------------")
    print("The {} step VeinAcc is:{}".format(i,VeinAcc))
    print("The {} step VeinSens is:{}".format(i,VeinSe))
    print("The {} step VeinSpec is:{}".format(i,VeinSp))
    print("The {} step VeinAUC is:{}".format(i,VeinAUC))
    print("-----------------------------------------------------------")
    print("The {} step VesselAcc is:{}".format(i,VesselAcc))
    print("The {} step VesselSens is:{}".format(i,VesselSe))
    print("The {} step VesselSpec is:{}".format(i,VesselSp))
    print("The {} step VesselAUC is:{}".format(i,VesselAUC))
    print("-----------------------------------------------------------")
    print("The {} step loss is :{}".format(i,loss_all/200))
    if not os.path.exists(savePath):
         file_w = open(savePath,'w')
    file_w = open(savePath,'r+')       
    file_w.read()
    file_w.write("=============================HRF===========================" + '\n' +
                 "The {} step ArteryAcc is:{}".format(i,ArteryAcc) + '\n' +
                 "The {} step ArterySens is:{}".format(i,ArterySe) + '\n' +
                 "The {} step ArterySpec is:{}".format(i,ArterySp) + '\n' +
                 "The {} step ArteryAUC is:{}".format(i,ArteryAUC) + '\n' +
                 "-----------------------------------------------------------" + '\n' +
                 "The {} step VeinAcc is:{}".format(i,VeinAcc) + '\n' +
                 "The {} step VeinSens is:{}".format(i,VeinSe) + '\n' +
                 "The {} step VeinSpec is:{}".format(i,VeinSp) + '\n' +
                 "The {} step VeinAUC is:{}".format(i,VeinAUC) + '\n' +
                 "-----------------------------------------------------------" + '\n' 
                 "The {} step VesselAcc is:{}".format(i,VesselAcc) + '\n'
                 "The {} step VesselSens is:{}".format(i,VesselSe) + '\n'
                 "The {} step VesselSpec is:{}".format(i,VesselSp) + '\n'
                 "The {} step VesselAUC is:{}".format(i,VesselAUC) + '\n'
                 "-----------------------------------------------------------" + '\n'
                 "The {} step loss is :{}".format(i,loss_all/200) + '\n') 
    file_w.close()
    
    
def Evalution(PredAll,LabelAll, MaskAll):
    y_scores, y_true = pred_only_FOV(PredAll,LabelAll, MaskAll)
    AUC = roc_auc_score(y_true,y_scores)           
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
    jaccard_index = jaccard_similarity_score(y_true, y_pred, normalize=True)
   
    #F1 score
    F1_score = f1_score(y_true, y_pred, labels=None, average='binary', sample_weight=None)
    return AUC,accuracy,specificity,sensitivity

def Evalution_AV(PredAll1,PredAll2,LabelAll1,LabelAll2, MaskAll):
    threshold_confusion = 0.5
    y_scores1, y_true1,y_scores2, y_true2 = pred_only_FOV_AV(PredAll1,PredAll2,LabelAll1,LabelAll2, MaskAll,threshold_confusion)
    AUC1 = roc_auc_score(y_true1,y_scores1)   
    AUC2 = roc_auc_score(y_true2,y_scores2)         
    y_pred1 = np.empty((y_scores1.shape[0]))
    y_pred2 = np.empty((y_scores2.shape[0]))
    for i in range(y_scores1.shape[0]):
        if y_scores1[i]>=threshold_confusion:
            y_pred1[i]=1
        else:
            y_pred1[i]=0
    for i in range(y_scores2.shape[0]):
        if y_scores2[i]>=threshold_confusion:
            y_pred2[i]=1
        else:
            y_pred2[i]=0
    confusion1 = confusion_matrix(y_true1, y_pred1)
    confusion2 = confusion_matrix(y_true2, y_pred2)
    
    accuracy1 = 0
    if float(np.sum(confusion1))!=0:
        accuracy1 = float(confusion1[0,0]+confusion1[1,1]-confusion2[1,0]+confusion1[0,1])/float(np.sum(confusion1))
        #accuracy1 = float(confusion1[0,0]+confusion1[1,1])/float(np.sum(confusion1))
    specificity1 = 0
    if float(confusion1[0,0]+confusion1[0,1])!=0:
        specificity1 = float(confusion2[1,1])/float(confusion2[1,1]+confusion1[0,1])
    sensitivity1 = 0
    if float(confusion1[1,1]+confusion1[1,0])!=0:
        sensitivity1 = float(confusion1[1,1])/float(confusion1[1,1]+confusion1[1,0])
        
    accuracy2 = 0
    if float(np.sum(confusion2))!=0:
        accuracy2 = float(confusion2[0,0]+confusion2[1,1]-confusion1[1,0]+confusion2[0,1])/float(np.sum(confusion2))
        #accuracy2 = float(confusion2[0,0]+confusion2[1,1])/float(np.sum(confusion2))
    specificity2 = 0
    if float(confusion2[0,0]+confusion2[0,1])!=0:
        specificity2 = float(confusion1[1,1])/float(confusion1[1,1]+confusion2[0,1])
    sensitivity2 = 0
    if float(confusion2[1,1]+confusion2[1,0])!=0:
        sensitivity2 = float(confusion2[1,1])/float(confusion2[1,1]+confusion2[1,0])
#    precision = 0
#    if float(confusion[1,1]+confusion[0,1])!=0:
#        precision = float(confusion[1,1])/float(confusion[1,1]+confusion[0,1])    
#    #Jaccard similarity index
#    jaccard_index = jaccard_similarity_score(y_true, y_pred, normalize=True)
   
    #F1 score
#    F1_score = f1_score(y_true, y_pred, labels=None, average='binary', sample_weight=None)
    return AUC1,accuracy1,specificity1,sensitivity1,AUC2,accuracy2,specificity2,sensitivity2

def inside_FOV_DRIVE(i, x, y, DRIVE_masks):
    assert (len(DRIVE_masks.shape)==4)  #4D arrays
    assert (DRIVE_masks.shape[1]==1)  #DRIVE masks is black and white
    # DRIVE_masks = DRIVE_masks/255.  #NOOO!! otherwise with float numbers takes forever!!

    if (x >= DRIVE_masks.shape[3] or y >= DRIVE_masks.shape[2]): #my image bigger than the original
        return False

    if (DRIVE_masks[i,0,y,x]>0):  #0==black pixels
        # print DRIVE_masks[i,0,y,x]  #verify it is working right
        return True
    else:
        return False
    
def pred_only_FOV(data_imgs,data_masks,original_imgs_border_masks):
    assert (len(data_imgs.shape)==4 and len(data_masks.shape)==4)  #4D arrays
    assert (data_imgs.shape[0]==data_masks.shape[0])
    assert (data_imgs.shape[2]==data_masks.shape[2])
    assert (data_imgs.shape[3]==data_masks.shape[3])
    assert (data_imgs.shape[1]==1 and data_masks.shape[1]==1)  #check the channel is 1
    height = data_imgs.shape[2]
    width = data_imgs.shape[3]
    new_pred_imgs = []
    new_pred_masks = []
    for i in range(data_imgs.shape[0]):  #loop over the full images
        for x in range(width):
            for y in range(height):
                if inside_FOV_DRIVE(i,x,y,original_imgs_border_masks)==True:
                    new_pred_imgs.append(data_imgs[i,:,y,x])
                    new_pred_masks.append(data_masks[i,:,y,x])
    new_pred_imgs = np.asarray(new_pred_imgs)
    new_pred_masks = np.asarray(new_pred_masks)
    return new_pred_imgs, new_pred_masks

def inside_FOV_DRIVE_AV(i, x, y,data_imgs1,data_imgs2, DRIVE_masks,threshold_confusion):
    assert (len(DRIVE_masks.shape)==4)  #4D arrays
    assert (DRIVE_masks.shape[1]==1)  #DRIVE masks is black and white
    # DRIVE_masks = DRIVE_masks/255.  #NOOO!! otherwise with float numbers takes forever!!

    if (x >= DRIVE_masks.shape[3] or y >= DRIVE_masks.shape[2]): #my image bigger than the original
        return False

    if (DRIVE_masks[i,0,y,x]>0)&((data_imgs1[i,0,y,x]>threshold_confusion)|(data_imgs2[i,0,y,x]>threshold_confusion)):  #0==black pixels
        # print DRIVE_masks[i,0,y,x]  #verify it is working right
        return True
    else:
        return False

def pred_only_FOV_AV(data_imgs1,data_imgs2,data_masks1,data_masks2,original_imgs_border_masks,threshold_confusion):
    assert (len(data_imgs1.shape)==4 and len(data_masks1.shape)==4)  #4D arrays
    assert (data_imgs1.shape[0]==data_masks1.shape[0])
    assert (data_imgs1.shape[2]==data_masks1.shape[2])
    assert (data_imgs1.shape[3]==data_masks1.shape[3])
    assert (data_imgs1.shape[1]==1 and data_masks1.shape[1]==1)  #check the channel is 1
    height = data_imgs1.shape[2]
    width = data_imgs1.shape[3]
    new_pred_imgs1 = []
    new_pred_masks1 = []
    new_pred_imgs2 = []
    new_pred_masks2 = []
    for i in range(data_imgs1.shape[0]):  #loop over the full images
        for x in range(width):
            for y in range(height):
                if inside_FOV_DRIVE_AV(i,x,y,data_masks1,data_masks2,original_imgs_border_masks,threshold_confusion)==True:
                    new_pred_imgs1.append(data_imgs1[i,:,y,x])
                    new_pred_masks1.append(data_masks1[i,:,y,x])
                    new_pred_imgs2.append(data_imgs2[i,:,y,x])
                    new_pred_masks2.append(data_masks2[i,:,y,x])
    new_pred_imgs1 = np.asarray(new_pred_imgs1)
    new_pred_masks1 = np.asarray(new_pred_masks1)
    new_pred_imgs2 = np.asarray(new_pred_imgs2)
    new_pred_masks2 = np.asarray(new_pred_masks2)
    return new_pred_imgs1, new_pred_masks1,new_pred_imgs2, new_pred_masks2

def GetResult(net,k,Vessel):
    if k<=9:
        ImgName = './data/AV_DRIVE/test/images/0' + str(k) + '_test.tif'
        LabelName = './data/AV_DRIVE/test/av/0' + str(k) + '_test.png'
        Vessel0 = np.transpose(Vessel[k-1,:,:,:],(1,2,0))
        MaskName = './data/AV_DRIVE/test/mask/0' + str(k) + '_test_mask.png'
        Mask0 = cv2.imread(MaskName)
        Mask = np.zeros((Mask0.shape[0],Mask0.shape[1]),np.float32)
        Mask[Mask0[:,:,2]>0] = 1
    elif k<=20:
        ImgName = './data/AV_DRIVE/test/images/' + str(k) + '_test.tif'
        LabelName = './data/AV_DRIVE/test/av/' + str(k) + '_test.png'
        Vessel0 = np.transpose(Vessel[k-1,:,:,:],(1,2,0))
        MaskName = './data/AV_DRIVE/test/mask/' + str(k) + '_test_mask.png'
        Mask0 = cv2.imread(MaskName)
        Mask = np.zeros((Mask0.shape[0],Mask0.shape[1]),np.float32)
        Mask[Mask0[:,:,2]>0] = 1
    else:
#        ImgName = './data/HRF_RevisedAVLabel/images/image' + str(k+10) + '.jpg'
#        LabelName = './data/HRF_RevisedAVLabel/Label/image' + str(k+10) + '.jpg'
        ImgPath = './data/HRF_RevisedAVLabel/images/test'
        ImgList0 = os.listdir(ImgPath)
        ImgList0 = natsort.natsorted(ImgList0)
        LabelPath = './data/HRF_RevisedAVLabel/ArteryVein/test'
        LabelList0 = os.listdir(LabelPath)
        LabelList0 = natsort.natsorted(LabelList0)
        ImgName = os.path.join(ImgPath, ImgList0[k-21])
        LabelName = os.path.join(LabelPath, LabelList0[k-21])
        Vessel0 = np.transpose(Vessel[k-21,:,:,:],(1,2,0))
    Img0 = cv2.imread(ImgName)
    
    
    Label0 = cv2.imread(LabelName)
    Img0 = cv2.resize(Img0,(565,584))
    Vessel0 = cv2.resize(Vessel0,(565,584))
    #Label0 = cv2.resize(Label0,(565,584))

    LabelArtery = np.zeros((Label0.shape[0], Label0.shape[1]), np.float32)
    LabelVein = np.zeros((Label0.shape[0], Label0.shape[1]), np.float32)
    LabelVessel = np.zeros((Label0.shape[0], Label0.shape[1]), np.float32)
    LabelArtery[(Label0[:,:,2]>=128)|(Label0[:,:,1]>=128)] = 1
    LabelArtery[(Label0[:,:,2]>=128)&(Label0[:,:,1]>=128)&(Label0[:,:,0]>=128)] = 0
    LabelVein[(Label0[:,:,1]>=128)|(Label0[:,:,0]>=128)] = 1
    LabelVein[(Label0[:,:,2]>=128)&(Label0[:,:,1]>=128)&(Label0[:,:,0]>=128)] = 0
    LabelVessel[(Label0[:,:,2]>=128)|(Label0[:,:,1]>=128)|(Label0[:,:,0]>=128)] = 1
    
    #print(ImgName)
    TempImg, TempMask = creatMask(Img0, threshold=10)
    ImgCropped, MaskCropped, cropLimit = cropImage(Img0, TempMask)

    IllumImage = illuminationCorrection(Img0, kernel_size=25, Mask=TempMask)
    ImgIllCropped, MaskCropped, cropLimit = cropImage(IllumImage, TempMask)
    #downsizeRatio = 512./np.maximum(ImgCropped.shape[0], ImgCropped.shape[1])
    #ImgResized = imageResize(ImgCropped, downsizeRatio=downsizeRatio)
    #MaskResized = imageResize(MaskCropped, downsizeRatio=downsizeRatio)
    #Img = ImgCropped  ##Replace the Image with Cropped Image
    #Mask = MaskCropped  ##Replace the Mask with cropped mask
    if k>20:
        Mask = TempMask
    Img = Img0
    ImgIllCropped = IllumImage
    
    
    
    
    
    # LabelImg = cv2.imread(LabelName)
    Mask = morphology.binary_erosion(Mask, morphology.disk(10))
    
    height, width = Img.shape[:2]
    #############################################
    # from lib.extract_patches import *
    # from lib.help_functions import *    
    n_classes = 3
    patch_height = 64
    patch_width = 64
    stride_height = 10
    stride_width = 10
    
    
    # Img = illuminationCorrection2(Img, kernel_size=35)
    Img = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)
    Img = np.float32(Img/255.)
    Img_enlarged = paint_border_overlap(Img, patch_height, patch_width, stride_height, stride_width)
    Vessel_enlarged = paint_border_overlap_trad(Vessel0, patch_height, patch_width, stride_height, stride_width)
    
    ImgIllCropped = cv2.cvtColor(ImgIllCropped, cv2.COLOR_BGR2RGB)
    ImgIllCropped = np.float32(ImgIllCropped/255.)
    ImgIll_enlarged = paint_border_overlap(ImgIllCropped, patch_height, patch_width, stride_height, stride_width)
#    print('Old shape:', Img.shape)
#    print('Enlarged shape:', Img_enlarged.shape)
    
    
    # ############################################################
    #load model and predict etc

    # from models.ResUnet_Nov9 import ResUnet
    # from models.ResUnet_middle import ResUnet
    
    
    #model_path = model_name
        
    Net = ResUnet_illum_tran_trad_conv(resnet='resnet18',  num_classes= n_classes)
    Net.load_state_dict(net)
    
    Net.cuda()
    Net.eval()
    
    
    patch_size = 64
    batch_size = 128
    
    patches_imgs = extract_ordered_overlap(Img_enlarged, patch_height, patch_width, stride_height, stride_width)
    patches_imgs = np.transpose(patches_imgs,(0,3,1,2))
    patches_imgs = Normalize(patches_imgs)
    
    patches_vessel1 = extract_ordered_overlap_trad(Vessel_enlarged, patch_height, patch_width, stride_height, stride_width,1)
    patches_vessel1 = np.transpose(patches_vessel1,(0,3,1,2))
    patches_vessel2 = extract_ordered_overlap_trad(Vessel_enlarged, patch_height, patch_width, stride_height, stride_width,2)
    patches_vessel2 = np.transpose(patches_vessel2,(0,3,1,2))
    patches_vessel3 = extract_ordered_overlap_trad(Vessel_enlarged, patch_height, patch_width, stride_height, stride_width,4)
    patches_vessel3 = np.transpose(patches_vessel3,(0,3,1,2))
    
    patches_imgsIll = extract_ordered_overlap(ImgIll_enlarged, patch_height, patch_width, stride_height, stride_width)
    patches_imgsIll = np.transpose(patches_imgsIll,(0,3,1,2))
    patches_imgsIll = Normalize(patches_imgsIll)
    
    
    patchNum = patches_imgs.shape[0]
    max_iter = int(np.ceil(patchNum/float(batch_size)))
    
    pred_patches = np.zeros((patchNum, n_classes, patch_size, patch_size), np.float32)
    for i in range(max_iter):
    
    
        begin_index = i*batch_size
        end_index = (i+1)*batch_size
    
        patches_temp1 = patches_imgs[begin_index:end_index, :, :, :]
        patches_temp2 = patches_imgsIll[begin_index:end_index, :, :, :]
        patches_temp3 = patches_vessel1[begin_index:end_index, :, :, :]
        patches_temp4 = patches_vessel2[begin_index:end_index, :, :, :]
        patches_temp5 = patches_vessel3[begin_index:end_index, :, :, :]
        #print(i, patches_temp.shape)
    
        patches_input_temp1 = torch.FloatTensor(patches_temp1)
        patches_input_temp1 = autograd.Variable(patches_input_temp1.cuda())
        patches_input_temp2 = torch.FloatTensor(patches_temp2)
        patches_input_temp2 = autograd.Variable(patches_input_temp2.cuda())
        patches_input_temp3 = torch.FloatTensor(patches_temp3)
        patches_input_temp3 = autograd.Variable(patches_input_temp3.cuda())
        patches_input_temp4 = torch.FloatTensor(patches_temp4)
        patches_input_temp4 = autograd.Variable(patches_input_temp4.cuda())
        patches_input_temp5 = torch.FloatTensor(patches_temp5)
        patches_input_temp5 = autograd.Variable(patches_input_temp5.cuda())
        output_temp = Net(patches_input_temp1,patches_input_temp2,patches_input_temp3,patches_input_temp4,patches_input_temp5)
        pred_patches_temp = np.float32(output_temp.data.cpu().numpy())
    
        pred_patches_temp_sigmoid = sigmoid(pred_patches_temp)
    
        pred_patches[begin_index:end_index, :,:,:] = pred_patches_temp_sigmoid
    
        del patches_input_temp1
        del patches_input_temp2
        del pred_patches_temp
        del patches_temp1
        del patches_temp2
        del output_temp
        del pred_patches_temp_sigmoid
    
    
    #print('pred_patches', pred_patches.shape)
    
    # ############################################################
    
    new_height, new_width = Img_enlarged.shape[0], Img_enlarged.shape[1]
    pred_img = recompone_overlap(pred_patches, new_height, new_width, stride_height, stride_width)  # predictions
    pred_img = pred_img[:,0:height,0:width]
    pred_img = kill_border(pred_img, Mask)
    
    
    
    
    ArteryPred = np.float32(pred_img[0,:,:])
    VeinPred = np.float32(pred_img[2,:,:])
    VesselPred = np.float32(pred_img[1,:,:])
    
    ArteryPred = ArteryPred[np.newaxis,:,:]
    VeinPred = VeinPred[np.newaxis,:,:]
    VesselPred = VesselPred[np.newaxis,:,:]
    LabelArtery = LabelArtery[np.newaxis,:,:]
    LabelVein = LabelVein[np.newaxis,:,:]
    LabelVessel = LabelVessel[np.newaxis,:,:]
    Mask = Mask[np.newaxis,:,:]
    
    
    return ArteryPred,VeinPred,VesselPred,LabelArtery,LabelVein,LabelVessel,Mask


def illuminationCorrection(Image, kernel_size, Mask):
    #input: original RGB image and kernel size
    #output: illumination corrected RGB image
    ## The return can be a RGB 3 channel image, but better to show the user the green channel only
    ##since green channel is more clear and has higher contrast

    Mask = np.uint8(Mask)
    Mask[Mask > 0] = 1
    Mask0 = Mask.copy()

    Img_pad = fakePad(Image, Mask, iterations=30)


    BackgroundIllumImage = cv2.medianBlur(Img_pad, ksize = kernel_size)

    maximumVal = np.max(BackgroundIllumImage)
    minimumVal = np.min(BackgroundIllumImage)
    constVal = maximumVal - 128

    BackgroundIllumImage[BackgroundIllumImage <=10] = 100
    IllumImage = Img_pad * (maximumVal / BackgroundIllumImage) - constVal
    IllumImage[IllumImage>255] = 255
    IllumImage[IllumImage<0] = 0
    IllumImage = np.uint8(IllumImage)

    IllumImage = cv2.bitwise_and(IllumImage, IllumImage, mask=Mask0)
    # IllumImage = cv2.medianBlur(IllumImage, ksize=3)

    return IllumImage

def ComputePara(net,savePath):
    params = list(net.parameters())
    k = 0
    if not os.path.exists(savePath):
         file_w = open(savePath,'w')
    file_w = open(savePath,'r+')  
    file_w.read()
    for i in params:
        l = 1
        print("layer structure:" + str(list(i.size())))
        file_w.write("layer structure:" + str(list(i.size())) + '\n') 
        for j in i.size():
            l *= j
        print("layer paramenters:"+str(l))
        file_w.write("layer paramenters:" + str(l) + '\n')
        k += l
    print("network paramenters:"+str(k))
    file_w.write("network paramenters:" + str(k) + '\n') 
    file_w.close()


def pixAcc(pred, targ):
    pred = pred>0
    targ = targ>0
    acc = (pred==targ).sum() /(targ>=0).sum()
    return acc

def pixSens(pred,targ):
    pred = pred > 0
    targ = targ > 0
    TP = ((pred==targ)&(pred>0)).sum()
    FN = ((pred!=targ)&(pred==0)).sum()
    return TP/(TP+FN)

def pixSpec(pred,targ):
    pred = pred > 0
    targ = targ > 0
    TN = ((pred==targ)&(pred==0)).sum()
    FP = ((pred!=targ)&(pred>0)).sum()
    return TN/(TN+FP)


def DataSet_random_illum(path): 
    
    ImgPath = path + "images/"
    LabelPath = path + "av/"
    ImgDir = os.listdir(ImgPath)
    LabelDir = os.listdir(LabelPath)
    
    Img0 = cv2.imread(ImgPath + ImgDir[0])
    Label0 = cv2.imread(LabelPath + LabelDir[0])
    
    Img0 = cv2.cvtColor(Img0, cv2.COLOR_BGR2RGB)
    Img0 = np.float32(Img0/255.)
    
    Img = np.zeros((len(ImgDir), 3, Img0.shape[0], Img0.shape[1]), np.float32)
    Img_illum = np.zeros((len(ImgDir), 3, Img0.shape[0], Img0.shape[1]), np.float32)
    Label = np.zeros((len(ImgDir), 3, Img0.shape[0], Img0.shape[1]), np.float32)
    
    
    for i in range(0,20):
        LabelArtery = np.zeros((Img0.shape[0], Img0.shape[1]), np.uint8)
        LabelVein = np.zeros((Img0.shape[0], Img0.shape[1]), np.uint8)
        LabelVessel = np.zeros((Img0.shape[0], Img0.shape[1]), np.uint8)
        
        Img0 = cv2.imread(ImgPath + str(i+21) + '_training.tif')
        Label0 = cv2.imread(LabelPath + str(i+21) + '_training.png')
        
        TempImg, TempMask = creatMask(Img0, threshold=10)
        ImgCropped, MaskCropped, cropLimit = cropImage(Img0, TempMask)
        IllumImage = illuminationCorrection(Img0, kernel_size=25, Mask=TempMask)
        ImgIllCropped, MaskCropped, cropLimit = cropImage(IllumImage, TempMask)
        
        
        
#        Img0 = cv2.cvtColor(Img0, cv2.COLOR_BGR2RGB)
#        Img0 = np.float32(Img0/255.)
        
        
#        downsizeRatio = 512./np.maximum(ImgCropped.shape[0], ImgCropped.shape[1])
#        ImgResized = imageResize(ImgCropped, downsizeRatio=downsizeRatio)
        
        #MaskResized = imageResize(MaskCropped, downsizeRatio=downsizeRatio)
        #Img = ImgResized  ##Replace the Image with Cropped Image
        #Mask = MaskResized  ##Replace the Mask with cropped mask
        
        LabelArtery[(Label0[:,:,2]==255)|(Label0[:,:,1]==255)] = 1
        LabelArtery[(Label0[:,:,2]==255)&(Label0[:,:,1]==255)&(Label0[:,:,0]==255)] = 0
        LabelVein[(Label0[:,:,1]==255)|(Label0[:,:,0]==255)] = 1
        LabelVein[(Label0[:,:,2]==255)&(Label0[:,:,1]==255)&(Label0[:,:,0]==255)] = 0
        LabelVessel[(Label0[:,:,2]==255)|(Label0[:,:,1]==255)|(Label0[:,:,0]==255)] = 1
        ImgCropped = cv2.cvtColor(ImgCropped, cv2.COLOR_BGR2RGB)
        ImgCropped = np.float32(ImgCropped/255.)
        ImgIllCropped = cv2.cvtColor(ImgIllCropped, cv2.COLOR_BGR2RGB)
        ImgIllCropped = np.float32(ImgIllCropped/255.)
        Img[i,:,:,:] = np.transpose(ImgCropped,(2,0,1))
        Img_illum[i,:,:,:] = np.transpose(ImgIllCropped,(2,0,1))
        Label[i,0,:,:] = LabelArtery
        Label[i,1,:,:] = LabelVein
        Label[i,2,:,:] = LabelVessel
        #Label = imageResize(Label, downsizeRatio=downsizeRatio)
        #Img_enlarged = paint_border_overlap(ImgResized, patch_height, patch_width, stride_height, stride_width)
    Img = Normalize(Img)
    Img_illum = Normalize(Img_illum)
    return Img,Img_illum,Label

def DataSet_random(path): 
    
    ImgPath = path + "images/"
    LabelPath = path + "av/"
    ImgDir = os.listdir(ImgPath)
    LabelDir = os.listdir(LabelPath)
    
    Img0 = cv2.imread(ImgPath + ImgDir[0])
    Label0 = cv2.imread(LabelPath + LabelDir[0])
    
    Img0 = cv2.cvtColor(Img0, cv2.COLOR_BGR2RGB)
    Img0 = np.float32(Img0/255.)
    
    Img = np.zeros((len(ImgDir), 3, Img0.shape[0], Img0.shape[1]), np.float32)
    Label = np.zeros((len(ImgDir), 3, Img0.shape[0], Img0.shape[1]), np.float32)
    
    
    for i in range(0,20):
        LabelArtery = np.zeros((Img0.shape[0], Img0.shape[1]), np.uint8)
        LabelVein = np.zeros((Img0.shape[0], Img0.shape[1]), np.uint8)
        LabelVessel = np.zeros((Img0.shape[0], Img0.shape[1]), np.uint8)
        Img0 = cv2.imread(ImgPath + str(i+21) + '_training.tif')
        Label0 = cv2.imread(LabelPath + str(i+21) + '_training.png')
        TempImg, TempMask = creatMask(Img0, threshold=10)
        ImgCropped, MaskCropped, cropLimit = cropImage(Img0, TempMask)
        
#        Img0 = cv2.cvtColor(Img0, cv2.COLOR_BGR2RGB)
#        Img0 = np.float32(Img0/255.)
        
        
#        downsizeRatio = 512./np.maximum(ImgCropped.shape[0], ImgCropped.shape[1])
#        ImgResized = imageResize(ImgCropped, downsizeRatio=downsizeRatio)
        
        #MaskResized = imageResize(MaskCropped, downsizeRatio=downsizeRatio)
        #Img = ImgResized  ##Replace the Image with Cropped Image
        #Mask = MaskResized  ##Replace the Mask with cropped mask
        
        LabelArtery[(Label0[:,:,2]==255)|(Label0[:,:,1]==255)] = 1
        LabelArtery[(Label0[:,:,2]==255)&(Label0[:,:,1]==255)&(Label0[:,:,0]==255)] = 0
        LabelVein[(Label0[:,:,1]==255)|(Label0[:,:,0]==255)] = 1
        LabelVein[(Label0[:,:,2]==255)&(Label0[:,:,1]==255)&(Label0[:,:,0]==255)] = 0
        LabelVessel[(Label0[:,:,2]==255)|(Label0[:,:,1]==255)|(Label0[:,:,0]==255)] = 1
        ImgCropped = cv2.cvtColor(ImgCropped, cv2.COLOR_BGR2RGB)
        ImgCropped = np.float32(ImgCropped/255.)
        Img[i,:,:,:] = np.transpose(ImgCropped,(2,0,1))
        Label[i,0,:,:] = LabelArtery
        Label[i,1,:,:] = LabelVein
        Label[i,2,:,:] = LabelVessel
        #Label = imageResize(Label, downsizeRatio=downsizeRatio)
        #Img_enlarged = paint_border_overlap(ImgResized, patch_height, patch_width, stride_height, stride_width)
    Img = Normalize(Img)
    return Img,Label

def DataSet(path,patch_size,step_size): 
    
    ImgPath = path + "images/"
    LabelPath = path + "av/"
    ImgDir = os.listdir(ImgPath)
    LabelDir = os.listdir(LabelPath)
    
    Img0 = cv2.imread(ImgPath + ImgDir[0])
    Label0 = cv2.imread(LabelPath + LabelDir[0])
    
    
    patch_height = patch_size
    patch_width = patch_size
    stride_height = step_size
    stride_width = step_size
        
    Img_enlarged = paint_border_overlap(Img0, patch_height, patch_width, stride_height, stride_width)
    patches_imgs = extract_ordered_overlap(Img_enlarged, patch_height, patch_width, stride_height, stride_width)
    Img = np.zeros((len(ImgDir),3, patches_imgs.shape[0],patch_height, patch_width), np.float32)
    Label = np.zeros((len(ImgDir),3,patches_imgs.shape[0],patch_height, patch_width), np.float32)
        
    for i in range(0,20):
        LabelArtery = np.zeros((patches_imgs.shape[0],patch_height, patch_width), np.uint8)
        LabelVein = np.zeros((patches_imgs.shape[0],patch_height, patch_width), np.uint8)
        LabelVessel = np.zeros((patches_imgs.shape[0],patch_height, patch_width), np.uint8)
        Img0 = cv2.imread(ImgPath + str(i+21) + '_training.tif')
        Label0 = cv2.imread(LabelPath + str(i+21) + '_training.png')
        
        
        
        TempImg, TempMask = creatMask(Img0, threshold=10)
        ImgCropped, MaskCropped, cropLimit = cropImage(Img0, TempMask)
#        downsizeRatio = 512./np.maximum(ImgCropped.shape[0], ImgCropped.shape[1])
#        ImgResized = imageResize(ImgCropped, downsizeRatio=downsizeRatio)
        
        #MaskResized = imageResize(MaskCropped, downsizeRatio=downsizeRatio)
        #Img = ImgResized  ##Replace the Image with Cropped Image
        #Mask = MaskResized  ##Replace the Mask with cropped mask
        
        Label_enlarged = paint_border_overlap(Label0, patch_height, patch_width, stride_height, stride_width)
        patches_labels = extract_ordered_overlap(Label_enlarged, patch_height, patch_width, stride_height, stride_width)

        
        
        LabelArtery[(patches_labels[:,:,:,2]==255)|(patches_labels[:,:,:,1]==255)] = 1
        LabelArtery[(patches_labels[:,:,:,2]==255)&(patches_labels[:,:,:,1]==255)&(patches_labels[:,:,:,0]==255)] = 0
        LabelVein[(patches_labels[:,:,:,1]==255)|(patches_labels[:,:,:,0]==255)] = 1
        LabelVein[(patches_labels[:,:,:,2]==255)&(patches_labels[:,:,:,1]==255)&(patches_labels[:,:,:,0]==255)] = 0
        LabelVessel[(patches_labels[:,:,:,2]==255)|(patches_labels[:,:,:,1]==255)|(patches_labels[:,:,:,0]==255)] = 1
        ImgCropped = cv2.cvtColor(ImgCropped, cv2.COLOR_BGR2RGB)
        ImgCropped = np.float32(ImgCropped/255.)
        Img_enlarged = paint_border_overlap(ImgCropped, patch_height, patch_width, stride_height, stride_width)
        patches_imgs = extract_ordered_overlap(Img_enlarged, patch_height, patch_width, stride_height, stride_width)
        
        Img[i,:,:,:,:] = np.transpose(patches_imgs,(3,0,1,2))
        Label[i,0,:,:,:] = LabelArtery
        Label[i,1,:,:,:] = LabelVein
        Label[i,2,:,:,:] = LabelVessel
        #Label = imageResize(Label, downsizeRatio=downsizeRatio)
        #Img_enlarged = paint_border_overlap(ImgResized, patch_height, patch_width, stride_height, stride_width)
    Img = Normalize(Img)
    return Img,Label
    
    
def TestAccuracy(net,k):
    if k<=9:
        ImgName = './data/AV_DRIVE/test/images/0' + str(k) + '_test.tif'
        LabelName = './data/AV_DRIVE/test/av/0' + str(k) + '_test.png'
    else:
        ImgName = './data/AV_DRIVE/test/images/' + str(k) + '_test.tif'
        LabelName = './data/AV_DRIVE/test/av/' + str(k) + '_test.png'
    Img0 = cv2.imread(ImgName)
    
    Label0 = cv2.imread(LabelName)
    LabelArtery = np.zeros((Img0.shape[0], Img0.shape[1]), np.float32)
    LabelVein = np.zeros((Img0.shape[0], Img0.shape[1]), np.float32)
    LabelVessel = np.zeros((Img0.shape[0], Img0.shape[1]), np.float32)
    LabelArtery[(Label0[:,:,2]==255)|(Label0[:,:,1]==255)] = 1
    LabelArtery[(Label0[:,:,2]==255)&(Label0[:,:,1]==255)&(Label0[:,:,0]==255)] = 0
    LabelVein[(Label0[:,:,1]==255)|(Label0[:,:,0]==255)] = 1
    LabelVein[(Label0[:,:,2]==255)&(Label0[:,:,1]==255)&(Label0[:,:,0]==255)] = 0
    LabelVessel[(Label0[:,:,2]==255)|(Label0[:,:,1]==255)|(Label0[:,:,0]==255)] = 1
    
    #print(ImgName)
    TempImg, TempMask = creatMask(Img0, threshold=10)
    ImgCropped, MaskCropped, cropLimit = cropImage(Img0, TempMask)
    IllumImage = illuminationCorrection(Img0, kernel_size=25, Mask=TempMask)
    ImgIllCropped, MaskCropped, cropLimit = cropImage(IllumImage, TempMask)
    #downsizeRatio = 512./np.maximum(ImgCropped.shape[0], ImgCropped.shape[1])
    #ImgResized = imageResize(ImgCropped, downsizeRatio=downsizeRatio)
    #MaskResized = imageResize(MaskCropped, downsizeRatio=downsizeRatio)
    Img = ImgCropped  ##Replace the Image with Cropped Image
    Mask = MaskCropped  ##Replace the Mask with cropped mask
    
    
    # LabelImg = cv2.imread(LabelName)
    Mask = morphology.binary_erosion(Mask, morphology.disk(10))
    
    height, width = Img.shape[:2]
    #############################################
    # from lib.extract_patches import *
    # from lib.help_functions import *    
    n_classes = 3
    patch_height = 64
    patch_width = 64
    stride_height = 10
    stride_width = 10
    
    
    # Img = illuminationCorrection2(Img, kernel_size=35)
    Img = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)
    Img = np.float32(Img/255.)
    Img_enlarged = paint_border_overlap(Img, patch_height, patch_width, stride_height, stride_width)
    
    ImgIllCropped = cv2.cvtColor(ImgIllCropped, cv2.COLOR_BGR2RGB)
    ImgIllCropped = np.float32(ImgIllCropped/255.)
    ImgIll_enlarged = paint_border_overlap(ImgIllCropped, patch_height, patch_width, stride_height, stride_width)
#    print('Old shape:', Img.shape)
#    print('Enlarged shape:', Img_enlarged.shape)
    
    
    # ############################################################
    #load model and predict etc

    # from models.ResUnet_Nov9 import ResUnet
    # from models.ResUnet_middle import ResUnet
    
    
    #model_path = model_name
        
    Net = ResUnet_illum(resnet='resnet18',  num_classes= n_classes)
    Net.load_state_dict(net)
    
    Net.cuda()
    Net.eval()
    
    
    patch_size = 64
    batch_size = 128
    
    patches_imgs = extract_ordered_overlap(Img_enlarged, patch_height, patch_width, stride_height, stride_width)
    patches_imgs = np.transpose(patches_imgs,(0,3,1,2))
    patches_imgs = Normalize(patches_imgs)
    patches_imgsIll = extract_ordered_overlap(ImgIll_enlarged, patch_height, patch_width, stride_height, stride_width)
    patches_imgsIll = np.transpose(patches_imgsIll,(0,3,1,2))
    patches_imgsIll = Normalize(patches_imgsIll)
    
    
    patchNum = patches_imgs.shape[0]
    max_iter = int(np.ceil(patchNum/float(batch_size)))
    
    pred_patches = np.zeros((patchNum, n_classes, patch_size, patch_size), np.float32)
    for i in range(max_iter):
    
    
        begin_index = i*batch_size
        end_index = (i+1)*batch_size
    
        patches_temp1 = patches_imgs[begin_index:end_index, :, :, :]
        patches_temp2 = patches_imgsIll[begin_index:end_index, :, :, :]
        #print(i, patches_temp.shape)
    
        patches_input_temp1 = torch.FloatTensor(patches_temp1)
        patches_input_temp1 = autograd.Variable(patches_input_temp1.cuda())
        patches_input_temp2 = torch.FloatTensor(patches_temp2)
        patches_input_temp2 = autograd.Variable(patches_input_temp2.cuda())
        output_temp = Net(patches_input_temp1,patches_input_temp2)
        pred_patches_temp = np.float32(output_temp.data.cpu().numpy())
    
        pred_patches_temp_sigmoid = sigmoid(pred_patches_temp)
    
        pred_patches[begin_index:end_index, :,:,:] = pred_patches_temp_sigmoid
    
        del patches_input_temp1
        del patches_input_temp2
        del pred_patches_temp
        del patches_temp1
        del patches_temp2
        del output_temp
        del pred_patches_temp_sigmoid
    
    
    #print('pred_patches', pred_patches.shape)
    
    # ############################################################
    
    new_height, new_width = Img_enlarged.shape[0], Img_enlarged.shape[1]
    pred_img = recompone_overlap(pred_patches, new_height, new_width, stride_height, stride_width)  # predictions
    pred_img = pred_img[:,0:height,0:width]
    pred_img = kill_border(pred_img, Mask)
    
    
    
    
    ArteryPred = np.float32(pred_img[0,:,:])
    VeinPred = np.float32(pred_img[2,:,:])
    VesselPred = np.float32(pred_img[1,:,:])
    
    ArrLabelArtery = LabelArtery.flatten()
    
    ArteryAUC = metrics.roc_auc_score(LabelArtery.flatten(),ArteryPred.flatten())
    VeinAUC = metrics.roc_auc_score(LabelVein.flatten(),VeinPred.flatten())
    VesselAUC = metrics.roc_auc_score(LabelVessel.flatten(),VesselPred.flatten())


    ArteryPred[ArteryPred>0.5] = 1
    VeinPred[VeinPred>0.5] = 1
    VesselPred[VesselPred>0.5] = 1
    ArteryPred[ArteryPred<=0.5] = 0
    VeinPred[VeinPred<=0.5] = 0
    VesselPred[VesselPred<=0.5] = 0
    
    
    ArteryAcc = pixAcc(ArteryPred,LabelArtery)
    VeinAcc = pixAcc(VeinPred,LabelVein)
    VesselAcc = pixAcc(VesselPred,LabelVessel)
    ArterySens = pixSens(ArteryPred,LabelArtery)
    VeinSens = pixSens(VeinPred,LabelVein)
    VesselSens = pixSens(VesselPred,LabelVessel)
    ArterySpec = pixSpec(ArteryPred,LabelArtery)
    VeinSpec = pixSpec(VeinPred,LabelVein)
    VesselSpec = pixSpec(VesselPred,LabelVessel)

    
    return ArteryAcc,VeinAcc,VesselAcc,ArterySens,VeinSens,VesselSens,ArterySpec,VeinSpec,VesselSpec,ArteryAUC,VeinAUC,VesselAUC
    ##################################################################################################
#    print("End of Image Processing >>>", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
#    plt.figure()
#    Images = [Img,  ArteryPred, VeinPred, VesselPred]
#    Titles = [ 'Img', 'ArteryPred', 'VeinPred',  'VesselPred']
#    
#    for i in range(0, len(Images)):
#        plt.subplot(2, 2, i + 1), plt.imshow(Images[i], 'gray'), plt.title(Titles[i])
#    plt.show()


def data_aug(data_mat_1,data_mat_2,label_mat,choice):
    data_mat_1 = np.transpose(data_mat_1,(1,2,0))
    data_mat_2 = np.transpose(data_mat_2,(1,2,0))
    label_mat = np.transpose(label_mat,(1,2,0))
    if choice==0:
        data_mat_1 = data_mat_1
        data_mat_2 = data_mat_2
        label_mat = label_mat
    elif choice==1:
        data_mat_1 = np.fliplr(data_mat_1)
        data_mat_2 = np.fliplr(data_mat_2)
        label_mat = np.fliplr(label_mat)
    elif choice==2: 
        data_mat_1 = np.flipud(data_mat_1)
        data_mat_2 = np.flipud(data_mat_2)
        label_mat = np.flipud(label_mat)
    elif choice==3:
        data_mat_1,data_mat_2,label_mat = data_augmentation1(data_mat_1,data_mat_2,label_mat)
    elif choice==4:
        data_mat_1,data_mat_2,label_mat = data_augmentation2(data_mat_1,data_mat_2,label_mat)
    elif choice==5:
        data_mat_1,data_mat_2,label_mat = data_augmentation3(data_mat_1,data_mat_2,label_mat)
    elif choice==6:
        data_mat_1,data_mat_2,label_mat = data_augmentation4(data_mat_1,data_mat_2,label_mat)
    
    data_mat_1 = np.transpose(data_mat_1,(2,0,1))
    data_mat_2 = np.transpose(data_mat_2,(2,0,1))
    label_mat = np.transpose(label_mat,(2,0,1))
    
    return data_mat_1,data_mat_2,label_mat

def data_augmentation1(image1,image2,image3):
    #image3 = np.expand_dims(image3,-1)
    [image1,image2,image3] = tl.prepro.rotation_multi([image1,image2,image3] , rg=90, is_random=True, fill_mode='constant')        
    [image1,image2,image3] = np.squeeze([image1,image2,image3]).astype(np.float32)
    return image1,image2,image3

def data_augmentation3(image1,image2,image3):
    #image3 = np.expand_dims(image3,-1)
    [image1,image2,image3] = tl.prepro.shift_multi([image1,image2,image3] ,  wrg=0.10,  hrg=0.10, is_random=True, fill_mode='constant')
    [image1,image2,image3] = np.squeeze([image1,image2,image3]).astype(np.float32)
    return image1,image2,image3 

def data_augmentation4(image1,image2,image3):
    #image3 = np.expand_dims(image3,-1)
    [image1,image2,image3] = tl.prepro.elastic_transform_multi([image1,image2,image3], alpha=720, sigma=24, is_random=True) 
    [image1,image2,image3] = np.squeeze([image1,image2,image3]).astype(np.float32)
    return image1,image2,image3

def data_augmentation2(image1,image2,image3):
    #image3 = np.expand_dims(image3,-1) 
    [image1,image2,image3] = tl.prepro.zoom_multi([image1,image2,image3] , zoom_range=[0.7, 1.2], is_random=True, fill_mode='constant')      
    [image1,image2,image3] = np.squeeze([image1,image2,image3]).astype(np.float32) 
    return image1,image2,image3    

def data_augmentation1_2(image1,image2,image3,image4):
    #image3 = np.expand_dims(image3,-1)
    [image1,image2,image3,image4] = tl.prepro.rotation_multi([image1,image2,image3,image4] , rg=90, is_random=True, fill_mode='constant')        
    [image1,image2,image3,image4] = np.squeeze([image1,image2,image3,image4]).astype(np.float32)
    return image1,image2,image3,image4

def data_augmentation3_2(image1,image2,image3,image4):
    #image3 = np.expand_dims(image3,-1)
    [image1,image2,image3,image4] = tl.prepro.shift_multi([image1,image2,image3,image4] ,  wrg=0.10,  hrg=0.10, is_random=True, fill_mode='constant')
    [image1,image2,image3,image4] = np.squeeze([image1,image2,image3,image4]).astype(np.float32)
    return image1,image2,image3,image4

def data_augmentation4_2(image1,image2,image3,image4):
    #image3 = np.expand_dims(image3,-1)
    [image1,image2,image3,image4] = tl.prepro.elastic_transform_multi([image1,image2,image3,image4], alpha=720, sigma=24, is_random=True) 
    [image1,image2,image3,image4] = np.squeeze([image1,image2,image3,image4]).astype(np.float32)
    return image1,image2,image3,image4

def data_augmentation2_2(image1,image2,image3,image4):
    #image3 = np.expand_dims(image3,-1) 
    [image1,image2,image3,image4] = tl.prepro.zoom_multi([image1,image2,image3,image4] , zoom_range=[0.7, 1.2], is_random=True, fill_mode='constant')      
    [image1,image2,image3,image4] = np.squeeze([image1,image2,image3,image4]).astype(np.float32) 
    return image1,image2,image3,image4

def data_aug2(data_mat_1, data_mat_2, data_mat_3,label_mat, choice):
    data_mat_1 = np.transpose(data_mat_1,(1,2,0))
    data_mat_2 = np.transpose(data_mat_2,(1,2,0))
    data_mat_3 = np.transpose(data_mat_3,(1,2,0))
    label_mat = np.transpose(label_mat,(1,2,0))
    if choice==0:
        data_mat_1 = data_mat_1
        data_mat_2 = data_mat_2
        data_mat_3 = data_mat_3
        label_mat = label_mat
    elif choice==1:
        data_mat_1 = np.fliplr(data_mat_1)
        data_mat_2 = np.fliplr(data_mat_2)
        data_mat_3 = np.fliplr(data_mat_3)
        label_mat = np.fliplr(label_mat)
    elif choice==2: 
        data_mat_1 = np.flipud(data_mat_1)
        data_mat_2 = np.flipud(data_mat_2)
        data_mat_3 = np.flipud(data_mat_3)
        label_mat = np.flipud(label_mat)
    elif choice==3:
        data_mat_1,data_mat_2,data_mat_3,label_mat = data_augmentation1_2(data_mat_1,data_mat_2,data_mat_3,label_mat)
    elif choice==4:
        data_mat_1,data_mat_2,data_mat_3,label_mat = data_augmentation2_2(data_mat_1,data_mat_2,data_mat_3,label_mat)
    elif choice==5:
        data_mat_1,data_mat_2,data_mat_3,label_mat = data_augmentation3_2(data_mat_1,data_mat_2,data_mat_3,label_mat)
    elif choice==6:
        data_mat_1,data_mat_2,data_mat_3,label_mat = data_augmentation4_2(data_mat_1,data_mat_2,data_mat_3,label_mat)
    
    data_mat_1 = np.transpose(data_mat_1,(2,0,1))
    data_mat_2 = np.transpose(data_mat_2,(2,0,1))
    data_mat_3 = np.transpose(data_mat_3,(2,0,1))
    label_mat = np.transpose(label_mat,(2,0,1))
    
    return data_mat_1,data_mat_2,data_mat_3,label_mat
