# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 20:28:12 2019

@author: Administrator
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import natsort
import pandas as pd

from skimage import morphology
from sklearn import metrics
from Tools.BGR2RGB import BGR2RGB
from Tools.BinaryPostProcessing import binaryPostProcessing3
#########################################
def softmax(x):
    e_x = np.exp(x-np.max(x))
    return e_x / e_x.sum()
#########################################
#def AVclassification_INSPIRE(ArteryPredAll,VeinPredAll,VesselPredAll,LabelArteryAll,LabelVeinAll,LabelVesselAll,MaskAll,1):
def AVclassification_INSPIRE(ArteryPredAll,VeinPredAll,VesselPredAll,LabelArteryAll,LabelVeinAll,LabelVesselAll,MaskAll):
    senList_sk = []
    specList_sk = []
    accList_sk = []
    for k in range(40):
        ImgNumber = k
        DF_disc = pd.read_excel(r'D:\项目\RBVS\data\INSPIRE_AVR\DiskParameters_INSPIRE.xls', sheet_name=0)
        label_folder = './data/INSPIRE_AVR/ResizedLabel_400/'
        
        
        
        labelList = os.listdir(label_folder)
        labelList = [x for x in labelList if x.__contains__('.tif')]
        labelList = natsort.natsorted(labelList)
        
        # for ImgNumber in range(len(ImgList)):
    
        
    
        Label = cv2.imread(label_folder+labelList[ImgNumber])
    
        Label = BGR2RGB(Label)
        
        
    
        Label = np.float32(Label/255.)
        height, width = Label.shape[:2]
        
        discCenter = (DF_disc.loc[ImgNumber, 'DiskCenterRow'], DF_disc.loc[ImgNumber, 'DiskCenterCol'])
        discRadius = DF_disc.loc[ImgNumber, 'DiskRadius']
        MaskDisc = np.ones((height, width), np.uint8)
        cv2.circle(MaskDisc, center=(discCenter[1], discCenter[0]), radius= discRadius, color=0, thickness=-1)
           
       
        ArteryProb = ArteryPredAll[k][0]
        VeinProb = VeinPredAll[k][0]
        VesselProb = VesselPredAll[k][0]
        
        ArteryLabel = Label[:,:,0] > 0
        VeinLabel = Label[:,:,2] > 0
        GreenLabel = Label[:,:,1] > 0
        ArteryLabel[GreenLabel>0] = 1
        VeinLabel[GreenLabel>0] = 1
        VesselLabel = np.bitwise_or(ArteryLabel>0, VeinLabel>0)
        
        ArteryLabel = morphology.binary_dilation(ArteryLabel, morphology.disk(5))
        VeinLabel = morphology.binary_dilation(VeinLabel, morphology.disk(5))
        
        
        VesselProb[MaskDisc == 0] = 0
        VesselLabel[MaskDisc == 0] = 0
        ArteryLabel[MaskDisc == 0] = 0
        VeinLabel[MaskDisc == 0] = 0
        ArteryProb[MaskDisc == 0] = 0
        VeinProb[MaskDisc == 0] = 0
        
        #########################################################
        """Only measure the AV classificaiton metrics on the segmented vessels, while the not segmented ones are not counted"""
        
        VesselSeg = VesselProb >= 0.5
        VesselSeg= binaryPostProcessing3(VesselSeg, removeArea=100, fillArea=20)
        
        vesselPixels = np.where(VesselSeg>0)
        
        ArteryProb2 = np.zeros((height,width))
        VeinProb2 = np.zeros((height,width))
        for i in range(len(vesselPixels[0])):
            row = vesselPixels[0][i]
            col = vesselPixels[1][i]
            probA = ArteryProb[row, col]
            probV = VeinProb[row, col]
            softmaxProb = softmax(np.array([probA, probV]))
            ArteryProb2[row, col] = softmaxProb[0]
            VeinProb2[row, col] = softmaxProb[1]
        
        
        ArteryLabelImg2= ArteryLabel.copy()
        VeinLabelImg2= VeinLabel.copy()
        ArteryLabelImg2 [VesselSeg == 0] = 0
        VeinLabelImg2 [VesselSeg == 0] = 0
        ArteryVeinLabelImg = np.zeros((height, width,3), np.uint8)
        ArteryVeinLabelImg[ArteryLabelImg2>0] = (255, 0, 0)
        ArteryVeinLabelImg[VeinLabelImg2>0] = (0, 0, 255)
        ArteryVeinLabelCommon = np.bitwise_and(ArteryLabelImg2>0, VeinLabelImg2>0)
        
        
        
        ArteryPred2 = ArteryProb2 >= 0.5
        VeinPred2 = VeinProb2 >= 0.5
        
        # ArteryPred2 = ArteryProb > 0.5
        # VeinPred2 = VeinProb > 0.5
        ArteryPred2= binaryPostProcessing3(ArteryPred2, removeArea=100, fillArea=20)
        VeinPred2= binaryPostProcessing3(VeinPred2, removeArea=100, fillArea=20)
        
        
        ##################################################################################################
        ##################################################################################################
        
        """Skeleton Performance Measurement"""
        Skeleton = np.uint8(morphology.skeletonize(VesselSeg))
        # ArterySkeletonLabel = cv2.bitwise_and(ArteryLabelImg2, ArteryLabelImg2, mask=Skeleton)
        # VeinSkeletonLabel = cv2.bitwise_and(VeinLabelImg2, VeinLabelImg2, mask=Skeleton)
        
        ArterySkeletonLabel = ArteryLabelImg2.copy()
        ArterySkeletonLabel[Skeleton==0] = 0
        VeinSkeletonLabel = VeinLabelImg2.copy()
        VeinSkeletonLabel[Skeleton==0] = 0
        
        
        # ArterySkeletonPred = cv2.bitwise_and(ArteryPred2, ArteryPred2, mask=Skeleton)
        # VeinSkeletonPred = cv2.bitwise_and(VeinPred2, VeinPred2, mask=Skeleton)
        
        ArterySkeletonPred = ArteryPred2.copy()
        ArterySkeletonPred[Skeleton == 0] = 0
        VeinSkeletonPred = VeinPred2.copy()
        VeinSkeletonPred[Skeleton == 0] = 0
        
        
        
        ArteryVeinPred_sk = np.zeros((height, width, 3), np.uint8)
        skeletonPixles = np.where(Skeleton > 0)
        
        TPa_sk = 0
        TNa_sk = 0
        FPa_sk = 0
        FNa_sk = 0
        for i in range(len(skeletonPixles[0])):
            row = skeletonPixles[0][i]
            col = skeletonPixles[1][i]
            if ArterySkeletonLabel[row, col] == 1 and ArterySkeletonPred[row, col] == 1:
                TPa_sk = TPa_sk + 1
                ArteryVeinPred_sk[row, col] = (255, 0, 0)
            elif VeinSkeletonLabel[row, col] == 1 and VeinSkeletonPred[row, col] == 1:
                TNa_sk = TNa_sk + 1
                ArteryVeinPred_sk[row, col] = (0, 0, 255)
            elif ArterySkeletonLabel[row, col] == 1 and VeinSkeletonPred[row, col] == 1 \
                    and ArteryVeinLabelCommon[row, col] == 0:
                FNa_sk = FNa_sk + 1
                ArteryVeinPred_sk[row, col] = (255, 255, 0)
            elif VeinSkeletonLabel[row, col] == 1 and ArterySkeletonPred[row, col] == 1 \
                    and ArteryVeinLabelCommon[row, col] == 0:
                FPa_sk = FPa_sk + 1
                ArteryVeinPred_sk[row, col] = (0, 255, 255)
            else:
                pass
        
        if TPa_sk + FNa_sk==0:
            sensitivity_sk = 0
        else:
            sensitivity_sk = TPa_sk / (TPa_sk + FNa_sk)
        if TNa_sk + FPa_sk==0:
            specificity_sk = 0
        else:
            specificity_sk = TNa_sk / (TNa_sk + FPa_sk)
        if TPa_sk + TNa_sk + FPa_sk + FNa_sk==0:
            acc_sk = 0
        else:
            acc_sk = (TPa_sk + TNa_sk) / (TPa_sk + TNa_sk + FPa_sk + FNa_sk)
        
        senList_sk.append(sensitivity_sk)
        specList_sk.append(specificity_sk)
        accList_sk.append(acc_sk)
        print('Skeletonal Metrics', acc_sk, sensitivity_sk, specificity_sk)    
    # print('Avg Pixel-wise Performance:', np.mean(accList), np.mean(senList), np.mean(specList))
    print('Avg Skeleton Performance:', np.mean(accList_sk), np.mean(senList_sk), np.mean(specList_sk))
    
    return np.mean(accList_sk), np.mean(senList_sk), np.mean(specList_sk)
    ######################################################
    ##################################################################################################



def AVclassification_INSPIRE2(ArteryPredAll,VeinPredAll,VesselPredAll,LabelArteryAll,LabelVeinAll,LabelVesselAll,MaskAll):
    senList_sk = []
    specList_sk = []
    accList_sk = []
    for k in range(40):
        ImgNumber = k
        DF_disc = pd.read_excel(r'D:\项目\RBVS\data\INSPIRE_AVR\DiskParameters_INSPIRE.xls', sheet_name=0)
        label_folder = './data/INSPIRE_AVR/ResizedLabel_400/'
        
        
        
        labelList = os.listdir(label_folder)
        labelList = [x for x in labelList if x.__contains__('.tif')]
        labelList = natsort.natsorted(labelList)
        
        # for ImgNumber in range(len(ImgList)):
    
        
    
        Label = cv2.imread(label_folder+labelList[ImgNumber])
    
        Label = BGR2RGB(Label)
        
        
    
        Label = np.float32(Label/255.)
        height, width = Label.shape[:2]
        
        discCenter = (DF_disc.loc[ImgNumber, 'DiskCenterRow'], DF_disc.loc[ImgNumber, 'DiskCenterCol'])
        discRadius = DF_disc.loc[ImgNumber, 'DiskRadius']
        MaskDisc = np.ones((2048, 2392), np.uint8)
        cv2.circle(MaskDisc, center=(discCenter[1], discCenter[0]), radius= discRadius, color=0, thickness=-1)
    
        # downsizeRatio = 1000 / (np.maximum(height, width))
        downsizeRatio = 400. / 2392 #(np.maximum(height, width))  ##TODO: set this to 400 if use the 400 size image; 600 if using 600 size image
        MaskDisc = cv2.resize(MaskDisc, dsize=None, fx=downsizeRatio, fy=downsizeRatio)
        height, width = Label.shape[:2]
    
    
        ArteryProb = ArteryPredAll[k][0]
        VeinProb = VeinPredAll[k][0]
        VesselProb = VesselPredAll[k][0]
    
        ArteryLabel = Label[:,:,0] > 0
        VeinLabel = Label[:,:,2] > 0
        WhiteLabel = Label[:,:,1] > 0
        ArteryLabel[WhiteLabel>0] = 0
        VeinLabel[WhiteLabel>0] = 0
        VesselLabel = np.bitwise_or(ArteryLabel>0, VeinLabel>0)
        #VesselLabel = np.bitwise_or(ArteryPredAll[k][0]>0.5, VeinPredAll[k][0]>0.5)
        VesselLabel = np.bitwise_or(VesselPredAll[k][0]>0.5,VesselPredAll[k][0]>0.5)
    
        VesselProb[MaskDisc == 0] = 0
        VesselLabel[MaskDisc == 0] = 0
        ArteryLabel[MaskDisc == 0] = 0
        VeinLabel[MaskDisc == 0] = 0
        ArteryProb[MaskDisc == 0] = 0
        VeinProb[MaskDisc == 0] = 0
    
        #########################################################
        """Only measure the AV classificaiton metrics on the segmented vessels, while the not segmented ones are not counted"""
    
        """Set up the ground-truth or the segmented pixels"""
        ##TODO: use the segmented vessel
        VesselSeg = VesselProb >= 0.5
        VesselSeg= binaryPostProcessing3(VesselSeg, removeArea=30, fillArea=10)
        VesselSeg[VesselLabel == 0] = 0
    
        vesselPixels = np.where(VesselLabel > 0)  ##VesselSeg / VesselLabel
        ###################################################
    
        ArteryProb2 = np.zeros((height,width))
        VeinProb2 = np.zeros((height,width))
        for i in range(len(vesselPixels[0])):
            row = vesselPixels[0][i]
            col = vesselPixels[1][i]
            probA = ArteryProb[row, col]
            probV = VeinProb[row, col]
            softmaxProb = softmax(np.array([probA, probV]))
            ArteryProb2[row, col] = softmaxProb[0]
            VeinProb2[row, col] = softmaxProb[1]
    
    
        ArteryLabelImg2= ArteryLabel.copy()
        VeinLabelImg2= VeinLabel.copy()
        ArteryLabelImg2 [VesselSeg == 0] = 0
        VeinLabelImg2 [VesselSeg == 0] = 0
        ArteryVeinLabelImg = np.zeros((height, width,3), np.uint8)
        ArteryVeinLabelImg[ArteryLabelImg2>0] = (255, 0, 0)
        ArteryVeinLabelImg[VeinLabelImg2>0] = (0, 0, 255)
        ArteryVeinLabelCommon = np.bitwise_and(ArteryLabelImg2>0, VeinLabelImg2>0)
    
    
    
        ArteryPred2 = ArteryProb2 >= 0.5
        VeinPred2 = VeinProb2 >= 0.5
    
    
        ##################################################################################################
        ##################################################################################################
    
        """Skeleton Performance Measurement"""
        Skeleton = np.uint8(morphology.skeletonize(VesselSeg))
    
        ArterySkeletonLabel = ArteryLabelImg2.copy()
        ArterySkeletonLabel[Skeleton==0] = 0
        VeinSkeletonLabel = VeinLabelImg2.copy()
        VeinSkeletonLabel[Skeleton==0] = 0
    
    
        ArterySkeletonPred = ArteryPred2.copy()
        ArterySkeletonPred[Skeleton == 0] = 0
        VeinSkeletonPred = VeinPred2.copy()
        VeinSkeletonPred[Skeleton == 0] = 0
    
    
    
        ArteryVeinPred_sk = np.zeros((height, width, 3), np.uint8)
        skeletonPixles = np.where(Skeleton > 0)
    
        TPa_sk = 0
        TNa_sk = 0
        FPa_sk = 0
        FNa_sk = 0
        for i in range(len(skeletonPixles[0])):
            row = skeletonPixles[0][i]
            col = skeletonPixles[1][i]
            if ArterySkeletonLabel[row, col] == 1 and ArterySkeletonPred[row, col] == 1:
                TPa_sk = TPa_sk + 1
                ArteryVeinPred_sk[row, col] = (255, 0, 0)
            elif VeinSkeletonLabel[row, col] == 1 and VeinSkeletonPred[row, col] == 1:
                TNa_sk = TNa_sk + 1
                ArteryVeinPred_sk[row, col] = (0, 0, 255)
            elif ArterySkeletonLabel[row, col] == 1 and VeinSkeletonPred[row, col] == 1 \
                    and ArteryVeinLabelCommon[row, col] == 0:
                FNa_sk = FNa_sk + 1
                ArteryVeinPred_sk[row, col] = (255, 255, 0)
            elif VeinSkeletonLabel[row, col] == 1 and ArterySkeletonPred[row, col] == 1 \
                    and ArteryVeinLabelCommon[row, col] == 0:
                FPa_sk = FPa_sk + 1
                ArteryVeinPred_sk[row, col] = (0, 255, 255)
            else:
                pass
        
        if TPa_sk + FNa_sk==0:
            sensitivity_sk = 0
        else:
            sensitivity_sk = TPa_sk / (TPa_sk + FNa_sk)
        if TNa_sk + FPa_sk==0:
            specificity_sk = 0
        else:
            specificity_sk = TNa_sk / (TNa_sk + FPa_sk)
        if TPa_sk + TNa_sk + FPa_sk + FNa_sk==0:
            acc_sk = 0
        else:
            acc_sk = (TPa_sk + TNa_sk) / (TPa_sk + TNa_sk + FPa_sk + FNa_sk)
        
        senList_sk.append(sensitivity_sk)
        specList_sk.append(specificity_sk)
        accList_sk.append(acc_sk)
        print('Skeletonal Metrics', acc_sk, sensitivity_sk, specificity_sk)    
    # print('Avg Pixel-wise Performance:', np.mean(accList), np.mean(senList), np.mean(specList))
    print('Avg Skeleton Performance:', np.mean(accList_sk), np.mean(senList_sk), np.mean(specList_sk))
    
    return np.mean(accList_sk), np.mean(senList_sk), np.mean(specList_sk)
    ######################################################
    ##################################################################################################