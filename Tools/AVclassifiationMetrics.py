
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


def AVclassifiationMetrics_new(PredAll1,PredAll2,VesselPredAll,LabelAll1,LabelAll2,LabelVesselAll,DataSet=0):
    
    senList = []
    specList = []
    accList = []
    
    for ImgNumber in range(PredAll1.shape[0]):
        ArteryLabel = LabelAll1[ImgNumber, 0, :, :]
        VeinLabel = LabelAll2[ImgNumber, 0, :, :]
        VesselLabel = LabelVesselAll[ImgNumber, 0, :, :]

    
        WhitePixels = np.bitwise_and(VesselLabel>0, np.bitwise_not(np.bitwise_or(ArteryLabel>0, VeinLabel>0)))
        GreenPixels = np.bitwise_and(ArteryLabel>0, VeinLabel>0)
    
    
        ArteryProb = PredAll1[ImgNumber, 0, :, :]
        VeinProb = PredAll2[ImgNumber, 0, :, :]
        VesselProb = VesselPredAll[ImgNumber, 0, :, :]
		
        # VesselProb = np.maximum(ArteryProb, VeinProb)
        # VesselProb = ArteryProb + VeinProb
    
        height,width =ArteryLabel.shape[:2]
    
        VesselSeg = VesselProb >= 0.5
        VesselSeg = VesselLabel
        # VesselSeg = np.bitwise_and(VesselSeg>0, np.bitwise_not(WhitePixels>0))
        # VesselSeg = np.bitwise_and(VesselSeg>0, np.bitwise_not(GreenPixels>0))
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
            
#        ArteryProb2 = ArteryProb
#        VeinProb2 = VeinProb
    
        ArteryLabelImg2= ArteryLabel.copy()
        VeinLabelImg2= VeinLabel.copy()
        ArteryLabelImg2 [VesselSeg == 0] = 0
        VeinLabelImg2 [VesselSeg == 0] = 0
        ArteryVeinLabelImg = np.zeros((height, width,3), np.uint8)
        ArteryVeinLabelImg[ArteryLabelImg2>0] = (255, 0, 0)
        ArteryVeinLabelImg[VeinLabelImg2>0] = (0, 0, 255)
        ArteryVeinLabelCommon = np.bitwise_and(ArteryLabelImg2>0, VeinLabelImg2>0)
    
    
    
        ArteryPred2 = ArteryProb2 > 0.5
        VeinPred2 = VeinProb2 > 0.5
    
        # ArteryPred2 = np.bitwise_and(ArteryPred2>0, np.bitwise_not(WhitePixels))
        # ArteryPred2 = np.bitwise_and(ArteryPred2>0, np.bitwise_not(GreenPixels))
        # VeinPred2 = np.bitwise_and(VeinPred2 > 0, np.bitwise_not(WhitePixels))
        # VeinPred2 = np.bitwise_and(VeinPred2 > 0, np.bitwise_not(GreenPixels))
        #
        # ArteryLabelImg2 = np.bitwise_and(ArteryLabelImg2 > 0, np.bitwise_not(WhitePixels))
        # ArteryLabelImg2 = np.bitwise_and(ArteryLabelImg2 > 0, np.bitwise_not(GreenPixels))
        # VeinLabelImg2 = np.bitwise_and(VeinLabelImg2 > 0, np.bitwise_not(WhitePixels))
        # VeinLabelImg2 = np.bitwise_and(VeinLabelImg2 > 0, np.bitwise_not(GreenPixels))
    
        ##################################################################################################
        """Get the ArteryVeinPredImg with Wrong Pixels Marked on the image"""
        ArteryVeinPredImg = np.zeros((height, width, 3), np.uint8)
        TPimg =  np.bitwise_and(ArteryPred2>0, ArteryLabelImg2>0)
        TNimg =  np.bitwise_and(VeinPred2>0, VeinLabelImg2>0)
        FPimg = np.bitwise_and(ArteryPred2>0, VeinLabelImg2>0)
        FPimg = np.bitwise_and(FPimg, np.bitwise_not(ArteryVeinLabelCommon))
        FNimg = np.bitwise_and(VeinPred2>0, ArteryLabelImg2>0)
        FNimg = np.bitwise_and(FNimg, np.bitwise_not(ArteryVeinLabelCommon))
        ArteryVeinPredImg[TPimg>0, :] = (255, 0, 0)
        ArteryVeinPredImg[TNimg>0, :] = (0, 0, 255)
        ArteryVeinPredImg[FPimg>0, :] = (0, 255, 255)
        ArteryVeinPredImg[FNimg>0, :] = (255, 255, 0)
    
    
        TPa = np.count_nonzero(TPimg)
        TNa = np.count_nonzero(TNimg)
        FPa = np.count_nonzero(FPimg)
        FNa = np.count_nonzero(FNimg)
        sensitivity = TPa/(TPa+FNa)
        specificity = TNa/(TNa + FPa)
        acc = (TPa + TNa) /(TPa + TNa + FPa + FNa)
        #print('Pixel-wise Metrics', ImgNumber, acc, sensitivity, specificity)
    
        senList.append(sensitivity)
        specList.append(specificity)
        accList.append(acc)
        
    return np.mean(accList), np.mean(specList),np.mean(senList)

def AVclassifiationMetrics_new2(PredAll1,PredAll2,VesselPredAll,LabelAll1,LabelAll2,LabelVesselAll,DataSet=0):
    
    senList = []
    specList = []
    accList = []
    
    for ImgNumber in range(PredAll1.shape[0]):
        ArteryLabel = LabelAll1[ImgNumber, 0, :, :]
        VeinLabel = LabelAll2[ImgNumber, 0, :, :]
        VesselLabel = LabelVesselAll[ImgNumber, 0, :, :]

    
        WhitePixels = np.bitwise_and(VesselLabel>0, np.bitwise_not(np.bitwise_or(ArteryLabel>0, VeinLabel>0)))
        GreenPixels = np.bitwise_and(ArteryLabel>0, VeinLabel>0)
    
    
        ArteryProb = PredAll1[ImgNumber, 0, :, :]
        VeinProb = PredAll2[ImgNumber, 0, :, :]
        VesselProb = VesselPredAll[ImgNumber, 0, :, :]
		
        # VesselProb = np.maximum(ArteryProb, VeinProb)
        # VesselProb = ArteryProb + VeinProb
    
        height,width =ArteryLabel.shape[:2]
    
        VesselSeg = VesselProb >= 0.5
        #VesselSeg = VesselLabel
        # VesselSeg = np.bitwise_and(VesselSeg>0, np.bitwise_not(WhitePixels>0))
        # VesselSeg = np.bitwise_and(VesselSeg>0, np.bitwise_not(GreenPixels>0))
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
            
#        ArteryProb2 = ArteryProb
#        VeinProb2 = VeinProb
    
        ArteryLabelImg2= ArteryLabel.copy()
        VeinLabelImg2= VeinLabel.copy()
        ArteryLabelImg2 [VesselSeg == 0] = 0
        VeinLabelImg2 [VesselSeg == 0] = 0
        ArteryVeinLabelImg = np.zeros((height, width,3), np.uint8)
        ArteryVeinLabelImg[ArteryLabelImg2>0] = (255, 0, 0)
        ArteryVeinLabelImg[VeinLabelImg2>0] = (0, 0, 255)
        ArteryVeinLabelCommon = np.bitwise_and(ArteryLabelImg2>0, VeinLabelImg2>0)
    
    
    
        ArteryPred2 = ArteryProb2 > 0.5
        VeinPred2 = VeinProb2 > 0.5
    
        # ArteryPred2 = np.bitwise_and(ArteryPred2>0, np.bitwise_not(WhitePixels))
        # ArteryPred2 = np.bitwise_and(ArteryPred2>0, np.bitwise_not(GreenPixels))
        # VeinPred2 = np.bitwise_and(VeinPred2 > 0, np.bitwise_not(WhitePixels))
        # VeinPred2 = np.bitwise_and(VeinPred2 > 0, np.bitwise_not(GreenPixels))
        #
        # ArteryLabelImg2 = np.bitwise_and(ArteryLabelImg2 > 0, np.bitwise_not(WhitePixels))
        # ArteryLabelImg2 = np.bitwise_and(ArteryLabelImg2 > 0, np.bitwise_not(GreenPixels))
        # VeinLabelImg2 = np.bitwise_and(VeinLabelImg2 > 0, np.bitwise_not(WhitePixels))
        # VeinLabelImg2 = np.bitwise_and(VeinLabelImg2 > 0, np.bitwise_not(GreenPixels))
    
        ##################################################################################################
        """Get the ArteryVeinPredImg with Wrong Pixels Marked on the image"""
        ArteryVeinPredImg = np.zeros((height, width, 3), np.uint8)
        TPimg =  np.bitwise_and(ArteryPred2>0, ArteryLabelImg2>0)
        TNimg =  np.bitwise_and(VeinPred2>0, VeinLabelImg2>0)
        FPimg = np.bitwise_and(ArteryPred2>0, VeinLabelImg2>0)
        FPimg = np.bitwise_and(FPimg, np.bitwise_not(ArteryVeinLabelCommon))
        FNimg = np.bitwise_and(VeinPred2>0, ArteryLabelImg2>0)
        FNimg = np.bitwise_and(FNimg, np.bitwise_not(ArteryVeinLabelCommon))
        ArteryVeinPredImg[TPimg>0, :] = (255, 0, 0)
        ArteryVeinPredImg[TNimg>0, :] = (0, 0, 255)
        ArteryVeinPredImg[FPimg>0, :] = (0, 255, 255)
        ArteryVeinPredImg[FNimg>0, :] = (255, 255, 0)
    
    
        TPa = np.count_nonzero(TPimg)
        TNa = np.count_nonzero(TNimg)
        FPa = np.count_nonzero(FPimg)
        FNa = np.count_nonzero(FNimg)
        sensitivity = TPa/(TPa+FNa)
        specificity = TNa/(TNa + FPa)
        acc = (TPa + TNa) /(TPa + TNa + FPa + FNa)
        #print('Pixel-wise Metrics', ImgNumber, acc, sensitivity, specificity)
    
        senList.append(sensitivity)
        specList.append(specificity)
        accList.append(acc)
        
    return np.mean(accList), np.mean(specList),np.mean(senList)

def AVclassifiationMetrics(PredAll1,PredAll2,VesselPredAll,LabelAll1,LabelAll2,LabelVesselAll,DataSet=0): 
    senList = []
    specList = []
    accList = []
    
    for k in range(PredAll1.shape[0]): 
    
        ImgNumber = k
        #folder = r'.\data\AV_DRIVE\test\images\\'
        if DataSet == 0:
            DF_disc = pd.read_excel(r'./Tools/DiskParameters_DRIVE_Test.xls', sheet_name=0)
        elif DataSet == 1:
            DF_disc = pd.read_excel(r'./Tools/HRF_DiscParameter.xls', sheet_name=0)
        VesselProb = VesselPredAll[k,0,:,:]
        VesselLabel = LabelVesselAll[k,0,:,:]
            
        ArteryLabel = LabelAll1[k, 0, :, :]
        VeinLabel = LabelAll2[k, 0, :, :]
        
        ArteryProb = PredAll1[k, 0,:,:]
        VeinProb = PredAll2[k, 0,:,:]
        
#        ImgList = os.listdir(folder)
#        ImgList = [x for x in ImgList if x.__contains__('.tif')]
#        ImgList = natsort.natsorted(ImgList)
        
        
        ###############################################
        
        
        height, width = VesselProb.shape[:2]
        
        discCenter = (DF_disc.loc[ImgNumber, 'DiskCenterRow'], DF_disc.loc[ImgNumber, 'DiskCenterCol'])
        discRadius = DF_disc.loc[ImgNumber, 'DiskRadius']
        MaskDisc = np.ones((height, width), np.uint8)
        cv2.circle(MaskDisc, center=(discCenter[1], discCenter[0]), radius= discRadius, color=0, thickness=-1)
        
        
        
        
        VesselProb = cv2.bitwise_and(VesselProb, VesselProb, mask=MaskDisc)
        VesselLabel = cv2.bitwise_and(VesselLabel, VesselLabel, mask=MaskDisc)
        ArteryLabel = cv2.bitwise_and(ArteryLabel, ArteryLabel, mask=MaskDisc)
        VeinLabel = cv2.bitwise_and(VeinLabel, VeinLabel, mask=MaskDisc)
        ArteryProb = cv2.bitwise_and(ArteryProb, ArteryProb, mask=MaskDisc)
        VeinProb = cv2.bitwise_and(VeinProb, VeinProb, mask=MaskDisc)
        
        #########################################################
        """Only measure the AV classificaiton metrics on the segmented vessels, while the not segmented ones are not counted"""
        
        Artery = ArteryProb>=0.5
        Vein = VeinProb>=0.5
        VesselSeg = Artery + Vein
        
        #VesselSeg = VesselProb >= 0.5
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
        """Get the ArteryVeinPredImg with Wrong Pixels Marked on the image"""
        ArteryVeinPredImg = np.zeros((height, width, 3), np.uint8)
        TPimg =  np.bitwise_and(ArteryPred2>0, ArteryLabelImg2>0)
        TNimg =  np.bitwise_and(VeinPred2>0, VeinLabelImg2>0)
        FPimg = np.bitwise_and(ArteryPred2>0, VeinLabelImg2>0)
        FPimg = np.bitwise_and(FPimg, np.bitwise_not(ArteryVeinLabelCommon))
        
        FNimg = np.bitwise_and(VeinPred2>0, ArteryLabelImg2>0)
        FNimg = np.bitwise_and(FNimg, np.bitwise_not(ArteryVeinLabelCommon))
        
        
        ArteryVeinPredImg[TPimg>0, :] = (255, 0, 0)
        ArteryVeinPredImg[TNimg>0, :] = (0, 0, 255)
        ArteryVeinPredImg[FPimg>0, :] = (0, 255, 255)
        ArteryVeinPredImg[FNimg>0, :] = (255, 255, 0)
        
        
        ##################################################################################################
        """Calculate sensitivity, specificity and accuracy"""
        
        
        # TPa = 0
        # TNa = 0
        # FPa = 0
        # FNa = 0
        # ArteryPixels = np.where(ArteryPred2 > 0)
        # VeinPixels = np.where(VeinPred2 > 0)
        # for i in range(len(ArteryPixels[0])):
        #     row = ArteryPixels[0][i]
        #     col = ArteryPixels[1][i]
        #     if ArteryLabelImg2[row, col] == 1: #or VeinLabelImg2[row, col] == 0: #if this pixel doesn't appear on the vein, count it as TP
        #         TPa = TPa +1
        #     elif VeinLabelImg2[row, col] == 1:
        #         FNa = FNa + 1
        #
        # for i in range(len(VeinPixels[0])):
        #     row = VeinPixels[0][i]
        #     col = VeinPixels[1][i]
        #     if VeinLabelImg2[row, col] == 1: #or ArteryLabelImg2[row, col] == 0: #if this pixel doesn't appear on the vein, count it as TP
        #         TNa = TNa +1
        #     elif ArteryLabelImg2[row, col] == 1:
        #         FPa = FPa + 1
        
        
        TPa = np.count_nonzero(TPimg)
        TNa = np.count_nonzero(TNimg)
        FPa = np.count_nonzero(FPimg)
        FNa = np.count_nonzero(FNimg)
        
        sensitivity = TPa/(TPa+FNa)
        specificity = TNa/(TNa + FPa)
        acc = (TPa + TNa) /(TPa + TNa + FPa + FNa)
        #print('Metrics', acc, sensitivity, specificity)
        
        senList.append(sensitivity)
        specList.append(specificity)
        accList.append(acc)
        # print('Avg Per:', np.mean(accList), np.mean(senList), np.mean(specList))
        
        
        
        
        ##################################################################################################
        #print("End of Image Processing >>>")
    #plt.figure()
    #Images = [BGR2RGB(Img), ArteryVeinLabelImg, ArteryVeinPredImg, ArteryProb2,
    #           ArteryPred2,  VeinPred2, ArteryVeinLabelCommon,MaskDisc]
    #Titles = [ 'ImgShow', 'ArteryVeinLabelImg',  'ArteryVeinPredImg',  'ArteryProb2',
    #            'ArteryPred2',  'VeinPred2', 'ArteryVeinLabelCommon','Mask']
    #
    #
    #for i in range(0, len(Images)):
    #    plt.subplot(2, 4, i + 1), plt.imshow(Images[i], 'gray'), plt.title(Titles[i])
    #    
    #plt.show()
    
    return sum(accList)/PredAll1.shape[0],sum(specList)/PredAll1.shape[0],sum(senList)/PredAll1.shape[0]

def AVclassifiationMetrics_skeletonPixles(PredAll1,PredAll2,VesselPredAll,LabelAll1,LabelAll2,LabelVesselAll,DataSet=0):
    
    if DataSet==0:
        ImgN = 20
        DF_disc = pd.read_excel('./Tools/DiskParameters_DRIVE_Test.xls', sheet_name=0)   
    else:
        ImgN = 15
        DF_disc = pd.read_excel('./Tools/HRF_DiscParameter.xls', sheet_name=0)
        
       
        
    senList = []
    specList = []
    accList = []
    
    senList_sk = []
    specList_sk = []
    accList_sk = []
    
    for ImgNumber in range(ImgN):
        
        height, width = PredAll1.shape[2:4]
    
        discCenter = (DF_disc.loc[ImgNumber, 'DiskCenterRow'], DF_disc.loc[ImgNumber, 'DiskCenterCol'])
        discRadius = DF_disc.loc[ImgNumber, 'DiskRadius']
        MaskDisc = np.ones((height, width), np.uint8)
        cv2.circle(MaskDisc, center=(discCenter[1], discCenter[0]), radius= discRadius, color=0, thickness=-1)
    
    
        VesselProb = VesselPredAll[ImgNumber, 0,:,:]
        VesselLabel = LabelVesselAll[ImgNumber, 0, :, :]
    
    
        ArteryLabel = LabelAll1[ImgNumber, 0, :, :]
        VeinLabel = LabelAll2[ImgNumber, 0, :, :]
    
        ArteryProb = PredAll1[ImgNumber, 0,:,:]
        VeinProb = PredAll2[ImgNumber, 0,:,:]
    
        VesselProb = cv2.bitwise_and(VesselProb, VesselProb, mask=MaskDisc)
        VesselLabel = cv2.bitwise_and(VesselLabel, VesselLabel, mask=MaskDisc)
        ArteryLabel = cv2.bitwise_and(ArteryLabel, ArteryLabel, mask=MaskDisc)
        VeinLabel = cv2.bitwise_and(VeinLabel, VeinLabel, mask=MaskDisc)
        ArteryProb = cv2.bitwise_and(ArteryProb, ArteryProb, mask=MaskDisc)
        VeinProb = cv2.bitwise_and(VeinProb, VeinProb, mask=MaskDisc)
    
        #########################################################
        """Only measure the AV classificaiton metrics on the segmented vessels, while the not segmented ones are not counted"""
    
#        Artery = ArteryProb>=0.5
#        Vein = VeinProb>=0.5
#        VesselSeg = Artery + Vein
        
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

        ArteryPred2= binaryPostProcessing3(ArteryPred2, removeArea=100, fillArea=20)
        VeinPred2= binaryPostProcessing3(VeinPred2, removeArea=100, fillArea=20)
    
    
        ##################################################################################################
        """Get the ArteryVeinPredImg with Wrong Pixels Marked on the image"""
        ArteryVeinPredImg = np.zeros((height, width, 3), np.uint8)
        TPimg =  np.bitwise_and(ArteryPred2>0, ArteryLabelImg2>0)
        TNimg =  np.bitwise_and(VeinPred2>0, VeinLabelImg2>0)
        FPimg = np.bitwise_and(ArteryPred2>0, VeinLabelImg2>0)
        FPimg = np.bitwise_and(FPimg, np.bitwise_not(ArteryVeinLabelCommon))
    
        FNimg = np.bitwise_and(VeinPred2>0, ArteryLabelImg2>0)
        FNimg = np.bitwise_and(FNimg, np.bitwise_not(ArteryVeinLabelCommon))
    
    
        ArteryVeinPredImg[TPimg>0, :] = (255, 0, 0)
        ArteryVeinPredImg[TNimg>0, :] = (0, 0, 255)
        ArteryVeinPredImg[FPimg>0, :] = (0, 255, 255)
        ArteryVeinPredImg[FNimg>0, :] = (255, 255, 0)
    
    
        ##################################################################################################
        """Calculate pixel-wise sensitivity, specificity and accuracy"""
    
    
    
        TPa = np.count_nonzero(TPimg)
        TNa = np.count_nonzero(TNimg)
        FPa = np.count_nonzero(FPimg)
        FNa = np.count_nonzero(FNimg)
    
        sensitivity = TPa/(TPa+FNa)
        specificity = TNa/(TNa + FPa)
        acc = (TPa + TNa) /(TPa + TNa + FPa + FNa)
        #print('Pixel-wise Metrics', acc, sensitivity, specificity)
    
        senList.append(sensitivity)
        specList.append(specificity)
        accList.append(acc)
        # print('Avg Per:', np.mean(accList), np.mean(senList), np.mean(specList))
    
        ##################################################################################################
        """Skeleton Performance Measurement"""
        Skeleton = np.uint8(morphology.skeletonize(VesselSeg))
        ArterySkeletonLabel = cv2.bitwise_and(ArteryLabelImg2, ArteryLabelImg2, mask=Skeleton)
        VeinSkeletonLabel = cv2.bitwise_and(VeinLabelImg2, VeinLabelImg2, mask=Skeleton)
    
        ArterySkeletonPred = cv2.bitwise_and(ArteryPred2, ArteryPred2, mask=Skeleton)
        VeinSkeletonPred = cv2.bitwise_and(VeinPred2, VeinPred2, mask=Skeleton)
    
        ArteryVeinPred_sk = np.zeros((height, width,3), np.uint8)
        skeletonPixles = np.where(Skeleton >0)
    
        TPa_sk = 0
        TNa_sk = 0
        FPa_sk = 0
        FNa_sk = 0
        for i in range(len(skeletonPixles[0])):
            row = skeletonPixles[0][i]
            col = skeletonPixles[1][i]
            if ArterySkeletonLabel[row, col] == 1 and ArterySkeletonPred[row, col] == 1:
                TPa_sk = TPa_sk +1
                ArteryVeinPred_sk[row, col] = (255, 0, 0)
            elif VeinSkeletonLabel[row, col] == 1 and VeinSkeletonPred[row, col] == 1:
                TNa_sk = TNa_sk + 1
                ArteryVeinPred_sk[row, col] = (0, 0, 255)
            elif ArterySkeletonLabel[row, col] == 1 and VeinSkeletonPred[row, col] == 1\
                    and ArteryVeinLabelCommon[row, col] == 0:
                FNa_sk = FNa_sk + 1
                ArteryVeinPred_sk[row, col] = (255, 255, 0)
            elif VeinSkeletonLabel[row, col] == 1 and ArterySkeletonPred[row, col] == 1\
                    and ArteryVeinLabelCommon[row, col] == 0:
                FPa_sk = FPa_sk + 1
                ArteryVeinPred_sk[row, col] = (0, 255, 255)
            else:
                pass
    
        sensitivity_sk = TPa_sk/(TPa_sk+FNa_sk)
        specificity_sk = TNa_sk/(TNa_sk + FPa_sk)
        acc_sk = (TPa_sk + TNa_sk) /(TPa_sk + TNa_sk + FPa_sk + FNa_sk)
    
        senList_sk.append(sensitivity_sk)
        specList_sk.append(specificity_sk)
        accList_sk.append(acc_sk)
        #print('Skeletonal Metrics', acc_sk, sensitivity_sk, specificity_sk)
    
    
    print('Avg Pixel-wise Performance:', np.mean(accList), np.mean(senList), np.mean(specList))
#    print('Avg Skeleton Performance:', np.mean(accList_sk), np.mean(senList_sk), np.mean(specList_sk))
    
    #return np.mean(accList_sk), np.mean(specList_sk),np.mean(senList_sk)
    return np.mean(accList), np.mean(specList),np.mean(senList)