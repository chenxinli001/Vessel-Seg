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
from models.ResUnet_small import ResUnet
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score


def modelEvalution(i,net,savePath):
    ArteryPredAll = np.zeros((20, 1, 584, 565), np.float32)
    VeinPredAll = np.zeros((20, 1, 584, 565), np.float32)
    VesselPredAll = np.zeros((20, 1, 584, 565), np.float32)
    LabelArteryAll = np.zeros((20, 1, 584, 565), np.float32)
    LabelVeinAll = np.zeros((20, 1, 584, 565), np.float32)
    LabelVesselAll = np.zeros((20, 1, 584, 565), np.float32)
    MaskAll = np.zeros((20, 1, 584, 565), np.float32)

    for k in range(20):
        ArteryPred,VeinPred,VesselPred,LabelArtery,LabelVein,LabelVessel,Mask = get_result(net,k+1)
        ArteryPredAll[k,:,:,:] = ArteryPred
        VeinPredAll[k,:,:,:] = VeinPred
        VesselPredAll[k,:,:,:] = VesselPred
        LabelArteryAll[k,:,:,:] = LabelArtery
        LabelVeinAll[k,:,:,:] = LabelVein
        LabelVesselAll[k,:,:,:] = LabelVessel
        MaskAll[k,:,:,:] = Mask
        #print (k)
        
    ArteryAUC,ArteryAcc,ArterySp,ArterySe = Evalution(ArteryPredAll,LabelArteryAll, MaskAll)
    VeinAUC,VeinAcc,VeinSp,VeinSe = Evalution(VeinPredAll,LabelVeinAll, MaskAll)
    VesselAUC,VesselAcc,VesselSp,VesselSe = Evalution(VesselPredAll,LabelVesselAll, MaskAll)
    
    
    print("===========================================================")
    print("The {} step ArteryAcc is:{}".format(i,ArteryAcc))
    print("The {} step ArterySens is:{}".format(i,ArterySe))
    print("The {} step ArterySpec is:{}".format(i,ArterySp))
    print("The {} step ArteryAUC is:{}".format(i,ArteryAUC))
    print("-----------------------------------------------------------")
    print("The {} step VeinAcc is:{}".format(i,VeinAcc))
    print("The {} step VeinSens is:{}".format(i,VesselSe))
    print("The {} step VeinSpec is:{}".format(i,VesselSp))
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
    file_w.write("===========================================================" + '\n' +
                 "The {} step ArteryAcc is:{}".format(i,ArteryAcc) + '\n' +
                 "The {} step ArterySens is:{}".format(i,ArterySe) + '\n' +
                 "The {} step ArterySpec is:{}".format(i,ArterySp) + '\n' +
                 "The {} step ArteryAUC is:{}".format(i,ArteryAUC) + '\n' +
                 "-----------------------------------------------------------" + '\n' +
                 "The {} step VeinAcc is:{}".format(i,VeinAcc) + '\n' +
                 "The {} step VeinSens is:{}".format(i,VesselSe) + '\n' +
                 "The {} step VeinSpec is:{}".format(i,VesselSp) + '\n' +
                 "The {} step VeinAUC is:{}".format(i,VeinAUC) + '\n' +
                 "-----------------------------------------------------------" + '\n' 
                 "The {} step VesselAcc is:{}".format(i,VesselAcc) + '\n'
                 "The {} step VesselSens is:{}".format(i,VesselSe) + '\n'
                 "The {} step VesselSpec is:{}".format(i,VesselSp) + '\n'
                 "The {} step VesselAUC is:{}".format(i,VesselAUC) + '\n') 
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

def get_result(net,k):
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
    
    
#    print('Old shape:', Img.shape)
#    print('Enlarged shape:', Img_enlarged.shape)
    
    
    # ############################################################
    #load model and predict etc

    # from models.ResUnet_Nov9 import ResUnet
    # from models.ResUnet_middle import ResUnet
    
    
    #model_path = model_name
    




    Net = ResUnet(resnet='resnet18',  num_classes= 3)
    Net.load_state_dict(net)
#    Net.load_state_dict(torch.load(model_path,map_location={'cuda:0':'cpu'}))
    

    
    Net.cuda()
    Net.eval()
    
    
    patch_size = 64
    batch_size = 64
    
    patches_imgs = extract_ordered_overlap(Img_enlarged, patch_height, patch_width, stride_height, stride_width)
    patches_imgs = np.transpose(patches_imgs,(0,3,1,2))
    patches_imgs = Normalize(patches_imgs)
    
    
    patchNum = patches_imgs.shape[0]
    max_iter = int(np.ceil(patchNum/float(batch_size)))
    
    pred_patches = np.zeros((patchNum, n_classes, patch_size, patch_size), np.float32)
    for i in range(max_iter):
    
    
        begin_index = i*batch_size
        end_index = (i+1)*batch_size
    
        patches_temp = patches_imgs[begin_index:end_index, :, :, :]
        #print(i, patches_temp.shape)
    
        patches_input_temp = torch.FloatTensor(patches_temp)
        patches_input_temp = autograd.Variable(patches_input_temp.cuda())
        output_temp = Net(patches_input_temp)
        pred_patches_temp = np.float32(output_temp.data.cpu().numpy())
    
        pred_patches_temp_sigmoid = sigmoid(pred_patches_temp)
    
        pred_patches[begin_index:end_index, :,:,:] = pred_patches_temp_sigmoid
    
        del patches_input_temp
        del pred_patches_temp
        del patches_temp
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

def DataSet(path): 
    
    ImgPath = path + "images/"
    LabelPath = path + "av/"
    ImgDir = os.listdir(ImgPath)
    LabelDir = os.listdir(LabelPath)
    
    Img0 = cv2.imread(ImgPath + ImgDir[0])
    Label0 = cv2.imread(LabelPath + LabelDir[0])
    
    
    patch_height = 64
    patch_width = 64
    stride_height = 10
    stride_width = 10
        
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
    
    
#    print('Old shape:', Img.shape)
#    print('Enlarged shape:', Img_enlarged.shape)
    
    
    # ############################################################
    #load model and predict etc

    # from models.ResUnet_Nov9 import ResUnet
    # from models.ResUnet_middle import ResUnet
    
    
    #model_path = model_name
        
    Net = ResUnet(resnet='resnet18',  num_classes= n_classes)
    Net.load_state_dict(net)
    
    Net.cuda()
    Net.eval()
    
    
    patch_size = 64
    batch_size = 128
    
    patches_imgs = extract_ordered_overlap(Img_enlarged, patch_height, patch_width, stride_height, stride_width)
    patches_imgs = np.transpose(patches_imgs,(0,3,1,2))
    patches_imgs = Normalize(patches_imgs)
    
    
    patchNum = patches_imgs.shape[0]
    max_iter = int(np.ceil(patchNum/float(batch_size)))
    
    pred_patches = np.zeros((patchNum, n_classes, patch_size, patch_size), np.float32)
    for i in range(max_iter):
    
    
        begin_index = i*batch_size
        end_index = (i+1)*batch_size
    
        patches_temp = patches_imgs[begin_index:end_index, :, :, :]
        #print(i, patches_temp.shape)
    
        patches_input_temp = torch.FloatTensor(patches_temp)
        patches_input_temp = autograd.Variable(patches_input_temp.cuda())
        output_temp = Net(patches_input_temp)
        pred_patches_temp = np.float32(output_temp.data.cpu().numpy())
    
        pred_patches_temp_sigmoid = sigmoid(pred_patches_temp)
    
        pred_patches[begin_index:end_index, :,:,:] = pred_patches_temp_sigmoid
    
        del patches_input_temp
        del pred_patches_temp
        del patches_temp
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

        
    
