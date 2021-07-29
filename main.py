# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 20:56:53 2019

@author: wenaoma
"""

import os
import numpy as np
import cv2
from skimage import morphology
import time
import natsort
import torch
import torch.autograd as autograd
from torch.autograd import Variable
from models.ResUnet_small import ResUnet,ResUnet_illum,ResUnet_illum_tran,ResUnet_illum_tran_trad,ResUnet_illum_tran_trad_conv,ResUnet_illum_tran_trad_conv2,ResUnet_illum_tran_trad_conv3,ResUnet_illum_tran_trad_conv_ds,ResUnet_illum_tran_trad_conv_ds2,ResUnet_illum_tran_trad_conv3_ds,ResUnet_illum_tran_ds_add_conv,ResUnet_illum_tran_ds3_add_conv,ResUnet_illum_tran_ds3_add_conv3
#from models.network import ResUnet_illum_tran_ds3_add_conv4_sSE_cSE_up,ResUnet_illum_tran_ds3_add_conv5_sSE_cSE_up,ResUnet_illum_tran_ds3_add_conv5_sSE_cSE_up_notra,ResUnet_illum_tran_ds3_add_conv5_sSE_cSE_up_nopre,ResUnet_illum_tran_ds3_add_conv5_sSE_cSE_up_nono,ResUnet_illum_tran_ds3_add_conv5_sSE_cSE_up_concat,ResUnet_illum_tran_ds3_add_conv5_sSE_cSE_up_nono2,ResUnet_illum_tran_ds3_add_conv5_sSE_cSE_up2
from models.network import *
from models.network2 import *
from models.network3 import *
from models.network4 import *
from torch.utils.data import DataLoader
from loss2 import CrossEntropyLoss2d,multiclassLoss,multiclassLoss_ds,multiclassLoss_ds3,multiclassLoss_ds4,multiclassLoss_ds5,FocalLoss2d,FocalLoss

from config import DefaultConfig
from opt_trad_ds3_ev import get_patch_trad_da,DataSet_random_illum_da, modelEvalution_rebuttal,DataSet,TestAccuracy,DataSet_random,ComputePara,DataSet_random_illum,modelEvalution,get_patch,get_patch_trad,get_patch_trad2,modelEvalution2,modelEvalution3
import matplotlib.pyplot as plt
import torch.nn.functional as F
from VesselSegProbMap.VesselSegmentation_ProbMap import VesselProMap,VesselProMap2

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def train():
    i=0
    n_classes = 3
    opt = DefaultConfig()
    
    net = ResUnet_illum_tran_ds3_add_conv5_sSE_cSE_up49(resnet='resnet18',  num_classes= n_classes)
    
    model_step = 55000
    modelName ='trad_conv1_0.33ds3_add5_bs16_up_scSE49_weight2_128_da_MIs'
    resultSavePath = './result_save2/' + modelName + '.txt'
    _dir ='./models_save2/'  + modelName + '/' 
    folder = os.path.exists(_dir)
    if not folder:
        os.mkdir(_dir)
    ComputePara(net,resultSavePath)
    if model_step != 0:
        model_path = _dir  + str(model_step) + '.pkl'
        net.load_state_dict(torch.load(model_path))
    net = net.cuda()
    net.train()
    max_step = 60000
    patch_size = 128
    batch_size = 16
    
    #load data
    train_data1,train_data2,label_data1 = DataSet_random_illum_da('./data/AV_DRIVE/training/images/','./data/AV_DRIVE/training/av/')
    train_data3 = VesselProMap('./data/AV_DRIVE/training/images')
    
    train_data4,train_data5,label_data2 = DataSet_random_illum_da('./data/HRF_RevisedAVLabel/images/train/','./data/HRF_RevisedAVLabel/ArteryVein/train/')
    train_data6 = VesselProMap2('./data/HRF_RevisedAVLabel/images/train')
    #epoch_step = train_data1.shape[0]*train_data1.shape[2]//batch_size

    #print("epoch_step:{}".format(epoch_step))
    #train_dataloader = DataLoader(train_data, opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    

#    for i in range(0, 3):
#        cv2.imwrite("./data/save_image/"+str(i)+".jpg",label_data[0,i,:,:]*255)

    
    #set loss function
    criterion = multiclassLoss_ds3()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.05,momentum = 0.9,weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size = 7500,gamma=0.5)   

    loss_all = 0

    for i in range(model_step,max_step+1):
        
        #data1, data2, label = get_patch(batch_size, patch_size, train_data1, train_data2, label_data)  
        data1, data2,data3,data4,data5, label = get_patch_trad_da(batch_size, patch_size, train_data1, train_data2,train_data3,label_data1,train_data4, train_data5,train_data6,label_data2)
        
        data_input1 = torch.FloatTensor(data1)
        data_input1 = autograd.Variable(data_input1.cuda())    
        data_input2 = torch.FloatTensor(data2)
        data_input2 = autograd.Variable(data_input2.cuda()) 
        data_input3 = torch.FloatTensor(data3)
        data_input3 = autograd.Variable(data_input3.cuda()) 
        data_input4 = torch.FloatTensor(data4)
        data_input4 = autograd.Variable(data_input4.cuda()) 
        data_input5 = torch.FloatTensor(data5)
        data_input5 = autograd.Variable(data_input5.cuda()) 
        label_input = torch.FloatTensor(label)
        label_input = autograd.Variable(label_input.cuda())
      
        optimizer.zero_grad()
        pre_target,ds1,ds2,ds3 = net(data_input1,data_input2,data_input3,data_input4,data_input5)

        loss = criterion(pre_target,ds1,ds2,ds3,label_input)
        loss.backward()
        scheduler.step(i)
        optimizer.step()
        loss_all += loss.item()/2.0

        if i%5000==0:
            torch.save(net.state_dict(),_dir+str(i)+'.pkl')
            torch.save(net,_dir+str(i)+'.pth')
            print("save model to {}".format(_dir))
        if i%200==0:
            if i>=55000:
                modelEvalution_rebuttal(i,net.state_dict(),resultSavePath,loss_all,_dir)
            print("-----------------------------------------------------------")
            print("The {} step loss is :{}".format(i,loss_all/200))
            loss_all = 0
            print("Step:{}".format(i))
            print(_dir)            

if __name__ == '__main__':
    train();  
