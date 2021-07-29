#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 20:23:33 2018

@author: maw
"""
from scipy import misc
import numpy as np


def DiceCompute(SEG,GT,strctures):
     isNaN = 1
     SEG_nzz = np.sum(SEG==strctures)
     GT_nzz = np.sum(GT==strctures)
     a,b = np.where((pred_img == label_img)&(pred_img==strctures)&(label_img==strctures))
     if (GT_nzz==0):
     #if ((SEG_nzz+GT_nzz)==0):
          dr = 0
          isNaN = 0
     else:
          dr = 2.0*len(a)/(SEG_nzz+GT_nzz)
     return dr,isNaN


def DiceCompute2(SEG,GT,strctures):
     isNaN = 1
     SEG_nzz = np.sum(SEG==strctures)
     GT_nzz = np.sum(GT==strctures)
     a,b = np.where((pred_img == label_img)&(pred_img==strctures)&(label_img==strctures))
     if (GT_nzz==0):
     #if ((SEG_nzz+GT_nzz)==0):
          dr = 0
          isNaN = 0
     else:
          dr = 2.0*len(a)/(SEG_nzz+GT_nzz)
     return dr,isNaN


def dc(input1, input2,Mask,strctures):
    r"""
    Dice coefficient
    
    Computes the Dice coefficient (also known as Sorensen index) between the binary
    objects in two images.
    
    The metric is defined as
    
    .. math::
        
        DC=\frac{2|A\cap B|}{|A|+|B|}
        
    , where :math:`A` is the first and :math:`B` the second set of samples (here: binary objects).
    
    Parameters
    ----------
    input1 : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    input2 : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    
    Returns
    -------
    dc : float
        The Dice coefficient between the object(s) in ```input1``` and the
        object(s) in ```input2```. It ranges from 0 (no overlap) to 1 (perfect overlap).
        
    Notes
    -----
    This is a real metric.
    """
    #input1 = np.atleast_1d(input1.astype(np.bool))
    #input2 = np.atleast_1d(input2.astype(np.bool))
    isNaN=1
    intersection = np.count_nonzero((input1==input2)&(input1==strctures)&(input2==strctures)&(Mask==1))
    
    size_i1 = np.count_nonzero((input1==strctures)&(Mask==1))
    size_i2 = np.count_nonzero((input2==strctures)&(Mask==1))
    
    try:
        dc = 2. * intersection / float(size_i1 + size_i2)
    except ZeroDivisionError:
        dc = 0.0
    
    return dc,isNaN

#val_subject = 1
#DataPath = './save_img'
#val_dic = {'1_start':1,'1_end':48,'2_start':49,'2_end':96,'3_start':97,'3_end':144,'4_start':145,'4_end':192,'5_start':193,'5_end':240}
#val_dic = {'1_start':40,'1_end':42,'2_start':57,'2_end':93,'3_start':106,'3_end':141,'4_start':152,'4_end':187,'5_start':198,'5_end':234}
#train_data_mat = range(1,241)
#val_subject_start = val_dic[str(val_subject) + '_start']
#val_subject_end = val_dic[str(val_subject) + '_end']
#val_data_mat = range(val_subject_start,val_subject_end+1)
#sum_dr_CSF = 0
#sum_dr_GM = 0
#sum_dr_WM = 0
#CSF_num = 0
#GM_num = 0
#WM_num = 0
#
#for i in val_data_mat:
#     #pred_path = DataPath + '/2D-Unet1_normal/{}/_im_logit_test_{:03d}.png'.format(val_subject,i)
#     #pred_path = DataPath + '/{}/_im_logit_test_{:03d}.png'.format(val_subject,i)
#     #pred_path = DataPath + '/2D-Unet1_normal_add3_AUG_add(3)/{}/_im_logit_test_{:03d}.png'.format(val_subject,i)
#     #pred_path = DataPath + '/2d_Unet_32_senet/{}/_im_logit_test_{:03d}.png'.format(val_subject,i)
#     #pred_path = DataPath + '/2d_Unet_100/{}/_im_logit_test_{:03d}.png'.format(val_subject,i)
#     pred_path = DataPath + '/3D_Unet_try/{}/_im_logit_test_{:03d}.png'.format(val_subject,i)
#     #pred_path = DataPath + '/2D-Unet1_concat3_de_new_50_3conv_senet+con_r10(try)/{}/_im_logit_test_{:03d}.png'.format(val_subject,i)
#     #pred_path = DataPath + '/2D-Unet1_concat3_de_new_50_3conv/{}/_im_logit_test_{:03d}.png'.format(val_subject,i)
#     #pred_path = DataPath + '/3D-Unet1_normal/{}/_im_logit_test_{:03d}.png'.format(val_subject,i)
#     label_path = DataPath + '/label/{}/_im_label_test_{:03d}.png'.format(val_subject,i)
#     pred_img = misc.imread(pred_path)
#     label_img = misc.imread(label_path)[:,:,0]
##     dr_CSF,CSF_isNaN = dc(pred_img,label_img,255)
##     dr_WM,WM_isNaN = dc(pred_img,label_img,170)
##     dr_GM,GM_isNaN = dc(pred_img,label_img,85)
#     dr_CSF,CSF_isNaN = DiceCompute(pred_img,label_img,255)
#     dr_WM,WM_isNaN = DiceCompute(pred_img,label_img,170)
#     dr_GM,GM_isNaN = DiceCompute(pred_img,label_img,85)
#     print ('CSF_%d:%g'%(i,dr_CSF))
#     print 'WM_%d:%g'%(i,dr_WM)
#     print 'GM_%d:%g'%(i,dr_GM)
#     sum_dr_CSF = sum_dr_CSF + dr_CSF
#     sum_dr_WM = sum_dr_WM + dr_WM
#     sum_dr_GM = sum_dr_GM + dr_GM
#     CSF_num = CSF_num + CSF_isNaN
#     GM_num = GM_num + GM_isNaN
#     WM_num = WM_num + WM_isNaN
#     
#
#print 'Dice_CSF:%g,num of img:%d'%(sum_dr_CSF/CSF_num,CSF_num)
#print 'Dice_WM:%g,num of img:%d'%(sum_dr_WM/WM_num,WM_num)   
#print 'Dice_GM:%g,num of img:%d'%(sum_dr_GM/GM_num,GM_num)      



     