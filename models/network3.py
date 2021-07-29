# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 11:37:28 2019

@author: wenaoma
"""

from torch import nn
import torch
from models.resnet import resnet34, resnet18, resnet50,resnet18_2,resnet18_3,resnet18_4,resnet18_5
import torch.nn.functional as F
from models.unet_parts import *
from models.layers import *

class ResUnet_illum_tran_ds3_add_conv5_sSE_cSE_up37(nn.Module):

    def __init__(self, resnet='resnet34', num_classes=2):
        super().__init__()

        cut = 7

        if resnet == 'resnet34':
            base_model = resnet34
        elif resnet == 'resnet18':
            base_model = resnet18
        elif resnet == 'resnet50':
            base_model = resnet50
        else:
            raise Exception('The Resnet Model only accept resnet18, resnet34 and resnet50')

        layers = list(base_model(pretrained=True).children())[:cut]
        base_layers = nn.Sequential(*layers)
        self.rn = base_layers

        self.conv1x1_5 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[2])]
        self.sfs.append(SaveFeatures(base_layers[4][1]))
        self.sfs.append(SaveFeatures(base_layers[5][1]))

        self.up2 = UnetBlock_tran8_scSE_add(256, 128,256)
        self.up3 = UnetBlock_tran8_scSE_add(256, 64,256)
        self.up4 = UnetBlock_tran8_scSE_add(256, 64,128)
        self.ec_block = expand_compress_block3(8)

        self.up5_2 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)

        self.ds_conv_1 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.ds_conv_2 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.ds_conv_3 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        
        self.ds_conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.ds_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.ds_conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.ds_conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.ds_conv5 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.ds_conv6 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(self.num_classes)
        self.bn2 = nn.BatchNorm2d(self.num_classes)
        self.bn3 = nn.BatchNorm2d(self.num_classes)
        self.bn4 = nn.BatchNorm2d(self.num_classes)

    def forward(self, x,y,pro1,pro2,pro3):

        x = torch.cat([x, y,pro1], dim=1)
        x = self.ec_block(x)
       
        
        x = F.relu(self.rn(x))

        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)

        x_out = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x_out = self.conv1x1_5(x_out)
        x_out = self.up5_2(x_out)


        if self.num_classes==1:
            output = x_out[:, 0]
        else:
            output = x_out[:, :self.num_classes]

        ds1 = F.upsample(self.sfs[2].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = self.ds_conv1(ds1)
        ds1 = F.upsample(ds1, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = self.ds_conv2(ds1)
        ds1 = F.upsample(ds1, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = self.ds_conv3(ds1)
        ds1 = self.ds_conv_1(ds1)
        ds1 = self.bn2(ds1)

        ds2 = F.upsample(self.sfs[1].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds2 = self.ds_conv4(ds2)
        ds2 = F.upsample(ds2, scale_factor=2, mode='bilinear', align_corners=True)
        ds2 = self.ds_conv5(ds2)
        ds2 = self.ds_conv_2(ds2)
        ds2 = self.bn3(ds2)
        
        ds3 = F.upsample(self.sfs[0].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds3 = self.ds_conv6(ds3)
        ds3 = self.ds_conv_3(ds3)
        ds3 = self.bn4(ds3)

        output = self.bn1(output)
        return output,ds1,ds2,ds3

    def close(self):
        for sf in self.sfs: sf.remove()   
        
        
class ResUnet_illum_tran_ds3_add_conv5_sSE_cSE_up38(nn.Module):

    def __init__(self, resnet='resnet34', num_classes=2):
        super().__init__()
        # super(ResUnet, self).__init__()
        # cut, lr_cut = [8, 6]
        cut, lr_cut = [7, 5]

        if resnet == 'resnet34':
            base_model = resnet34
        elif resnet == 'resnet18':
            base_model = resnet18
        elif resnet == 'resnet50':
            base_model = resnet50
        else:
            raise Exception('The Resnet Model only accept resnet18, resnet34 and resnet50')

        layers = list(base_model(pretrained=True).children())[:cut]
        base_layers = nn.Sequential(*layers)
        self.rn = base_layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        #self.conv2 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3,
        #                       bias=False)

        self.conv1x1 = nn.Conv2d(128, 64, kernel_size=1, padding=0)
        self.conv1x1_2 = nn.Conv2d(8, 3, kernel_size=1, padding=0)
        self.conv1x1_3 = nn.Conv2d(2, 32, kernel_size=1, padding=0)
        self.conv1x1_4 = nn.Conv2d(8, 8, kernel_size=1, padding=0)
        self.conv1x1_5 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[i]) for i in [2, 4]]
        self.sfs.append(SaveFeatures(base_layers[5][0]))
        #self.up1 = UnetBlock0(512, 256, 256)
        #self.up1 = UnetBlock(256, 128, 256)
        self.up2 = UnetBlock_tran8_3x3(256, 128,256)
        self.up3 = UnetBlock_tran8_3x3(128, 64,256)
        self.up4 = UnetBlock_tran8_3x3(64, 64,128)
        self.ec_block = expand_compress_block3(8)
        # self.up5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)
        # self.up5_1 = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.up5_2 = nn.Conv2d(256, self.num_classes, 1)

        #self.up5_1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up5_1 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.up5_2 = nn.Conv2d(32, self.num_classes, kernel_size=1, padding=0)
        
        self.ds_deconv_1 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv_1_ = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv2_1 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.ds_conv_1 = nn.Conv2d(32, self.num_classes, kernel_size=1, padding=0)
        self.bn_1 = nn.BatchNorm2d(self.num_classes)

        self.ds_deconv_2 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv_2_ = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_conv_2 = nn.Conv2d(32, self.num_classes, kernel_size=1, padding=0)
        self.bn_2 = nn.BatchNorm2d(self.num_classes)
        
        self.ds_deconv_3 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_conv_3 = nn.Conv2d(32, self.num_classes, kernel_size=1, padding=0)
        self.bn_3 = nn.BatchNorm2d(self.num_classes)

        self.ds_conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.ds_conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.ds_conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.ds_conv4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.ds_conv5 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.ds_conv6 = nn.Conv2d(64, 32, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(self.num_classes)

    def forward(self, x,y,pro1,pro2,pro3):
        #x = self.conv1(x)
        #y = self.conv1(y)
        x = torch.cat([x, y,pro1], dim=1)
        x = self.ec_block(x)
       
        
        x = F.relu(self.rn(x))
        # x = self.up1(x, self.sfs[2].features)
        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)
        # x = self.up4(x, self.sfs[0].features)
        # x_out = self.up5(x)
        x_out = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x_out = self.conv1x1_5(x_out)
        x_out = self.up5_2(x_out)


        if self.num_classes==1:
            output = x_out[:, 0]
        else:
            output = x_out[:, :self.num_classes]

        ds1 = F.upsample(self.sfs[2].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = self.ds_conv1(ds1)
        ds1 = F.upsample(ds1, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = self.ds_conv2(ds1)
        ds1 = F.upsample(ds1, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = self.ds_conv3(ds1)
        ds1 = self.ds_conv_1(ds1)
        ds1 = self.bn_1(ds1)

        ds2 = F.upsample(self.sfs[1].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds2 = self.ds_conv4(ds2)
        ds2 = F.upsample(ds2, scale_factor=2, mode='bilinear', align_corners=True)
        ds2 = self.ds_conv5(ds2)
        ds2 = self.ds_conv_2(ds2)
        ds2 = self.bn_2(ds2)
        
        ds3 = F.upsample(self.sfs[0].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds3 = self.ds_conv6(ds3)
        ds3 = self.ds_conv_3(ds3)
        ds3 = self.bn_3(ds3)

        output = self.bn1(output)
        return output,ds1,ds2,ds3

    def close(self):
        for sf in self.sfs: sf.remove()      
        
        
        
        
class ResUnet_illum_tran_ds3_add_conv5_sSE_cSE_up39(nn.Module):

    def __init__(self, resnet='resnet34', num_classes=2):
        super().__init__()

        cut = 7

        if resnet == 'resnet34':
            base_model = resnet34
        elif resnet == 'resnet18':
            base_model = resnet18
        elif resnet == 'resnet50':
            base_model = resnet50
        else:
            raise Exception('The Resnet Model only accept resnet18, resnet34 and resnet50')

        layers = list(base_model(pretrained=True).children())[:cut]
        base_layers = nn.Sequential(*layers)
        self.rn = base_layers



        self.conv1x1_5 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[2])]
        self.sfs.append(SaveFeatures(base_layers[4][1]))
        self.sfs.append(SaveFeatures(base_layers[5][1]))


        self.up2 = UnetBlock_tran8_1x1(256, 128,256)
        self.up3 = UnetBlock_tran8_1x1(128, 64,256)
        self.up4 = UnetBlock_tran8_1x1(64, 64,128)
        self.ec_block = expand_compress_block3(8)
        self.up5_2 = nn.Conv2d(32, self.num_classes, kernel_size=1, padding=0)
        
        self.ds_conv_1 = nn.Conv2d(32, self.num_classes, kernel_size=1, padding=0)
        self.bn_1 = nn.BatchNorm2d(self.num_classes)

        self.ds_conv_2 = nn.Conv2d(32, self.num_classes, kernel_size=1, padding=0)
        self.bn_2 = nn.BatchNorm2d(self.num_classes)
        
        self.ds_conv_3 = nn.Conv2d(32, self.num_classes, kernel_size=1, padding=0)
        self.bn_3 = nn.BatchNorm2d(self.num_classes)

        self.ds_conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.ds_conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.ds_conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.ds_conv4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.ds_conv5 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.ds_conv6 = nn.Conv2d(64, 32, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(self.num_classes)

    def forward(self, x,y,pro1,pro2,pro3):

        x = torch.cat([x, y,pro1], dim=1)
        x = self.ec_block(x)
       
        
        x = F.relu(self.rn(x))

        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)

        x_out = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x_out = self.conv1x1_5(x_out)
        x_out = self.up5_2(x_out)


        if self.num_classes==1:
            output = x_out[:, 0]
        else:
            output = x_out[:, :self.num_classes]

        ds1 = F.upsample(self.sfs[2].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = self.ds_conv1(ds1)
        ds1 = F.upsample(ds1, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = self.ds_conv2(ds1)
        ds1 = F.upsample(ds1, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = self.ds_conv3(ds1)
        ds1 = self.ds_conv_1(ds1)
        ds1 = self.bn_1(ds1)

        ds2 = F.upsample(self.sfs[1].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds2 = self.ds_conv4(ds2)
        ds2 = F.upsample(ds2, scale_factor=2, mode='bilinear', align_corners=True)
        ds2 = self.ds_conv5(ds2)
        ds2 = self.ds_conv_2(ds2)
        ds2 = self.bn_2(ds2)
        
        ds3 = F.upsample(self.sfs[0].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds3 = self.ds_conv6(ds3)
        ds3 = self.ds_conv_3(ds3)
        ds3 = self.bn_3(ds3)

        output = self.bn1(output)
        return output,ds1,ds2,ds3

    def close(self):
        for sf in self.sfs: sf.remove()  
        
        
class ResUnet_illum_tran_ds3_add_conv5_sSE_cSE_up40(nn.Module):

    def __init__(self, resnet='resnet34', num_classes=2):
        super().__init__()
        # super(ResUnet, self).__init__()
        # cut, lr_cut = [8, 6]
        cut, lr_cut = [7, 5]

        if resnet == 'resnet34':
            base_model = resnet34
        elif resnet == 'resnet18':
            base_model = resnet18
        elif resnet == 'resnet50':
            base_model = resnet50
        else:
            raise Exception('The Resnet Model only accept resnet18, resnet34 and resnet50')

        layers = list(base_model(pretrained=True).children())[:cut]
        base_layers = nn.Sequential(*layers)
        self.rn = base_layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        #self.conv2 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3,
        #                       bias=False)

        self.conv1x1 = nn.Conv2d(128, 64, kernel_size=1, padding=0)
        self.conv1x1_2 = nn.Conv2d(8, 3, kernel_size=1, padding=0)
        self.conv1x1_3 = nn.Conv2d(2, 32, kernel_size=1, padding=0)
        self.conv1x1_4 = nn.Conv2d(8, 8, kernel_size=1, padding=0)
        self.conv1x1_5 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[2])]
        self.sfs.append(SaveFeatures(base_layers[4][1]))
        self.sfs.append(SaveFeatures(base_layers[5][1]))
        #self.up1 = UnetBlock0(512, 256, 256)
        #self.up1 = UnetBlock(256, 128, 256)
        self.up2 = UnetBlock_tran8_scSE(256, 128,256)
        self.up3 = UnetBlock_tran8_scSE(256, 64,256)
        self.up4 = UnetBlock_tran8_scSE(256, 64,128)
        self.ec_block = expand_compress_block3(8)
        # self.up5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)
        # self.up5_1 = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.up5_2 = nn.Conv2d(256, self.num_classes, 1)

        #self.up5_1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up5_1 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.up5_2 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        
        self.ds_deconv_1 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv_1_ = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv2_1 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.ds_conv_1 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.bn_1 = nn.BatchNorm2d(self.num_classes)

        self.ds_deconv_2 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv_2_ = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_conv_2 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.bn_2 = nn.BatchNorm2d(self.num_classes)
        
        self.ds_deconv_3 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_conv_3 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.bn_3 = nn.BatchNorm2d(self.num_classes)

        self.ds_conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.ds_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.ds_conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.ds_conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.ds_conv5 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.ds_conv6 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(self.num_classes)

    def forward(self, x,y,pro1,pro2,pro3):
        #x = self.conv1(x)
        #y = self.conv1(y)
        x = torch.cat([x, y,pro1], dim=1)
        x = self.ec_block(x)
       
        
        x = F.relu(self.rn(x))
        # x = self.up1(x, self.sfs[2].features)
        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)
        # x = self.up4(x, self.sfs[0].features)
        # x_out = self.up5(x)
        x_out = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x_out = self.conv1x1_5(x_out)
        x_out = self.up5_2(x_out)


        if self.num_classes==1:
            output = x_out[:, 0]
        else:
            output = x_out[:, :self.num_classes]

        ds1 = F.upsample(self.sfs[2].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = self.ds_conv1(ds1)
        ds1 = F.upsample(ds1, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = self.ds_conv2(ds1)
        ds1 = F.upsample(ds1, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = self.ds_conv3(ds1)
        ds1 = self.ds_conv_1(ds1)
        ds1 = self.bn_1(ds1)

        ds2 = F.upsample(self.sfs[1].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds2 = self.ds_conv4(ds2)
        ds2 = F.upsample(ds2, scale_factor=2, mode='bilinear', align_corners=True)
        ds2 = self.ds_conv5(ds2)
        ds2 = self.ds_conv_2(ds2)
        ds2 = self.bn_2(ds2)
        
        ds3 = F.upsample(self.sfs[0].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds3 = self.ds_conv6(ds3)
        ds3 = self.ds_conv_3(ds3)
        ds3 = self.bn_3(ds3)

        output = self.bn1(output)
        return output,ds1,ds2,ds3

    def close(self):
        for sf in self.sfs: sf.remove()     
        
        
class ResUnet_illum_tran_ds3_add_conv5_sSE_cSE_up41(nn.Module):

    def __init__(self, resnet='resnet34', num_classes=2):
        super().__init__()

        cut = 7

        if resnet == 'resnet34':
            base_model = resnet34
        elif resnet == 'resnet18':
            base_model = resnet18
        elif resnet == 'resnet50':
            base_model = resnet50
        else:
            raise Exception('The Resnet Model only accept resnet18, resnet34 and resnet50')

        layers = list(base_model(pretrained=True).children())[:cut]
        base_layers = nn.Sequential(*layers)
        self.rn = base_layers



        self.conv1x1_5 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[2])]
        self.sfs.append(SaveFeatures(base_layers[4][1]))
        self.sfs.append(SaveFeatures(base_layers[5][1]))


        self.up2 = UnetBlock_tran10(256, 128)
        self.up3 = UnetBlock_tran10(128, 64)
        self.up4 = UnetBlock_tran10(64, 64)
        self.ec_block = expand_compress_block3(8)
        self.up5_2 = nn.Conv2d(32, self.num_classes, kernel_size=1, padding=0)
        
        self.ds_conv_1 = nn.Conv2d(32, self.num_classes, kernel_size=1, padding=0)
        self.bn_1 = nn.BatchNorm2d(self.num_classes)

        self.ds_conv_2 = nn.Conv2d(32, self.num_classes, kernel_size=1, padding=0)
        self.bn_2 = nn.BatchNorm2d(self.num_classes)
        
        self.ds_conv_3 = nn.Conv2d(32, self.num_classes, kernel_size=1, padding=0)
        self.bn_3 = nn.BatchNorm2d(self.num_classes)

        self.ds_conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.ds_conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.ds_conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.ds_conv4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.ds_conv5 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.ds_conv6 = nn.Conv2d(64, 32, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(self.num_classes)

    def forward(self, x,y,pro1,pro2,pro3):

        x = torch.cat([x, y,pro1], dim=1)
        x = self.ec_block(x)
       
        
        x = F.relu(self.rn(x))

        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)

        x_out = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x_out = self.conv1x1_5(x_out)
        x_out = self.up5_2(x_out)


        if self.num_classes==1:
            output = x_out[:, 0]
        else:
            output = x_out[:, :self.num_classes]

        ds1 = F.upsample(self.sfs[2].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = self.ds_conv1(ds1)
        ds1 = F.upsample(ds1, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = self.ds_conv2(ds1)
        ds1 = F.upsample(ds1, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = self.ds_conv3(ds1)
        ds1 = self.ds_conv_1(ds1)
        ds1 = self.bn_1(ds1)

        ds2 = F.upsample(self.sfs[1].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds2 = self.ds_conv4(ds2)
        ds2 = F.upsample(ds2, scale_factor=2, mode='bilinear', align_corners=True)
        ds2 = self.ds_conv5(ds2)
        ds2 = self.ds_conv_2(ds2)
        ds2 = self.bn_2(ds2)
        
        ds3 = F.upsample(self.sfs[0].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds3 = self.ds_conv6(ds3)
        ds3 = self.ds_conv_3(ds3)
        ds3 = self.bn_3(ds3)

        output = self.bn1(output)
        return output,ds1,ds2,ds3

    def close(self):
        for sf in self.sfs: sf.remove()    
        
        
        
class ResUnet_illum_tran_ds3_add_conv5_sSE_cSE_up42(nn.Module):

    def __init__(self, resnet='resnet34', num_classes=2):
        super().__init__()

        cut = 7

        if resnet == 'resnet34':
            base_model = resnet34
        elif resnet == 'resnet18':
            base_model = resnet18
        elif resnet == 'resnet50':
            base_model = resnet50
        else:
            raise Exception('The Resnet Model only accept resnet18, resnet34 and resnet50')

        layers = list(base_model(pretrained=True).children())[:cut]
        base_layers = nn.Sequential(*layers)
        self.rn = base_layers



        self.conv1x1_5 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[2])]
        self.sfs.append(SaveFeatures(base_layers[4][1]))
        self.sfs.append(SaveFeatures(base_layers[5][1]))


        self.up2 = UnetBlock_tran10(256, 128)
        self.up3 = UnetBlock_tran10(128, 64)
        self.up4 = UnetBlock_tran10(64, 64)
        self.ec_block = expand_compress_block3(8)
        self.up5_2 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        
        self.ds_conv_1 = nn.Conv2d(32, self.num_classes, kernel_size=1, padding=0)
        self.bn_1 = nn.BatchNorm2d(self.num_classes)

        self.ds_conv_2 = nn.Conv2d(32, self.num_classes, kernel_size=1, padding=0)
        self.bn_2 = nn.BatchNorm2d(self.num_classes)
        
        self.ds_conv_3 = nn.Conv2d(32, self.num_classes, kernel_size=1, padding=0)
        self.bn_3 = nn.BatchNorm2d(self.num_classes)

        self.ds_conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.ds_conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.ds_conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.ds_conv4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.ds_conv5 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.ds_conv6 = nn.Conv2d(64, 32, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(self.num_classes)

    def forward(self, x,y,pro1,pro2,pro3):

        x = torch.cat([x, y,pro1], dim=1)
        x = self.ec_block(x)
       
        
        x = F.relu(self.rn(x))

        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)

        x_out = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x_out = self.conv1x1_5(x_out)
        x_out = self.up5_2(x_out)


        if self.num_classes==1:
            output = x_out[:, 0]
        else:
            output = x_out[:, :self.num_classes]

        ds1 = F.upsample(self.sfs[2].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = self.ds_conv1(ds1)
        ds1 = F.upsample(ds1, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = self.ds_conv2(ds1)
        ds1 = F.upsample(ds1, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = self.ds_conv3(ds1)
        ds1 = self.ds_conv_1(ds1)
        ds1 = self.bn_1(ds1)

        ds2 = F.upsample(self.sfs[1].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds2 = self.ds_conv4(ds2)
        ds2 = F.upsample(ds2, scale_factor=2, mode='bilinear', align_corners=True)
        ds2 = self.ds_conv5(ds2)
        ds2 = self.ds_conv_2(ds2)
        ds2 = self.bn_2(ds2)
        
        ds3 = F.upsample(self.sfs[0].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds3 = self.ds_conv6(ds3)
        ds3 = self.ds_conv_3(ds3)
        ds3 = self.bn_3(ds3)

        output = self.bn1(output)
        return output,ds1,ds2,ds3

    def close(self):
        for sf in self.sfs: sf.remove()             
        
class ResUnet_illum_tran_ds3_add_conv5_sSE_cSE_up43(nn.Module):

    def __init__(self, resnet='resnet34', num_classes=2):
        super().__init__()

        cut = 7

        if resnet == 'resnet34':
            base_model = resnet34
        elif resnet == 'resnet18':
            base_model = resnet18
        elif resnet == 'resnet50':
            base_model = resnet50
        else:
            raise Exception('The Resnet Model only accept resnet18, resnet34 and resnet50')

        layers = list(base_model(pretrained=True).children())[:cut]
        base_layers = nn.Sequential(*layers)
        self.rn = base_layers



        self.conv1x1_5 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[2])]
        self.sfs.append(SaveFeatures(base_layers[4][1]))
        self.sfs.append(SaveFeatures(base_layers[5][1]))


        self.up2 = UnetBlock_tran10_scSE(256, 128)
        self.up3 = UnetBlock_tran10_scSE(128, 64)
        self.up4 = UnetBlock_tran10_scSE(64, 64)
        self.ec_block = expand_compress_block3(8)
        self.up5_2 = nn.Conv2d(32, self.num_classes, kernel_size=1, padding=0)
        
        self.ds_conv_1 = nn.Conv2d(32, self.num_classes, kernel_size=1, padding=0)
        self.bn_1 = nn.BatchNorm2d(self.num_classes)

        self.ds_conv_2 = nn.Conv2d(32, self.num_classes, kernel_size=1, padding=0)
        self.bn_2 = nn.BatchNorm2d(self.num_classes)
        
        self.ds_conv_3 = nn.Conv2d(32, self.num_classes, kernel_size=1, padding=0)
        self.bn_3 = nn.BatchNorm2d(self.num_classes)

        self.ds_conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.ds_conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.ds_conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.ds_conv4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.ds_conv5 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.ds_conv6 = nn.Conv2d(64, 32, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(self.num_classes)

    def forward(self, x,y,pro1,pro2,pro3):

        x = torch.cat([x, y,pro1], dim=1)
        x = self.ec_block(x)
       
        
        x = F.relu(self.rn(x))

        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)

        x_out = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x_out = self.conv1x1_5(x_out)
        x_out = self.up5_2(x_out)


        if self.num_classes==1:
            output = x_out[:, 0]
        else:
            output = x_out[:, :self.num_classes]

        ds1 = F.upsample(self.sfs[2].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = self.ds_conv1(ds1)
        ds1 = F.upsample(ds1, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = self.ds_conv2(ds1)
        ds1 = F.upsample(ds1, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = self.ds_conv3(ds1)
        ds1 = self.ds_conv_1(ds1)
        ds1 = self.bn_1(ds1)

        ds2 = F.upsample(self.sfs[1].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds2 = self.ds_conv4(ds2)
        ds2 = F.upsample(ds2, scale_factor=2, mode='bilinear', align_corners=True)
        ds2 = self.ds_conv5(ds2)
        ds2 = self.ds_conv_2(ds2)
        ds2 = self.bn_2(ds2)
        
        ds3 = F.upsample(self.sfs[0].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds3 = self.ds_conv6(ds3)
        ds3 = self.ds_conv_3(ds3)
        ds3 = self.bn_3(ds3)

        output = self.bn1(output)
        return output,ds1,ds2,ds3

    def close(self):
        for sf in self.sfs: sf.remove()        
        
        
class ResUnet_illum_tran_ds3_add_conv5_sSE_cSE_up44(nn.Module):

    def __init__(self, resnet='resnet34', num_classes=2):
        super().__init__()

        cut = 7

        if resnet == 'resnet34':
            base_model = resnet34
        elif resnet == 'resnet18':
            base_model = resnet18
        elif resnet == 'resnet50':
            base_model = resnet50
        else:
            raise Exception('The Resnet Model only accept resnet18, resnet34 and resnet50')

        layers = list(base_model(pretrained=True).children())[:cut]
        base_layers = nn.Sequential(*layers)
        self.rn = base_layers



        self.conv1x1_5 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[2])]
        self.sfs.append(SaveFeatures(base_layers[4][1]))
        self.sfs.append(SaveFeatures(base_layers[5][1]))


        self.up2 = UnetBlock_tran10_bn(256, 128)
        self.up3 = UnetBlock_tran10_bn(128, 64)
        self.up4 = UnetBlock_tran10_bn(64, 64)
        self.ec_block = expand_compress_block3(8)
        self.up5_2 = nn.Conv2d(32, self.num_classes, kernel_size=1, padding=0)
        
        self.ds_conv_1 = nn.Conv2d(32, self.num_classes, kernel_size=1, padding=0)
        self.bn_1 = nn.BatchNorm2d(self.num_classes)

        self.ds_conv_2 = nn.Conv2d(32, self.num_classes, kernel_size=1, padding=0)
        self.bn_2 = nn.BatchNorm2d(self.num_classes)
        
        self.ds_conv_3 = nn.Conv2d(32, self.num_classes, kernel_size=1, padding=0)
        self.bn_3 = nn.BatchNorm2d(self.num_classes)

        self.ds_conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.ds_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.ds_conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.ds_conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.ds_conv5 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.ds_conv6 = nn.Conv2d(64, 32, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(self.num_classes)      
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(32)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm2d(32)
        self.bn7 = nn.BatchNorm2d(32)

    def forward(self, x,y,pro1,pro2,pro3):

        x = torch.cat([x, y,pro1], dim=1)
        x = self.ec_block(x)
       
        
        x = F.relu(self.rn(x))

        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)

        x_out = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x_out = self.conv1x1_5(x_out)
        x_out = self.up5_2(x_out)


        if self.num_classes==1:
            output = x_out[:, 0]
        else:
            output = x_out[:, :self.num_classes]

        ds1 = F.upsample(self.sfs[2].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = F.relu(self.bn2(self.ds_conv1(ds1)))
        ds1 = F.upsample(ds1, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = F.relu(self.bn3(self.ds_conv2(ds1)))
        ds1 = F.upsample(ds1, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = F.relu(self.bn4(self.ds_conv3(ds1)))
        ds1 = self.ds_conv_1(ds1)
        ds1 = self.bn_1(ds1)

        ds2 = F.upsample(self.sfs[1].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds2 = F.relu(self.bn5(self.ds_conv4(ds2)))
        ds2 = F.upsample(ds2, scale_factor=2, mode='bilinear', align_corners=True)
        ds2 = F.relu(self.bn6(self.ds_conv5(ds2)))
        ds2 = self.ds_conv_2(ds2)
        ds2 = self.bn_2(ds2)
        
        ds3 = F.upsample(self.sfs[0].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds3 = F.relu(self.bn7(self.ds_conv6(ds3)))
        ds3 = self.ds_conv_3(ds3)
        ds3 = self.bn_3(ds3)

        output = self.bn1(output)
        return output,ds1,ds2,ds3

    def close(self):
        for sf in self.sfs: sf.remove()        
        
        
class ResUnet_illum_tran_ds3_add_conv5_sSE_cSE_up45(nn.Module):

    def __init__(self, resnet='resnet34', num_classes=2):
        super().__init__()

        cut = 7

        if resnet == 'resnet34':
            base_model = resnet34
        elif resnet == 'resnet18':
            base_model = resnet18
        elif resnet == 'resnet50':
            base_model = resnet50
        else:
            raise Exception('The Resnet Model only accept resnet18, resnet34 and resnet50')

        layers = list(base_model(pretrained=True).children())[:cut]
        base_layers = nn.Sequential(*layers)
        self.rn = base_layers



        self.conv1x1_5 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[2])]
        self.sfs.append(SaveFeatures(base_layers[4][1]))
        self.sfs.append(SaveFeatures(base_layers[5][1]))


        self.up2 = UnetBlock_tran10_bn_dp(256, 128)
        self.up3 = UnetBlock_tran10_bn_dp(128, 64)
        self.up4 = UnetBlock_tran10_bn_dp(64, 64)
        self.ec_block = expand_compress_block3(8)
        self.up5_2 = nn.Conv2d(32, self.num_classes, kernel_size=1, padding=0)
        
        self.ds_conv_1 = nn.Conv2d(32, self.num_classes, kernel_size=1, padding=0)
        self.bn_1 = nn.BatchNorm2d(self.num_classes)

        self.ds_conv_2 = nn.Conv2d(32, self.num_classes, kernel_size=1, padding=0)
        self.bn_2 = nn.BatchNorm2d(self.num_classes)
        
        self.ds_conv_3 = nn.Conv2d(32, self.num_classes, kernel_size=1, padding=0)
        self.bn_3 = nn.BatchNorm2d(self.num_classes)

        self.ds_conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.ds_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.ds_conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.ds_conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.ds_conv5 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.ds_conv6 = nn.Conv2d(64, 32, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(self.num_classes)      
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(32)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm2d(32)
        self.bn7 = nn.BatchNorm2d(32)

    def forward(self, x,y,pro1,pro2,pro3):

        x = torch.cat([x, y,pro1], dim=1)
        x = self.ec_block(x)
       
        
        x = F.relu(self.rn(x))

        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)

        x_out = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x_out = self.conv1x1_5(x_out)
        x_out = self.up5_2(x_out)


        if self.num_classes==1:
            output = x_out[:, 0]
        else:
            output = x_out[:, :self.num_classes]

        ds1 = F.upsample(self.sfs[2].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = F.relu(self.bn2(self.ds_conv1(ds1)))
        ds1 = F.upsample(ds1, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = F.relu(self.bn3(self.ds_conv2(ds1)))
        ds1 = F.upsample(ds1, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = F.relu(self.bn4(self.ds_conv3(ds1)))
        ds1 = self.ds_conv_1(ds1)
        ds1 = self.bn_1(ds1)

        ds2 = F.upsample(self.sfs[1].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds2 = F.relu(self.bn5(self.ds_conv4(ds2)))
        ds2 = F.upsample(ds2, scale_factor=2, mode='bilinear', align_corners=True)
        ds2 = F.relu(self.bn6(self.ds_conv5(ds2)))
        ds2 = self.ds_conv_2(ds2)
        ds2 = self.bn_2(ds2)
        
        ds3 = F.upsample(self.sfs[0].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds3 = F.relu(self.bn7(self.ds_conv6(ds3)))
        ds3 = self.ds_conv_3(ds3)
        ds3 = self.bn_3(ds3)

        output = self.bn1(output)
        return output,ds1,ds2,ds3

    def close(self):
        for sf in self.sfs: sf.remove()    
        
        
class ResUnet_illum_tran_ds3_add_conv5_sSE_cSE_up46(nn.Module):

    def __init__(self, resnet='resnet34', num_classes=2):
        super().__init__()

        cut = 7

        if resnet == 'resnet34':
            base_model = resnet34
        elif resnet == 'resnet18':
            base_model = resnet18
        elif resnet == 'resnet50':
            base_model = resnet50
        else:
            raise Exception('The Resnet Model only accept resnet18, resnet34 and resnet50')

        layers = list(base_model(pretrained=True).children())[:cut]
        base_layers = nn.Sequential(*layers)
        self.rn = base_layers



        self.conv1x1_5 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[2])]
        self.sfs.append(SaveFeatures(base_layers[4][1]))
        self.sfs.append(SaveFeatures(base_layers[5][1]))


        self.up2 = UnetBlock_tran10_bn(256, 128)
        self.up3 = UnetBlock_tran10_bn(128, 64)
        self.up4 = UnetBlock_tran10_bn(64, 64)
        self.ec_block = expand_compress_block3(8)
        self.up5_2 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        
        self.ds_conv_1 = nn.Conv2d(32, self.num_classes, kernel_size=1, padding=0)
        self.bn_1 = nn.BatchNorm2d(self.num_classes)

        self.ds_conv_2 = nn.Conv2d(32, self.num_classes, kernel_size=1, padding=0)
        self.bn_2 = nn.BatchNorm2d(self.num_classes)
        
        self.ds_conv_3 = nn.Conv2d(32, self.num_classes, kernel_size=1, padding=0)
        self.bn_3 = nn.BatchNorm2d(self.num_classes)

        self.ds_conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.ds_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.ds_conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.ds_conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.ds_conv5 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.ds_conv6 = nn.Conv2d(64, 32, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(self.num_classes)      
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(32)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm2d(32)
        self.bn7 = nn.BatchNorm2d(32)

    def forward(self, x,y,pro1,pro2,pro3):

        x = torch.cat([x, y,pro1], dim=1)
        x = self.ec_block(x)
       
        
        x = F.relu(self.rn(x))

        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)

        x_out = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x_out = self.conv1x1_5(x_out)
        x_out = self.up5_2(x_out)


        if self.num_classes==1:
            output = x_out[:, 0]
        else:
            output = x_out[:, :self.num_classes]

        ds1 = F.upsample(self.sfs[2].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = F.relu(self.bn2(self.ds_conv1(ds1)))
        ds1 = F.upsample(ds1, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = F.relu(self.bn3(self.ds_conv2(ds1)))
        ds1 = F.upsample(ds1, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = F.relu(self.bn4(self.ds_conv3(ds1)))
        ds1 = self.ds_conv_1(ds1)
        ds1 = self.bn_1(ds1)

        ds2 = F.upsample(self.sfs[1].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds2 = F.relu(self.bn5(self.ds_conv4(ds2)))
        ds2 = F.upsample(ds2, scale_factor=2, mode='bilinear', align_corners=True)
        ds2 = F.relu(self.bn6(self.ds_conv5(ds2)))
        ds2 = self.ds_conv_2(ds2)
        ds2 = self.bn_2(ds2)
        
        ds3 = F.upsample(self.sfs[0].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds3 = F.relu(self.bn7(self.ds_conv6(ds3)))
        ds3 = self.ds_conv_3(ds3)
        ds3 = self.bn_3(ds3)

        output = self.bn1(output)
        return output,ds1,ds2,ds3

    def close(self):
        for sf in self.sfs: sf.remove()   
        
class ResUnet_illum_tran_ds3_add_conv5_sSE_cSE_up47(nn.Module):

    def __init__(self, resnet='resnet34', num_classes=2):
        super().__init__()

        cut = 7

        if resnet == 'resnet34':
            base_model = resnet34
        elif resnet == 'resnet18':
            base_model = resnet18
        elif resnet == 'resnet50':
            base_model = resnet50
        else:
            raise Exception('The Resnet Model only accept resnet18, resnet34 and resnet50')

        layers = list(base_model(pretrained=True).children())[:cut]
        base_layers = nn.Sequential(*layers)
        self.rn = base_layers



        self.conv1x1_5 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[2])]
        self.sfs.append(SaveFeatures(base_layers[4][1]))
        self.sfs.append(SaveFeatures(base_layers[5][1]))


        self.up2 = UnetBlock_tran10_bn_dp(256, 128)
        self.up3 = UnetBlock_tran10_bn_dp(128, 64)
        self.up4 = UnetBlock_tran10_bn_dp(64, 64)
        self.ec_block = expand_compress_block3(8)
        self.up5_2 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        
        self.ds_conv_1 = nn.Conv2d(32, self.num_classes, kernel_size=1, padding=0)
        self.bn_1 = nn.BatchNorm2d(self.num_classes)

        self.ds_conv_2 = nn.Conv2d(32, self.num_classes, kernel_size=1, padding=0)
        self.bn_2 = nn.BatchNorm2d(self.num_classes)
        
        self.ds_conv_3 = nn.Conv2d(32, self.num_classes, kernel_size=1, padding=0)
        self.bn_3 = nn.BatchNorm2d(self.num_classes)

        self.ds_conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.ds_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.ds_conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.ds_conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.ds_conv5 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.ds_conv6 = nn.Conv2d(64, 32, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(self.num_classes)      
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(32)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm2d(32)
        self.bn7 = nn.BatchNorm2d(32)

    def forward(self, x,y,pro1,pro2,pro3):

        x = torch.cat([x, y,pro1], dim=1)
        x = self.ec_block(x)
       
        
        x = F.relu(self.rn(x))

        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)

        x_out = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x_out = self.conv1x1_5(x_out)
        x_out = self.up5_2(x_out)


        if self.num_classes==1:
            output = x_out[:, 0]
        else:
            output = x_out[:, :self.num_classes]

        ds1 = F.upsample(self.sfs[2].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = F.relu(self.bn2(self.ds_conv1(ds1)))
        ds1 = F.upsample(ds1, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = F.relu(self.bn3(self.ds_conv2(ds1)))
        ds1 = F.upsample(ds1, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = F.relu(self.bn4(self.ds_conv3(ds1)))
        ds1 = self.ds_conv_1(ds1)
        ds1 = self.bn_1(ds1)

        ds2 = F.upsample(self.sfs[1].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds2 = F.relu(self.bn5(self.ds_conv4(ds2)))
        ds2 = F.upsample(ds2, scale_factor=2, mode='bilinear', align_corners=True)
        ds2 = F.relu(self.bn6(self.ds_conv5(ds2)))
        ds2 = self.ds_conv_2(ds2)
        ds2 = self.bn_2(ds2)
        
        ds3 = F.upsample(self.sfs[0].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds3 = F.relu(self.bn7(self.ds_conv6(ds3)))
        ds3 = self.ds_conv_3(ds3)
        ds3 = self.bn_3(ds3)

        output = self.bn1(output)
        return output,ds1,ds2,ds3

    def close(self):
        for sf in self.sfs: sf.remove()        
        
        
class ResUnet_illum_tran_ds3_add_conv5_sSE_cSE_up44_wopro(nn.Module):

    def __init__(self, resnet='resnet34', num_classes=2):
        super().__init__()

        cut = 7

        if resnet == 'resnet34':
            base_model = resnet34
        elif resnet == 'resnet18':
            base_model = resnet18
        elif resnet == 'resnet50':
            base_model = resnet50
        else:
            raise Exception('The Resnet Model only accept resnet18, resnet34 and resnet50')

        layers = list(base_model(pretrained=True).children())[:cut]
        base_layers = nn.Sequential(*layers)
        self.rn = base_layers



        self.conv1x1_5 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[2])]
        self.sfs.append(SaveFeatures(base_layers[4][1]))
        self.sfs.append(SaveFeatures(base_layers[5][1]))


        self.up2 = UnetBlock_tran10_bn(256, 128)
        self.up3 = UnetBlock_tran10_bn(128, 64)
        self.up4 = UnetBlock_tran10_bn(64, 64)
        self.ec_block = expand_compress_block3(6)
        self.up5_2 = nn.Conv2d(32, self.num_classes, kernel_size=1, padding=0)
        
        self.ds_conv_1 = nn.Conv2d(32, self.num_classes, kernel_size=1, padding=0)
        self.bn_1 = nn.BatchNorm2d(self.num_classes)

        self.ds_conv_2 = nn.Conv2d(32, self.num_classes, kernel_size=1, padding=0)
        self.bn_2 = nn.BatchNorm2d(self.num_classes)
        
        self.ds_conv_3 = nn.Conv2d(32, self.num_classes, kernel_size=1, padding=0)
        self.bn_3 = nn.BatchNorm2d(self.num_classes)

        self.ds_conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.ds_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.ds_conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.ds_conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.ds_conv5 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.ds_conv6 = nn.Conv2d(64, 32, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(self.num_classes)      
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(32)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm2d(32)
        self.bn7 = nn.BatchNorm2d(32)

    def forward(self, x,y,pro1,pro2,pro3):

        x = torch.cat([x, y], dim=1)
        x = self.ec_block(x)
       
        
        x = F.relu(self.rn(x))

        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)

        x_out = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x_out = self.conv1x1_5(x_out)
        x_out = self.up5_2(x_out)


        if self.num_classes==1:
            output = x_out[:, 0]
        else:
            output = x_out[:, :self.num_classes]

        ds1 = F.upsample(self.sfs[2].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = F.relu(self.bn2(self.ds_conv1(ds1)))
        ds1 = F.upsample(ds1, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = F.relu(self.bn3(self.ds_conv2(ds1)))
        ds1 = F.upsample(ds1, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = F.relu(self.bn4(self.ds_conv3(ds1)))
        ds1 = self.ds_conv_1(ds1)
        ds1 = self.bn_1(ds1)

        ds2 = F.upsample(self.sfs[1].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds2 = F.relu(self.bn5(self.ds_conv4(ds2)))
        ds2 = F.upsample(ds2, scale_factor=2, mode='bilinear', align_corners=True)
        ds2 = F.relu(self.bn6(self.ds_conv5(ds2)))
        ds2 = self.ds_conv_2(ds2)
        ds2 = self.bn_2(ds2)
        
        ds3 = F.upsample(self.sfs[0].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds3 = F.relu(self.bn7(self.ds_conv6(ds3)))
        ds3 = self.ds_conv_3(ds3)
        ds3 = self.bn_3(ds3)

        output = self.bn1(output)
        return output,ds1,ds2,ds3

    def close(self):
        for sf in self.sfs: sf.remove()   

class ResUnet_illum_tran_ds3_add_conv5_sSE_cSE_up44_woillum(nn.Module):

    def __init__(self, resnet='resnet34', num_classes=2):
        super().__init__()

        cut = 7

        if resnet == 'resnet34':
            base_model = resnet34
        elif resnet == 'resnet18':
            base_model = resnet18
        elif resnet == 'resnet50':
            base_model = resnet50
        else:
            raise Exception('The Resnet Model only accept resnet18, resnet34 and resnet50')

        layers = list(base_model(pretrained=True).children())[:cut]
        base_layers = nn.Sequential(*layers)
        self.rn = base_layers



        self.conv1x1_5 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[2])]
        self.sfs.append(SaveFeatures(base_layers[4][1]))
        self.sfs.append(SaveFeatures(base_layers[5][1]))


        self.up2 = UnetBlock_tran10_bn(256, 128)
        self.up3 = UnetBlock_tran10_bn(128, 64)
        self.up4 = UnetBlock_tran10_bn(64, 64)
        self.ec_block = expand_compress_block3(5)
        self.up5_2 = nn.Conv2d(32, self.num_classes, kernel_size=1, padding=0)
        
        self.ds_conv_1 = nn.Conv2d(32, self.num_classes, kernel_size=1, padding=0)
        self.bn_1 = nn.BatchNorm2d(self.num_classes)

        self.ds_conv_2 = nn.Conv2d(32, self.num_classes, kernel_size=1, padding=0)
        self.bn_2 = nn.BatchNorm2d(self.num_classes)
        
        self.ds_conv_3 = nn.Conv2d(32, self.num_classes, kernel_size=1, padding=0)
        self.bn_3 = nn.BatchNorm2d(self.num_classes)

        self.ds_conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.ds_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.ds_conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.ds_conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.ds_conv5 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.ds_conv6 = nn.Conv2d(64, 32, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(self.num_classes)      
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(32)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm2d(32)
        self.bn7 = nn.BatchNorm2d(32)

    def forward(self, x,y,pro1,pro2,pro3):

        x = torch.cat([x, pro1], dim=1)
        x = self.ec_block(x)
       
        
        x = F.relu(self.rn(x))

        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)

        x_out = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x_out = self.conv1x1_5(x_out)
        x_out = self.up5_2(x_out)


        if self.num_classes==1:
            output = x_out[:, 0]
        else:
            output = x_out[:, :self.num_classes]

        ds1 = F.upsample(self.sfs[2].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = F.relu(self.bn2(self.ds_conv1(ds1)))
        ds1 = F.upsample(ds1, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = F.relu(self.bn3(self.ds_conv2(ds1)))
        ds1 = F.upsample(ds1, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = F.relu(self.bn4(self.ds_conv3(ds1)))
        ds1 = self.ds_conv_1(ds1)
        ds1 = self.bn_1(ds1)

        ds2 = F.upsample(self.sfs[1].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds2 = F.relu(self.bn5(self.ds_conv4(ds2)))
        ds2 = F.upsample(ds2, scale_factor=2, mode='bilinear', align_corners=True)
        ds2 = F.relu(self.bn6(self.ds_conv5(ds2)))
        ds2 = self.ds_conv_2(ds2)
        ds2 = self.bn_2(ds2)
        
        ds3 = F.upsample(self.sfs[0].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds3 = F.relu(self.bn7(self.ds_conv6(ds3)))
        ds3 = self.ds_conv_3(ds3)
        ds3 = self.bn_3(ds3)

        output = self.bn1(output)
        return output,ds1,ds2,ds3

    def close(self):
        for sf in self.sfs: sf.remove()         
        